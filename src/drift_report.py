"""Мониторинг после развертывания: дрейф данных, дрейф таргета, деградация
качества -- через Evidently AI.

Скрипт строит:
  * data_drift.html  -- DataDriftPreset: пофичевые тесты сдвига распределений;
  * target_drift.html-- TargetDriftPreset: сдвиг распределения таргета/прогноза;
  * data_quality.html-- DataQualityPreset: пропуски, константы, дубликаты, статистики;
  * classification.html -- ClassificationPreset: качество на свежей разметке;
  * drift_tests.json -- TestSuite с бинарным вердиктом pass/fail для CI и алертов.

Reference = train-выборка (данные, на которых модель обучалась).
Current   = свежий production-трафик. Для демонстрации его можно синтезировать
флагом --simulate-drift: сдвигаем MonthlyCharges/tenure и структуру контрактов,
т.е. воспроизводим covariate shift при неизменной зависимости P(y|X).

Запуск:
    python src/drift_report.py                        # test как current
    python src/drift_report.py --simulate-drift       # искусственный дрейф
    python src/drift_report.py --fail-on-drift        # код 1 при дрейфе -> алерт в CI
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset, DataStabilityTestPreset
from evidently.test_suite import TestSuite

from config import MODELS_DIR, PARAMS, PROCESSED_DIR, REPORTS_DIR

TARGET = PARAMS["dataset"]["target"]
ID_COL = PARAMS["dataset"]["id_col"]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def simulate_drift(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Искусственный covariate shift: клиенты стали дороже и моложе по стажу,
    выросла доля месячных контрактов. Зависимость P(churn | X) не трогаем."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["MonthlyCharges"] = out["MonthlyCharges"] * rng.normal(1.35, 0.06, len(out))
    out["tenure"] = np.clip(
        (out["tenure"] * rng.normal(0.55, 0.08, len(out))).round(), 0, None
    ).astype(int)
    out["TotalCharges"] = out["MonthlyCharges"] * out["tenure"]
    flip = rng.random(len(out)) < 0.35
    out.loc[flip, "Contract"] = "Month-to-month"
    flip2 = rng.random(len(out)) < 0.30
    out.loc[flip2, "InternetService"] = "Fiber optic"
    return out


def add_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    X = df.drop(columns=[c for c in (TARGET, ID_COL) if c in df.columns])
    out = df.copy()
    out["prediction"] = model.predict_proba(X)[:, 1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate-drift", action="store_true",
                    help="синтезировать дрейф во current-выборке")
    ap.add_argument("--out-dir", default=str(REPORTS_DIR / "monitoring"))
    ap.add_argument("--fail-on-drift", action="store_true",
                    help="вернуть код 1 при обнаружении дрейфа (для алертов и CI); "
                         "по умолчанию 0, чтобы стадия dvc repro не падала на отчёте")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODELS_DIR / "model.joblib")
    reference = pd.read_csv(PROCESSED_DIR / "train.csv").drop(columns=[ID_COL])
    current = pd.read_csv(PROCESSED_DIR / "test.csv").drop(columns=[ID_COL])
    if args.simulate_drift:
        current = simulate_drift(current)
        print("[drift] current-выборка искусственно сдвинута (covariate shift)")

    reference = add_predictions(reference, model)
    current = add_predictions(current, model)

    column_mapping = ColumnMapping(
        target=TARGET,
        prediction="prediction",
        numerical_features=NUMERIC,
        categorical_features=[c for c in reference.columns
                              if c not in NUMERIC + [TARGET, "prediction"]],
    )

    reports = {
        "data_drift.html": Report(metrics=[DataDriftPreset()]),
        "target_drift.html": Report(metrics=[TargetDriftPreset()]),
        "data_quality.html": Report(metrics=[DataQualityPreset()]),
        "classification.html": Report(metrics=[ClassificationPreset(probas_threshold=0.5)]),
    }
    for name, report in reports.items():
        report.run(reference_data=reference, current_data=current,
                   column_mapping=column_mapping)
        report.save_html(str(out_dir / name))
        print(f"[drift] отчёт -> {out_dir / name}")

    # --- TestSuite: машиночитаемый вердикт для алертов и CI ---
    suite = TestSuite(tests=[DataDriftTestPreset(), DataStabilityTestPreset()])
    suite.run(reference_data=reference, current_data=current,
              column_mapping=column_mapping)
    suite.save_html(str(out_dir / "drift_tests.html"))
    result = suite.as_dict()

    summary = {
        "all_passed": bool(result["summary"]["all_passed"]),
        "total_tests": result["summary"]["total_tests"],
        "success": result["summary"]["success_tests"],
        "failed": result["summary"]["failed_tests"],
        "simulated_drift": args.simulate_drift,
        "failed_test_names": [t["name"] + ": " + t["description"]
                              for t in result["tests"] if t["status"] != "SUCCESS"][:25],
    }
    (out_dir / "drift_tests.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[drift] тестов всего: {summary['total_tests']}, "
          f"пройдено: {summary['success']}, упало: {summary['failed']}")
    if not summary["all_passed"]:
        print("[drift] ДРЕЙФ ОБНАРУЖЕН -- требуется анализ и, возможно, переобучение:")
        for name in summary["failed_test_names"][:10]:
            print(f"  - {name}")
        # с --fail-on-drift код 1 триггерит алерт Alertmanager / автопереобучение
        return 1 if args.fail_on_drift else 0
    print("[drift] значимого дрейфа не обнаружено")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
