"""Этап 5 жизненного цикла: валидация модели + quality gate для CI/CD.

Автоматический тест качества, который в CI/CD решает,
можно ли продвигать модель в production.
Скрипт возвращает код 1, если модель не прошла гейт -> пайплайн падает,
и плохая модель физически не может быть задеплоена.

Проверки:
  1. test ROC-AUC >= params.yaml:evaluate.min_roc_auc  -- абсолютный порог качества;
  2. модель не хуже предыдущей на baseline_metrics.json -- защита от регрессии;
  3. разрыв train/test ROC-AUC < 0.10                   -- защита от переобучения;
  4. модель десериализуется и выдаёт корректные вероятности на smoke-выборке.
"""
from __future__ import annotations

import json
import sys

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import METRICS_DIR, MODELS_DIR, PARAMS, PROCESSED_DIR

BASELINE_PATH = METRICS_DIR / "baseline_metrics.json"
MAX_OVERFIT_GAP = 0.10
MAX_REGRESSION = 0.01  # допустимое падение ROC-AUC относительно baseline


def main() -> int:
    min_auc = PARAMS["evaluate"]["min_roc_auc"]
    metrics = json.loads((METRICS_DIR / "metrics.json").read_text(encoding="utf-8"))
    failures: list[str] = []

    test_auc = metrics["test_roc_auc"]
    print(f"[gate] test ROC-AUC = {test_auc:.4f} (порог {min_auc})")
    if test_auc < min_auc:
        failures.append(f"ROC-AUC {test_auc:.4f} ниже порога {min_auc}")

    gap = metrics["overfit_gap_roc_auc"]
    print(f"[gate] разрыв train-test ROC-AUC = {gap:.4f} (макс {MAX_OVERFIT_GAP})")
    if gap > MAX_OVERFIT_GAP:
        failures.append(f"признак переобучения: разрыв {gap:.4f} > {MAX_OVERFIT_GAP}")

    if BASELINE_PATH.exists():
        base = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        delta = test_auc - base["test_roc_auc"]
        print(f"[gate] baseline ROC-AUC = {base['test_roc_auc']:.4f}, "
              f"дельта = {delta:+.4f}")
        if delta < -MAX_REGRESSION:
            failures.append(f"регрессия качества: {delta:+.4f} относительно baseline")
    else:
        print("[gate] baseline отсутствует -- текущая модель станет baseline")

    # smoke-тест сериализованной модели: ровно то, что поедет в контейнер
    model_path = MODELS_DIR / "model.joblib"
    pipe = joblib.load(model_path)
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X = test_df.drop(columns=[PARAMS["dataset"]["target"], PARAMS["dataset"]["id_col"]])
    proba = pipe.predict_proba(X)[:, 1]
    if not ((proba >= 0).all() and (proba <= 1).all()):
        failures.append("модель выдаёт вероятности вне [0, 1]")
    reloaded_auc = roc_auc_score(test_df[PARAMS["dataset"]["target"]], proba)
    print(f"[gate] ROC-AUC перезагруженной модели = {reloaded_auc:.4f}")
    if abs(reloaded_auc - test_auc) > 1e-6:
        failures.append("метрика сериализованной модели не совпадает с залогированной")

    if failures:
        print("\n[gate] МОДЕЛЬ НЕ ПРОШЛА ВАЛИДАЦИЮ:")
        for f in failures:
            print(f"  - {f}")
        return 1

    BASELINE_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print("\n[gate] все проверки пройдены -- модель допущена к развертыванию")
    return 0


if __name__ == "__main__":
    sys.exit(main())
