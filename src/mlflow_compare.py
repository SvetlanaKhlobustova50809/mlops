"""Сравнение прогонов эксперимента через MLflow Tracking API.

Показывает, как читать результаты экспериментов программно -- так строятся
автоматические отчёты и гейты "новая модель лучше текущей".

Запуск: python src/mlflow_compare.py
"""
from __future__ import annotations

import mlflow
import pandas as pd

from config import PARAMS


def main() -> None:
    cfg = PARAMS["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])

    exp = mlflow.get_experiment_by_name(cfg["experiment_name"])
    if exp is None:
        raise SystemExit(f"эксперимент '{cfg['experiment_name']}' не найден -- "
                         f"сначала выполните: python src/train.py")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.test_roc_auc DESC"],
    )
    if runs.empty:
        raise SystemExit("в эксперименте нет прогонов")

    cols = {
        "tags.mlflow.runName": "run",
        "params.model_type": "модель",
        "metrics.cv_roc_auc_mean": "CV ROC-AUC",
        "metrics.test_roc_auc": "test ROC-AUC",
        "metrics.test_pr_auc": "test PR-AUC",
        "metrics.test_f1": "test F1",
        "metrics.overfit_gap_roc_auc": "overfit gap",
        "tags.dataset_sha256": "data sha256",
    }
    available = {k: v for k, v in cols.items() if k in runs.columns}
    table = runs[list(available)].rename(columns=available).copy()
    if "data sha256" in table.columns:
        table["data sha256"] = table["data sha256"].str.slice(0, 10)
    num = table.select_dtypes("number").columns
    table[num] = table[num].round(4)

    pd.set_option("display.width", 200)
    print(f"\nЭксперимент: {cfg['experiment_name']} "
          f"(experiment_id={exp.experiment_id}), прогонов: {len(runs)}\n")
    print(table.to_string(index=False))

    best = runs.iloc[0]
    print(f"\nЛучший прогон: {best.get('tags.mlflow.runName')} "
          f"(run_id={best['run_id']}), test ROC-AUC = "
          f"{best['metrics.test_roc_auc']:.4f}")

    # Model Registry: какие версии зарегистрированы и в какой стадии
    client = mlflow.MlflowClient()
    name = cfg["registered_model_name"]
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except Exception:
        versions = []
    if versions:
        print(f"\nMLflow Model Registry -- модель '{name}':")
        for v in sorted(versions, key=lambda x: int(x.version)):
            aliases = ", ".join(v.aliases) if v.aliases else "-"
            print(f"  version {v.version}: aliases=[{aliases}], "
                  f"run_id={v.run_id}, status={v.status}")
            if v.tags:
                print(f"               tags: {dict(v.tags)}")
        newest = max(versions, key=lambda x: int(x.version)).version
        print("\nПродвижение версии в production (современный API -- алиасы):")
        print(f"  client.set_registered_model_alias('{name}', 'production', "
              f"'{newest}')")
        print(f"  сервис забирает модель по алиасу: "
              f"MLFLOW_MODEL_URI=models:/{name}@production")
    else:
        print(f"\nв Model Registry нет модели '{name}' -- "
              f"зарегистрировать: python src/train.py --register")


if __name__ == "__main__":
    main()
