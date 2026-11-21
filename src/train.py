"""Этапы 3-4 жизненного цикла: экспериментирование и тренировка модели
с полным трекингом в MLflow.

Что логируется в MLflow Tracking и почему это полный набор для
воспроизводимости эксперимента:
  params   -- все гиперпараметры из params.yaml + тип модели + состав признаков;
  metrics  -- roc_auc / pr_auc / f1 / precision / recall / accuracy на train, CV и test;
  tags     -- git commit и dirty-флаг, sha256 обучающих данных, версии библиотек,
              имя датасета Kaggle, автор, стадия;
  artifacts-- params.yaml, dataset.meta.json, отчёт валидации данных, графики,
              classification report, requirements.txt (lock зависимостей);
  model    -- сама модель через mlflow.sklearn.log_model с signature и input_example,
              плюс (опционально) регистрация в MLflow Model Registry.

Запуск:
    python src/train.py
    python src/train.py --model-type random_forest --register
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import METRICS_DIR, MODELS_DIR, PARAMS, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ROOT

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


# --------------------------------------------------------------------------- #
# Метаданные окружения -- без них эксперимент не воспроизводим
# --------------------------------------------------------------------------- #
def git_info() -> dict[str, str]:
    def _run(*args: str) -> str:
        try:
            return subprocess.check_output(
                args, cwd=ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return "unknown"

    commit = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain")
    return {
        "git_commit": commit,
        "git_branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": "true" if status and status != "unknown" else "false",
    }


def dataset_meta() -> dict:
    path = RAW_DIR / "dataset.meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# --------------------------------------------------------------------------- #
# Модель
# --------------------------------------------------------------------------- #
def build_estimator(model_type: str, cfg: dict, random_state: int):
    if model_type == "logistic_regression":
        return LogisticRegression(random_state=random_state, **cfg["logistic_regression"])
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1,
                                      **cfg["random_forest"])
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state,
                                          **cfg["gradient_boosting"])
    raise ValueError(f"неизвестный model_type: {model_type}")


def build_pipeline(model_type: str, cat_features: list[str], num_features: list[str],
                   cfg: dict, random_state: int) -> Pipeline:
    """Весь препроцессинг внутри Pipeline -> fit только на train, никакой утечки.

    Это же гарантирует, что в production применяются ровно те же преобразования,
    что при обучении (нет training/serving skew).
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        [("num", numeric, num_features), ("cat", categorical, cat_features)],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", build_estimator(model_type, cfg, random_state)),
    ])


def compute_metrics(y_true, y_pred, y_proba, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_roc_auc": float(roc_auc_score(y_true, y_proba)),
        f"{prefix}_pr_auc": float(average_precision_score(y_true, y_proba)),
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_plots(pipe: Pipeline, X_test, y_test, y_proba, out_dir: Path) -> list[Path]:
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(5, 4.2))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="model")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
    ax.set_title("ROC-кривая (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    p = out_dir / "roc_curve.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax, name="model")
    ax.set_title("Precision-Recall кривая (test)")
    fig.tight_layout()
    p = out_dir / "pr_curve.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax, colorbar=False,
                                          display_labels=["no churn", "churn"])
    ax.set_title("Матрица ошибок (test)")
    fig.tight_layout()
    p = out_dir / "confusion_matrix.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    return paths


def save_feature_importance(pipe: Pipeline, X_test, y_test, out_dir: Path,
                            random_state: int) -> tuple[Path, Path]:
    """Permutation importance по исходным колонкам -- интерпретируемость модели."""
    r = permutation_importance(pipe, X_test, y_test, n_repeats=5,
                               random_state=random_state, scoring="roc_auc", n_jobs=-1)
    imp = (pd.DataFrame({"feature": X_test.columns,
                         "importance_mean": r.importances_mean,
                         "importance_std": r.importances_std})
           .sort_values("importance_mean", ascending=False))
    csv_path = out_dir / "feature_importance.csv"
    imp.to_csv(csv_path, index=False)

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("падение ROC-AUC при перемешивании признака")
    ax.set_title("Permutation importance (top-15)")
    fig.tight_layout()
    png_path = out_dir / "feature_importance.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return csv_path, png_path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение модели churn с трекингом в MLflow")
    ap.add_argument("--model-type", default=None,
                    choices=["logistic_regression", "random_forest", "gradient_boosting"],
                    help="переопределяет params.yaml:train.model_type")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--register", action="store_true",
                    help="зарегистрировать модель в MLflow Model Registry")
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()

    train_cfg = PARAMS["train"]
    mlf_cfg = PARAMS["mlflow"]
    target = PARAMS["dataset"]["target"]
    id_col = PARAMS["dataset"]["id_col"]
    model_type = args.model_type or train_cfg["model_type"]
    random_state = train_cfg["random_state"]
    threshold = PARAMS["evaluate"]["threshold"]

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    drop_cols = [target, id_col]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target]

    num_features = [c for c in NUMERIC_FEATURES if c in X_train.columns]
    cat_features = [c for c in X_train.columns if c not in num_features]

    pipe = build_pipeline(model_type, cat_features, num_features, train_cfg, random_state)

    # ------------------- MLflow: трекинг эксперимента ---------------------- #
    mlflow.set_tracking_uri(mlf_cfg["tracking_uri"])
    mlflow.set_experiment(mlf_cfg["experiment_name"])

    run_name = args.run_name or (
        f"{model_type}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        gi = git_info()
        dmeta = dataset_meta()

        # --- теги: происхождение кода, данных и окружения ---
        mlflow.set_tags({
            **gi,
            "dataset": dmeta.get("kaggle_ref", "blastchar/telco-customer-churn"),
            "dataset_sha256": dmeta.get("sha256", "unknown"),
            "dataset_source": dmeta.get("source", ""),
            "task": "binary_classification/churn",
            "stage": "experimentation",
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        })

        # --- параметры: гиперпараметры + описание данных ---
        mlflow.log_params({
            "model_type": model_type,
            "random_state": random_state,
            "threshold": threshold,
            "test_size": PARAMS["prepare"]["test_size"],
            "stratify": PARAMS["prepare"]["stratify"],
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "n_numeric_features": len(num_features),
            "n_categorical_features": len(cat_features),
            "cv_folds": args.cv_folds,
        })
        # гиперпараметры конкретного алгоритма -- с префиксом, чтобы не смешивались
        for k, v in train_cfg[model_type].items():
            mlflow.log_param(f"{model_type}__{k}", v)

        # --- кросс-валидация на train (честная оценка до взгляда на test) ---
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                             random_state=random_state)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)
        mlflow.log_metric("cv_roc_auc_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_roc_auc_std", float(cv_scores.std()))
        for i, s in enumerate(cv_scores, start=1):
            mlflow.log_metric("cv_roc_auc_fold", float(s), step=i)
        print(f"[train] CV ROC-AUC = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # --- обучение финальной модели ---
        pipe.fit(X_train, y_train)

        proba_train = pipe.predict_proba(X_train)[:, 1]
        proba_test = pipe.predict_proba(X_test)[:, 1]
        pred_train = (proba_train >= threshold).astype(int)
        pred_test = (proba_test >= threshold).astype(int)

        metrics = {
            **compute_metrics(y_train, pred_train, proba_train, "train"),
            **compute_metrics(y_test, pred_test, proba_test, "test"),
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "overfit_gap_roc_auc": float(roc_auc_score(y_train, proba_train)
                                         - roc_auc_score(y_test, proba_test)),
        }
        mlflow.log_metrics(metrics)

        print(f"[train] test ROC-AUC = {metrics['test_roc_auc']:.4f} | "
              f"PR-AUC = {metrics['test_pr_auc']:.4f} | "
              f"F1 = {metrics['test_f1']:.4f} | "
              f"accuracy = {metrics['test_accuracy']:.4f}")

        # --- артефакты: графики, отчёты, конфиги, lock зависимостей ---
        plots = save_plots(pipe, X_test, y_test, proba_test, REPORTS_DIR)
        fi_csv, fi_png = save_feature_importance(pipe, X_test, y_test, REPORTS_DIR,
                                                 random_state)

        clf_report = classification_report(y_test, pred_test, digits=4,
                                           target_names=["no churn", "churn"])
        clf_path = REPORTS_DIR / "classification_report.txt"
        clf_path.write_text(clf_report, encoding="utf-8")
        print(clf_report)

        for p in [*plots, fi_csv, fi_png, clf_path]:
            mlflow.log_artifact(str(p), artifact_path="reports")
        mlflow.log_artifact(str(ROOT / "params.yaml"), artifact_path="config")
        mlflow.log_artifact(str(ROOT / "requirements.txt"), artifact_path="config")
        if (RAW_DIR / "dataset.meta.json").exists():
            mlflow.log_artifact(str(RAW_DIR / "dataset.meta.json"), artifact_path="data")
        if (REPORTS_DIR / "data_validation.json").exists():
            mlflow.log_artifact(str(REPORTS_DIR / "data_validation.json"),
                                artifact_path="data")

        # --- модель: signature + input_example -> контракт на вход/выход ---
        signature = infer_signature(X_test, proba_test)
        log_kwargs = dict(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(3),
            pip_requirements=str(ROOT / "requirements.txt"),
        )
        if args.register:
            log_kwargs["registered_model_name"] = mlf_cfg["registered_model_name"]
        mlflow.sklearn.log_model(**log_kwargs)

        # --- локальная копия модели (её кладём в Docker-образ) ---
        model_path = MODELS_DIR / "model.joblib"
        joblib.dump(pipe, model_path)

        model_card = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_name": run_name,
            "model_type": model_type,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": dmeta,
            "git": gi,
            "env": {
                "python": platform.python_version(),
                "sklearn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "features": {"numeric": num_features, "categorical": cat_features},
            "hyperparameters": train_cfg[model_type],
            "threshold": threshold,
            "metrics": metrics,
        }
        card_path = MODELS_DIR / "model_card.json"
        card_path.write_text(json.dumps(model_card, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        mlflow.log_artifact(str(card_path), artifact_path="model_card")

        metrics_path = METRICS_DIR / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                encoding="utf-8")

        print(f"[train] модель        -> {model_path}")
        print(f"[train] model card    -> {card_path}")
        print(f"[train] метрики       -> {metrics_path}")
        print(f"[train] MLflow run_id = {run.info.run_id}")
        print(f"[train] MLflow UI: mlflow ui --backend-store-uri {mlf_cfg['tracking_uri']}")


if __name__ == "__main__":
    sys.exit(main())
