"""Этап 2 жизненного цикла: подготовка данных + data validation + семплирование.

Что делается:
  * загрузка сырого CSV;
  * data validation -- проверка схемы, типов и допустимых значений;
  * очистка известных дефектов датасета (TotalCharges приходит строкой с пробелами);
  * стратифицированное семплирование на train/test;
  * сохранение processed-выборок и отчёта о валидации.

Защита от data leakage: split делается ДО любого fit-преобразования,
а все трансформации (impute/scale/one-hot) лежат внутри sklearn Pipeline и
обучаются только на train внутри train.py.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config import PARAMS, PROCESSED_DIR, RAW_DIR, REPORTS_DIR

RAW_CSV = RAW_DIR / "telco_churn.csv"

# Ожидаемая схема данных: имя -> тип. Нарушение схемы = падение пайплайна в CI.
EXPECTED_SCHEMA = {
    "customerID": "object",
    "gender": "object",
    "SeniorCitizen": "int64",
    "Partner": "object",
    "Dependents": "object",
    "tenure": "int64",
    "PhoneService": "object",
    "MultipleLines": "object",
    "InternetService": "object",
    "OnlineSecurity": "object",
    "OnlineBackup": "object",
    "DeviceProtection": "object",
    "TechSupport": "object",
    "StreamingTV": "object",
    "StreamingMovies": "object",
    "Contract": "object",
    "PaperlessBilling": "object",
    "PaymentMethod": "object",
    "MonthlyCharges": "float64",
    "TotalCharges": "object",  # дефект источника: строка вместо float
    "Churn": "object",
}

NUMERIC_RANGES = {
    "tenure": (0, 100),
    "MonthlyCharges": (0.0, 1000.0),
    "TotalCharges": (0.0, 100000.0),
    "SeniorCitizen": (0, 1),
}


def validate_raw(df: pd.DataFrame) -> dict:
    """Data validation: схема, пропуски, дубликаты, диапазоны, баланс классов."""
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = sorted(set(EXPECTED_SCHEMA) - set(df.columns))
    extra_cols = sorted(set(df.columns) - set(EXPECTED_SCHEMA))
    if missing_cols:
        errors.append(f"отсутствуют обязательные колонки: {missing_cols}")
    if extra_cols:
        warnings.append(f"неожидаемые колонки (schema drift): {extra_cols}")

    for col, expected in EXPECTED_SCHEMA.items():
        if col in df.columns and str(df[col].dtype) != expected:
            warnings.append(f"{col}: тип {df[col].dtype}, ожидался {expected}")

    target = PARAMS["dataset"]["target"]
    id_col = PARAMS["dataset"]["id_col"]

    if target in df.columns:
        bad_labels = set(df[target].dropna().unique()) - {"Yes", "No"}
        if bad_labels:
            errors.append(f"недопустимые значения таргета: {sorted(bad_labels)}")
        if df[target].isna().any():
            errors.append("есть пропуски в таргете")

    if id_col in df.columns and df[id_col].duplicated().any():
        errors.append(f"дубликаты по {id_col}: {int(df[id_col].duplicated().sum())}")

    full_dups = int(df.duplicated().sum())
    if full_dups:
        warnings.append(f"полные дубликаты строк: {full_dups}")

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "errors": errors,
        "warnings": warnings,
        "null_counts": {k: int(v) for k, v in df.isna().sum().items() if v},
        "target_distribution": (
            df[target].value_counts(normalize=True).round(4).to_dict()
            if target in df.columns else {}
        ),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges: 11 строк с пробелом вместо числа (клиенты с tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    target = PARAMS["dataset"]["target"]
    df[target] = (df[target] == "Yes").astype(int)
    return df


def validate_post_clean(df: pd.DataFrame) -> list[str]:
    """Проверка допустимых диапазонов после приведения типов."""
    errors = []
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        out = int(((s < lo) | (s > hi)).sum())
        if out:
            errors.append(f"{col}: {out} значений вне диапазона [{lo}, {hi}]")
    return errors


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} не найден. Сначала выполните: python src/download_data.py"
        )

    df = pd.read_csv(RAW_CSV)
    report = validate_raw(df)

    df = clean(df)
    report["errors_post_clean"] = validate_post_clean(df)
    report["errors"] += report["errors_post_clean"]

    report_path = REPORTS_DIR / "data_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for w in report["warnings"]:
        print(f"[validate][WARN] {w}")
    if report["errors"]:
        for e in report["errors"]:
            print(f"[validate][ERROR] {e}")
        raise SystemExit("data validation не пройдена -- пайплайн остановлен")
    print(f"[validate] OK: {report['n_rows']} строк, {report['n_cols']} колонок")
    print(f"[validate] баланс таргета: {report['target_distribution']}")

    p = PARAMS["prepare"]
    target = PARAMS["dataset"]["target"]
    # Стратифицированное семплирование: доля класса Churn=1 (~26.5%) сохраняется
    # и в train, и в test -> метрики на тесте несмещённые.
    strat = df[target] if p["stratify"] else None
    train_df, test_df = train_test_split(
        df,
        test_size=p["test_size"],
        random_state=p["random_state"],
        stratify=strat,
    )

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[prepare] train: {len(train_df)} строк "
          f"(churn={train_df[target].mean():.4f}) -> {train_path.name}")
    print(f"[prepare] test:  {len(test_df)} строк "
          f"(churn={test_df[target].mean():.4f}) -> {test_path.name}")


if __name__ == "__main__":
    main()
