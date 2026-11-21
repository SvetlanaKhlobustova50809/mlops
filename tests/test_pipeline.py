"""Тесты, которые гоняет CI: код, данные, схема, пайплайн.

Разделены по тому, ЧТО проверяют:
  * test_data_*   -- качество и схема данных (data validation в CI);
  * test_model_*  -- сериализованная модель: контракт, детерминизм, качество;
  * test_api_*    -- работоспособность сервиса и валидация входов;
  * test_repro_*  -- наличие артефактов воспроизводимости.

Запуск: pytest -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN = ROOT / "data" / "processed" / "train.csv"
TEST = ROOT / "data" / "processed" / "test.csv"
MODEL = ROOT / "models" / "model.joblib"
METRICS = ROOT / "metrics" / "metrics.json"
CARD = ROOT / "models" / "model_card.json"

pytestmark = pytest.mark.skipif(
    not (TRAIN.exists() and MODEL.exists()),
    reason="сначала выполните: make all (download -> prepare -> train)",
)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    return pd.read_csv(TEST)


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL)


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(METRICS.read_text(encoding="utf-8"))


# ------------------------------- данные ------------------------------------ #
def test_data_no_leakage_between_splits():
    """Пересечение train и test по customerID = data leakage."""
    tr = pd.read_csv(TRAIN)
    te = pd.read_csv(TEST)
    assert set(tr["customerID"]) & set(te["customerID"]) == set()


def test_data_target_is_binary(test_df):
    assert set(test_df["Churn"].unique()) <= {0, 1}


def test_data_no_nulls_in_features(test_df):
    assert test_df.drop(columns=["customerID"]).isna().sum().sum() == 0


def test_data_stratification_preserved():
    """Доля класса в train и test должна совпадать -- иначе split сломан."""
    tr = pd.read_csv(TRAIN)["Churn"].mean()
    te = pd.read_csv(TEST)["Churn"].mean()
    assert abs(tr - te) < 0.01


def test_data_schema_matches_training_contract(model, test_df):
    """Схема данных == то, что ждёт обученный ColumnTransformer."""
    expected = set(model.named_steps["preprocess"].feature_names_in_)
    actual = set(test_df.drop(columns=["Churn", "customerID"]).columns)
    assert expected == actual


# -------------------------------- модель ----------------------------------- #
def test_model_predicts_valid_probabilities(model, test_df):
    proba = model.predict_proba(test_df.drop(columns=["Churn", "customerID"]))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert len(proba) == len(test_df)


def test_model_is_deterministic(model, test_df):
    """Два вызова на одном входе -> идентичный результат (воспроизводимость)."""
    X = test_df.drop(columns=["Churn", "customerID"]).head(100)
    assert (model.predict_proba(X) == model.predict_proba(X)).all()


def test_model_handles_unknown_category(model, test_df):
    """handle_unknown='ignore': новая категория в production не роняет сервис."""
    X = test_df.drop(columns=["Churn", "customerID"]).head(1).copy()
    X.loc[:, "PaymentMethod"] = "Crypto wallet"  # категории не было в train
    proba = model.predict_proba(X)[:, 1]
    assert 0 <= float(proba[0]) <= 1


def test_model_meets_quality_gate(metrics):
    """Порог качества из params.yaml -- тот же, что в src/evaluate.py."""
    import yaml

    min_auc = yaml.safe_load((ROOT / "params.yaml").read_text())["evaluate"]["min_roc_auc"]
    assert metrics["test_roc_auc"] >= min_auc


def test_model_not_overfitted(metrics):
    assert metrics["overfit_gap_roc_auc"] < 0.10


def test_model_beats_trivial_baseline(metrics):
    """Модель должна быть осмысленно лучше случайного угадывания."""
    assert metrics["test_roc_auc"] > 0.70
    assert metrics["test_pr_auc"] > 0.40


# --------------------------------- API ------------------------------------- #
@pytest.fixture(scope="module")
def client():
    import serve

    with TestClient(serve.app) as c:
        yield c


def test_api_health(client):
    assert client.get("/health").status_code == 200


def test_api_ready(client):
    assert client.get("/ready").status_code == 200


def test_api_model_info_exposes_provenance(client):
    """Сервис обязан отдавать происхождение модели: run_id, датасет, хеш данных."""
    body = client.get("/model").json()
    for field in ("run_id", "dataset", "dataset_sha256", "model_type", "metrics"):
        assert body.get(field), f"нет поля {field} в /model"


def test_api_predict_single(client, test_df):
    row = test_df.drop(columns=["Churn", "customerID"]).iloc[0].to_dict()
    r = client.post("/predict", json={"data": row})
    assert r.status_code == 200, r.text
    pred = r.json()["predictions"][0]
    assert 0 <= pred["churn_probability"] <= 1
    assert pred["churn"] in (0, 1)
    assert pred["risk_band"] in ("low", "medium", "high")


def test_api_predict_batch(client, test_df):
    rows = test_df.drop(columns=["Churn", "customerID"]).head(5).to_dict("records")
    r = client.post("/predict", json={"data": rows})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 5


def test_api_rejects_invalid_schema(client):
    """Валидация входов: битый тип -> 422, модель не вызывается."""
    r = client.post("/predict", json={"data": {"gender": "Female", "tenure": "abc"}})
    assert r.status_code == 422


def test_api_metrics_endpoint_is_prometheus_format(client):
    text = client.get("/metrics").text
    assert "churn_requests_total" in text
    assert "# TYPE" in text


# --------------------------- воспроизводимость ------------------------------ #
def test_repro_model_card_is_complete():
    """Model card -- минимальный набор для воспроизведения эксперимента."""
    card = json.loads(CARD.read_text(encoding="utf-8"))
    assert card["run_id"]
    assert card["dataset"]["sha256"]
    assert card["dataset"]["kaggle_ref"]
    assert card["hyperparameters"]
    assert card["env"]["sklearn"]
    assert "git_commit" in card["git"]


def test_repro_requirements_are_pinned():
    """Все зависимости рантайма закреплены на точную версию (==)."""
    lines = [
        line.strip()
        for line in (ROOT / "requirements-serve.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    for line in lines:
        assert "==" in line, f"версия не зафиксирована: {line}"
