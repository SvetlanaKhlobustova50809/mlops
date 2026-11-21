"""Этап 6 жизненного цикла: развертывание модели как REST API (FastAPI).

Приложение упаковывается в Docker-образ и запускается в Kubernetes
как Deployment + Service (манифесты -- в папке k8s/).

Эндпоинты:
  GET  /health   -- liveness probe для Kubernetes (процесс жив);
  GET  /ready    -- readiness probe (модель загружена, можно слать трафик);
  GET  /model    -- версия модели, git-коммит, хеш данных, метрики (из model card);
  GET  /metrics  -- счётчики в формате Prometheus для мониторинга;
  POST /predict  -- предсказание по одному или списку клиентов.

Загрузка модели управляется переменными окружения:
  MODEL_PATH=/app/models/model.joblib               -- файл из образа, или
  MLFLOW_MODEL_URI=models:/telco-churn-clf/Production -- из MLflow Model Registry.
Второй вариант -- продакшн-путь: образ неизменен, версия модели берётся из реестра.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("churn-service")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
MODEL_CARD_PATH = Path(os.getenv("MODEL_CARD_PATH", ROOT / "models" / "model_card.json"))
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

app = FastAPI(
    title="Telco Churn Prediction Service",
    description="Модель оттока клиентов, обученная на датасете Kaggle Telco Customer Churn",
    version=os.getenv("SERVICE_VERSION", "1.0.0"),
)

STATE: Dict[str, Any] = {"model": None, "card": {}, "source": None}

# ВНИМАНИЕ: счётчики живут в памяти процесса. При WORKERS > 1 каждый uvicorn-воркер
# ведёт свои счётчики, и GET /metrics вернёт данные только того воркера, который
# обслужил запрос. Поэтому по умолчанию WORKERS=1, а горизонтальное масштабирование
# делается репликами Pod'ов в Kubernetes (см. k8s/deployment.yaml + hpa.yaml) --
# там каждый Pod скрейпится Prometheus отдельно и метрики корректно агрегируются.
# Для multi-worker внутри одного контейнера нужен prometheus_client в режиме
# multiprocess (PROMETHEUS_MULTIPROC_DIR) либо отдельный exporter-сайдкар.
STATS: Dict[str, float] = {
    "requests_total": 0, "predictions_total": 0, "errors_total": 0,
    "latency_sum_seconds": 0.0, "positive_predictions_total": 0,
    "proba_sum": 0.0,
}


# ----------------------------- схемы запроса ------------------------------- #
class Customer(BaseModel):
    """Контракт входных данных = валидация схемы на границе сервиса.

    Pydantic отбивает запросы с неверными типами/пропущенными полями до
    попадания в модель -- это data validation в рантайме.
    """
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, le=120)
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["Yes"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class PredictRequest(BaseModel):
    data: Union[Customer, List[Customer]]
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Prediction(BaseModel):
    churn_probability: float
    churn: int
    risk_band: str


class PredictResponse(BaseModel):
    # protected_namespaces=() -- поле model_version иначе конфликтует с
    # зарезервированным в pydantic префиксом model_
    model_config = {"protected_namespaces": ()}

    model_version: str
    threshold: float
    latency_ms: float
    predictions: List[Prediction]


def risk_band(p: float) -> str:
    if p >= 0.7:
        return "high"
    if p >= 0.4:
        return "medium"
    return "low"


# ------------------------------- lifecycle --------------------------------- #
@app.on_event("startup")
def load_model() -> None:
    if MLFLOW_MODEL_URI:
        import mlflow.sklearn  # импорт здесь, чтобы образ без MLflow тоже работал

        STATE["model"] = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
        STATE["source"] = MLFLOW_MODEL_URI
        log.info(f"модель загружена из MLflow Model Registry: {MLFLOW_MODEL_URI}")
    else:
        if not MODEL_PATH.exists():
            log.error(f"файл модели не найден: {MODEL_PATH}")
            return
        STATE["model"] = joblib.load(MODEL_PATH)
        STATE["source"] = str(MODEL_PATH)
        log.info(f"модель загружена из файла: {MODEL_PATH}")

    if MODEL_CARD_PATH.exists():
        STATE["card"] = json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))


# ------------------------------- endpoints --------------------------------- #
@app.get("/health")
def health() -> dict:
    """Liveness probe: процесс отвечает."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness probe: модель загружена. Пока не ok -- Service не льёт трафик."""
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model_source": STATE["source"]}


@app.get("/model")
def model_info() -> dict:
    card = STATE["card"]
    return {
        "service_version": app.version,
        "model_source": STATE["source"],
        "model_type": card.get("model_type"),
        "run_id": card.get("run_id"),
        "git_commit": card.get("git", {}).get("git_commit"),
        "dataset": card.get("dataset", {}).get("kaggle_ref"),
        "dataset_sha256": card.get("dataset", {}).get("sha256"),
        "trained_at_utc": card.get("created_at_utc"),
        "metrics": card.get("metrics", {}),
        "default_threshold": THRESHOLD,
    }


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """Метрики для Prometheus: трафик, задержка, доля положительных предсказаний.

    Средняя предсказанная вероятность -- дешёвый прокси-детектор дрейфа:
    её сдвиг виден без разметки, до появления факта оттока.
    """
    n = max(STATS["predictions_total"], 1)
    lines = [
        "# HELP churn_requests_total Total prediction requests",
        "# TYPE churn_requests_total counter",
        f"churn_requests_total {int(STATS['requests_total'])}",
        "# HELP churn_predictions_total Total scored customers",
        "# TYPE churn_predictions_total counter",
        f"churn_predictions_total {int(STATS['predictions_total'])}",
        "# HELP churn_errors_total Total failed requests",
        "# TYPE churn_errors_total counter",
        f"churn_errors_total {int(STATS['errors_total'])}",
        "# HELP churn_latency_seconds_sum Cumulative inference latency",
        "# TYPE churn_latency_seconds_sum counter",
        f"churn_latency_seconds_sum {STATS['latency_sum_seconds']:.6f}",
        "# HELP churn_positive_rate Share of predicted churners",
        "# TYPE churn_positive_rate gauge",
        f"churn_positive_rate {STATS['positive_predictions_total'] / n:.6f}",
        "# HELP churn_mean_probability Mean predicted probability (drift proxy)",
        "# TYPE churn_mean_probability gauge",
        f"churn_mean_probability {STATS['proba_sum'] / n:.6f}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if STATE["model"] is None:
        STATS["errors_total"] += 1
        raise HTTPException(status_code=503, detail="model not loaded")

    started = time.perf_counter()
    STATS["requests_total"] += 1
    rows = req.data if isinstance(req.data, list) else [req.data]
    threshold = req.threshold if req.threshold is not None else THRESHOLD

    try:
        X = pd.DataFrame([r.model_dump() for r in rows])
        proba = STATE["model"].predict_proba(X)[:, 1]
    except Exception as exc:
        STATS["errors_total"] += 1
        log.exception("inference failed")
        raise HTTPException(status_code=400, detail=f"inference error: {exc}") from exc

    latency = time.perf_counter() - started
    STATS["predictions_total"] += len(proba)
    STATS["latency_sum_seconds"] += latency
    STATS["proba_sum"] += float(proba.sum())
    STATS["positive_predictions_total"] += int((proba >= threshold).sum())

    # структурный лог каждого запроса -- вход для офлайн-анализа дрейфа
    log.info(f"predict n={len(proba)} mean_proba={float(proba.mean()):.4f} "
             f"latency_ms={latency * 1000:.2f}")

    return PredictResponse(
        model_version=str(STATE["card"].get("run_id", "unknown")),
        threshold=threshold,
        latency_ms=round(latency * 1000, 3),
        predictions=[
            Prediction(churn_probability=round(float(p), 6),
                       churn=int(p >= threshold),
                       risk_band=risk_band(float(p)))
            for p in proba
        ],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
