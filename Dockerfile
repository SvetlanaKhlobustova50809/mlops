# ---------------------------------------------------------------------------
# Упаковка обученной ML-модели в контейнер.
#
# Multi-stage build:
#   builder -- собирает зависимости в изолированный venv;
#   runtime -- тонкий финальный образ: python slim + venv + модель + код сервиса.
# Это даёт маленький образ без компиляторов и кеша pip внутри.
#
# Принципы, важные для MLOps:
#   * версии базового образа и всех пакетов зафиксированы -> воспроизводимый артефакт;
#   * модель и код лежат в образе -> образ самодостаточен (immutable artifact);
#   * процесс запускается от непривилегированного пользователя;
#   * есть HEALTHCHECK -> Docker/Kubernetes видят состояние контейнера;
#   * конфигурация только через ENV -> один образ для dev/stage/prod (12-factor).
# ---------------------------------------------------------------------------
FROM python:3.11.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt


FROM python:3.11.10-slim AS runtime

LABEL org.opencontainers.image.title="telco-churn-service" \
      org.opencontainers.image.description="REST-сервис модели оттока клиентов (Kaggle Telco Customer Churn)" \
      org.opencontainers.image.source="https://github.com/SvetlanaKhlobustova50809/mlops"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_CARD_PATH=/app/models/model_card.json \
    THRESHOLD=0.5 \
    PORT=8000 \
    WORKERS=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Сначала код и модель -- слои кешируются независимо от зависимостей
COPY src/serve.py src/serve.py
COPY models/model.joblib models/model_card.json models/
COPY params.yaml params.yaml

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn serve:app --app-dir src --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]
