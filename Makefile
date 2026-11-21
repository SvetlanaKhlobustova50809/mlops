# Единая точка входа во все этапы жизненного цикла.
# Одинаковые команды локально и в CI -> нет расхождения "у меня работало".

PY      ?= .venv/bin/python
IMAGE   ?= telco-churn-service
TAG     ?= 1.0.0
PORT    ?= 8000
MLRUNS  ?= ./mlruns

.PHONY: help venv download prepare train evaluate monitor drift-demo sampling test \
        all repro docker-build docker-run docker-stop compose-up compose-down \
        mlflow-ui mlflow-list k8s-apply k8s-delete k8s-validate clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## создать окружение и установить зависимости
	uv venv --python 3.11 .venv
	.venv/bin/python -m pip install -q --upgrade pip
	uv pip install -r requirements.txt

# ---------------------------- фазы ML-цикла -------------------------------- #
download: ## 1. сбор данных: скачать датасет с Kaggle + зафиксировать sha256
	$(PY) src/download_data.py

prepare: ## 2. подготовка: валидация данных + стратифицированный split
	$(PY) src/prepare_data.py

train: ## 3-4. обучение с трекингом в MLflow
	$(PY) src/train.py

train-register: ## обучение + регистрация модели в MLflow Model Registry
	$(PY) src/train.py --register

train-all: ## сравнить три алгоритма как отдельные MLflow-эксперименты
	$(PY) src/train.py --model-type logistic_regression --run-name baseline-logreg
	$(PY) src/train.py --model-type random_forest       --run-name rf-400
	$(PY) src/train.py --model-type gradient_boosting   --run-name gb-300

evaluate: ## 5. валидация: quality gate для CI/CD
	$(PY) src/evaluate.py

monitor: ## 6+. мониторинг дрейфа через Evidently
	$(PY) src/drift_report.py

drift-demo: ## демонстрация детекции дрейфа на искусственно сдвинутых данных
	$(PY) src/drift_report.py --simulate-drift --out-dir reports/monitoring_drift_demo

sampling: ## сравнение методов семплирования
	$(PY) src/sampling_demo.py

test: ## тесты кода, данных, модели и API
	$(PY) -m pytest

all: download prepare train evaluate test ## полный прогон пайплайна

# ------------------------------- DVC --------------------------------------- #
repro: ## воспроизвести пайплайн через DVC (пересчитает только изменённое)
	.venv/bin/dvc repro

dvc-push: ## выгрузить данные и модели в DVC remote
	.venv/bin/dvc push

dvc-pull: ## забрать версию данных, соответствующую текущему Git-коммиту
	.venv/bin/dvc pull

dag: ## показать граф пайплайна
	.venv/bin/dvc dag

# ------------------------------ MLflow ------------------------------------- #
mlflow-ui: ## открыть MLflow UI на http://localhost:5000
	.venv/bin/mlflow ui --backend-store-uri $(MLRUNS) --host 0.0.0.0 --port 5000

mlflow-list: ## сравнить прогоны в консоли
	$(PY) src/mlflow_compare.py

# ------------------------------ Docker ------------------------------------- #
docker-build: ## собрать образ с моделью
	docker build -t $(IMAGE):$(TAG) .

docker-run: ## запустить контейнер
	docker run -d --name churn-api -p $(PORT):8000 $(IMAGE):$(TAG)

docker-smoke: ## smoke-тест запущенного контейнера
	curl -fsS http://localhost:$(PORT)/health && echo
	curl -fsS http://localhost:$(PORT)/ready && echo
	curl -fsS -X POST http://localhost:$(PORT)/predict \
	  -H 'Content-Type: application/json' -d @examples/request_batch.json && echo

docker-stop: ## остановить и удалить контейнер
	-docker rm -f churn-api

compose-up: ## поднять сервис + MLflow Tracking Server
	docker compose up -d --build

compose-down:
	docker compose down

# ---------------------------- Kubernetes ----------------------------------- #
k8s-validate: ## проверить манифесты без применения
	kubectl apply --dry-run=client -f k8s/

k8s-apply: ## выпустить модель в кластер
	kubectl apply -f k8s/

k8s-status:
	kubectl -n ml-prod get deploy,rs,pod,svc,ingress,hpa,cronjob

k8s-rollout: ## обновить образ модели без простоя
	kubectl -n ml-prod set image deploy/churn-api api=$(IMAGE):$(TAG)
	kubectl -n ml-prod rollout status deploy/churn-api

k8s-rollback: ## откатиться на предыдущую версию модели
	kubectl -n ml-prod rollout undo deploy/churn-api

k8s-delete:
	-kubectl delete -f k8s/

clean: ## удалить сгенерированные артефакты (данные и модели не трогаем)
	rm -rf .pytest_cache **/__pycache__ reports/monitoring reports/monitoring_drift_demo
