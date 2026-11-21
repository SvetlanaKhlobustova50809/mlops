# Telco Churn — сквозной MLOps-пайплайн

Учебный, но полностью рабочий MLOps-проект на датасете
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn):
от загрузки данных до модели в контейнере, с трекингом экспериментов,
версионированием данных, CI/CD и мониторингом дрейфа.

Задача — предсказать отток клиента телеком-оператора (бинарная классификация,
7043 записи, 20 признаков, доля положительного класса 26.5%).

## Что внутри

| Фаза жизненного цикла | Реализация |
|---|---|
| Сбор данных | загрузка с Kaggle + фиксация `sha256` версии данных |
| Подготовка | валидация схемы/типов/диапазонов, стратифицированный split |
| Экспериментирование | трекинг параметров, метрик и артефактов в MLflow |
| Тренировка | `sklearn.Pipeline`, 3 алгоритма на сравнение |
| Валидация | quality gate, блокирующий деплой слабой модели |
| Развертывание | FastAPI в multi-stage Docker-образе, манифесты Kubernetes |
| Мониторинг | отчёты и тесты дрейфа в Evidently, метрики для Prometheus |
| Переобучение | CronJob по расписанию + триггер по дрейфу |

## Результаты

Сравнение алгоритмов (5-fold стратифицированная CV на train, holdout 20%):

| модель | CV ROC-AUC | test ROC-AUC | test PR-AUC | test F1 | overfit gap |
|---|---|---|---|---|---|
| **gradient_boosting** | **0.8481 ± 0.0108** | **0.8428** | **0.6591** | 0.5835 | 0.0523 |
| random_forest | 0.8450 | 0.8424 | 0.6509 | 0.6318 | 0.0860 |
| logistic_regression | 0.8462 | 0.8421 | 0.6343 | 0.6040 | 0.0072 |

Выбран `GradientBoostingClassifier` (`n_estimators=300, learning_rate=0.05,
max_depth=3, subsample=0.9`). Четыре независимых прогона этой конфигурации дали
побитово идентичные метрики. Сильнейшие предикторы по permutation importance —
`Contract`, `tenure`, `InternetService`, `MonthlyCharges`: уходят в основном
клиенты на месячном контракте с небольшим стажем и дорогим тарифом на оптике.

Релизная модель прослеживается до кода и данных (`models/model_card.json`):

```
run_id         434d75f995c64aeea9d47f58f9210bf4
git_commit     5ce663a38a82599b6c2fc63b07503cf89421ae65   (git_dirty: false)
dataset        blastchar/telco-customer-churn
dataset sha256 88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a
env            python 3.11.15, sklearn 1.5.2, pandas 2.2.3, numpy 2.0.2
registry       telco-churn-clf v2 (alias production), v1 (alias previous)
```

## Быстрый старт

```bash
# окружение (Python 3.11)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt          # либо: make venv

# полный прогон: данные -> модель -> валидация -> тесты
make all

# эксперименты
make mlflow-ui          # http://localhost:5000
make mlflow-list        # сравнение прогонов в консоли

# сервис модели в контейнере
make docker-build && make docker-run && make docker-smoke

# мониторинг дрейфа
make monitor            # отчёты Evidently на свежих данных
make drift-demo         # детекция искусственного covariate shift
```

`make help` выводит все цели с описанием.

> Загрузка датасета работает и без токена Kaggle: `src/download_data.py`
> сначала пробует официальный Kaggle API (`~/.kaggle/kaggle.json`), а при его
> отсутствии — публичный download-endpoint Kaggle.

## API сервиса

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | liveness probe — процесс жив |
| GET | `/ready` | readiness probe — модель загружена |
| GET | `/model` | версия модели, run_id, git-коммит, хеш датасета, метрики |
| GET | `/metrics` | счётчики в формате Prometheus |
| POST | `/predict` | предсказание для одного клиента или батча |

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d @examples/request_batch.json
```

```json
{"predictions": [{"churn_probability": 0.837942, "churn": 1, "risk_band": "high"},
                 {"churn_probability": 0.034539, "churn": 0, "risk_band": "low"}]}
```

Клиент с месячным контрактом и стажем 1 месяц — риск 0.84; лояльный клиент с
двухлетним контрактом и стажем 68 месяцев — 0.03. Запрос с нарушенной схемой
отбивается кодом 422 до вызова модели.

Модель можно брать не из образа, а из реестра — тогда образ остаётся неизменным,
а версия управляется алиасом в MLflow:

```bash
docker run -e MLFLOW_MODEL_URI=models:/telco-churn-clf@production \
           -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
           telco-churn-service:1.0.0
```

Локальный стенд «сервис + MLflow Tracking Server» — `docker compose up -d --build`.

## Структура

```
.
├── params.yaml                единый источник гиперпараметров
├── requirements.txt           зависимости разработки (все версии ==)
├── requirements-serve.txt     минимальные зависимости рантайма сервиса
├── Makefile                   одинаковые команды локально и в CI
│
├── src/
│   ├── config.py              загрузка params.yaml, пути проекта
│   ├── download_data.py       сбор данных: Kaggle + sha256 версии данных
│   ├── prepare_data.py        валидация данных + стратифицированный split
│   ├── train.py               обучение + полный трекинг в MLflow
│   ├── evaluate.py            quality gate: блокирует деплой слабой модели
│   ├── serve.py               REST API на FastAPI
│   ├── drift_report.py        мониторинг дрейфа через Evidently
│   ├── mlflow_compare.py      сравнение прогонов и работа с Model Registry
│   └── sampling_demo.py       сравнение методов семплирования
│
├── Dockerfile                 multi-stage образ с моделью
├── docker-compose.yml         локальный стенд: сервис + MLflow Server
│
├── dvc.yaml                   DAG: download -> prepare -> train -> evaluate -> monitor
├── .dvc/config                DVC remote
│
├── k8s/                       Namespace, ConfigMap, Secret, Deployment, Service,
│                              Ingress, HPA, PDB, canary, CronJob (переобучение
│                              и проверка дрейфа)
├── .github/workflows/ci.yml   lint -> pipeline -> build-image -> deploy
├── scripts/check_canary.sh    автопроверка canary по метрикам Prometheus
├── tests/test_pipeline.py     20 тестов: данные, модель, API, воспроизводимость
│
├── data/                      raw + processed (версионируются DVC, не Git)
├── models/                    model.joblib + model_card.json
├── metrics/                   metrics.json + baseline_metrics.json
├── examples/                  примеры запросов к API
└── reports/                   графики, отчёты, протоколы прогонов
```

## Пайплайн DVC

```
download -> prepare -> train -> evaluate
                         \
                          -> monitor
```

Воспроизводится одной командой `dvc repro`: пересчитываются только стадии, у
которых изменились код, параметры или входные данные. В Git попадают код,
конфигурации, `.dvc`-файлы и `dvc.lock`; данные и модели — в S3-совместимый
remote:

```bash
dvc remote add -d s3remote s3://bucket/dvcstore
dvc add data/raw/telco_churn.csv   # создаёт .dvc-файл и правит .gitignore
git add data/raw/telco_churn.csv.dvc data/raw/.gitignore && git commit -m "data v1"
dvc push                            # данные уходят в remote
# на другой машине: git pull && dvc pull -> та же версия данных, что в коммите
```

## Мониторинг дрейфа

`src/drift_report.py` строит пять артефактов (reference = обучающая выборка,
current = свежие данные): `data_drift.html`, `target_drift.html`,
`data_quality.html`, `classification.html` и `drift_tests.json` с вердиктом
pass/fail для CI и алертов.

Флаг `--simulate-drift` воспроизводит covariate shift (`MonthlyCharges` ×1.35,
сжатый `tenure`, больше месячных контрактов) при неизменной зависимости
`P(churn|X)` — Evidently это уверенно находит:

```
$ python src/drift_report.py --simulate-drift
[drift] тестов всего: 68, пройдено: 61, упало: 7
  MonthlyCharges  0.709 | tenure 0.606 | TotalCharges 0.291
  InternetService 0.121 | Contract 0.118          (порог 0.1)
```

Флаг `--fail-on-drift` заставляет скрипт вернуть код 1 — так он вызывается из
CronJob, и ненулевой код превращается в алерт и может триггерить переобучение.

## Как закрыты ключевые проблемы эксплуатации ML

| Проблема | Решение |
|---|---|
| **Воспроизводимость** | git-коммит + `dataset_sha256` + `params.yaml` + lock зависимостей + `model_card.json`, всё связано через MLflow run_id |
| **Data leakage** | split до любого `fit`; весь препроцессинг внутри `sklearn.Pipeline`; тест на пересечение train/test по `customerID` |
| **Дрейф данных** | Evidently по расписанию, `--fail-on-drift` → алерт; `churn_mean_probability` в `/metrics` как прокси-детектор без разметки |
| **Качество данных** | валидация схемы, типов, диапазонов и дубликатов в `prepare_data.py`; pydantic-схема на границе API |
| **Деплой слабой модели** | `evaluate.py` возвращает код 1 при ROC-AUC ниже порога, регрессии против baseline или признаках переобучения |
| **Training/serving skew** | один и тот же объект `Pipeline` обучается и сериализуется; `handle_unknown='ignore'` для новых категорий |
| **Непонятно, что в проде** | аннотации `mlops/mlflow-run-id` и `mlops/dataset-sha256` в Deployment; эндпоинт `/model` отдаёт происхождение модели |
| **Риск при релизе** | canary 10% через Ingress, автопроверка метрик, `rollout undo` при неудаче |
| **Git перегружен бинарниками** | в Git только код, конфиги и `.dvc`-метаданные; данные и модели — в DVC remote |

## Качество кода

```
$ ruff check src tests
All checks passed!
$ make test
20 passed
```

## Известные ограничения

- **Счётчики в `/metrics` живут в памяти процесса.** При `WORKERS > 1` каждый
  uvicorn-воркер ведёт свои счётчики, поэтому по умолчанию `WORKERS=1`, а
  масштабирование делается репликами Pod'ов — Prometheus скрейпит каждый Pod
  отдельно и агрегирует корректно. Для multi-worker внутри одного контейнера
  нужен `prometheus_client` в режиме multiprocess.
- **DVC remote настроен на локальный каталог**, а не на S3 — в учебной среде нет
  облачных кредов. Переключение — одна команда `dvc remote add -d ... s3://...`.
- **Два теста `DataStabilityTestPreset` падают и на неизменённых данных** — это
  артефакт сравнения выборок разного размера (test = 20% от train), а не дрейф:
  `Number of Rows` ожидает одинаковый размер, `Share of Out-of-Range Values`
  реагирует на одно значение `MonthlyCharges` за пределами диапазона train.
  Пофичевые тесты дрейфа при этом не срабатывают ни разу.
- **`plotly` закреплён на 5.24.1**: Evidently 0.4.x несовместим с plotly ≥ 6
  при рендеринге HTML-отчётов.
- Манифесты Kubernetes проверены на корректность синтаксиса и структуры, но не
  применялись в реальном кластере.

## Лицензия

MIT
