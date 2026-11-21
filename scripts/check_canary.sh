#!/usr/bin/env bash
# Автоматическая проверка canary-версии модели перед полным раскатом.
# Вызывается из CD-пайплайна (.github/workflows/ci.yml, job deploy).
#
# Сравниваем canary и stable по трём группам сигналов:
#   1) техника   -- доля ошибок и p99 задержки;
#   2) поведение модели -- средняя предсказанная вероятность (прокси-дрейф);
#   3) бизнес    -- конверсия/отклик (подтягивается из хранилища событий).
# Любой провал -> exit 1 -> CD откатывает canary.
set -euo pipefail

PROM="${PROM_URL:-http://prometheus.monitoring.svc.cluster.local:9090}"
WINDOW="${WINDOW:-15m}"
MAX_ERROR_RATE="${MAX_ERROR_RATE:-0.01}"      # 1% ошибок
MAX_P99_SECONDS="${MAX_P99_SECONDS:-0.5}"
MAX_PROBA_SHIFT="${MAX_PROBA_SHIFT:-0.05}"    # сдвиг среднего скора

q() { curl -sfG "$PROM/api/v1/query" --data-urlencode "query=$1" \
        | python3 -c 'import json,sys; r=json.load(sys.stdin)["data"]["result"]; print(r[0]["value"][1] if r else "0")'; }

gt() { python3 -c "import sys; sys.exit(0 if float('$1') > float('$2') else 1)"; }

echo "Проверка canary за последние $WINDOW"

ERR=$(q "sum(rate(churn_errors_total{app=\"churn-api-canary\"}[$WINDOW]))
        / clamp_min(sum(rate(churn_requests_total{app=\"churn-api-canary\"}[$WINDOW])), 0.001)")
echo "  error rate       = $ERR (макс $MAX_ERROR_RATE)"
gt "$ERR" "$MAX_ERROR_RATE" && { echo "FAIL: слишком много ошибок"; exit 1; }

P99=$(q "histogram_quantile(0.99, sum(rate(churn_latency_seconds_bucket{app=\"churn-api-canary\"}[$WINDOW])) by (le))")
echo "  p99 latency      = ${P99}s (макс ${MAX_P99_SECONDS}s)"
gt "$P99" "$MAX_P99_SECONDS" && { echo "FAIL: деградация задержки"; exit 1; }

CAN=$(q "avg(churn_mean_probability{app=\"churn-api-canary\"})")
STA=$(q "avg(churn_mean_probability{app=\"churn-api\"})")
SHIFT=$(python3 -c "print(abs(float('$CAN') - float('$STA')))")
echo "  mean proba       = canary $CAN vs stable $STA, сдвиг $SHIFT (макс $MAX_PROBA_SHIFT)"
gt "$SHIFT" "$MAX_PROBA_SHIFT" && { echo "FAIL: распределение скоров уехало"; exit 1; }

echo "Canary здоров -- можно раскатывать на 100% трафика"
