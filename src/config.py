"""Загрузка конфигурации пайплайна из params.yaml.

Весь пайплайн читает гиперпараметры только отсюда: ни одного "магического числа"
в коде обучения -> эксперимент однозначно описывается парой (git commit, params.yaml).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]

PARAMS_PATH = Path(os.getenv("PARAMS_PATH", ROOT / "params.yaml"))

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "metrics"
REPORTS_DIR = ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, METRICS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_params(path: Path | str = PARAMS_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


PARAMS = load_params()
