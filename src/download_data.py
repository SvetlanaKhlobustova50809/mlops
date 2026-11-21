"""Этап 1 жизненного цикла: сбор данных (data ingestion).

Датасет: Kaggle "Telco Customer Churn" (blastchar/telco-customer-churn).

Две стратегии загрузки:
  1) официальный Kaggle API (kaggle.json в ~/.kaggle) -- основной путь для CI;
  2) публичный download-endpoint Kaggle через requests -- fallback, когда
     токена нет (например, на машине проверяющего).

После загрузки считается SHA-256 сырого CSV и пишется в data/raw/dataset.meta.json.
Этот хеш -- "отпечаток" версии данных: он логируется в MLflow вместе с моделью,
что и обеспечивает воспроизводимость эксперимента.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import PARAMS, RAW_DIR

KAGGLE_ANON_URL = "https://www.kaggle.com/api/v1/datasets/download/{ref}"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _try_kaggle_api(ref: str, dest: Path) -> bool:
    """Загрузка через официальный Kaggle API (нужен ~/.kaggle/kaggle.json)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(ref, path=str(dest), unzip=True, quiet=False)
        print("[download] источник: Kaggle API (kaggle.json)")
        return True
    except Exception as exc:  # нет токена / нет сети / нет пакета
        print(f"[download] Kaggle API недоступен ({type(exc).__name__}: {exc}); "
              f"пробую публичный endpoint")
        return False


def _try_kaggle_anonymous(ref: str, dest: Path) -> bool:
    """Fallback: публичный endpoint Kaggle, работает без авторизации."""
    url = KAGGLE_ANON_URL.format(ref=ref)
    resp = requests.get(url, timeout=120, allow_redirects=True)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(dest)
    print(f"[download] источник: {url}")
    return True


def main() -> None:
    ref = PARAMS["dataset"]["kaggle_ref"]
    csv_name = PARAMS["dataset"]["csv_name"]
    target_csv = RAW_DIR / "telco_churn.csv"

    if not _try_kaggle_api(ref, RAW_DIR):
        _try_kaggle_anonymous(ref, RAW_DIR)

    downloaded = RAW_DIR / csv_name
    if not downloaded.exists():
        candidates = sorted(RAW_DIR.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"CSV не найден в {RAW_DIR}")
        downloaded = candidates[0]

    if downloaded.resolve() != target_csv.resolve():
        shutil.move(str(downloaded), target_csv)

    digest = sha256_file(target_csv)
    meta = {
        "kaggle_ref": ref,
        "source": f"https://www.kaggle.com/datasets/{ref}",
        "local_path": str(target_csv.relative_to(RAW_DIR.parent.parent)),
        "sha256": digest,
        "size_bytes": target_csv.stat().st_size,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = RAW_DIR / "dataset.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[download] {target_csv} ({meta['size_bytes']} байт)")
    print(f"[download] sha256 = {digest}")
    print(f"[download] метаданные версии данных -> {meta_path}")


if __name__ == "__main__":
    main()
