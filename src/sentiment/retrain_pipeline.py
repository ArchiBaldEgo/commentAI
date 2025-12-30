"""Пайплайн переобучения: объединяем базовую разметку с фидбеком и тренируем новую версию.

Результат копируем в `models/production`, чтобы сервис начал использовать свежую модель.
"""
import json
import time
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path("data")
FEEDBACK_FILE = DATA_DIR / "feedback_buffer.jsonl"
LABELED_FILE = DATA_DIR / "reviews_labeled.csv"
MODELS_DIR = Path("models")
PROD_DIR = MODELS_DIR / "production"
VERSIONS_DIR = MODELS_DIR / "versions"
RETRAIN_STATUS = DATA_DIR / "retrain_status.json"


def merge_labeled_and_feedback() -> tuple[Path, int, int, int]:
    """Собираем единый CSV для обучения из `reviews_labeled.csv` и feedback_buffer.jsonl.
    Дубликаты по тексту удаляем — берём последнюю версию.

    Возвращаем путь к CSV и статистику: (base_rows, fb_rows, total_rows).
    """
    if not LABELED_FILE.exists():
        raise RuntimeError("Base labeled dataset not found: data/reviews_labeled.csv")

    df_base = pd.read_csv(LABELED_FILE)
    df_base = df_base.rename(columns={"sentiment": "label"})
    frames = []
    base_rows = 0
    fb_rows = 0

    if {"text", "label"}.issubset(df_base.columns):
        df_base = df_base[["text", "label"]].dropna()
        base_rows = len(df_base)
        frames.append(df_base)

    if FEEDBACK_FILE.exists():
        rows = []
        with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("text") and rec.get("label"):
                        rows.append({"text": rec["text"], "label": rec["label"]})
                except Exception:
                    continue
        if rows:
            df_fb = pd.DataFrame(rows)[["text", "label"]].dropna()
            fb_rows = len(df_fb)
            frames.append(df_fb)

    if not frames:
        raise RuntimeError("No data for retrain: neither labeled CSV nor feedback produced usable rows")

    df_all = pd.concat(frames, ignore_index=True)
    # drop duplicates by text
    df_all = df_all.drop_duplicates(subset=["text"]).reset_index(drop=True)
    total_rows = len(df_all)
    out_path = DATA_DIR / "merged_for_train.csv"
    df_all.to_csv(out_path, index=False)
    return out_path, base_rows, fb_rows, total_rows


def run_retrain(class_weight: Optional[str] = "balanced", char_ngrams: bool = True) -> str:
    """Запускаем переобучение модели.

    1. Объединяем базовый датасет и feedback.
    2. Обучаем модель в новой версионной папке.
    3. Обновляем `models/production` и записываем подробный статус в `data/retrain_status.json`.

    Возвращаем путь к созданной версии.
    """
    from .train import main as train_main

    t0 = time.time()
    status: dict = {
        "started_at": t0,
        "status": "started",
        "error": None,
    }

    try:
        merged, base_rows, fb_rows, total_rows = merge_labeled_and_feedback()
        status.update({
            "base_rows": base_rows,
            "feedback_rows": fb_rows,
            "total_rows": total_rows,
            "merged_path": str(merged),
        })

        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        version_dir = VERSIONS_DIR / f"v_{int(t0)}"
        version_dir.mkdir(parents=True, exist_ok=True)

        args = [
            "--data", str(merged),
            "--model-dir", str(version_dir),
        ]
        if class_weight:
            args += ["--class-weight", class_weight]
        if char_ngrams:
            args += ["--char-ngrams"]

        # Запускаем тренер
        train_main(args=args)

        # Обновляем продакшен‑модель
        PROD_DIR.mkdir(parents=True, exist_ok=True)
        src_model = version_dir / "model.joblib"
        src_meta = version_dir / "meta.json"
        if src_model.exists():
            shutil.copyfile(src_model, PROD_DIR / "model.joblib")
        if src_meta.exists():
            # Обогащаем мета‑файл версией
            try:
                meta = json.loads(src_meta.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            meta["version"] = version_dir.name
            (PROD_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # Маркер версии
        (PROD_DIR / "VERSION").write_text(version_dir.name, encoding="utf-8")

        status["status"] = "success"
        status["version_dir"] = str(version_dir)
    except Exception as e:  # noqa: BLE001
        status["status"] = "error"
        status["error"] = str(e)
    finally:
        status["finished_at"] = time.time()
        try:
            RETRAIN_STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    return status.get("version_dir", "")
