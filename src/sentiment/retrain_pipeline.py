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


def merge_labeled_and_feedback() -> Path:
    """Собираем единый CSV для обучения из `reviews_labeled.csv` и feedback_buffer.jsonl.
    Дубликаты по тексту удаляем — берём последнюю версию.
    """
    if not LABELED_FILE.exists():
        raise RuntimeError("Base labeled dataset not found: data/reviews_labeled.csv")
    df_base = pd.read_csv(LABELED_FILE)
    df_base = df_base.rename(columns={"sentiment": "label"})
    frames = [df_base[["text", "label"]]]

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
            df_fb = pd.DataFrame(rows)
            frames.append(df_fb[["text", "label"]])

    df_all = pd.concat(frames, ignore_index=True)
    # drop duplicates by text
    df_all = df_all.drop_duplicates(subset=["text"]).reset_index(drop=True)
    out_path = DATA_DIR / "merged_for_train.csv"
    df_all.to_csv(out_path, index=False)
    return out_path


def run_retrain(class_weight: Optional[str] = "balanced", char_ngrams: bool = True) -> str:
    """Запускаем обучение с дефолтом под онлайн‑режим (Hashing+SGD),
    раскладываем в версионированную папку и обновляем продакшен‑модель.
    Возвращаем путь к версии.
    """
    from .train import main as train_main
    merged = merge_labeled_and_feedback()
    version_dir = MODELS_DIR / f"version_{int(time.time())}"
    version_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "--data", str(merged),
        "--model-dir", str(version_dir),
    ]
    if class_weight:
        args += ["--class-weight", class_weight]
    if char_ngrams:
        args += ["--char-ngrams"]
    # Use hashing + SGD as default for production online friendliness
    args += ["--hashing", "--algo", "sgd"]

    # Invoke training function
    train_main(args=args)
    # Copy resulting model into production dir so inference uses the latest
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    src_model = version_dir / "model.joblib"
    src_meta = version_dir / "meta.json"
    if src_model.exists():
        shutil.copyfile(src_model, PROD_DIR / "model.joblib")
    if src_meta.exists():
        shutil.copyfile(src_meta, PROD_DIR / "meta.json")
    # Keep version marker
    (PROD_DIR / "VERSION").write_text(version_dir.name)
    return str(version_dir)
