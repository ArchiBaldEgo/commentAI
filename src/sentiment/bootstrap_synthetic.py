"""Генерация синтетических отзывов для дообучения модели.

Скрипт создаёт расширенный CSV `data/reviews_labeled_synth.csv`
примерно из 500 примеров (neg/neu/pos) и запускает ретрейн.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .retrain_pipeline import run_retrain, DATA_DIR


def build_synthetic_dataset(target_size: int = 500) -> Path:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    base = DATA_DIR / "reviews_labeled.csv"
    out = DATA_DIR / "reviews_labeled_synth.csv"

    if base.exists():
        df = pd.read_csv(base)
    else:
        df = pd.DataFrame(columns=["text", "label"])

    base_rows = len(df)

    # Простые заготовки фраз
    pos_phrases = [
        "Очень понравилось, всё отлично",
        "Шикарный товар, рекомендую",
        "Отличное качество, буду брать ещё",
        "В целом доволен покупкой",
        "Работает идеально, без нареканий",
    ]
    neg_phrases = [
        "Совсем не понравилось, разочарован",
        "Плохое качество, не советую",
        "Товар сломался через неделю",
        "Полнейший отстой, зря купил",
        "Очень разочарован сервисом и качеством",
    ]
    neu_phrases = [
        "Нормальный товар, без восторга",
        "Что-то среднее, есть плюсы и минусы",
        "В целом нормально, ожидал большего",
        "Обычный товар, работает как должен",
        "Среднее качество, в целом ок",
    ]

    rows = []
    rng = random.Random(42)
    labels_cycle = ["neg", "neu", "pos"]
    i = 0
    while base_rows + len(rows) < target_size:
        label = labels_cycle[i % 3]
        if label == "pos":
            text = rng.choice(pos_phrases)
        elif label == "neg":
            text = rng.choice(neg_phrases)
        else:
            text = rng.choice(neu_phrases)
        # слегка варьируем текст
        suffix = rng.choice(["", "!", ".", " :)", " :(", " в целом"])
        rows.append({"text": f"{text}{suffix}", "label": label})
        i += 1

    if rows:
        df_new = pd.DataFrame(rows)
        df = pd.concat([df, df_new], ignore_index=True)

    df.to_csv(out, index=False)
    return out


def main() -> None:
    csv_path = build_synthetic_dataset(500)
    # временно переиспользуем пайплайн ретрейна, указав другой базовый CSV
    # просто заменим LABELED_FILE на сгенерированный в текущем процессе
    from . import retrain_pipeline as rp

    rp.LABELED_FILE = csv_path  # type: ignore[attr-defined]
    run_retrain()


if __name__ == "__main__":
    main()
