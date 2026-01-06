"""Тренировка продакшен‑модели: HashingVectorizer + TF‑IDF + SGDClassifier.

Выбор сделан в пользу онлайновых свойств и стабильности в проде:
- HashingVectorizer не требует словаря и экономит память
- SGDClassifier поддерживает partial_fit для дообучения
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pandas as pd
from pandas.errors import ParserError
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
import joblib

from .preprocess import PreprocessTransformer


def build_pipeline(char_ngrams: bool = True, n_features_word: int = 8000, n_features_char: int = 10000, class_weight=None):
    """Конструируем пайплайн признаков и классификатор.
    По умолчанию включены символьные n-граммы — помогают на «шумных» текстах.
    """
    if char_ngrams:
        featurizer = FeatureUnion([
            ('word', Pipeline([
                ('hv', HashingVectorizer(n_features=n_features_word, alternate_sign=False, ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer())
            ])),
            ('char', Pipeline([
                ('hv', HashingVectorizer(analyzer='char', n_features=n_features_char, ngram_range=(3, 5), alternate_sign=False)),
                ('tfidf', TfidfTransformer())
            ]))
        ])
    else:
        featurizer = Pipeline([
            ('hv', HashingVectorizer(n_features=n_features_word, alternate_sign=False, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer())
        ])

    # Логистическая регрессия на стероидах (через SGD) — поддерживает онлайн‑обучение
    clf = SGDClassifier(loss='log_loss', max_iter=10, class_weight=class_weight, random_state=42)

    pipeline = Pipeline([
        ('prep', PreprocessTransformer()),
        ('tfidf', featurizer),
        ('clf', clf)
    ])
    return pipeline


def main(args: list[str] | None = None):
    """Точка входа для обучения из CLI и как функция.
    На выход кладём артефакты в указанную директорию: model.joblib и meta.json.
    """
    parser = argparse.ArgumentParser(description="Train Hashing+SGD sentiment model")
    parser.add_argument('--data', required=True, help='CSV with columns: text,label')
    parser.add_argument('--model-dir', required=True, help='Output directory for model')
    parser.add_argument('--class-weight', default=None, help='e.g. balanced')
    parser.add_argument('--char-ngrams', action='store_true', help='Include char ngrams branch')
    ns = parser.parse_args(args=args)

    try:
        df = pd.read_csv(ns.data)
    except ParserError:
        # Учебный fallback: если в тексте есть запятые без кавычек, pd.read_csv падает.
        # Тогда парсим строку вручную: последняя "колонка" — label, остальное — text.
        rows = []
        with open(ns.data, "r", encoding="utf-8") as f:
            header_seen = False
            for line in f:
                line = line.strip("\n\r")
                if not line.strip():
                    continue
                if not header_seen:
                    header_seen = True
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                label = parts[-1].strip().lower()
                text = ",".join(parts[:-1]).strip()
                rows.append({"text": text, "label": label})
        df = pd.DataFrame(rows)
    label_col = 'label' if 'label' in df.columns else 'sentiment'
    if 'text' not in df.columns or label_col not in df.columns:
        raise ValueError('CSV must contain columns: text and label/sentiment')
    # Clean data: drop NaNs, trim spaces, keep only allowed labels
    df = df[['text', label_col]].dropna()
    df['text'] = df['text'].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    allowed = {'neg', 'neu', 'pos'}
    df = df[df[label_col].isin(allowed)]
    if df.empty:
        raise ValueError('No valid rows after cleaning. Allowed labels: neg|neu|pos')
    X = df['text'].tolist()
    y = df[label_col].tolist()

    pipe = build_pipeline(char_ngrams=ns.char_ngrams, class_weight=ns.class_weight)
    pipe.fit(X, y)

    out_dir = Path(ns.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / 'model.joblib')
    meta = {
        'algorithm': 'sgd',
        'vectorizer': 'hashing+tfidf',
        'char_ngrams': bool(ns.char_ngrams),
        'classes': getattr(getattr(pipe, 'classes_', None), 'tolist', lambda: list(getattr(pipe, 'classes_', [])))()
    }
    with (out_dir / 'meta.json').open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved model to {out_dir}")  # маленькое счастье тренера

if __name__ == '__main__':
    main()
