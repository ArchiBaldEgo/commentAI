"""Инференс и онлайн‑обновление модели.

Содержит кешированную загрузку модели (чтобы не читать файл при каждом запросе),
а также безопасный partial_fit с блокировкой, чтобы не повредить артефакт.
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Optional
import threading

import joblib
import numpy as np
import json
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

from .preprocess import PreprocessTransformer

DEFAULT_MODEL_DIR = Path("models/production")


_MODEL_UPDATE_LOCK = threading.Lock()

def _resolve_paths(model_path: Optional[str] = None) -> Tuple[Path, Path]:
    """Возвращает пути к model.joblib и meta.json. Принимает как путь к файлу, так и к директории."""
    if model_path:
        p = Path(model_path)
        if p.is_dir():
            return p / "model.joblib", p / "meta.json"
        else:
            return p, p.with_suffix(".meta.json")
    else:
        return DEFAULT_MODEL_DIR / "model.joblib", DEFAULT_MODEL_DIR / "meta.json"

def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("prep", PreprocessTransformer()),
        ("hv", HashingVectorizer(n_features=2 ** 20, alternate_sign=False, ngram_range=(1, 2))),
        ("clf", SGDClassifier(loss="log_loss", max_iter=5, random_state=42)),
    ])

def build_default_pipeline() -> Pipeline:
    hv = HashingVectorizer(n_features=2**16, alternate_sign=False)
    clf = SGDClassifier(loss="log_loss", max_iter=5, tol=1e-3)
    pipe = Pipeline([
        ("prep", PreprocessTransformer()),
        ("hv", hv),
        ("clf", clf),
    ])
    # Initialize classifier with dummy samples so it's fitted
    dummy_texts = ["хорошо", "плохо"]
    Xv = hv.transform(dummy_texts)
    y = np.array(["pos", "neg"])  # ensure both classes present
    clf.partial_fit(Xv, y, classes=np.array(["neg", "pos"]))
    return pipe

@lru_cache(maxsize=1)
def load_model_cached(model_path: Optional[str] = None) -> Pipeline:
    """Грузим модель, если нет — создаём базовую пайплайн‑модель."""
    p, meta = _resolve_paths(model_path)
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            pass
    # Fallback: свежий пайплайн
    return build_default_pipeline()

def save_model(model: Pipeline, model_path: Optional[str] = None) -> None:
    """Сохраняем модель и обновляем метаданные классов."""
    p, meta = _resolve_paths(model_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    try:
        classes = list(getattr(model, "classes_", ["neg", "neu", "pos"]))
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({"classes": classes}, f, ensure_ascii=False)
    except Exception:
        pass

def online_partial_fit(texts: List[str], labels: List[str], model_path: Optional[str] = None) -> bool:
    """Incrementally update classifier with new labeled examples.
    Requires the classifier to support partial_fit (e.g., SGDClassifier).
    Returns True if updated, False otherwise.
    """
    if not texts:
        return False
    model = load_model_cached(model_path)
    clf = model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "partial_fit"):
        return False
    with _MODEL_UPDATE_LOCK:
        # partial_fit на всём пайплайне — sklearn будет вызывать предварительную обработку корректно
        classes = getattr(clf, "classes_", None)
        kwargs = {"classes": classes} if classes is not None else {"classes": ["neg", "neu", "pos"]}
        model.partial_fit(texts, labels, **kwargs)  # type: ignore
        save_model(model, model_path)
    return True


def predict_proba_texts(model, texts: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Предсказываем вероятности и метки для списка текстов.
    Возвращаем (probs, predicted_labels, classes).
    """
    # Всегда работаем через Pipeline: prep -> hv -> clf
    clf = model.named_steps.get("clf")
    prep = model.named_steps.get("prep")
    featurizer = model.named_steps.get("hv") or model.named_steps.get("tfidf")
    processed = prep.transform(texts)
    Xv = featurizer.transform(processed)
    # If predict_proba isn't available, fall back to decision_function
    try:
        probs = clf.predict_proba(Xv)
    except Exception:
        # Map decision function to pseudo-probabilities via softmax
        scores = clf.decision_function(Xv)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
    classes = list(getattr(clf, "classes_", ["neg", "neu", "pos"]))

    preds_idx = probs.argmax(axis=1)
    pred_labels = [classes[i] for i in preds_idx]
    return probs, pred_labels, classes
