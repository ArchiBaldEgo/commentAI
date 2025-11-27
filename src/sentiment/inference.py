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

DEFAULT_MODEL_PATH = Path("models/production/model.joblib")


_MODEL_UPDATE_LOCK = threading.Lock()

def _resolve_path(model_path: Optional[str] = None) -> Path:
    return Path(model_path) if model_path else DEFAULT_MODEL_PATH

@lru_cache(maxsize=1)
def load_model_cached(model_path: Optional[str] = None):
    """Лениво грузим модель с диска и кешируем в памяти.
    Если путь не указан — используем продакшен‑модель.
    """
    p = _resolve_path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)

def save_model(model, model_path: Optional[str] = None):
    """Сохраняем модель обратно на диск (перезаписываем артефакт)."""
    p = _resolve_path(model_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)

def online_partial_fit(texts: List[str], labels: List[str], model_path: Optional[str] = None) -> bool:
    """Incrementally update classifier with new labeled examples.
    Requires the classifier to support partial_fit (e.g., SGDClassifier).
    Returns True if updated, False otherwise.
    """
    if not texts:
        return False
    model = load_model_cached(model_path)
    # Pipeline or estimator
    clf = None
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
    if clf is None and hasattr(model, "partial_fit"):
        clf = model
    if clf is None or not hasattr(clf, "partial_fit"):
        return False
    with _MODEL_UPDATE_LOCK:
        # NB: блокировка на время обновления — чтобы два потока не писали модель одновременно
        # Use pipeline.partial_fit if available to apply preprocess/vectorizer
        if hasattr(model, "partial_fit"):
            # Determine classes only if classifier is not initialized
            classes = getattr(clf, "classes_", None)
            kwargs = {"classes": classes} if classes is not None else {}
            model.partial_fit(texts, labels, **kwargs)  # type: ignore
        else:
            # Fallback: manually transform then call clf.partial_fit
            prep = model.named_steps.get("prep")
            vec = model.named_steps.get("tfidf")
            Xt = vec.transform(prep.transform(texts))
            classes = getattr(clf, "classes_", None)
            kwargs = {"classes": classes} if classes is not None else {}
            clf.partial_fit(Xt, labels, **kwargs)
        save_model(model, model_path)
    return True


def predict_proba_texts(model, texts: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Предсказываем вероятности и метки для списка текстов.
    Возвращаем (probs, predicted_labels, classes).
    """
    # Support sklearn Pipeline with preprocess + vectorizer + classifier
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)  # type: ignore
        classes = list(getattr(model, "classes_", []))
    else:
        # If pipeline, try to access clf
        try:
            clf = model.named_steps["clf"]
            prep = model.named_steps["prep"]
            vec = model.named_steps["tfidf"]
            processed = prep.transform(texts)
            Xv = vec.transform(processed)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(Xv)
            else:
                df = clf.decision_function(Xv)
                if df.ndim == 1:
                    df = np.vstack([-df, df]).T
                ex = np.exp(df - df.max(axis=1, keepdims=True))
                probs = ex / ex.sum(axis=1, keepdims=True)
            classes = list(getattr(clf, "classes_", []))
        except Exception:
            raise RuntimeError("Model does not support probability inference")

    preds_idx = probs.argmax(axis=1)
    pred_labels = [classes[i] for i in preds_idx]
    return probs, pred_labels, classes
