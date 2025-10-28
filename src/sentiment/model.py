from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from .preprocess import PreprocessTransformer

MODEL_FILENAME = "model.joblib"
META_FILENAME = "meta.json"

@dataclass
class ModelMeta:
    model_type: str
    classes: List[str]
    vectorizer: str
    algorithm: str
    version: str = "1.0"

class SentimentClassifier:
    def __init__(self, pipeline: Optional[Pipeline] = None):
        if pipeline is None:
            pipeline = Pipeline([
                ("prep", PreprocessTransformer()),
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
                ("clf", LogisticRegression(max_iter=1000, n_jobs=None))
            ])
        self.pipeline = pipeline

    def fit(self, X: Iterable[str], y: Iterable[str]):
        self.pipeline.fit(list(X), list(y))
        return self

    def predict(self, texts: Iterable[str]) -> List[str]:
        return list(self.pipeline.predict(list(texts)))

    def predict_proba(self, texts: Iterable[str]):
        clf = self.pipeline.named_steps.get("clf")
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(self.pipeline[:-1].transform(list(texts)))  # type: ignore
        raise AttributeError("Classifier does not support predict_proba")

    def evaluate(self, X: Iterable[str], y: Iterable[str]) -> str:
        preds = self.predict(X)
        return classification_report(list(y), preds)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.pipeline, os.path.join(path, MODEL_FILENAME))
        meta = ModelMeta(
            model_type="sentiment",
            classes=list(self.pipeline.classes_),
            vectorizer="tfidf",
            algorithm="logistic_regression"
        )
        with open(os.path.join(path, META_FILENAME), "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SentimentClassifier":
        pipeline = joblib.load(os.path.join(path, MODEL_FILENAME))
        return cls(pipeline=pipeline)
