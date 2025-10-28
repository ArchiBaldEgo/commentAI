from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any

import logging
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

from .preprocess import PreprocessTransformer
from .config import get_settings, Settings


# --------- Rating/label mapping ---------
Label = Literal["neg", "neu", "pos"]


def label_from_stars(stars: int) -> Label:
    if stars <= 2:
        return "neg"
    if stars == 3:
        return "neu"
    return "pos"


def stars_from_proba(proba: Dict[str, float]) -> int:
    # Expectation mapping: 1*P(neg) + 3*P(neu) + 5*P(pos)
    pneg = float(proba.get("neg", 0.0))
    pneu = float(proba.get("neu", 0.0))
    ppos = float(proba.get("pos", 0.0))
    exp = 1 * pneg + 3 * pneu + 5 * ppos
    s = int(round(min(5, max(1, exp))))
    return s


# --------- Online model (hashing + SGD) ---------


@dataclass
class OnlineModelConfig:
    n_features: int = 2 ** 20
    ngram_range: tuple[int, int] = (1, 2)
    random_state: int = 42
    max_iter: int = 5


class OnlineSentiment:
    """Incremental model: HashingVectorizer + SGDClassifier(log-loss)"""

    def __init__(self, cfg: OnlineModelConfig | None = None):
        if cfg is None:
            cfg = OnlineModelConfig()
        self.cfg = cfg
        self.pipeline: Pipeline = Pipeline([
            ("prep", PreprocessTransformer()),
            (
                "hv",
                HashingVectorizer(
                    n_features=cfg.n_features,
                    alternate_sign=False,
                    ngram_range=cfg.ngram_range,
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    max_iter=cfg.max_iter,
                    random_state=cfg.random_state,
                ),
            ),
        ])
        self._classes_: List[str] | None = None

    @property
    def classes_(self) -> List[str]:
        if self._classes_ is None:
            # default order
            self._classes_ = ["neg", "neu", "pos"]
        return self._classes_

    def predict(self, texts: List[str]) -> List[str]:
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(self.pipeline[:-1].transform(texts))  # type: ignore
        raise HTTPException(status_code=500, detail="Classifier has no predict_proba")

    def partial_fit(self, texts: List[str], labels: List[str]):
        classes = self.classes_
        self.pipeline.partial_fit(texts, labels, classes=classes)  # type: ignore

    # ---- persistence ----
    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.pipeline, os.path.join(model_dir, "online_model.joblib"))
        with open(os.path.join(model_dir, "online_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"classes": self.classes_}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "OnlineSentiment":
        pipe = joblib.load(os.path.join(model_dir, "online_model.joblib"))
        inst = cls()
        inst.pipeline = pipe
        meta_path = os.path.join(model_dir, "online_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                inst._classes_ = list(meta.get("classes", ["neg", "neu", "pos"]))
        return inst


# --------- Feedback storage (JSONL, backend-agnostic) ---------


class JsonlFeedbackStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def append(self, item: Dict[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def write_all(self, items: List[Dict[str, Any]]):
        with open(self.path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")


# --------- FastAPI app ---------


class ServerState:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = self._load_or_init_model(model_dir)
        self.store = JsonlFeedbackStore(os.path.join(model_dir, "feedback.jsonl"))

    def _load_or_init_model(self, model_dir: str) -> OnlineSentiment:
        model_path = os.path.join(model_dir, "online_model.joblib")
        if os.path.exists(model_path):
            return OnlineSentiment.load(model_dir)
        return OnlineSentiment()


class PredictIn(BaseModel):
    texts: List[str] = Field(default_factory=list)


class PredictOutItem(BaseModel):
    text: str
    label: Label
    proba: Dict[str, float]
    stars: int


class PredictOut(BaseModel):
    items: List[PredictOutItem]


class FeedbackIn(BaseModel):
    text: str
    rating: Optional[int] = Field(None, ge=1, le=5)
    label: Optional[Label] = None


class FitOut(BaseModel):
    trained: int


class TrainBatchIn(BaseModel):
    clear_after: bool = True
    limit: Optional[int] = None
    shuffle: bool = True


class TrainBatchOut(BaseModel):
    seen: int
    trained: int
    remaining: int


class FeedbackBulkItem(BaseModel):
    text: str
    rating: Optional[int] = Field(None, ge=1, le=5)
    label: Optional[Label] = None


class FeedbackBulkIn(BaseModel):
    items: List[FeedbackBulkItem]
    store_only: bool = False  # если True, только сохраняем, без обучения


class FeedbackBulkOut(BaseModel):
    received: int
    stored: int
    trained: int


def create_app(model_dir: str | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    if model_dir is None:
        model_dir = settings.model_dir
    state = ServerState(model_dir)

    # Logging basic config
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

    app = FastAPI(title=settings.app_name, version=settings.app_version)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional API Key dependency
    def require_api_key(x_api_key: str | None = Header(default=None)):
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    dep = [Depends(require_api_key)] if settings.api_key else []

    @app.get("/health", dependencies=dep)
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictOut, dependencies=dep)
    def predict(inp: PredictIn):
        if not inp.texts:
            raise HTTPException(400, "texts is empty")
        labels = state.model.predict(inp.texts)
        probas = state.model.predict_proba(inp.texts)
        classes = state.model.classes_
        items: List[PredictOutItem] = []
        for t, lbl, pr in zip(inp.texts, labels, probas):
            proba_dict = {c: float(p) for c, p in zip(classes, pr)}
            stars = stars_from_proba(proba_dict)
            items.append(PredictOutItem(text=t, label=lbl, proba=proba_dict, stars=stars))
        return PredictOut(items=items)

    @app.post("/feedback", response_model=FitOut, dependencies=dep)
    def feedback(inp: FeedbackIn):
        if not inp.text:
            raise HTTPException(400, "text is required")
        if inp.label is None and inp.rating is None:
            raise HTTPException(400, "Either label or rating must be provided")
        label = inp.label or label_from_stars(int(inp.rating))  # type: ignore[arg-type]

        # Persist feedback for auditing
        state.store.append({"text": inp.text, "label": label})

        # Online learn immediately
        state.model.partial_fit([inp.text], [label])
        # Persist updated model (best-effort)
        try:
            state.model.save(state.model_dir)
        except Exception:
            pass
        return FitOut(trained=1)

    @app.post("/train_batch", response_model=TrainBatchOut, dependencies=dep)
    def train_batch(inp: TrainBatchIn):
        items = state.store.read_all()
        seen = len(items)
        if seen == 0:
            return TrainBatchOut(seen=0, trained=0, remaining=0)
        if inp.shuffle:
            import random
            random.shuffle(items)
        if inp.limit is not None:
            items_to_train = items[: int(inp.limit)]
            remaining_items = items[int(inp.limit) :]
        else:
            items_to_train = items
            remaining_items = []

        texts = [it.get("text", "") for it in items_to_train if it.get("text")]
        labels = [it.get("label") for it in items_to_train if it.get("label")]
        # align lengths
        paired = [(t, l) for t, l in zip(texts, labels) if t and l]
        if not paired:
            return TrainBatchOut(seen=seen, trained=0, remaining=len(remaining_items))
        t_train, y_train = zip(*paired)
        state.model.partial_fit(list(t_train), list(y_train))
        try:
            state.model.save(state.model_dir)
        except Exception:
            pass
        if inp.clear_after:
            state.store.write_all(remaining_items)
            remaining = len(remaining_items)
        else:
            remaining = seen - len(paired)
        return TrainBatchOut(seen=seen, trained=len(paired), remaining=remaining)

    @app.post("/feedback_bulk", response_model=FeedbackBulkOut, dependencies=dep)
    def feedback_bulk(inp: FeedbackBulkIn):
        if not inp.items:
            raise HTTPException(400, "items is empty")
        # Store all
        stored = 0
        texts: List[str] = []
        labels: List[str] = []
        for it in inp.items:
            if not it.text:
                continue
            lbl = it.label or (label_from_stars(int(it.rating)) if it.rating is not None else None)
            if lbl is None:
                # store even without label for дальнейшей ручной разметки
                state.store.append({"text": it.text, "label": None})
                stored += 1
                continue
            state.store.append({"text": it.text, "label": lbl})
            stored += 1
            if not inp.store_only:
                texts.append(it.text)
                labels.append(lbl)
        trained = 0
        if texts and labels and not inp.store_only:
            state.model.partial_fit(texts, labels)
            try:
                state.model.save(state.model_dir)
            except Exception:
                pass
            trained = len(texts)
        return FeedbackBulkOut(received=len(inp.items), stored=stored, trained=trained)

    @app.get("/version", dependencies=dep)
    def version() -> Dict[str, str]:
        return {"version": settings.app_version}

    return app


# --------- CLI entry (uvicorn) ---------


def run(host: str = "0.0.0.0", port: int = 8000, model_dir: str = "models/online"):
    import uvicorn

    os.makedirs(model_dir, exist_ok=True)
    app = create_app(model_dir)
    uvicorn.run(app, host=host, port=port)

# Module-level ASGI app for uvicorn CLI
settings = get_settings()
app = create_app(settings.model_dir, settings)
