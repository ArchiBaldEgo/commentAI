"""FastAPI сервис для анализа тональности.

Здесь собраны:
- базовые эндпоинты (/health, /labels, /predict, /product/score)
- сбор обратной связи (/feedback) и фоновое дообучение
- метрики Prometheus и простое логирование в JSONL
- троттлинг по IP и защита по X-API-Key

Цель — быть практичным и неприхотливым в продакшене.
"""

from pathlib import Path
from typing import List, Optional, Dict
import threading
import time
import os
import logging
import json
from collections import deque

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .inference import load_model_cached, predict_proba_texts, online_partial_fit
from .storage import init_storage, store_feedback, store_product_score
from .retrain_pipeline import run_retrain, RETRAIN_STATUS

DATA_DIR = Path("data")
SCORES_LOG = DATA_DIR / "product_scores.csv"
SCORES_LOG.parent.mkdir(parents=True, exist_ok=True)

LABEL_TO_SCORE = {"neg": 1.0, "neu": 3.0, "pos": 5.0}

app = FastAPI(title="Sentiment Service", version="1.0")

# Security: API key via header X-API-Key
API_KEY = os.getenv("API_KEY")

def require_api_key(request: Request):
    """Простая проверка ключа API через заголовок X-API-Key.
    Если переменная окружения API_KEY не задана — авторизация отключена.
    """
    if not API_KEY:
        return None  # auth disabled if not set
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# Prometheus metrics
REQ_COUNT = Counter("sentiment_requests_total", "Total requests", ["endpoint", "method", "status"])
REQ_LATENCY = Histogram("sentiment_request_latency_seconds", "Request latency", ["endpoint", "method"])
ONLINE_UPDATES = Counter("sentiment_online_updates_total", "Online partial_fit updates")

def track(endpoint: str, method: str, status: int, start: float):
    """Учёт запросов/латентности в метриках Prometheus."""
    REQ_COUNT.labels(endpoint=endpoint, method=method, status=str(status)).inc()
    REQ_LATENCY.labels(endpoint=endpoint, method=method).observe(time.time() - start)

# Logging setup (JSON lines)
logger = logging.getLogger("sentiment.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

def get_client_ip(request: Request) -> str:
    """Аккуратно достаём IP клиента, учитывая X-Forwarded-For за прокси."""
    fwd = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    client = request.client.host if request.client else "unknown"
    return client

# Rate limit (sliding window per IP)
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "120"))
WHITELIST_PATHS = set((os.getenv("RATE_LIMIT_WHITELIST", "/health,/metrics").split(",")))
_RL_STATE: dict[str, deque] = {}
_RL_LOCK = threading.Lock()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Логируем каждый HTTP‑запрос в одну строку JSON — удобно парсить."""
    start = time.time()
    ip = get_client_ip(request)
    ua = request.headers.get("user-agent", "")
    path = request.url.path
    method = request.method
    try:
        response = await call_next(request)
        status = response.status_code
        duration = time.time() - start
        logger.info(json.dumps({
            "event": "http_request",
            "ip": ip,
            "method": method,
            "path": path,
            "status": status,
            "duration_ms": int(duration * 1000),
            "ua": ua
        }, ensure_ascii=False))
        return response
    except Exception as e:
        duration = time.time() - start
        logger.exception(json.dumps({
            "event": "http_error",
            "ip": ip,
            "method": method,
            "path": path,
            "status": 500,
            "duration_ms": int(duration * 1000),
            "error": str(e),
            "ua": ua
        }, ensure_ascii=False))
        raise

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Очень простой rate‑limit per IP (скользящее окно).
    Белый список путей можно задать через RATE_LIMIT_WHITELIST.
    """
    path = request.url.path
    if path in WHITELIST_PATHS or RATE_LIMIT_MAX <= 0:
        return await call_next(request)
    ip = get_client_ip(request)
    now = time.time()
    key = ip
    with _RL_LOCK:
        dq = _RL_STATE.get(key)
        if dq is None:
            dq = deque()
            _RL_STATE[key] = dq
        # prune old
        cutoff = now - RATE_LIMIT_WINDOW
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_MAX:
            # metrics for 429
            REQ_COUNT.labels(endpoint=path, method=request.method, status="429").inc()
            return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})
        dq.append(now)
    return await call_next(request)

# Auto-retrain background worker
AUTO_RETRAIN_ENABLED = os.getenv("AUTO_RETRAIN_ENABLED", "0") in ("1", "true", "True")
AUTO_RETRAIN_INTERVAL = int(os.getenv("AUTO_RETRAIN_INTERVAL_SECONDS", "600"))
AUTO_RETRAIN_MIN_ITEMS = int(os.getenv("AUTO_RETRAIN_MIN_ITEMS", "50"))
STATE_FILE = DATA_DIR / "auto_retrain.state"

def _load_state():
    try:
        import json as _json
        return _json.loads(STATE_FILE.read_text())
    except Exception:
        return {"processed_lines": 0, "last_retrain_ts": 0.0}

def _save_state(state):
    try:
        import json as _json
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(_json.dumps(state))
    except Exception:
        pass

def _count_lines(p: Path) -> int:
    try:
        with p.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def _auto_retrain_worker():
    """Фоновый воркер: периодически триггерит переобучение, если накопились новые фидбеки."""
    logger.info("auto_retrain: worker started")
    while True:
        try:
            state = _load_state()
            now = time.time()
            total = _count_lines(DATA_DIR / "feedback_buffer.jsonl")
            delta = max(0, total - int(state.get("processed_lines", 0)))
            due_time = (now - float(state.get("last_retrain_ts", 0.0))) >= AUTO_RETRAIN_INTERVAL
            if delta >= AUTO_RETRAIN_MIN_ITEMS or (due_time and delta > 0):
                logger.info(f"auto_retrain: triggering retrain, new_items={delta}")
                run_retrain()
                state["processed_lines"] = total
                state["last_retrain_ts"] = now
                _save_state(state)
        except Exception as e:
            logger.error(json.dumps({"event":"auto_retrain_error","error":str(e)}))
        time.sleep(max(5, min(30, AUTO_RETRAIN_INTERVAL // 6 or 10)))

@app.on_event("startup")
def _startup_autoretrain():
    """На старте поднимаем фоновые потоки (автотрен, онлайн‑батчи) и инициализируем сторедж."""
    if AUTO_RETRAIN_ENABLED:
        t = threading.Thread(target=_auto_retrain_worker, daemon=True)
        t.start()
    # Start online micro-batch worker if enabled
    if os.getenv("ONLINE_LEARNING", "0") in ("1", "true", "True"):
        t2 = threading.Thread(target=_online_microbatch_worker, daemon=True)
        t2.start()
    # Initialize storage (DB if configured)
    init_storage()

# Online learning micro-batching (hybrid mode)
ONLINE_BATCH_SIZE = int(os.getenv("ONLINE_BATCH_SIZE", "16"))
ONLINE_BATCH_INTERVAL = int(os.getenv("ONLINE_BATCH_INTERVAL_SECONDS", "30"))
ONLINE_MAX_LOAD = float(os.getenv("ONLINE_MAX_LOAD", "4.0"))
_ONLINE_QUEUE: deque[tuple[str, str]] = deque()
_ONLINE_LOCK = threading.Lock()
_ONLINE_LAST_TS = 0.0

def _system_load_ok() -> bool:
    """Проверяем, не перегружена ли система, чтобы не злить соседние сервисы."""
    try:
        la1, _, _ = os.getloadavg()
        # allow slight headroom for small machines
        return la1 <= ONLINE_MAX_LOAD
    except Exception:
        return True

def _online_microbatch_worker():
    """Фоновая микробатч‑очередь для online partial_fit.
    Ставит обновления модельки «не сразу», а небольшими пачками,
    с дедупликацией и ограничением по нагрузке системы.
    """
    global _ONLINE_LAST_TS
    logger.info("online_fit: micro-batch worker started")
    while True:
        try:
            time.sleep(1)
            now = time.time()
            with _ONLINE_LOCK:
                qlen = len(_ONLINE_QUEUE)
                should_time = (now - _ONLINE_LAST_TS) >= ONLINE_BATCH_INTERVAL
                if qlen == 0 or (qlen < ONLINE_BATCH_SIZE and not should_time):
                    continue
                # collect up to batch size
                batch = []
                take = min(ONLINE_BATCH_SIZE, qlen)
                for _ in range(take):
                    batch.append(_ONLINE_QUEUE.popleft())
            if not _system_load_ok():
                # push back if load is high
                with _ONLINE_LOCK:
                    for item in reversed(batch):
                        _ONLINE_QUEUE.appendleft(item)
                time.sleep(5)
                continue
            # deduplicate by text (last label wins)
            uniq = {}
            for text, label in batch:
                if text and label:
                    uniq[text] = label
            texts = list(uniq.keys())
            labels = list(uniq.values())
            if texts:
                ok = online_partial_fit(texts, labels)
                if ok:
                    ONLINE_UPDATES.inc()
                    _ONLINE_LAST_TS = now
        except Exception as e:
            logger.error(json.dumps({"event":"online_fit_error","error":str(e)}))


class PredictRequest(BaseModel):
    texts: List[str]
    model_path: Optional[str] = None


class PredictResponse(BaseModel):
    predictions: List[str]
    probabilities: List[List[float]]
    labels: List[str]
    model_version: Optional[str] = None


class FeedbackItem(BaseModel):
    text: str
    label: str
    source: Optional[str] = None
    user_id: Optional[str] = None


class FeedbackBatch(BaseModel):
    items: List[FeedbackItem]


class ProductScoreRequest(BaseModel):
    productId: str
    texts: List[str]
    model_path: Optional[str] = None


class ProductScoreResponse(BaseModel):
    productId: str
    score: float
    details: Dict[str, int]


@app.get("/health")
def health(request: Request):
    """Быстрая проверка доступности сервиса."""
    start = time.time()
    resp = {"status": "ok"}
    track("/health", request.method, 200, start)
    return resp


@app.get("/labels")
def labels(request: Request):
    """Возвращает список меток, известных текущей модели."""
    model = load_model_cached()
    try:
        clf = model.named_steps["clf"]
        cls = [str(c) for c in list(getattr(clf, "classes_", []))]
    except Exception:
        cls = list(getattr(model, "classes_", ["neg", "neu", "pos"]))
    start = time.time()
    resp = {"labels": cls}
    track("/labels", request.method, 200, start)
    return resp


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request, _: None = Depends(require_api_key)):
    """Классифицирует список текстов и возвращает вероятности/метки."""
    start = time.time()
    if not req.texts:
        track("/predict", request.method, 400, start)
        raise HTTPException(400, "texts must be non-empty")
    # Простая защита от слишком больших запросов
    if len(req.texts) > 200:
        track("/predict", request.method, 413, start)
        raise HTTPException(413, "too many texts in one request (max 200)")
    if any(len(t) > 5000 for t in req.texts):
        track("/predict", request.method, 413, start)
        raise HTTPException(413, "text too long (max 5000 characters)")
    model = load_model_cached(req.model_path)

    # Пытаемся вытащить версию модели из meta.json
    version: Optional[str] = None
    try:
        import json as _json
        from pathlib import Path as _Path

        base = req.model_path or "models/production"
        meta_path = _Path(base)
        if meta_path.is_dir():
            meta_path = meta_path / "meta.json"
        if meta_path.exists():
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            version = meta.get("version")
    except Exception:
        version = None

    probs, preds, classes = predict_proba_texts(model, req.texts)
    track("/predict", request.method, 200, start)
    return PredictResponse(
        predictions=preds,
        probabilities=[list(map(float, p)) for p in probs],
        labels=classes,
        model_version=version,
    )


@app.post("/product/score", response_model=ProductScoreResponse)
def product_score(req: ProductScoreRequest, request: Request, _: None = Depends(require_api_key)):
    """Считает среднюю «оценку товара» по предсказанным меткам и логирует результат."""
    start = time.time()
    if not req.texts:
        track("/product/score", request.method, 400, start)
        raise HTTPException(400, "texts must be non-empty")
    model = load_model_cached(req.model_path)
    probs, preds, classes = predict_proba_texts(model, req.texts)
    # Map labels to scores and average
    counts = {"neg": 0, "neu": 0, "pos": 0}
    points = []
    for label in preds:
        counts[label] = counts.get(label, 0) + 1
        points.append(LABEL_TO_SCORE.get(label, 3.0))
    score = float(sum(points) / len(points)) if points else 0.0
    # Persist via storage (file + optional DB)
    store_product_score(req.productId, score, counts)
    track("/product/score", request.method, 200, start)
    return ProductScoreResponse(productId=req.productId, score=score, details=counts)


@app.post("/feedback")
def feedback(batch: FeedbackBatch, request: Request, _: None = Depends(require_api_key)):
    """Принимает пользовательскую разметку и кладёт её в файл/БД, а также в очередь online‑обучения."""
    start = time.time()
    # Валидация меток на входе
    allowed = {"neg", "neu", "pos"}
    for it in batch.items:
        if it.label not in allowed:
            track("/feedback", request.method, 400, start)
            raise HTTPException(400, f"invalid label '{it.label}', allowed: neg|neu|pos")
    # Persist feedback (file + optional DB)
    items = [{
        "text": it.text,
        "label": it.label,
        "source": it.source,
        "user_id": it.user_id,
    } for it in batch.items]
    store_feedback(items)
    saved = len(items)
    # Enqueue for online micro-batch processing (hybrid mode)
    if os.getenv("ONLINE_LEARNING", "0") in ("1", "true", "True"):
        with _ONLINE_LOCK:
            for it in batch.items:
                if it.text and it.label:
                    _ONLINE_QUEUE.append((it.text, it.label))
    track("/feedback", request.method, 200, start)
    return {"saved": saved}


@app.post("/retrain")
def retrain(request: Request, _: None = Depends(require_api_key)):
    """Асинхронно запускает переобучение модели (фоновый поток)."""
    start = time.time()
    def task():
        try:
            run_retrain()
        except Exception as e:
            print("Retrain error:", e)

    threading.Thread(target=task, daemon=True).start()
    track("/retrain", request.method, 200, start)
    return {"status": "started"}


@app.get("/retrain/status")
def retrain_status(_: None = Depends(require_api_key)):
    """Возвращает информацию о последнем запуске ретрейна из файла статуса."""
    try:
        import json as _json

        if not RETRAIN_STATUS.exists():
            raise HTTPException(404, "retrain status not found")
        data = _json.loads(RETRAIN_STATUS.read_text(encoding="utf-8"))
        return data
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"cannot read retrain status: {e}")

@app.get("/metrics")
def metrics():
    """Метрики Prometheus для скрейпинга."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
