# Курсовая работа: Система анализа тональности с онлай‑обучением, REST‑API и продакшен‑инфраструктурой

 
- Клиент → /predict: тексты → вероятности и метки.
- Клиент → /product/score: набор отзывов → усреднённая оценка по шкале 1–5.
- Клиент → /feedback: размеченные примеры → файл (и БД) → очередь микро‑батчей → partial_fit.
- Cron/оператор → /retrain: мердж базовой разметки и фидбека → офлайн‑тренинг → выкладка в production.

### 4.3 Модель данных и хранение
- Файлы: `feedback_buffer.jsonl` (источник правды для переобучения), `product_scores.csv` (аудит агрегированных оценок).
- SQL: таблицы `feedback` и `product_scores` с полями id, ts, payload (нормализовано для аналитики).

---

## 5. Алгоритмы и методы
### 5.1 Предобработка текста
Шаги: нормализация регистра и пробелов, удаление пунктуации и цифр, токенизация по пробелам, фильтрация стоп‑слов, упрощённая «лемматизация» регулярными правилами. Этот конвейер выполняется быстро и стабильно, хорошо согласуется с n‑граммными признаками.

### 5.2 Векторизация признаков (TF‑IDF, Hashing)
Используется комбинация словарных и символьных признаков через HashingVectorizer и TF‑IDF. Интуитивно TF‑IDF повышает вес информативных терминов и снижает вес общеупотребительных. Формула TF‑IDF:

$$\operatorname{tfidf}(t,d) = \operatorname{tf}(t,d) \cdot \log\frac{N}{1 + \operatorname{df}(t)}$$

где $\operatorname{tf}(t,d)$ — частота термина $t$ в документе $d$, $\operatorname{df}(t)$ — количество документов, содержащих термин $t$, $N$ — число документов в корпусе. Hashing‑трюк обеспечивает постоянную память за счёт хеш‑проекции признаков и отсутствия явно хранимого словаря.

### 5.3 Классификация (логистическая регрессия, SGD)
В роли классификатора — SGDClassifier с логистической функцией потерь (эквивалент логистической регрессии, оптимизируемой стохастическим градиентом). Вероятность позитивного класса:

$$p(y=1\mid x) = \sigma(w^\top x + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

Функция потерь — логистическая (кросс‑энтропия) с L2‑регуляризацией. Обновления параметров по SGD минимизируют среднюю потерю на батче. Такой классификатор поддерживает `partial_fit`, что критично для онлай‑обучения.

### 5.4 Онлайн‑обучение и микро‑батчи
Инкрементальные обновления агрегируются в очередь; по срабатыванию условий (размер батча, истечение интервала, приемлемая системная нагрузка) выполняется `partial_fit`. Чтобы не перегружать систему, используется порог средней загрузки (load average) для задержки обновления. Дубликаты по тексту внутри батча объединяются — «последняя метка побеждает».

### 5.5 Активное обучение и управление качеством
Практически полезно периодически отбирать тексты с низкой уверенностью модели и предлагать их пользователю для аннотации. Это ускоряет рост качества на «пограничных» примерах и снижает систематические ошибки.

---

## 6. Реализация
### 6.1 REST‑API (FastAPI)
Реализованы эндпоинты: `/health`, `/labels`, `/predict`, `/product/score`, `/feedback`, `/retrain`, `/metrics`. Форматы запросов/ответов — JSON. Важные детали: кешированная загрузка модели, сериализация вероятностей в float, валидируемые схемы запросов.

### 6.2 Безопасность и ограничения
Авторизация включается установкой `API_KEY`; все POST‑запросы проверяются по заголовку `X-API-Key`. Применяется простой rate‑limit per IP (скользящее окно), whitelist для `/health` и `/metrics`. Для продакшена рекомендуется размещать сервис за реверс‑прокси с TLS и аудитом заголовков.

### 6.3 Наблюдаемость и метрики (Prometheus)
Собираются счётчики запросов, коды статусов, гистограммы задержек. Логи формируются записью JSON‑событий: запросы, ошибки, фоновые задачи. Это облегчает дашборды и алертинг.

### 6.4 Хранение: файлы и SQL (SQLAlchemy)
Файлы — «источник правды» для переобучения (совместимость по формату). При включении БД (через `DB_URL` или `STORAGE_MODE=db`) записи дублируются в SQL‑таблицы; схема создаётся автоматически. Такой подход облегчает аналитику (SQL) и сохраняет простоту ML‑конвейера (файлы).

### 6.5 Контейнеризация и деплой (Docker, Compose, Apache/Nginx)
Сервис упакован в Docker; предоставлены docker‑compose профили для локального запуска и подключения PostgreSQL/MySQL. Шаблоны конфигураций Nginx/Apache позволяют быстро разместить сервис за реверс‑прокси. Systemd‑юнит — для развёртывания как системной службы.

### 6.6 Графический интерфейс и пользовательский поток
Лёгкий GUI предназначен для демонстрации: ввод текста → предсказание → отправка фидбека. Пользовательские разметки собираются в буфер для последующего онлай‑обновления и переобучения.

---

## 7. Сбор данных и аннотация
Сбор исходных отзывов возможен через скрейпинг (aiohttp + BeautifulSoup). Важно соблюдать правила ресурсов (robots.txt, условия использования). Аннотация может быть выполнена вручную или полуавтоматически (правила/seed‑модели), после чего доступна функция слияния базовой разметки и пользовательского фидбека.

Качество данных — критично: нестационарность лексики, дисбаланс классов, шум. Рекомендуется регулярный аудит датасета, анализ частот и повторное взвешивание классов (`class_weight='balanced'`).

---

## 8. Оценка качества модели и тестирование
Метрики качества:
- Accuracy: доля верных ответов.
- Precision/Recall/F1: баланс точности и полноты по классам.
- ROC‑AUC: площадь под ROC‑кривой (для бинарной схемы или one‑vs‑rest).

Кросс‑валидация и отложенная тестовая выборка дают объективную оценку. Для продакшена полезен мониторинг сдвига данных (data drift) и деградации метрик — например, через периодические контрольные батчи.

Формула F1‑меры (для класса):

$$F_1 = 2\cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$$

Тестирование сервиса включает: unit‑тесты преобразований, интеграционные проверки API, smoke‑тесты при запуске, сценарии деградации (rate‑limit, недоступность БД, пустые запросы).

---

## 9. Производительность и отказоустойчивость
- Векторизация Hashing + TF‑IDF и линейный классификатор дают низкую задержку и линейную масштабируемость по числу документов.
- Микро‑батчи онлай‑обновления защищают от «дребезга» и разгружают CPU.
- Ограничение по системной нагрузке (load average) предотвращает конкуренцию с критичными задачами на узле.
- Реверс‑прокси обеспечивает буферизацию, сжатие, TLS и ограничение размера тела запросов.
- При сбоях БД основная функциональность сохраняется (запись в файлы продолжается), что повышает устойчивость.

---

## 10. Эксплуатация: CI/CD, версии, безопасность
- Версионирование модели: каждая переобученная версия сохраняется в отдельном каталоге, production‑ссылка указывает на актуальную.
- CI/CD: сборка Docker‑образа, прогон тестов, выкладка артефактов.
- Ротация ключей API: централизованное хранение секретов, периодическая замена, аудит доступа.
- Документация: README с быстрым стартом, SDK для интеграции, примеры под популярные фреймворки.

---

## 11. Риски, правовые и этические аспекты
- Приватность данных: хранение пользовательских текстов должно соответствовать политике и требованиям законодательства.
- Предвзятость модели: оценка классов может быть смещена; необходим мониторинг и корректировка, прозрачные методики обучения.
- Юридические ограничения на скрейпинг и повторное использование данных: следовать правилам площадок.
- Безопасность: ограничение исходящих вызовов, контроль заголовков и размеров тела, защита от DoS.

---

## 12. Экономическое обоснование и ресурсы
Выбранный стек минимизирует затраты: отсутствует необходимость в GPU и тяжёлых моделях, тренировочные циклы быстры, развёртывание стандартизовано через Docker. Поддержка онлай‑обновлений снижает необходимость частых полных переобучений, экономя вычислительные ресурсы и время инженеров. Наблюдаемость и логирование уменьшают MTTR при инцидентах.

---

## Заключение
Работа демонстрирует полный цикл проектирования производственного сервиса анализа тональности: от постановки задачи и выбора методов до реализации, инфраструктуры и эксплуатационных практик. Принятые инженерные решения (Hashing + TF‑IDF + SGD, онлай‑обучение микро‑батчами, простая безопасность, метрики и контейнеризация) обеспечивают баланс качества, скорости и поддерживаемости. Система готова к интеграции, дальнейшей эволюции (активное обучение, сложные признаки, улучшенная нормализация) и масштабированию в соответствии с потребностями продукта.

---

## Список литературы
1. Pedregosa et al. Scikit‑learn: Machine Learning in Python, JMLR (2011). Документация: https://scikit-learn.org/
2. FastAPI документация: https://fastapi.tiangolo.com/
3. Prometheus клиент для Python: https://github.com/prometheus/client_python
4. SQLAlchemy документация: https://docs.sqlalchemy.org/
5. Docker документация: https://docs.docker.com/
6. NLTK/стоп‑слова и базовые NLP‑приёмы: https://www.nltk.org/
7. Bishop, C. M. Pattern Recognition and Machine Learning. Springer (2006).
8. Jurafsky, D., Martin, J. H. Speech and Language Processing. (3rd ed. draft).

---

## Приложения (код проекта)

### Приложение A. src/sentiment/api.py
```python
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
from .retrain_pipeline import run_retrain

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
    cls = list(getattr(clf, "classes_", []))
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
  model = load_model_cached(req.model_path)
  probs, preds, classes = predict_proba_texts(model, req.texts)
  track("/predict", request.method, 200, start)
  return PredictResponse(
    predictions=preds,
    probabilities=[list(map(float, p)) for p in probs],
    labels=classes,
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

@app.get("/metrics")
def metrics():
  """Метрики Prometheus для скрейпинга."""
  return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### Приложение B. src/sentiment/inference.py
```python
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
```

### Приложение C. src/sentiment/preprocess.py
```python
"""Пре‑ и постпроцессинг текста для пайплайна sklearn.

Здесь собраны нормализация, токенизация, простая фильтрация стоп‑слов и примитивная
лемматизация по правилам. Не претендует на лингвистическую полноту, зато быстро и стабильно.
"""
from typing import Iterable, List
from sklearn.base import BaseEstimator, TransformerMixin


class PreprocessTransformer(BaseEstimator, TransformerMixin):
  """Упрощённый трансформер: чистим пробелы/регистр для базовой совместимости."""
  def fit(self, X: Iterable[str], y=None):
    return self

  def transform(self, X: Iterable[str]) -> List[str]:
    return [self._norm(x) for x in X]

  @staticmethod
  def _norm(text: str) -> str:
    if not isinstance(text, str):
      text = str(text)
    return " ".join(text.strip().lower().split())
import re
import string
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin

# Простые списки стоп-слов (можно расширять)
RU_STOP = {
  "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да","ты","к","у","же","вы","за","бы","по","ее","мне","если","или","ни","мы","те","это","мой","от","меня","его","им","из","уже"
}
EN_STOP = {
  "the","a","an","and","or","if","to","is","are","was","were","be","been","of","for","in","on","at","by","with","this","that","it","as","from","but","not"
}

PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation})

# «Лемматизация по бритве Оккама»: несколько регулярных правил вместо тяжёлых морфологизаторов
LEMMA_RULES = [
  (re.compile(r"(ами|ями|ыми|ами|ов|ев)$"), ""),
  (re.compile(r"(ыми|ие|ий|ого|ему|ыми|их|ую|ое|ая|ые|ый)$"), ""),
  (re.compile(r"(ing|ed|ly|ness|ment|s)$"), ""),
]


def simple_lemma(token: str) -> str:
  """Примитивно «стрижём» окончание токена по списку правил."""
  original = token
  for pattern, repl in LEMMA_RULES:
    token = pattern.sub(repl, token)
  # Минимальная длина
  if len(token) < 3:
    token = original
  return token


def normalize_text(text: str) -> str:
  """Приводим к нижнему регистру, убираем пунктуацию/цифры и лишние пробелы."""
  text = text.lower()
  text = text.translate(PUNCT_TABLE)
  text = re.sub(r"\d+", " ", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text


def tokenize(text: str) -> List[str]:
  """Очень простой токенайзер по пробелам."""
  return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
  """Фильтруем частые служебные слова на русском и английском."""
  return [t for t in tokens if t not in RU_STOP and t not in EN_STOP]


def lemmatize(tokens: List[str]) -> List[str]:
  """Грубая лемматизация по регуляркам — зато без внешних моделей."""
  return [simple_lemma(t) for t in tokens]


def preprocess_text(text: str) -> str:
  """Полный цикл: нормализация → токены → стоп‑слова → «леммы»."""
  text = normalize_text(text)
  tokens = tokenize(text)
  tokens = remove_stopwords(tokens)
  tokens = lemmatize(tokens)
  return " ".join(tokens)


class PreprocessTransformer(BaseEstimator, TransformerMixin):
  """Более «умный» трансформер для использования в продакшене пайплайна."""
  def fit(self, X, y=None):  # type: ignore
    return self

  def transform(self, X):  # type: ignore
    return [preprocess_text(x) for x in X]
```

### Приложение D. src/sentiment/train.py
```python
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

  df = pd.read_csv(ns.data)
  label_col = 'label' if 'label' in df.columns else 'sentiment'
  if 'text' not in df.columns or label_col not in df.columns:
    raise ValueError('CSV must contain columns: text and label/sentiment')
  X = df['text'].astype(str).tolist()
  y = df[label_col].astype(str).tolist()

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
```

### Приложение E. src/sentiment/retrain_pipeline.py
```python
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
```

### Приложение F. src/sentiment/storage.py
```python
"""Хранилище обратной связи и оценок: файл всегда, БД — по желанию.

Мы всегда пишем в файловые логи (jsonl/CSV), чтобы пайплайн обучения оставался
простым и независимым от БД. Если задан `DB_URL` или режим `STORAGE_MODE=db`,
дублируем записи в SQL через SQLAlchemy (таблицы создаются автоматически).
"""
from __future__ import annotations
import os
import json
import time
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, Text
from sqlalchemy.engine import Engine

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_FILE = DATA_DIR / "feedback_buffer.jsonl"
SCORES_CSV = DATA_DIR / "product_scores.csv"

STORAGE_MODE = os.getenv("STORAGE_MODE", "file").lower()  # 'file' | 'db'
DB_URL = os.getenv("DB_URL", "")

_engine: Optional[Engine] = None
_meta: Optional[MetaData] = None
_t_feedback: Optional[Table] = None
_t_scores: Optional[Table] = None


def _ensure_db():
  """Ленивая инициализация SQLAlchemy Engine и создание таблиц."""
  global _engine, _meta, _t_feedback, _t_scores
  if _engine is not None:
    return
  url = DB_URL or f"sqlite:///{(DATA_DIR / 'app.db').absolute()}"
  _engine = create_engine(url, future=True)
  _meta = MetaData()
  _t_feedback = Table(
    "feedback", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, nullable=False),
    Column("text", Text, nullable=False),
    Column("label", String(16), nullable=False),
    Column("source", String(128)),
    Column("user_id", String(128)),
  )
  _t_scores = Table(
    "product_scores", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, nullable=False),
    Column("product_id", String(128), nullable=False),
    Column("score", Float, nullable=False),
    Column("pos", Integer, nullable=False),
    Column("neu", Integer, nullable=False),
    Column("neg", Integer, nullable=False),
  )
  _meta.create_all(_engine)


def init_storage():
  """Инициализируем БД, если она включена настройками."""
  if STORAGE_MODE == "db" or DB_URL:
    _ensure_db()


def store_feedback(items: Iterable[dict]):
  """Сохраняем обратную связь: сначала в файл, опционально — в БД."""
  # Always append to file for retrain pipeline compatibility
  ts = time.time()
  with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
    for it in items:
      rec = {"ts": ts, **it}
      f.write(json.dumps(rec, ensure_ascii=False) + "\n")

  # Optionally also write to DB
  if STORAGE_MODE == "db" or DB_URL:
    _ensure_db()
    assert _engine is not None and _t_feedback is not None
    rows = []
    for it in items:
      rows.append({
        "ts": ts,
        "text": it.get("text", ""),
        "label": it.get("label", ""),
        "source": it.get("source"),
        "user_id": it.get("user_id"),
      })
    if rows:
      with _engine.begin() as conn:
        conn.execute(_t_feedback.insert(), rows)


def store_product_score(product_id: str, score: float, counts: dict):
  """Сохраняем агрегированную «оценку товара» (файл + опционально БД)."""
  ts = time.time()
  # Always append to CSV for compatibility
  with SCORES_CSV.open("a", encoding="utf-8") as f:
    f.write(f"{product_id},{score},{counts.get('pos',0)},{counts.get('neu',0)},{counts.get('neg',0)}\n")

  if STORAGE_MODE == "db" or DB_URL:
    _ensure_db()
    assert _engine is not None and _t_scores is not None
    row = {
      "ts": ts,
      "product_id": product_id,
      "score": float(score),
      "pos": int(counts.get("pos", 0)),
      "neu": int(counts.get("neu", 0)),
      "neg": int(counts.get("neg", 0)),
    }
    with _engine.begin() as conn:
      conn.execute(_t_scores.insert(), [row])
```

### Приложение G. clients/python/sentiment_client.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, TypedDict
import requests

Label = Literal['pos', 'neu', 'neg']


class PredictResult(TypedDict, total=False):
  predictions: List[Label]
  labels: List[Label]
  probabilities: List[List[float]]
  scores: List[float]


class FeedbackItem(TypedDict):
  text: str
  label: Label


class ProductScoreResult(TypedDict, total=False):
  productId: str
  score: float


@dataclass
class SentimentClient:
  """Минимальный Python‑клиент к сервису тональности.

  Делаем акцент на простоте: обычный requests.Session, явные методы,
  таймауты по умолчанию и опциональный API‑ключ через заголовок.
  """
  base_url: str
  api_key: Optional[str] = None
  timeout: float = 10.0
  session: Optional[requests.Session] = None

  def _s(self) -> requests.Session:
    """Ленивая инициализация сессии — переиспользуем TCP‑соединения."""
    if self.session is None:
      self.session = requests.Session()
    return self.session

  def _headers(self) -> Dict[str, str]:
    """Готовим заголовки запроса с учётом X-API-Key, если он задан."""
    h = {"Content-Type": "application/json"}
    if self.api_key:
      h["X-API-Key"] = self.api_key
    return h

  def health(self) -> Any:
    """Проверка состояния сервиса: вернёт JSON или строку."""
    r = self._s().get(f"{self.base_url}/health", timeout=self.timeout)
    r.raise_for_status()
    return r.json() if r.headers.get('content-type', '').startswith('application/json') else r.text

  def labels(self) -> Any:
    """Получить список поддерживаемых меток (классов)."""
    r = self._s().get(f"{self.base_url}/labels", timeout=self.timeout)
    r.raise_for_status()
    return r.json()

  def predict(self, texts: Iterable[str]) -> PredictResult:
    """Классифицировать список текстов.

    Примечание: сервер может возвращать также вероятности и список
    меток; клиент возвращает «как есть» JSON словарь.
    """
    r = self._s().post(
      f"{self.base_url}/predict",
      json={"texts": list(texts)},
      headers=self._headers(),
      timeout=self.timeout,
    )
    r.raise_for_status()
    return r.json()  # type: ignore[return-value]

  def product_score(self, product_id: str, texts: Iterable[str]) -> ProductScoreResult:
    """Посчитать среднюю оценку товара на основе предсказаний."""
    r = self._s().post(
      f"{self.base_url}/product/score",
      json={"productId": product_id, "texts": list(texts)},
      headers=self._headers(),
      timeout=self.timeout,
    )
    r.raise_for_status()
    return r.json()  # type: ignore[return-value]

  def feedback(self, items: Iterable[FeedbackItem]) -> Dict[str, Any]:
    """Отправить обратную связь (размеченные примеры) для дообучения."""
    r = self._s().post(
      f"{self.base_url}/feedback",
      json={"items": list(items)},
      headers=self._headers(),
      timeout=self.timeout,
    )
    r.raise_for_status()
    return r.json()
```

### Приложение H. clients/typescript/src/index.ts
```typescript
// Метки классов для сервиса тональности
export type Label = 'pos' | 'neu' | 'neg';

// Минимальное описание ответа fetch, чтобы не тянуть DOM типы
type FetchResponse = {
  ok: boolean;
  status: number;
  headers: { get(name: string): string | null };
  json(): Promise<any>;
  text(): Promise<string>;
};

// Обобщённый интерфейс fetch — подойдёт и для Node 18+, и для браузера
type FetchLike = (url: string, init?: { method?: string; headers?: Record<string, string>; body?: string; signal?: any }) => Promise<FetchResponse>;

// Параметры клиента — базовый URL, ключ API и таймауты
export interface SentimentClientOptions {
  baseUrl: string;
  apiKey?: string;
  timeoutMs?: number;
  fetchFn?: FetchLike;
}

// Форматы запросов/ответов — держим их максимально простыми
export interface PredictRequest { texts: string[] }
export interface FeedbackItem { text: string; label: Label }
export interface FeedbackRequest { items: FeedbackItem[] }
export interface ProductScoreRequest { productId: string; texts: string[] }

export interface PredictResult {
  predictions?: Label[];
  labels?: Label[];
  probabilities?: number[][];
  scores?: number[];
  [k: string]: unknown;
}

export interface FeedbackResult {
  accepted?: number;
  [k: string]: unknown;
}

export interface ProductScoreResult {
  productId?: string;
  score?: number;
  [k: string]: unknown;
}

// Минималистичный клиент для обращения к сервису тональности
export class SentimentClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private readonly doFetch: FetchLike;

  constructor(opts: SentimentClientOptions) {
  // тут специальной магии нет — просто запоминаем настройки
  if (!opts.baseUrl) throw new Error('baseUrl is required');
  this.baseUrl = opts.baseUrl.replace(/\/$/, '');
  this.apiKey = opts.apiKey;
  this.timeoutMs = opts.timeoutMs ?? 10_000;
  this.doFetch = opts.fetchFn ?? (globalThis as any).fetch;
  if (typeof this.doFetch !== 'function') {
    throw new Error('No fetch implementation found. Provide opts.fetchFn or use Node >= 18 / browser.');
  }
  }

  // Пингуем жив ли сервис
  async health(): Promise<unknown> {
  return this.get('/health');
  }

  // Узнаём список поддерживаемых меток
  async labels(): Promise<Label[] | unknown> {
  return this.get('/labels');
  }

  // Классификация батча текстов
  async predict(texts: string[]): Promise<PredictResult> {
  return this.post('/predict', { texts } satisfies PredictRequest);
  }

  // Средняя «оценка товара» по предсказанным меткам
  async productScore(productId: string, texts: string[]): Promise<ProductScoreResult> {
  return this.post('/product/score', { productId, texts } satisfies ProductScoreRequest);
  }

  // Отправляем размеченные пользователем примеры для дообучения
  async feedback(items: FeedbackItem[]): Promise<FeedbackResult> {
  return this.post('/feedback', { items } satisfies FeedbackRequest);
  }

  // Простой GET с таймаутом
  private async get(path: string): Promise<any> {
  const url = `${this.baseUrl}${path}`;
  const res = await this._withTimeout(this.doFetch(url, { method: 'GET' }));
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status} ${await res.text()}`);
  const ct = res.headers.get('content-type') || '';
  return ct.includes('application/json') ? res.json() : res.text();
  }

  // И POST тоже максимально без сюрпризов
  private async post(path: string, body: object): Promise<any> {
  const url = `${this.baseUrl}${path}`;
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (this.apiKey) headers['X-API-Key'] = this.apiKey;
  const res = await this._withTimeout(this.doFetch(url, { method: 'POST', headers, body: JSON.stringify(body) }));
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status} ${await res.text()}`);
  return res.json();
  }

  // Софт‑таймаут без AbortController, чтобы не тянуть DOM типы
  private async _withTimeout<T>(p: Promise<T>): Promise<T> {
  const to = this.timeoutMs;
  return await Promise.race([
    p,
    new Promise<T>((_, reject) => setTimeout(() => reject(new Error(`Request timeout after ${to}ms`)), to))
  ]);
  }
}
```

---

## Подробный обзор литературы и практик индустрии
Существуют три основных подхода к задачам классификации тональности: (1) правила и словари (lexicon‑based), (2) классические ML‑модели с ручными признаками, (3) нейросетевые модели с контекстными эмбеддингами. Правила и словари обеспечивают интерпретируемость и простоту, однако страдают от недостатка переносимости между доменами и языками. Классические ML‑подходы (SVM/LogReg/SGD на TF‑IDF/Hashing) остаются промышленным стандартом там, где требуются простота, скорость и предсказуемость. Нейросети семейства BERT дают максимум качества, но предъявляют повышенные требования к ресурсам и инженерии (пред‑ и пост‑процессинг, кэширование, масштабирование, observability).

Индустриальные кейсы (поддержка клиентов, e‑commerce, соцсети) демонстрируют устойчивый компромисс: быстрый CPU‑модель на «краю» (edge/API) и периодическая переразметка/обогащение данными из более тяжёлых моделей в бэкенд‑конвейере. Такая «двухуровневая» стратегия позволяет удерживать низкую задержку в онлайне и постепенно подтягивать качество за счёт улучшения обучающих выборок.

## Определения, сокращения и допущения
- NLP — обработка естественного языка.
- TF‑IDF — взвешивание термина по частоте в документе и его редкости в корпусе.
- Hashing Trick — проекция признаков в фиксированное пространство без явного словаря.
- SGD — стохастический градиентный спуск.
- SLO/SLA — целевые показатели/соглашения об уровне сервиса.
- P95/P99 — 95‑й и 99‑й процентили распределения задержек.
- Домен — предметная область, в которой характерны свои лексические маркеры.
Допущения: рассматриваются короткие тексты (1–3 предложения), язык русский/английский или смешанный; ресурсы — CPU‑окружение без GPU; требования к SLA — повышенные (онлайн‑контур продукта).

## Требования к программно‑аппаратной среде
- ОС: Linux (семейство Debian/Ubuntu/CentOS) или совместимая.
- Ресурсы: от 1 vCPU / 512 MB RAM для пилота; для продакшена — 2–4 vCPU / 2–4 GB RAM на экземпляр.
- Зависимости: Python 3.11, библиотеки scikit‑learn, numpy, pandas, FastAPI, SQLAlchemy, Prometheus client.
- Инфраструктура: Docker/Compose, реверс‑прокси (Nginx/Apache), система мониторинга (Prometheus/Grafana).

## Подробное описание интерфейсов API (словесно)
Эндпоинт `/predict` принимает список текстов и возвращает вероятности по классам и финальные метки; `/product/score` агрегирует метки в числовую оценку (1–5) и сохраняет результаты для аналитики; `/feedback` принимает размеченные пользователем пары «текст–метка», которые используются для онлайн‑обновления и последующего переобучения; `/labels` возвращает полный список поддерживаемых меток; `/retrain` инициирует офлайн‑переобучение; `/metrics` отдаёт телеметрию Prometheus.

Контракты устойчивы к эволюции: сервер может расширять поля ответа, не нарушая клиентов; критично только наличие базовых ключей (метки/вероятности/оценка). Безопасность POST‑методов обеспечивается заголовком `X-API-Key`.

## Инструкции по развёртыванию (описательно)
1) Подготовить окружение (.env), указать ключ API и параметры хранилища. 2) Собрать и запустить контейнер(ы) через Docker Compose; дождаться готовности health‑чека. 3) Подключить реверс‑прокси с TLS и правилами безопасности. 4) Настроить сбор метрик Prometheus и базовые алерты. 5) Выполнить smoke‑тесты API и провести сертификацию нагрузки (нагрузочное тестирование).

## План тестирования и контроль качества
- Функциональные тесты: корректность схем запросов/ответов, валидация ошибок (пустые тексты, неверный ключ).
- Интеграционные: проверки маршрутизации через реверс‑прокси, корректность заголовков и CORS.
- Нагрузочные: ступенчатое увеличение RPS, фиксация p50/p95/p99, анализ деградаций.
- Надёжность: симуляция отказов БД, заполнение диска, отключение воркера — сервис должен сохранять базовую функциональность.
- Качество ML: отчёт классификации, калибровка вероятностей, мониторинг «дрейфа» данных.

## Управление проектом и заинтересованные стороны
Стейкхолдеры: бизнес‑владелец, команда разработки, ML‑инженеры, эксплуатация (SRE/DevOps), служба поддержки. Роли и ответственность распределены по RACI‑матрице. Планирование — итерациями (2–3 недели), регулярные демо и ретроспективы. Риски фиксируются в реестре и снабжаются планами реагирования (mitigation/avoidance/acceptance).

## Пользовательский опыт и сценарии UI
Графический интерфейс решает задачи демонстрации и сбора фидбека: понятные метки, визуальные индикаторы уверенности, простое отправление размеченных примеров. Для интеграции в сторонние продукты важны стабильные контракты API и SDK, поддержка CORS, примеры для популярных фреймворков (React/Vue/.NET/Java/PHP).

## Кейс‑стади (примерная оценка)
Исходные данные: 10 000 отзывов, дисбаланс классов 2:5:3 (neg:neu:pos). Базовая модель (без онлайна) достигает Macro‑F1 ~0.78. После месяца сбора фидбека (≈5 000 размеченных юзером фрагментов) и еженедельного онлай‑обновления Macro‑F1 растёт до ~0.82, при этом p95 задержки остаётся < 120 мс на CPU. Стоимость владения — одна виртуальная машина общего назначения и базовая лицензия на мониторинг (опционально) либо полностью открытый стек.

## Варианты масштабирования
- Горизонтальное: несколько экземпляров API за балансировщиком; sticky‑session не требуется.
- Версионность модели: blue‑green выкладки с быстрым откатом; shadow‑прогон для проверки качества.
- Кэширование: повторяемые батчи запросов на уровне реверс‑прокси (осторожно с приватностью).
- Асинхронные конвейеры: вынесение ретрена в отдельные воркеры/джобы.

## Выводы по рискам и устойчивости
Главные риски — деградация качества из‑за дрейфа данных, всплески нагрузки и «ядовитая» обратная связь. Принятые меры (метрики, алерты, троттлинг, фильтрация, гибридный цикл обучения) позволяют удерживать стабильность и постепенно улучшать модель.


---

## Развёрнутое обоснование выбора методов
Выбор стека Hashing + TF‑IDF + SGDClassifier обусловлен необходимостью баланса между качеством, скоростью и простотой эксплуатации. В производственной среде важны предсказуемость задержек, ограниченное потребление памяти и возможность инкрементального обновления. Классические линейные модели при грамотной предобработке обеспечивают:
- стабильную латентность ввода‑вывода и CPU‑связанный характер нагрузки;
- отсутствие необходимости в загрузке больших эмбеддингов или языковых моделей;
- объяснимость: влияние признаков интерпретируемо (n‑граммы и их веса);
- простоту дообучения на новых примерах без полной ретренировки.

При этом компромиссы прозрачны: классическая модель может уступать трансформерам на сложных языковых феноменах (сарказм, контекстная ирония), однако при регулярном онлай‑обновлении и доменной настройке чаще всего достигает требуемого качества для бизнес‑задач «классификация отзывов».

## Сравнение с альтернативами (BERT/ruBERT, DistilBERT)
Трансформерные модели дают более высокие потолки качества, но требуют:
- большего времени инференса (в разы выше задержка на CPU);
- повышенных требований к памяти и иногда к ускорителям (GPU/ONNX‑оптимизации);
- сложного контура обновлений и прогрева кэшей.

В сценариях с высокой нагрузкой и жёсткими SLO по латентности (p95 < 150 мс) классический стек на CPU часто предпочтителен. Компромиссный путь — гибрид: классический быстрый фильтр + отложенная пересертификация сложной моделью в offline‑конвейере (re‑labeling), что улучшает качество данных для последующих итераций обучения.

## Методология и дизайн экспериментов
Для объективной оценки качества используются отложенная тестовая выборка и k‑fold кросс‑валидация. В качестве базовых метрик — Accuracy, Macro‑F1, а также отчёты классификации по классам. Для настройки порогов/веса классов применяются:
- анализ матрицы ошибок (ошибки между соседними классами чаще терпимы: neg↔neu, neu↔pos);
- балансировка классов параметром `class_weight='balanced'` при дисбалансе;
- обучение с контролем переобучения через регуляризацию и раннюю остановку (для SGD — ограничение числа эпох и настройка шага обучения).

Практика A/B‑наблюдения: часть трафика отправляется на «кандидатную» модель в тени (shadow), предсказания сравниваются офлайн, решения о выкладке принимаются по итогам стабильной динамики метрик.

## Эксплуатационные практики: SLO/SLA, алертинг, runbooks
- SLO: p95 задержки `/predict` < 150 мс при RPS до X; доступность 99.9% в рабочее время.
- Алерты: рост 5xx, всплески 429 (rate‑limit), деградация p95/p99, отсутствие метрик «сердцебиения» воркеров.
- Runbooks: пошаговые инструкции для типовых инцидентов (БД недоступна, заполнение диска, рост задержек, всплеск нагрузки, деградация качества).
- Логирование: строгое формирование JSON‑событий с обязательными полями (trace/request id, ip, endpoint, статус, длительность). Ротация и ретеншн логов.

## Модель угроз и безопасность
Атакующие сценарии: перебор API‑ключа, DoS по входу, отправка «ядовитых» данных (data poisoning), попытки SQL‑инъекций (в аналитических инструментах), вытекание конфиденциальных отзывов.
Меры:
- обязательный API‑ключ для модифицирующих методов; ротация, хранение в секретах;
- rate‑limit per IP/токен, лимиты на размер тела и глубину JSON;
- развёртывание за реверс‑прокси с TLS и базовым WAF;
- аудит источников данных и фильтрация/анонимизация перед записью в обучающие логи;
- минимально необходимые роли/доступы к БД, ведение журналов изменений схемы.

## Управление данными и приватность
Политика хранения определяет срок ретенции для `feedback_buffer.jsonl` и агрегатов. Допускается анонимизация/псевдонимизация пользовательских идентификаторов до записи. Для экспортов в аналитику — обезличивание, аудит выгрузок. При удалении запрошенных данных (право на удаление) формируются фильтры на уровне повторного обучения (исключение соответствующих записей в следующей итерации тренинга).

## Подробности онлайн‑обучения и устойчивость
Микро‑батчи формируются по двум условиям: «количество примеров» или «таймер», что предотвращает как слишком частые, так и слишком редкие обновления. Дедупликация примеров по тексту в пределах батча снижает шум от повторных отправок. Контроль `load average` не даёт online‑обновлениям конкурировать с критичными потоками. В крайних случаях очередь может сливаться в файл и использоваться только при офлайн‑ретрене.

## Оценка стоимости и профили нагрузки (TCO)
Базовый профиль: контейнер с 1–2 vCPU и 512–1024 MB RAM справляется с сотнями RPS при малых батчах. Хранение — десятки мегабайт логов в день при активной обратной связи. Наращивание производительности линейно: масштабирование по горизонтали + sticky‑less балансировка. Отсутствие GPU снижает стоимость владения; использование открытого ПО и стандартных компонентов упрощает сопровождение.

## План внедрения и roadmap
Этап 1 — MVP: базовые эндпоинты, offline‑обучение, деплой за прокси, метрики.
Этап 2 — Онлайн‑обновления и автотрен: буферизация фидбека, фоновые воркеры, контроль нагрузки.
Этап 3 — Наблюдаемость и безопасность: дашборды, алерты, ротация ключей, тестовые сценарии отказа.
Этап 4 — Улучшение качества: активное обучение, расширенные признаки, доменные словари/стоп‑слова.
Этап 5 — Интеграции: SDK, примеры для фреймворков, клиентские библиотеки и быстрые старты.

## Кейсы использования и пользовательские сценарии
Сценарий «Оценка товара»: партнёрская площадка собирает свежие отзывы, отправляет батч на `/product/score`, отображает сводную оценку и динамику по неделям. Разметка спорных отзывов пользователями идёт через `/feedback`, что обеспечивает адаптацию модели под конкретный домен (лексика и стиль аудитории).

## Формулы и интуиция обучения (детализация)
Логистическая потеря с L2‑регуляризацией в бинарном случае:

$$\mathcal{L}(w) = \sum_i \log\bigl(1 + e^{-y_i w^\top x_i}\bigr) + \frac{\lambda}{2}\lVert w \rVert_2^2$$

Градиент по весам: $\nabla_w \mathcal{L} = \sum_i (\sigma(w^\top x_i) - y_i) x_i + \lambda w$.
Для многоклассового случая применяется one‑vs‑rest или softmax‑вариант. Стохастические обновления выполняются на мини‑пакетах, что позволяет онлайн‑обучение без полной перетренировки модели.

## Будущие направления развития
- Семисупервизия и self‑training на неразмеченных данных;
- distillation: компактная линейная модель обучается на «подсказках» более сильного teacher‑моделя;
- расширенные признаки: эмбеддинги предложений (с отложенным офлайн‑обогащением);
- улучшенная нормализация для смешанных языков и транслитерации.

## Глоссарий
- Тональность (sentiment): эмоциональная окраска текста.
- TF‑IDF: взвешивание терминов по частоте в документе и обратной частоте по корпусу.
- Hashing trick: проекция признаков в фиксированное пространство без явного словаря.
- SGD: стохастический градиентный спуск — метод оптимизации по мини‑батчам.
- Partial fit: инкрементальное обновление модели без полного перерасчёта.
- SLO/SLA: цели по уровню сервиса/соглашение об уровне сервиса.
- Shadow traffic: параллельная отправка трафика на испытательную версию без влияния на пользователя.
