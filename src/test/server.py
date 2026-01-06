from pathlib import Path
from typing import List
import sys
import os
import json
import time
from threading import Thread


def _get_app_root() -> Path:
    """Корень проекта/папки релиза.

    - В режиме .exe (PyInstaller) считаем корнем папку, где лежит exe.
      Рядом с ней должны лежать папки data/ и models/.
    - В режиме разработки — корень репозитория.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


APP_ROOT = _get_app_root()

# ВАЖНО: чтобы относительные пути (data/, models/) работали одинаково
# в dev и в собранном приложении, фиксируем рабочую директорию.
try:
    os.chdir(APP_ROOT)
except OSError:
    pass

# Обеспечиваем доступность пакета sentiment.
SRC_DIR = APP_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from sentiment.inference import load_model_cached, predict_proba_texts, online_partial_fit
from sentiment.retrain_pipeline import run_retrain
from sentiment.storage import init_storage, store_feedback

# Исправляем путь к шаблонам для PyInstaller
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # В режиме exe (onefile) шаблоны лежат в _MEIPASS/templates
    BASE_DIR = Path(sys._MEIPASS)
else:
    # В режиме разработки или onedir — шаблоны в папке рядом с server.py
    BASE_DIR = Path(__file__).resolve().parent

TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(title="commentAI test GUI")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Debug: убедимся, что шаблоны найдены
if not TEMPLATES_DIR.exists():
    raise FileNotFoundError(
        f"Папка с шаблонами не найдена: {TEMPLATES_DIR}\n"
        f"APP_ROOT={APP_ROOT}, BASE_DIR={BASE_DIR}, frozen={getattr(sys, 'frozen', False)}"
    )

COMMENTS: List[dict] = []
LAST_PROBS: dict | None = None  # вероятности для последнего комментария
LAST_PRED_LABEL: str | None = None
LAST_SAVED_LABEL: str | None = None
LAST_MODEL_VERSION: str | None = None


@app.on_event("startup")
def _startup_seed_data() -> None:
    # Чтобы на новом устройстве (Windows) буфер не был пустым,
    # а преподавателю сразу показывались готовые примеры.
    init_storage()
    _load_history()


def _read_json_file(path: Path) -> dict | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0
    except OSError:
        return 0


def _get_model_version() -> str:
    """Версия текущей модели (models/production/VERSION или meta.json)."""
    prod = APP_ROOT / "models" / "production"
    version_file = prod / "VERSION"
    if version_file.exists():
        try:
            return version_file.read_text(encoding="utf-8").strip() or "unknown"
        except Exception:
            pass
    meta = _read_json_file(prod / "meta.json")
    if meta and isinstance(meta.get("version"), str):
        return meta["version"]
    return "unknown"


def label_to_score(label: str, probs: dict | None = None) -> int:
    """Преобразуем метку в оценку 1–3.

    фиксированная шкала
    neg -> 1, neu -> 2, pos -> 3.

    Параметр probs оставлен для совместимости (не используется).
    """
    _ = probs
    if label == "neg":
        return 1
    if label == "neu":
        return 2
    if label == "pos":
        return 3
    return 2


def _load_history() -> None:
    """Загружаем историю комментариев из feedback_buffer.jsonl при старте.

    Это нужно только для демонстрации: чтобы после
    перезапуска сервера на странице снова отображались предыдущие
    комментарии и их оценки.
    """
    from sentiment.storage import FEEDBACK_FILE

    if not FEEDBACK_FILE.exists():
        return

    try:
        with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text")
                label = rec.get("label")
                pred_label = rec.get("predicted")
                if not text or not label:
                    continue
                score = label_to_score(str(label))
                COMMENTS.append({
                    "text": text,
                    "label": str(label),
                    "pred_label": str(pred_label) if pred_label else None,
                    "score": score,
                })
    except OSError:
        # если файл по какой-то причине недоступен — просто пропускаем
        pass


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if COMMENTS:
        avg_score = sum(c["score"] for c in COMMENTS) / len(COMMENTS)
    else:
        avg_score = None

    # Данные для демонстрации преподавателю
    from sentiment.storage import FEEDBACK_FILE

    retrain_status = _read_json_file(APP_ROOT / "data" / "retrain_status.json")
    model_version = _get_model_version()
    feedback_lines = _count_lines(FEEDBACK_FILE)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "comments": COMMENTS[::-1],  # последние сверху
            "avg_score": avg_score,
            "last_probs": LAST_PROBS,
            "last_pred_label": LAST_PRED_LABEL,
            "last_saved_label": LAST_SAVED_LABEL,
            "model_version": model_version,
            "retrain_status": retrain_status,
            "feedback_lines": feedback_lines,
            "app_root": str(APP_ROOT),
            "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


@app.post("/predict")
async def predict(comment: str = Form(...), label_override: str = Form("model")):
    comment = comment.strip()
    if not comment:
        return RedirectResponse("/", status_code=303)

    model = load_model_cached(None)
    probs, preds, labels = predict_proba_texts(model, [comment])
    pred_label = preds[0]

    # Для доказуемого обучения: преподаватель может выбрать "правильную" метку.
    allowed = {"neg", "neu", "pos"}
    saved_label = pred_label
    user_overrode = label_override in allowed
    if user_overrode:
        saved_label = label_override

    # Если пользователь поправил метку — сразу дообучим модель и
    # пересчитаем вероятности, чтобы в UI было видно изменение.
    if user_overrode:
        try:
            online_partial_fit([comment], [saved_label], None)
            # модель в кеше уже обновлена, но пересчёт вероятностей нужен для отображения
            probs, preds, labels = predict_proba_texts(model, [comment])
            pred_label = preds[0]
        except Exception:
            pass

    global LAST_PROBS
    LAST_PROBS = {labels[i]: float(probs[0][i]) for i in range(len(labels))}

    global LAST_PRED_LABEL, LAST_SAVED_LABEL, LAST_MODEL_VERSION
    LAST_PRED_LABEL = pred_label
    LAST_SAVED_LABEL = saved_label
    LAST_MODEL_VERSION = _get_model_version()

    # Важно для UX: если пользователь выставил метку вручную,
    # оценка/среднее должны следовать этой метке.
    score = label_to_score(saved_label, LAST_PROBS)

    COMMENTS.append({
        "text": comment,
        "label": saved_label,
        "pred_label": pred_label,
        "score": score,
    })

    # пишем в feedback_buffer.jsonl через общий сторедж (файл + опционально БД)
    try:
        store_feedback([{
            "text": comment,
            "label": saved_label,
            "source": "gui",
            "user_id": None,
            "predicted": pred_label,
            "probs": LAST_PROBS,
            "model_version": LAST_MODEL_VERSION,
        }])
    except Exception:
        pass
    return RedirectResponse("/", status_code=303)


@app.post("/retrain_now")
async def retrain_now():
    """Ручной запуск переобучения (для демонстрации преподавателю)."""

    def task() -> None:
        try:
            run_retrain()
        except Exception:
            pass

    Thread(target=task, daemon=True).start()
    return RedirectResponse("/", status_code=303)


if __name__ == "__main__":
    import uvicorn
    from time import sleep

    # На старте инициализируем сторедж и поднимаем историю комментариев.
    init_storage()
    _load_history()

    def auto_retrain_loop() -> None:
        while True:
            try:
                run_retrain()
            except Exception:
                # для простого GUI можно молча игнорировать ошибки
                pass
            sleep(10)

    # фоновая переобучалка каждые 10 секунд
    t = Thread(target=auto_retrain_loop, daemon=True)
    t.start()

    uvicorn.run(app, host="127.0.0.1", port=9000)
