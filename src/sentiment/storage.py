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
SEED_FEEDBACK_FILE = DATA_DIR / "seed_feedback.jsonl"
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

    # На свежем устройстве feedback_buffer.jsonl может отсутствовать
    # (его нет в git по умолчанию). Чтобы у преподавателя сразу были
    # «готовые слова», подсеваем буфер из seed/CSV.
    try:
        ensure_feedback_seed(min_lines=int(os.getenv("FEEDBACK_SEED_MIN_LINES", "200")))
    except Exception:
        pass


def _iter_seed_examples() -> Iterable[dict]:
    """Итерируем стартовые примеры для заполнения feedback_buffer.jsonl.

    Источники (в порядке приоритета):
    1) data/seed_feedback.jsonl (tracked in git)
    2) data/reviews_labeled.csv + data/hard_cases_labeled.csv
    """
    # 1) Seed JSONL, если он существует
    if SEED_FEEDBACK_FILE.exists():
        try:
            with SEED_FEEDBACK_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    text = rec.get("text")
                    label = rec.get("label")
                    if isinstance(text, str) and text.strip() and isinstance(label, str) and label.strip():
                        yield {"text": " ".join(text.split()), "label": label.strip().lower(), "source": "seed"}
        except OSError:
            pass

    # 2) CSV-датасеты
    for csv_name in ("reviews_labeled.csv", "hard_cases_labeled.csv"):
        p = DATA_DIR / csv_name
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                header_seen = False
                for raw_line in f:
                    line = raw_line.strip("\n\r")
                    if not line.strip():
                        continue
                    if not header_seen:
                        header_seen = True
                        continue
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) < 2:
                        continue
                    label = parts[-1].strip().lower()
                    text = ",".join(parts[:-1]).strip()
                    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
                        text = text[1:-1]
                    text = " ".join(text.split())
                    if not text:
                        continue
                    if label not in {"neg", "neu", "pos"}:
                        continue
                    yield {"text": text, "label": label, "source": f"seed:{csv_name}"}
        except OSError:
            continue


def ensure_feedback_seed(min_lines: int = 40) -> int:
    """Гарантирует, что feedback_buffer.jsonl не пустой.

    Если файл отсутствует или содержит слишком мало строк —
    добавляем примеры из seed/CSV, с дедупликацией по тексту.
    Возвращает количество добавленных строк.
    """
    min_lines = max(0, int(min_lines))

    def _count_lines(p: Path) -> int:
        try:
            with p.open("r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0
        except OSError:
            return 0

    current = _count_lines(FEEDBACK_FILE)
    if current >= min_lines:
        return 0

    seen: set[str] = set()
    if FEEDBACK_FILE.exists():
        try:
            with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    text = rec.get("text")
                    if isinstance(text, str) and text.strip():
                        seen.add(" ".join(text.split()).lower())
        except OSError:
            pass

    # Дозаполняем до min_lines
    to_write: list[dict] = []
    for rec in _iter_seed_examples():
        text = rec.get("text")
        label = rec.get("label")
        if not isinstance(text, str) or not text.strip():
            continue
        if label not in {"neg", "neu", "pos"}:
            continue
        key = " ".join(text.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        to_write.append({
            "ts": time.time(),
            "text": " ".join(text.split()),
            "label": label,
            "source": rec.get("source") or "seed",
            "user_id": None,
        })
        if current + len(to_write) >= min_lines:
            break

    if not to_write:
        return 0

    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        for rec in to_write:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(to_write)


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
