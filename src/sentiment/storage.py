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
