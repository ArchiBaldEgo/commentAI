"""Backfill data/feedback_buffer.jsonl from labeled CSVs.

Зачем:
- Веб-демо и ретрейн используют data/feedback_buffer.jsonl как буфер примеров.
- Если хочется, чтобы в demo уже было "много данных", можно дозаполнить JSONL
  из текущих CSV датасетов.

Безопасность:
- Скрипт НЕ удаляет существующий feedback_buffer.jsonl.
- Он только добавляет отсутствующие тексты (дедуп по text.lower()).

Формат JSONL совместим с sentiment.retrain_pipeline.merge_labeled_and_feedback():
- обязательные поля: text, label
- дополнительные поля игнорируются пайплайном, но полезны для демо
"""

from __future__ import annotations

import json
import time
import re
from pathlib import Path


ALLOWED = {"neg", "neu", "pos"}
_COMMA_SPACING = re.compile(r"\s*,\s*")


def iter_csv_rows(path: Path):
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        header_seen = False
        for raw_line in f:
            line = raw_line.strip("\n\r")
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
            if text.startswith('"') and text.endswith('"') and len(text) >= 2:
                text = text[1:-1]

            text = _COMMA_SPACING.sub(", ", text)
            text_norm = " ".join(text.split())
            if not text_norm:
                continue
            if label not in ALLOWED:
                continue

            yield text_norm, label


def load_existing_texts(jsonl_path: Path) -> set[str]:
    seen: set[str] = set()
    if not jsonl_path.exists():
        return seen

    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
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

    return seen


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    jsonl_path = data_dir / "feedback_buffer.jsonl"
    sources = [
        data_dir / "reviews_labeled.csv",
        data_dir / "hard_cases_labeled.csv",
    ]

    seen = load_existing_texts(jsonl_path)
    before = len(seen)

    to_append: list[dict] = []
    now = time.time()

    for src in sources:
        for text, label in iter_csv_rows(src):
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            to_append.append({
                "text": text,
                "label": label,
                "source": "csv_backfill",
                "ts": now,
            })

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    if to_append:
        with jsonl_path.open("a", encoding="utf-8") as f:
            for rec in to_append:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                appended += 1

    print(f"feedback_jsonl: {jsonl_path}")
    print(f"existing_unique_texts_before: {before}")
    print(f"appended_new_records: {appended}")
    print(f"existing_unique_texts_after: {len(seen)}")


if __name__ == "__main__":
    main()
