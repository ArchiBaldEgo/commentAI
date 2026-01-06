from __future__ import annotations

from pathlib import Path
import re

ALLOWED = {"neg", "neu", "pos"}


_COMMA_SPACING = re.compile(r"\s*,\s*")


def iter_rows(path: Path):
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

            # Делаем текст «читабельным»: единый пробел после запятых + схлопываем пробелы
            text = _COMMA_SPACING.sub(", ", text)
            text_norm = " ".join(text.split())
            if not text_norm:
                continue
            if label not in ALLOWED:
                continue
            yield text_norm, label


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base_path = root / "data" / "reviews_labeled.csv"
    hard_path = root / "data" / "hard_cases_labeled.csv"
    synth_path = root / "data" / "reviews_labeled_synth.csv"

    if not base_path.exists():
        raise SystemExit(f"Base file not found: {base_path}")
    if not synth_path.exists():
        raise SystemExit(f"Synth file not found: {synth_path}")

    synth_rows = list(iter_rows(synth_path))

    def merge_into(target: Path) -> None:
        if not target.exists():
            return
        existing = list(iter_rows(target))
        seen = set()
        merged: list[tuple[str, str]] = []

        def add(rows):
            for text, label in rows:
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append((text, label))

        add(existing)
        add(synth_rows)

        out_lines = ["text,label\n"]
        out_lines += [f"{text},{label}\n" for text, label in merged]
        target.write_text("".join(out_lines), encoding="utf-8")
        print(f"Merged into {target} | existing={len(existing)} synth={len(synth_rows)} merged={len(merged)}")

    # Эти файлы реально используются при (пере)обучении:
    # - data/reviews_labeled.csv (база)
    # - data/hard_cases_labeled.csv (дополнительные примеры)
    merge_into(base_path)
    merge_into(hard_path)


if __name__ == "__main__":
    main()
