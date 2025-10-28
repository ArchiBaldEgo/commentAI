import hashlib
import os
from typing import Iterable, List, Optional
import pandas as pd

HASH_COL = "_hash"


def text_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode('utf-8')).hexdigest()


def load_dataset(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["text", "sentiment"])  # empty


def append_texts(path: str, texts: Iterable[str], default_label: Optional[str] = None) -> int:
    df = load_dataset(path)
    # Ensure columns
    if 'text' not in df.columns:
        df['text'] = []
    if 'sentiment' not in df.columns:
        df['sentiment'] = []

    existing_hashes = set(df['text'].dropna().map(text_hash)) if not df.empty else set()

    new_rows = []
    for t in texts:
        h = text_hash(t)
        if h in existing_hashes:
            continue
        new_rows.append({"text": t, "sentiment": default_label if default_label else ""})
        existing_hashes.add(h)

    if not new_rows:
        return 0

    df_new = pd.DataFrame(new_rows)
    df_out = pd.concat([df, df_new], ignore_index=True)
    df_out.to_csv(path, index=False)
    return len(new_rows)


def balance_downsample(df: pd.DataFrame, label_col: str = 'sentiment', random_state: int = 42) -> pd.DataFrame:
    groups = df.groupby(label_col)
    min_count = groups.size().min()
    return groups.sample(min_count, random_state=random_state).reset_index(drop=True)


def unlabeled_rows(path: str) -> pd.DataFrame:
    df = load_dataset(path)
    return df[df['sentiment'].isna() | (df['sentiment'].astype(str).str.strip() == '')]
