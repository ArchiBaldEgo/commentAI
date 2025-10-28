import re
import string
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
from .lexicon import RU_POS_LEXICON, RU_NEG_LEXICON, RU_PRICE, RU_DELIVERY, RU_QUALITY, RU_SERVICE

# Простые списки стоп-слов (можно расширять)
RU_STOP = {
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да","ты","к","у","же","вы","за","бы","по","ее","мне","если","или","ни","мы","те","это","мой","от","меня","его","им","из","уже"
}
EN_STOP = {
    "the","a","an","and","or","if","to","is","are","was","were","be","been","of","for","in","on","at","by","with","this","that","it","as","from","but","not"
}

PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation})

LEMMA_RULES = [
    (re.compile(r"(ами|ями|ыми|ами|ов|ев)$"), ""),
    (re.compile(r"(ыми|ие|ий|ого|ему|ыми|их|ую|ое|ая|ые|ый)$"), ""),
    (re.compile(r"(ing|ed|ly|ness|ment|s)$"), ""),
]


def simple_lemma(token: str) -> str:
    original = token
    for pattern, repl in LEMMA_RULES:
        token = pattern.sub(repl, token)
    # Минимальная длина
    if len(token) < 3:
        token = original
    return token


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in RU_STOP and t not in EN_STOP]


def lemmatize(tokens: List[str]) -> List[str]:
    return [simple_lemma(t) for t in tokens]


def preprocess_text(text: str) -> str:
    text = normalize_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    # Лексикон: добавим специальные маркеры для усиления сигналов
    toks_set = set(tokens)
    tags: List[str] = []
    if toks_set & RU_POS_LEXICON:
        tags.append("__kw_pos")
    if toks_set & RU_NEG_LEXICON:
        tags.append("__kw_neg")
    if toks_set & RU_PRICE:
        tags.append("__aspect_price")
    if toks_set & RU_DELIVERY:
        tags.append("__aspect_delivery")
    if toks_set & RU_QUALITY:
        tags.append("__aspect_quality")
    if toks_set & RU_SERVICE:
        tags.append("__aspect_service")
    if tags:
        tokens.extend(tags)
    return " ".join(tokens)


class PreprocessTransformer(BaseEstimator, TransformerMixin):
    """Sklearn совместимый трансформер для пайплайна."""
    def fit(self, X, y=None):  # type: ignore
        return self

    def transform(self, X):  # type: ignore
        return [preprocess_text(x) for x in X]
