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
