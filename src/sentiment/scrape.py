import asyncio
import re
from dataclasses import dataclass
from typing import List, Optional, Iterable, Dict, Any

import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent  # может выбрасывать ошибки сети

DEFAULT_TIMEOUT = 20

@dataclass
class FetchedPage:
    url: str
    status: int
    text: str

USER_AGENT = UserAgent()

async def fetch(session: aiohttp.ClientSession, url: str) -> FetchedPage:
    try:
        async with session.get(url, timeout=DEFAULT_TIMEOUT) as resp:
            txt = await resp.text(errors="ignore")
            return FetchedPage(url=url, status=resp.status, text=txt)
    except Exception as e:
        return FetchedPage(url=url, status=0, text=str(e))

async def fetch_all(urls: List[str], concurrency: int = 5) -> List[FetchedPage]:
    connector = aiohttp.TCPConnector(limit=concurrency, ssl=False)
    headers = {"User-Agent": USER_AGENT.random}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = [fetch(session, u) for u in urls]
        return await asyncio.gather(*tasks)

CSSSelect = str

@dataclass
class ExtractRule:
    container: CSSSelect  # CSS селектор контейнера отзыва
    text: Optional[CSSSelect] = None  # если нужно до конкретного узла


def extract_reviews(html: str, rule: ExtractRule) -> List[str]:
    soup = BeautifulSoup(html, 'lxml')
    out: List[str] = []
    containers = soup.select(rule.container)
    for c in containers:
        node = c
        if rule.text:
            t = c.select_one(rule.text)
            if t:
                node = t
        text = node.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        if text and len(text.split()) >= 2:
            out.append(text)
    return out

async def collect(urls: List[str], rule: ExtractRule) -> List[str]:
    pages = await fetch_all(urls)
    reviews: List[str] = []
    for p in pages:
        if p.status == 200:
            reviews.extend(extract_reviews(p.text, rule))
    # dedup preserve order
    seen = set()
    uniq = []
    for r in reviews:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq

if __name__ == "__main__":
    import sys, json
    # Пример: python -m src.sentiment.scrape "https://example.com/page1" "https://example.com/page2" 'div.review' 'p.text'
    if len(sys.argv) < 3:
        print("Usage: python -m src.sentiment.scrape <url1> <url2> ... <container_selector> [text_selector]", file=sys.stderr)
        sys.exit(1)
    *url_list, container_selector = sys.argv[1:-1], sys.argv[-1]
    text_selector = None
    # Упростим интерфейс для демо
    rule = ExtractRule(container=container_selector, text=text_selector)
    res = asyncio.run(collect(url_list, rule))
    print(json.dumps(res, ensure_ascii=False, indent=2))
