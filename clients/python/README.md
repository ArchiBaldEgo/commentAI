# Sentiment SDK (Python)

Minimal Python client for the sentiment API.

Install deps:

```bash
cd clients/python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Usage:

```python
from sentiment_client import SentimentClient

client = SentimentClient(base_url='http://localhost:8000', api_key='secret')
print(client.health())
print(client.labels())
print(client.predict(['Отлично', 'Плохо']))
print(client.product_score('SKU-1', ['Хорошо', 'Плохо']))
client.feedback([{'text': 'Хорошо', 'label': 'pos'}])
```