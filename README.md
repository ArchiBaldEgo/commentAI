# commentAI — сервис анализа тональности

Лёгкий в развёртывании сервис на FastAPI для классификации тональности (neg/neu/pos) с поддержкой гибридного непрерывного обучения, метрик, ограничений по запросам, безопасностью по API‑ключу и универсальным слоем хранения (файлы/БД).

## Особенности
- HTTP API совместимо «с любым стеком» (любой сайт/бэкенд).
- Онлайн‑обучение (микро‑батчи + периодический ретрейн) — без простоя.
- Минимальная настройка: `.env` + `docker-compose up -d`.
- Хранилище: файлы по умолчанию; опционально любая БД (SQLAlchemy).
- Метрики Prometheus, rate‑limit, JSON‑логирование.

## Структура
```
src/sentiment/
	api.py               # FastAPI приложение
	inference.py         # загрузка модели, predict_proba, online partial_fit
	retrain_pipeline.py  # слияние фидбека + базового датасета, тренировка, promotion
	train.py             # Hashing+SGD тренер для прод-модели
	preprocess.py        # простая нормализация текста
	storage.py           # файл/БД хранилище (SQLAlchemy)
data/                  # данные и артефакты (feedback, csv, sqlite)
models/production/     # актуальная модель (model.joblib, meta.json)
deploy/{systemd,nginx,apache}/  # шаблоны деплоя
```

## Быстрый старт
Вариант A (Docker):
```bash
cp .env.example .env
docker-compose up --build -d
# сервис на http://localhost:8000
# для PostgreSQL: make docker-pg
# для MySQL:     make docker-mysql
```
Вариант B (venv):
- PostgreSQL через Compose:
```bash
cp .env.example .env
bash scripts/run_docker.sh pg
# DB_URL уже проставлен на postgres:5432
```
- MySQL через Compose:
```bash
cp .env.example .env
bash scripts/run_docker.sh mysql
# DB_URL уже проставлен на mysql:3306
```
```bash
make bootstrap   # создаст venv, установит deps, обучит модель
make api         # запустит API на 8000
# Проверка готовности:
make check       # ждёт подъём /health
```

## Обучение модели вручную
```bash
source .venv/bin/activate
PYTHONPATH=src python -m sentiment.train --data data/reviews_labeled.csv --model-dir models/production
```
Файл `data/reviews_labeled.csv` должен содержать колонки `text,label` со значениями `neg|neu|pos`.

## Гибридный режим обучения
- Онлайн: установите `ONLINE_LEARNING=1` — фидбек из `/feedback` будет добавляться в микро‑батчи для `partial_fit`.
- Периодический ретрейн: включите `AUTO_RETRAIN_ENABLED=1` и настройте интервалы; либо вызывайте `/retrain` вручную.

## Запуск с БД (PostgreSQL/MySQL)
Переменные окружения:
- `STORAGE_MODE=db` — включить запись в БД.
- `DB_URL` — строка подключения SQLAlchemy: `postgresql+psycopg2://user:pass@host:5432/db` или `mysql+pymysql://user:pass@host:3306/db`.

Docker Compose оверлеи:
```bash
# PostgreSQL
docker compose -f docker-compose.yml -f deploy/compose.postgres.yml up -d
# MySQL
docker compose -f docker-compose.yml -f deploy/compose.mysql.yml up -d
```
Таблицы создаются автоматически: `feedback`, `product_scores`. Даже при режиме БД записи также дублируются в файлы (`data/feedback_buffer.jsonl`, `data/product_scores.csv`).

## Конфигурация (.env)
- Security: `API_KEY=secret`
- Онлайн‑обучение (гибрид):
	- `ONLINE_LEARNING=1`
	- `ONLINE_BATCH_SIZE=16`, `ONLINE_BATCH_INTERVAL_SECONDS=30`, `ONLINE_MAX_LOAD=4.0`
- Авто‑ретрейн: `AUTO_RETRAIN_ENABLED=1`, `AUTO_RETRAIN_INTERVAL_SECONDS=600`, `AUTO_RETRAIN_MIN_ITEMS=50`
- Rate‑limit: `RATE_LIMIT_WINDOW_SECONDS=60`, `RATE_LIMIT_MAX_REQUESTS=120`, `RATE_LIMIT_WHITELIST=/health,/metrics`
- Хранилище: `STORAGE_MODE=file|db`, `DB_URL=` (пусто → SQLite в `data/app.db`)
	- PostgreSQL: `postgresql+psycopg2://user:pass@host:5432/db`
	- MySQL: `mysql+pymysql://user:pass@host:3306/db`
	- Для Docker Compose используйте оверлеи `deploy/compose.postgres.yml` и `deploy/compose.mysql.yml`.

## Запуск с БД (PostgreSQL/MySQL)

- Переменные окружения:
	- `STORAGE_MODE=db` — включить запись в БД
	- `DB_URL` — строка подключения SQLAlchemy
- Docker Compose оверлеи:
	- PostgreSQL:
		```bash
		docker compose -f docker-compose.yml -f deploy/compose.postgres.yml up -d
		```
	- MySQL:
		```bash
		docker compose -f docker-compose.yml -f deploy/compose.mysql.yml up -d
		```
Таблицы создаются автоматически: `feedback` и `product_scores`.

## Эндпоинты
- `GET /health` — состояние
- `GET /labels` — список классов
- `POST /predict` — пакетные предсказания
	- тело: `{ "texts": ["Отличный товар", "Плохо"], "model_path": null }`
	- заголовок: `X-API-Key: <API_KEY>`
- `POST /product/score` — средняя оценка товара (1..5) + распределение
	- тело: `{ "productId": "SKU-1", "texts": [..] }`
- `POST /feedback` — загрузка размеченных примеров (собирается очередь для онлайн‑обучения)
	- тело: `{ "items": [{"text":"...","label":"pos"}] }`
- `POST /retrain` — фоноваое переобучение (сливает feedback + labeled, обучает, продвигает в production)
- `GET /metrics` — Prometheus метрики (запросы, латентность, `sentiment_online_updates_total`)

Примеры:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/labels
curl -X POST http://localhost:8000/predict \
	-H 'Content-Type: application/json' -H 'X-API-Key: secret' \
	-d '{"texts":["Отличный товар","Плохая упаковка"]}'
curl -X POST http://localhost:8000/feedback \
	-H 'Content-Type: application/json' -H 'X-API-Key: secret' \
	-d '{"items":[{"text":"Супер","label":"pos"},{"text":"Плохо","label":"neg"}]}'
curl http://localhost:8000/metrics
```

## Хранилище данных
- По умолчанию всё пишется в файлы: `data/feedback_buffer.jsonl`, `data/product_scores.csv`.
- Режим БД: установите `STORAGE_MODE=db` и (необязательно) `DB_URL`; таблицы `feedback` и `product_scores` создадутся автоматически.
- Файловые артефакты сохраняются и при DB‑режиме (совместимость для ретрейна).

## Модель и обучение
- Продовая модель: Hashing+SGD (`models/production/model.joblib`).
- Онлайн‑обучение: микро‑батчи из `/feedback` с дедупликацией и ограничением нагрузки по loadavg.
- Авто‑ретрейн: периодически переобучает на объединении базы + фидбек.
- Ручной тренинг (пример):
```bash
PYTHONPATH=src python -m sentiment.train --data data/sample_reviews.csv --model-dir models/production --char-ngrams
```

## Логирование и безопасность
- JSON‑логи запросов: метод, путь, статус, время, IP, UA.
- Rate‑limit с белым списком путей (`/health`, `/metrics`).
- Безопасность: `X-API-Key` обязателен для всех POST (если `API_KEY` задан).

## Метрики и мониторинг
- `sentiment_requests_total{endpoint,method,status}`
- `sentiment_request_latency_seconds{endpoint,method}`
- `sentiment_online_updates_total`

## Деплой
- Docker/Compose: `.env` + `docker-compose up -d`.
- systemd: `deploy/systemd/sentiment.service` (обновите `WorkingDirectory` и путь к venv), затем:
```bash
sudo cp -r . /opt/sentiment
cd /opt/sentiment && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
sudo cp deploy/systemd/sentiment.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now sentiment
```
- Nginx: `deploy/nginx/sentiment.conf` (proxy на 127.0.0.1:8000)
- Apache: `deploy/apache/sentiment.conf`

## Troubleshooting
- Нет модели: обучите базовую — `make train`.
- ImportError (SQLAlchemy/драйвер): `pip install -r requirements.txt` или добавьте нужный драйвер.
- 401 Unauthorized: проверьте заголовок `X-API-Key` и значение `API_KEY`.
- 429 Too Many Requests: уменьшите нагрузку или увеличьте лимиты (`RATE_LIMIT_*`).
- Высокая загрузка: увеличьте `ONLINE_BATCH_INTERVAL_SECONDS`, уменьшите `ONLINE_BATCH_SIZE`, понизьте `ONLINE_MAX_LOAD`.
- БД недоступна: проверьте `DB_URL` и сетевую доступность; при необходимости вернитесь к `STORAGE_MODE=file`.

## Лицензия
MIT

## TL;DR
- Docker без БД: `cp .env.example .env && bash scripts/run_docker.sh && make check`
- Docker + PostgreSQL: `cp .env.example .env && make docker-pg && make check`
- Docker + MySQL: `cp .env.example .env && make docker-mysql && make check`
- Локально: `make bootstrap && make api && make check`

## Примеры интеграции
- JavaScript (fetch, браузер):
```js
const API = 'http://your-host:8000';
const API_KEY = 'secret';

async function predict(texts) {
	const res = await fetch(`${API}/predict`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'X-API-Key': API_KEY
		},
		body: JSON.stringify({ texts })
	});
	if (!res.ok) throw new Error(await res.text());
	return await res.json();
}

async function sendFeedback(items) {
	const res = await fetch(`${API}/feedback`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
		body: JSON.stringify({ items })
	});
	if (!res.ok) throw new Error(await res.text());
	return await res.json();
}

// Пример
predict(['Отличный товар', 'Плохая упаковка']).then(console.log);
```

- Node.js (axios):
```js
import axios from 'axios';
const api = axios.create({ baseURL: 'http://your-host:8000', headers: { 'X-API-Key': 'secret' }});

const pred = await api.post('/predict', { texts: ['Классно', 'Ужасно'] });
console.log(pred.data);

await api.post('/feedback', { items: [{ text: 'Классно', label: 'pos' }] });
```

- Python (requests):
```python
import requests
API = 'http://your-host:8000'
headers = {'X-API-Key': 'secret', 'Content-Type': 'application/json'}

r = requests.post(f'{API}/predict', json={'texts': ['Отлично', 'Плохо']}, headers=headers)
r.raise_for_status()
print(r.json())

fb = requests.post(f'{API}/feedback', json={'items': [{'text':'Средне','label':'neu'}]}, headers=headers)
fb.raise_for_status()
print(fb.json())
```

- PHP (cURL):
```php
<?php
$ch = curl_init('http://your-host:8000/predict');
$payload = json_encode([ 'texts' => ['Отлично', 'Плохо'] ]);
curl_setopt_array($ch, [
	CURLOPT_POST => 1,
	CURLOPT_HTTPHEADER => [ 'Content-Type: application/json', 'X-API-Key: secret' ],
	CURLOPT_POSTFIELDS => $payload,
	CURLOPT_RETURNTRANSFER => 1
]);
$resp = curl_exec($ch);
if ($resp === false) { throw new Exception(curl_error($ch)); }
echo $resp; 
```

- Java (OkHttp):
```java
OkHttpClient client = new OkHttpClient();
MediaType JSON = MediaType.parse("application/json; charset=utf-8");
RequestBody body = RequestBody.create(JSON, "{\"texts\":[\"Отлично\",\"Плохо\"]}");
Request req = new Request.Builder()
		.url("http://your-host:8000/predict")
		.addHeader("X-API-Key", "secret")
		.post(body)
		.build();
try (Response res = client.newCall(req).execute()) {
	if (!res.isSuccessful()) throw new IOException("Unexpected " + res);
	System.out.println(res.body().string());
}
```

- C# (.NET HttpClient):
```csharp
using var http = new HttpClient { BaseAddress = new Uri("http://your-host:8000") };
http.DefaultRequestHeaders.Add("X-API-Key", "secret");
var resp = await http.PostAsync("/predict", new StringContent("{\"texts\":[\"Отлично\"]}", Encoding.UTF8, "application/json"));
resp.EnsureSuccessStatusCode();
Console.WriteLine(await resp.Content.ReadAsStringAsync());
```

Подсказки:
- Для `/product/score` тело: `{ "productId": "SKU-1", "texts": ["..."] }`.
- Для `/feedback` тело: `{ "items": [{"text":"...","label":"pos|neu|neg"}] }`.
- Всегда добавляйте заголовок `X-API-Key` для POST, если `API_KEY` включён на сервере.

## SDKs
- TypeScript: см. `clients/typescript`
	- Установка: `cd clients/typescript && npm install && npm run build`
	- Использование:
		```ts
		import { SentimentClient } from './clients/typescript/dist/index.js';
		const client = new SentimentClient({ baseUrl: 'http://localhost:8000', apiKey: 'secret' });
		const res = await client.predict(['Отлично', 'Плохо']);
		```
- Python: см. `clients/python`
	- Установка: `cd clients/python && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
	- Использование:
		```python
		from sentiment_client import SentimentClient
		c = SentimentClient(base_url='http://localhost:8000', api_key='secret')
		print(c.predict(['good', 'bad']))
		```

## Сниппеты фреймворков
- React (hooks):
	```tsx
	import { useEffect, useState } from 'react';
	import { SentimentClient } from '../clients/typescript/dist/index.js';

	const client = new SentimentClient({ baseUrl: import.meta.env.VITE_API_URL, apiKey: import.meta.env.VITE_API_KEY });

	export function UsePredictions({ texts }: { texts: string[] }) {
		const [data, setData] = useState<any>(null);
		const [err, setErr] = useState<string | null>(null);
		useEffect(() => {
			client.predict(texts).then(setData).catch(e => setErr(String(e)));
		}, [texts]);
		if (err) return <div>Error: {err}</div>;
		return <pre>{JSON.stringify(data, null, 2)}</pre>;
	}
	```

- Vue 3 (Composition API):
	```ts
	import { ref, onMounted } from 'vue';

	export function useSentiment(texts: string[], baseUrl: string, apiKey?: string) {
		const data = ref<any>(null);
		const error = ref<string | null>(null);
		onMounted(async () => {
			try {
				const res = await fetch(`${baseUrl}/predict`, {
					method: 'POST',
					headers: { 'Content-Type': 'application/json', ...(apiKey ? { 'X-API-Key': apiKey } : {}) },
					body: JSON.stringify({ texts })
				});
				if (!res.ok) throw new Error(await res.text());
				data.value = await res.json();
			} catch (e:any) { error.value = String(e.message || e); }
		});
		return { data, error };
	}
	```

- Laravel (HTTP client):
	```php
	use Illuminate\Support\Facades\Http;

	$resp = Http::withHeaders([
			'X-API-Key' => env('SENTIMENT_API_KEY', 'secret')
		])->post(env('SENTIMENT_URL', 'http://localhost:8000').'/predict', [
			'texts' => ['Отлично', 'Плохо']
		])->json();
	```

- Spring Boot (WebClient):
	```java
	@Bean
	WebClient sentimentWebClient(WebClient.Builder b) {
		return b.baseUrl("http://localhost:8000").defaultHeader("X-API-Key", "secret").build();
	}

	public Mono<Map> predict(WebClient wc, List<String> texts) {
		return wc.post().uri("/predict")
			.bodyValue(Map.of("texts", texts))
			.retrieve().bodyToMono(Map.class);
	}
	```

- .NET (IHttpClientFactory):
	```csharp
	public class SentimentService {
		private readonly HttpClient _http;
		public SentimentService(HttpClient http) { _http = http; }
		public async Task<string> PredictAsync(IEnumerable<string> texts) {
			using var content = new StringContent(System.Text.Json.JsonSerializer.Serialize(new { texts }), System.Text.Encoding.UTF8, "application/json");
			var req = new HttpRequestMessage(HttpMethod.Post, "/predict");
			req.Headers.Add("X-API-Key", "secret");
			req.Content = content;
			var resp = await _http.SendAsync(req);
			resp.EnsureSuccessStatusCode();
			return await resp.Content.ReadAsStringAsync();
		}
	}
	// Program.cs
	builder.Services.AddHttpClient<SentimentService>(c => c.BaseAddress = new Uri("http://localhost:8000"));
	```

## Справочник переменных окружения
- `API_KEY`: ключ для заголовка `X-API-Key` (обязателен для POST, если задан)
- Онлайн‑обучение:
	- `ONLINE_LEARNING` (0/1): включить очередь микро‑батчей
	- `ONLINE_BATCH_SIZE` (16): размер пакета partial_fit
	- `ONLINE_BATCH_INTERVAL_SECONDS` (30): макс. интервал между пакетами
	- `ONLINE_MAX_LOAD` (4.0): порог loadavg (1m), выше — откладывать обучение
- Авто‑ретрейн:
	- `AUTO_RETRAIN_ENABLED` (0/1)
	- `AUTO_RETRAIN_INTERVAL_SECONDS` (600)
	- `AUTO_RETRAIN_MIN_ITEMS` (50)
- Rate‑limit:
	- `RATE_LIMIT_WINDOW_SECONDS` (60)
	- `RATE_LIMIT_MAX_REQUESTS` (120)
	- `RATE_LIMIT_WHITELIST` (/health,/metrics)
- Хранилище:
	- `STORAGE_MODE` (file|db)
	- `DB_URL` (по умолчанию SQLite в `data/app.db`)
	- Postgres: `PG_DB`, `PG_USER`, `PG_PASSWORD`
	- MySQL: `MYSQL_DB`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_ROOT_PASSWORD`
- Логи/уровень:
	- `LOG_LEVEL` (INFO|DEBUG|WARN|ERROR)

## Makefile — шпаргалка
- `make bootstrap` — venv + зависимости + обучить стартовую модель
- `make api` — запустить API локально
- `make train` — переобучить прод‑модель на sample датасете
- `make docker-up` / `make docker-down` — базовый compose
- `make docker-pg` — API + PostgreSQL
- `make docker-mysql` — API + MySQL
- `make check` — дождаться `/health`

## Рекомендации для продакшна
- Uvicorn/Workers: используйте процесс‑менеджер (systemd/Supervisor) и несколько воркеров через gunicorn+uvicorn workers при высоком трафике.
- Reverse proxy: включите прокидывание `X-Forwarded-*`, rate‑limit на уровне Nginx/Apache, TLS termination.
- Secrets: храните `API_KEY` и `DB_URL` в секретах (Docker/Swarm/K8s/OS env), не коммитьте .env.
- Мониторинг: подключите Prometheus scrape к `/metrics`; графики латентности/ошибок и счётчик `sentiment_online_updates_total`.
- Ротация данных: периодически архивируйте `data/feedback_buffer.jsonl` и чистите старые версии моделей.
- Бэкапы БД: для Postgres/MySQL настройте регулярные дампы/снапшоты volume’ов.


## Быстрый старт для разработчика
```bash
make bootstrap   # создаст venv, установит зависимости, подготовит данные и обучит модель
make api         # запустит API на 8000
# или одним шагом через Docker
bash scripts/run_docker.sh
```

## Подключение любой БД через SQLAlchemy
- Выберите драйвер (уже предустановлены: PostgreSQL `psycopg2-binary`, MySQL `pymysql`).
- В `.env` установите:
	- `STORAGE_MODE=db`
	- `DB_URL=postgresql+psycopg2://user:pass@host:5432/dbname` (пример)
	- При пустом `DB_URL` используется SQLite в `data/app.db`.
- Таблицы создаются автоматически: `feedback`, `product_scores`.

## Шаблоны для сервера
- systemd: `deploy/systemd/sentiment.service` (укажите путь к проекту и пользователю)
- Nginx: `deploy/nginx/sentiment.conf`
- Apache: `deploy/apache/sentiment.conf`
