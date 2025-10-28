# Система анализа тональности текстовых отзывов

Простой проект для обучения модели классификации тональности (позитив / негатив / нейтраль) на основе TF-IDF + Logistic Regression.

## Быстрая интеграция: REST API + самообучение

Добавлен лёгкий сервис FastAPI с онлайновым обучением (HashingVectorizer + SGDClassifier). Он подходит для подключения из любого бэкенда/языка через HTTP/JSON и умеет:
- Предсказывать тональность и оценку в звёздах (1–5) по вероятностям классов
- Принимать обратную связь (label или рейтинг 1–5) и дообучаться на лету (partial_fit)

Запуск сервера:

```powershell
sentiment-cli serve --host 0.0.0.0 --port 8000 --model-dir models/online
```

Эндпоинты:
- POST /predict
	- Вход: { "texts": ["Очень плохо", "Отлично"] }
	- Выход: { "items": [{"text":"...","label":"neg|neu|pos","proba":{"neg":..,"neu":..,"pos":..},"stars":1..5}, ...] }
- POST /feedback
	- Вход: { "text": "Товар супер", "rating": 5 } или { "text": "Плохо", "label": "neg" }
	- Эффект: модель дообучается на одном примере и сохраняется в `models/online`
- GET /health — проверка готовности

Пакетное дообучение (без сервера):
```powershell
sentiment-cli train-batch --model-dir models\online --limit 100 --clear-after
```
Или через REST:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/train_batch" -Method POST -ContentType "application/json" -Body (@{ clear_after = $true; limit = 100 } | ConvertTo-Json)
```

Примечания:
- Используется HashingVectorizer: новые слова автоматически учитываются без переобучения словаря.
- Маппинг 1–5 звёзд: ожидание по вероятностям — 1*P(neg) + 3*P(neu) + 5*P(pos), округление к целому в [1;5].
- Обратная связь складывается в `models/online/feedback.jsonl` для аудита.

## Новые возможности автоматизации
- Единый CLI (`sentiment-cli`) с командами: train, predict, collect, label, evaluate, gui
- Makefile: install, train, predict, collect, label, pipeline
- Конфиг `config.yaml` для пайплайна
- Active learning (select_uncertain)
- GUI на tkinter с сохранением обратной связи пользователя и переобучением

## GUI
Запуск:
```bash
sentiment-cli gui
```
Функции:
- Загрузка модели
- Ввод текста и предсказание с вероятностями
- Сохранение пользовательской корректировки метки в `data/user_feedback.csv`
- Переобучение: перенос новых размеченных примеров в `data/reviews_labeled.csv` и повторный запуск `train`

Формат feedback (TSV):
```
timestamp	text	prediction	user_label
```

## Инкрементальное обучение
Используйте Hashing + SGD:
```bash
python -m src.sentiment.train --data data/reviews_labeled.csv --model-dir models/sgd --algo sgd --hashing --partial --char-ngrams
```

## Docker

Собрать и запустить через docker-compose (с Nginx реверс‑прокси, rate limit и gzip):
```powershell
docker compose up --build
```
Сервис будет доступен на http://localhost (через Nginx порт 80). Данные модели сохраняются в `./models`.

Замечания по прод-настройкам:
- Nginx проксирует к сервису Uvicorn (несколько воркеров), включает gzip, keep‑alive и rate limit (10 rps, burst 20).
- Дополнительно: лимиты одновременно по IP, по ключу (`X-API-Key`) и по эндпоинту (URI). Для тяжёлых маршрутов (`/feedback_bulk`, `/train_batch`) — более строгие лимиты.
- Логирование в JSON (access) на stdout контейнера; error лог — в stderr (формат Nginx). Для полностью JSON error‑логов нужен сторонний модуль — при необходимости можно подключить.
- Меняйте воркеры и таймауты через переменные окружения `UVICORN_WORKERS`, `UVICORN_TIMEOUT` в `docker-compose.yml`.
- Для защиты включите API‑ключ: добавьте `API_KEY` в `.env` и передавайте заголовок `X-API-Key`.

## Отбор неуверенных примеров
```bash
python -m src.sentiment.select_uncertain --model-dir models/production --data data/reviews_labeled.csv --threshold 0.5 --output data/uncertain.csv
```

## Сбор → разметка → обучение цикл
1. collect (парсинг)
2. label (ручная разметка)
3. train (обучение)
4. gui (сбор пользовательских поправок)
5. select_uncertain (active learning) → возврат к шагу 2

## Упаковка в исполняемый файл (PyInstaller)
Установка:
```bash
pip install pyinstaller
```
Сборка GUI exe (Windows пример):
```bash
pyinstaller --name sentiment_gui --onefile --noconsole -p src --add-data "models/production:model" src/sentiment/gui.py
```
CLI вариант:
```bash
pyinstaller --name sentiment_cli --onefile -p src -m src.sentiment.cli
```
Примечания:
- Убедитесь, что директория `models/production` существует и содержит обученную модель (model.joblib + meta.json)
- Для корректной загрузки данных можно копировать их рядом или упаковать через `--add-data`

## Рekomendacii по улучшению
- Добавить spaCy / pymorphy2 для лемматизации
- Кэшировать предобработку (joblib Memory)
- Логи и метрики в JSON
- REST API (FastAPI) для сервера

## Лицензия
MIT
