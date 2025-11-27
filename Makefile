SHELL := /bin/bash

.PHONY: install run api stop docker-up docker-down train bootstrap docker-pg docker-mysql check

install:
	python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

api:
	source .venv/bin/activate && uvicorn src.sentiment.api:app --host 0.0.0.0 --port 8000 --reload

run: install api

stop:
	pkill -f "uvicorn src.sentiment.api:app" || true

train:
	source .venv/bin/activate && PYTHONPATH=src python -m sentiment.train --data data/sample_reviews.csv --model-dir models/production --char-ngrams

bootstrap:
	bash scripts/bootstrap.sh

docker-up:
	[ -f .env ] || cp .env.example .env
	docker-compose up --build -d

docker-down:
	docker-compose down

docker-pg:
	[ -f .env ] || cp .env.example .env
	bash scripts/run_docker.sh pg

docker-mysql:
	[ -f .env ] || cp .env.example .env
	bash scripts/run_docker.sh mysql

check:
	bash scripts/wait_for_api.sh http://127.0.0.1:8000/health 30
PY=python
PKG=src/sentiment
DATA=data
MODEL_DIR=models/auto
CONFIG=config.yaml

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

train:
	$(PY) -m src.sentiment.train --data $(DATA)/sample_reviews.csv --model-dir $(MODEL_DIR) --class-weight balanced --char-ngrams

predict:
	$(PY) -m src.sentiment.predict --model-dir $(MODEL_DIR) --text "Отличный товар" --text "Очень плохо" --proba

collect:
	$(PY) -m src.sentiment.collect_reviews --urls-file urls.txt --container "div.review" --output-csv $(DATA)/reviews_raw.csv

label:
	$(PY) -m src.sentiment.label_unlabeled --data $(DATA)/reviews_raw.csv

pipeline:
	$(PY) -m src.sentiment.pipeline --config $(CONFIG)

serve:
	$(PY) -m src.sentiment.cli serve --host 0.0.0.0 --port 8000 --model-dir models/online

.PHONY: install train predict collect label pipeline
