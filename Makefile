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

# Docker Compose overlays
docker-pg:
	[ -f .env ] || cp .env.example .env
	docker compose -f docker-compose.yml -f deploy/compose.postgres.yml up -d

docker-mysql:
	[ -f .env ] || cp .env.example .env
	docker compose -f docker-compose.yml -f deploy/compose.mysql.yml up -d

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
