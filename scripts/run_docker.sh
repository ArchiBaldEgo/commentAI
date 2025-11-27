#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="base"
if [[ "${1:-}" == "pg" ]]; then
	EXTRA="-f docker-compose.postgres.yml"
	echo "Starting with PostgreSQL"
elif [[ "${1:-}" == "mysql" ]]; then
	EXTRA="-f docker-compose.mysql.yml"
	echo "Starting with MySQL"
else
	EXTRA=""
	echo "Starting API only (no DB service). To use DB: run with 'pg' or 'mysql' argument."
fi

[ -f .env ] || cp .env.example .env
docker-compose -f docker-compose.yml ${EXTRA} up --build -d
echo "Service running at http://localhost:8000"
