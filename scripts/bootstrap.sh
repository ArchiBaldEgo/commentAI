#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure venv and deps
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt

# Create sample data if missing
if [ ! -f data/sample_reviews.csv ]; then
  mkdir -p data
  cat > data/sample_reviews.csv <<'CSV'
text,label
"Отличный товар, рекомендую!",pos
"Ужасное качество, разочарован",neg
"Нормально, но могло быть лучше",neu
"Очень быстрая доставка и хорошая упаковка",pos
"Сломалось через неделю, не советую",neg
"Обычный продукт, без восторга",neu
CSV
fi

# Train production model if missing
if [ ! -f models/production/model.joblib ]; then
  mkdir -p models/production
  PYTHONPATH=src python -m sentiment.train --data data/sample_reviews.csv --model-dir models/production --char-ngrams || {
    echo "Training failed" >&2
    exit 1
  }
fi

echo "Bootstrap complete. Start API:"
echo "  source .venv/bin/activate && uvicorn src.sentiment.api:app --host 0.0.0.0 --port 8000 --reload"
