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
