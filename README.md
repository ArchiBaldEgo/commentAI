# Система анализа тональности текстовых отзывов

Простой проект для обучения модели классификации тональности (позитив / негатив / нейтраль) на основе TF-IDF + Logistic Regression.

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
