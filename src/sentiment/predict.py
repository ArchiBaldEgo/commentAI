import json
from typing import List, Optional
import click
import pandas as pd
from .model import SentimentClassifier

@click.command()
@click.option('--model-dir', required=True, type=click.Path(exists=True), help='Директория сохраненной модели')
@click.option('--text', multiple=True, help='Текст для анализа (можно указать несколько)')
@click.option('--file', 'file_path', type=click.Path(exists=True), help='CSV файл с колонкой text')
@click.option('--text-col', default='text', show_default=True, help='Имя колонки в файле')
@click.option('--proba', is_flag=True, help='Возвращать вероятности классов')
@click.option('--output', type=click.Path(), help='Путь для сохранения JSON результата')
def main(model_dir, text, file_path, text_col, proba, output):
    if not text and not file_path:
        raise click.UsageError('Нужно указать либо --text, либо --file')

    model = SentimentClassifier.load(model_dir)

    texts: List[str] = []
    if file_path:
        df = pd.read_csv(file_path)
        if text_col not in df.columns:
            raise click.UsageError(f'Нет колонки {text_col} в файле')
        texts.extend(df[text_col].astype(str).tolist())
    if text:
        texts.extend(list(text))

    preds = model.predict(texts)

    result = []
    if proba:
        # Получим вероятности через pipeline напрямую для упрощения
        clf = model.pipeline.named_steps['clf']
        tfidf = model.pipeline.named_steps['tfidf']
        prep = model.pipeline.named_steps['prep']
        processed = prep.transform(texts)
        X_vec = tfidf.transform(processed)
        probas = clf.predict_proba(X_vec)
        classes = list(clf.classes_)
        for t, pred, pr in zip(texts, preds, probas):
            result.append({'text': t, 'prediction': pred, 'proba': {c: float(p) for c, p in zip(classes, pr)}})
    else:
        for t, pred in zip(texts, preds):
            result.append({'text': t, 'prediction': pred})

    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(json_str)
    print(json_str)

if __name__ == '__main__':
    main()
