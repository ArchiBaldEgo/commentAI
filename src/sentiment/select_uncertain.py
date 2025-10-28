import csv
import json
import click
import pandas as pd
from .model import SentimentClassifier

@click.command()
@click.option('--model-dir', required=True, type=click.Path(exists=True))
@click.option('--data', required=True, type=click.Path(exists=True), help='CSV с текстами (колонка text)')
@click.option('--output', default='data/uncertain.csv', show_default=True)
@click.option('--threshold', default=0.45, show_default=True, help='Максимум вероятности ниже порога → отбор')
@click.option('--limit', default=200, show_default=True, help='Максимум строк в выводе')
@click.option('--json-out', is_flag=True, help='Вывести JSON вместо CSV')
@click.option('--text-col', default='text', show_default=True)
@click.option('--label-col', default='sentiment', show_default=True)
def main(model_dir, data, output, threshold, limit, json_out, text_col, label_col):
    df = pd.read_csv(data)
    if text_col not in df.columns:
        raise click.UsageError(f'Нет колонки {text_col}')

    model = SentimentClassifier.load(model_dir)
    texts = df[text_col].astype(str).tolist()
    clf = model.pipeline.named_steps['clf']

    # Получаем вероятности через pipeline (как в predict)
    prep = model.pipeline.named_steps['prep']
    vec = model.pipeline.named_steps['tfidf']
    processed = prep.transform(texts)

    # FeatureUnion / Pipeline поддержка
    X_vec = vec.transform(processed)
    probas = clf.predict_proba(X_vec)  # type: ignore
    classes = list(clf.classes_)

    uncertain_rows = []
    for i, (t, pr) in enumerate(zip(texts, probas)):
        max_p = float(pr.max())
        if max_p < threshold:
            uncertain_rows.append({
                'text': t,
                'max_proba': max_p,
                'distribution': {c: float(p) for c, p in zip(classes, pr)}
            })
        if len(uncertain_rows) >= limit:
            break

    if json_out:
        print(json.dumps(uncertain_rows, ensure_ascii=False, indent=2))
        return

    # Запись в CSV
    import csv
    with open(output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'max_proba', 'distribution_json'])
        for row in uncertain_rows:
            writer.writerow([row['text'], row['max_proba'], json.dumps(row['distribution'], ensure_ascii=False)])
    print(f"Сохранено: {len(uncertain_rows)} строк в {output}")

if __name__ == '__main__':
    main()
