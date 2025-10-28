import click
import pandas as pd
from sklearn.metrics import classification_report
from .model import SentimentClassifier

@click.command()
@click.option('--data', 'data_path', required=True, type=click.Path(exists=True), help='CSV файл с тестовыми данными')
@click.option('--model-dir', required=True, type=click.Path(exists=True), help='Директория сохраненной модели')
@click.option('--text-col', default='text', show_default=True)
@click.option('--label-col', default='sentiment', show_default=True)
def main(data_path, model_dir, text_col, label_col):
    df = pd.read_csv(data_path)
    model = SentimentClassifier.load(model_dir)
    X = df[text_col].astype(str).tolist()
    y = df[label_col].astype(str).tolist()
    report = model.evaluate(X, y)
    print(report)

if __name__ == '__main__':
    main()
