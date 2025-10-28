import json
import os
from pathlib import Path
import click
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from .model import SentimentClassifier
from .utils import set_seed

@click.command()
@click.option('--data', 'data_path', required=True, type=click.Path(exists=True), help='CSV файл с данными')
@click.option('--text-col', default='text', show_default=True, help='Имя колонки с текстом')
@click.option('--label-col', default='sentiment', show_default=True, help='Имя колонки с меткой')
@click.option('--model-dir', default='models/default', show_default=True, help='Директория для сохранения модели')
@click.option('--test-size', default=0.2, show_default=True, help='Доля теста')
@click.option('--seed', default=42, show_default=True, help='Seed')
@click.option('--cv', default=3, show_default=True, help='Кросс-валидация для грид-серча')
@click.option('--grid', is_flag=True, help='Запустить grid search гиперпараметров')
@click.option('--class-weight', 'class_weight', default=None, show_default=True, help='class_weight для LogisticRegression (например balanced)')
@click.option('--max-features', default=8000, show_default=True, help='tfidf max_features')
@click.option('--char-ngrams', is_flag=True, help='Добавить символьные ngrams (Char TF-IDF)')
@click.option('--verbose', is_flag=True, help='Подробный вывод')
@click.option('--hashing', is_flag=True, help='Использовать HashingVectorizer (+TfidfTransformer) вместо TF-IDF')
@click.option('--algo', type=click.Choice(['logreg','sgd']), default='logreg', show_default=True, help='Алгоритм классификатора')
@click.option('--partial', is_flag=True, help='Инкрементальное partial_fit для SGD (игнорирует grid)')
def main(data_path, text_col, label_col, model_dir, test_size, seed, cv, grid, class_weight, max_features, char_ngrams, verbose, hashing, algo, partial):
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from .preprocess import PreprocessTransformer

    set_seed(seed)
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[text_col, label_col])
    X = df[text_col].astype(str).tolist()
    y = df[label_col].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    text_pre = PreprocessTransformer()

    if hashing:
        # Hashing (не обучается, детерминированно), затем TfidfTransformer
        if char_ngrams:
            featurizer = FeatureUnion([
                ('word', Pipeline([
                    ('hv', HashingVectorizer(n_features=max_features, alternate_sign=False, ngram_range=(1,2))),
                    ('tfidf', TfidfTransformer())
                ])),
                ('char', Pipeline([
                    ('hv', HashingVectorizer(analyzer='char', n_features=10000, ngram_range=(3,5), alternate_sign=False)),
                    ('tfidf', TfidfTransformer())
                ]))
            ])
        else:
            featurizer = Pipeline([
                ('hv', HashingVectorizer(n_features=max_features, alternate_sign=False, ngram_range=(1,2))),
                ('tfidf', TfidfTransformer())
            ])
    else:
        if char_ngrams:
            featurizer = FeatureUnion([
                ('word', TfidfVectorizer(max_features=max_features, ngram_range=(1,2))),
                ('char', TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=10000))
            ])
        else:
            featurizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))

    if algo == 'logreg':
        clf = LogisticRegression(max_iter=1500, class_weight=class_weight)
    else:
        clf = SGDClassifier(loss='log_loss', max_iter=10, class_weight=class_weight, random_state=seed)

    pipeline = Pipeline([
        ('prep', text_pre),
        ('tfidf', featurizer),
        ('clf', clf)
    ])

    model = SentimentClassifier(pipeline=pipeline)

    if algo == 'logreg' and grid and not hashing and not partial:
        param_grid = {
            'tfidf__word__max_features': [5000, max_features] if char_ngrams else [max_features],
            'clf__C': [0.5, 1.0, 2.0]
        }
        gs = GridSearchCV(model.pipeline, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        model.pipeline = gs.best_estimator_
        if verbose:
            print(f"Лучшие параметры: {gs.best_params_}")

    if algo == 'sgd' and partial:
        # partial_fit по батчам
        import numpy as np
        classes = sorted(set(y_train))
        batch_size = max(32, len(X_train)//20)
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            model.pipeline.partial_fit(xb, yb, classes=classes)  # type: ignore
        if verbose:
            print("Partial fit завершен")
    else:
        model.fit(X_train, y_train)

    report = model.evaluate(X_test, y_test)
    print(report)

    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    with open(os.path.join(model_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == '__main__':
    main()
