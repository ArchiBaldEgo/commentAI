import os
from src.sentiment.model import SentimentClassifier
from src.sentiment.train import main as train_main
import pandas as pd
from click.testing import CliRunner

def test_training_and_prediction(tmp_path):
    # Prepare small dataset
    data_path = tmp_path / 'data.csv'
    pd.DataFrame({
        'text': [
            'Отличный продукт',
            'Очень плохо',
            'Нормально',
            'Great item',
            'Bad quality',
            'Average experience'
        ],
        'sentiment': ['pos','neg','neu','pos','neg','neu']
    }).to_csv(data_path, index=False)

    model_dir = tmp_path / 'model'

    runner = CliRunner()
    result = runner.invoke(train_main, [
        '--data', str(data_path),
        '--model-dir', str(model_dir),
        '--test-size', '0.5',
        '--seed', '7'
    ])
    assert result.exit_code == 0, result.output

    clf = SentimentClassifier.load(str(model_dir))
    preds = clf.predict(['Отличный продукт', 'Очень плохо'])
    assert len(preds) == 2
    assert set(preds).issubset({'pos','neg','neu'})
