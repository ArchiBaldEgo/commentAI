import click
from . import train as train_module
from . import predict as predict_module
from . import collect_reviews as collect_module
from . import label_unlabeled as label_module
from . import evaluate as evaluate_module  # may be empty placeholder
from . import gui as gui_module
import uvicorn
from . import api as api_module

@click.group()
def main():
    """Единый CLI для операций sentiment."""
    pass

@main.command()
@click.argument('args', nargs=-1)
def train(args):
    """Проксирование к train.py"""
    train_module.main.main(args=list(args))  # type: ignore

@main.command()
@click.argument('args', nargs=-1)
def predict(args):
    predict_module.main.main(args=list(args))  # type: ignore

@main.command()
@click.argument('args', nargs=-1)
def collect(args):
    collect_module.main.main(args=list(args))  # type: ignore

@main.command()
@click.argument('args', nargs=-1)
def label(args):
    label_module.main.main(args=list(args))  # type: ignore

@main.command()
@click.argument('args', nargs=-1)
def evaluate(args):
    evaluate_module.main.main(args=list(args))  # type: ignore

@main.command()
def gui():
    """Запуск графического интерфейса"""
    gui_module.run()

@main.command()
@click.option('--host', default='0.0.0.0', show_default=True)
@click.option('--port', default=8000, type=int, show_default=True)
@click.option('--workers', default=1, type=int, show_default=True)
def api(host, port, workers):
    """Запуск REST API (FastAPI)"""
    uvicorn.run("src.sentiment.api:app", host=host, port=port, workers=workers, reload=False)

if __name__ == '__main__':
    main()
