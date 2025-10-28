import click
from . import train as train_module
from . import predict as predict_module
from . import collect_reviews as collect_module
from . import label_unlabeled as label_module
from . import evaluate as evaluate_module  # may be empty placeholder
from . import gui as gui_module
from . import server as server_module

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
@click.option('--host', default='0.0.0.0', show_default=True, help='Хост для сервера')
@click.option('--port', default=8000, show_default=True, help='Порт для сервера')
@click.option('--model-dir', default='models/online', show_default=True, help='Директория модели для онлайна')
def serve(host, port, model_dir):
    """Запуск REST API сервера (FastAPI) с онлайновым самообучением"""
    server_module.run(host=host, port=port, model_dir=model_dir)


@main.command("train-batch")
@click.option('--model-dir', default='models/online', show_default=True, help='Директория модели/feedback')
@click.option('--clear-after', is_flag=True, default=True, help='Очистить feedback.jsonl после обучения')
@click.option('--limit', type=int, default=None, help='Ограничить число примеров для обучения')
def train_batch(model_dir, clear_after, limit):
    """Пакетное дообучение из feedback.jsonl (без запуска сервера)"""
    from .server import ServerState
    import random
    state = ServerState(model_dir)
    items = state.store.read_all()
    random.shuffle(items)
    if limit is not None:
        items_to_train = items[:limit]
        remaining = items[limit:]
    else:
        items_to_train = items
        remaining = []
    texts = [it.get('text', '') for it in items_to_train if it.get('text')]
    labels = [it.get('label') for it in items_to_train if it.get('label')]
    paired = [(t, l) for t, l in zip(texts, labels) if t and l]
    if paired:
        t_train, y_train = zip(*paired)
        state.model.partial_fit(list(t_train), list(y_train))
        state.model.save(state.model_dir)
    if clear_after:
        state.store.write_all(remaining)
    click.echo(f"seen={len(items)} trained={len(paired)} remaining={(len(remaining) if clear_after else len(items)-len(paired))}")

if __name__ == '__main__':
    main()
