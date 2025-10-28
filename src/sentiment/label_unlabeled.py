import click
import pandas as pd
from .dataset_tools import unlabeled_rows, load_dataset

LABELS = ["pos", "neg", "neu"]

@click.command()
@click.option('--data', 'data_path', required=True, type=click.Path(exists=True), help='CSV датасет')
@click.option('--save-every', default=10, show_default=True, help='Сохранять каждые N размеченных')
@click.option('--label-col', default='sentiment', show_default=True)
@click.option('--text-col', default='text', show_default=True)
def main(data_path, save_every, label_col, text_col):
    df = load_dataset(data_path)
    changed = 0
    for idx, row in df.iterrows():
        label_val = str(row.get(label_col, '')).strip()
        if label_val:
            continue
        text = row[text_col]
        print('\n---')
        print(text)
        lab = input(f"Метка ({'/'.join(LABELS)} / skip / stop): ").strip()
        if lab == 'stop':
            break
        if lab == 'skip' or lab not in LABELS:
            continue
        df.at[idx, label_col] = lab
        changed += 1
        if changed % save_every == 0:
            df.to_csv(data_path, index=False)
            print(f"Промежуточное сохранение ({changed})")
    if changed:
        df.to_csv(data_path, index=False)
        print(f"Сохранено. Новых меток: {changed}")
    else:
        print("Нет новых строк для разметки")

if __name__ == '__main__':
    main()
