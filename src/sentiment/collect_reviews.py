import asyncio
import json
from typing import Optional
import click
from .scrape import ExtractRule, collect
from .dataset_tools import append_texts

@click.command()
@click.option('--urls-file', type=click.Path(exists=True), required=True, help='Файл со списком URL (по одному в строке)')
@click.option('--container', required=True, help='CSS селектор контейнера отзыва')
@click.option('--text-sel', default=None, help='Внутренний селектор текста (опц.)')
@click.option('--output-csv', default='data/reviews_raw.csv', show_default=True, help='CSV куда дописываем')
@click.option('--label', default=None, help='Присвоить метку всем новым (опц.)')
@click.option('--print-json', is_flag=True, help='Вывести JSON вместо CSV')
def main(urls_file, container, text_sel, output_csv, label, print_json):
    with open(urls_file, 'r', encoding='utf-8') as f:
        urls = [l.strip() for l in f if l.strip()]
    rule = ExtractRule(container=container, text=text_sel)
    reviews = asyncio.run(collect(urls, rule))
    if print_json:
        print(json.dumps(reviews, ensure_ascii=False, indent=2))
        return
    added = append_texts(output_csv, reviews, default_label=label)
    print(f"Добавлено новых строк: {added}")

if __name__ == '__main__':
    main()
