import os
import yaml
import subprocess
import sys
from pathlib import Path

# Простой оркестратор: scrape (если urls.txt существует) -> train -> (опц.) evaluate

def run(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        print(f"Команда завершилась с кодом {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)


def pipeline(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    paths = cfg['paths']

    urls_file = paths.get('urls_file')
    if urls_file and os.path.exists(urls_file):
        scrape_cfg = cfg.get('scrape', {})
        container = scrape_cfg.get('container')
        text_sel = scrape_cfg.get('text_selector')
        cmd = [sys.executable, '-m', 'src.sentiment.collect_reviews', '--urls-file', urls_file, '--container', container, '--output-csv', paths['raw']]
        if text_sel:
            cmd += ['--text-sel', text_sel]
        run(cmd)

    # Ожидаем, что пользователь сделал разметку raw -> labeled
    train_cfg = cfg['train']
    train_cmd = [sys.executable, '-m', 'src.sentiment.train', '--data', paths['labeled'], '--model-dir', paths['model_dir'], '--test-size', str(train_cfg['test_size']), '--seed', str(train_cfg['seed']), '--max-features', str(train_cfg['max_features'])]
    if train_cfg.get('class_weight'):
        train_cmd += ['--class-weight', train_cfg['class_weight']]
    if train_cfg.get('char_ngrams'):
        train_cmd += ['--char-ngrams']
    if train_cfg.get('grid'):
        train_cmd += ['--grid']
    run(train_cmd)

    eval_cfg = cfg.get('evaluate')
    if eval_cfg:
        data = eval_cfg.get('data')
        if data and os.path.exists(data):
            eval_cmd = [sys.executable, '-m', 'src.sentiment.evaluate', '--data', data, '--model-dir', paths['model_dir']]
            run(eval_cmd)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    pipeline(args.config)
