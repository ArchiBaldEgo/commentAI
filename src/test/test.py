import sys
from pathlib import Path

# Обеспечиваем доступность пакета sentiment при запуске из корня проекта
BASE_DIR = Path(__file__).resolve().parents[1]  # .../src
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sentiment.inference import load_model_cached, predict_proba_texts


LABEL_TO_SCORE = {"neg": 1.0, "neu": 3.0, "pos": 5.0}


def sentiment_to_stars(label: str) -> int:
    return int(LABEL_TO_SCORE.get(label, 3.0))


def main() -> None:
    print("=== commentAI: тест модели тональности ===")
    print("Введите несколько комментариев. Пустая строка — закончить.\n")

    texts: list[str] = []
    while True:
        try:
            line = input("Комментарий> ").strip()
        except EOFError:
            break
        if not line:
            break
        texts.append(line)

    if not texts:
        print("Нет введённых комментариев — выход.")
        return

    # Загружаем модель из стандартной папки models/production
    model = load_model_cached(None)
    probs, preds, labels = predict_proba_texts(model, texts)

    print("\nРезультаты:\n")
    for i, text in enumerate(texts):
        label = preds[i]
        stars = sentiment_to_stars(label)
        print(f"[{i+1}] {text}")
        print(f"   Тональность: {label}  | рейтинг: {stars}/5")
    print("\nГотово.")


if __name__ == "__main__":
    main()
