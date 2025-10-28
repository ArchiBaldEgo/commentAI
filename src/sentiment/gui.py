import os
import threading
import time
import queue
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from .model import SentimentClassifier
from .preprocess import preprocess_text

FEEDBACK_FILE = 'data/user_feedback.csv'
DEFAULT_MODEL_DIR = 'models/production'

class SentimentGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Sentiment Analyzer')
        self.model_dir = tk.StringVar(value=DEFAULT_MODEL_DIR)
        self.status_var = tk.StringVar(value='Загрузите модель')
        self.text_input = tk.Text(root, height=6, width=80)
        self.result_var = tk.StringVar()
        self.proba_tree = None
        self.model = None
        self.feedback_label_var = tk.StringVar(value='')
        self._build()

    def _build(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill='both', expand=True)

        top = ttk.Frame(frm)
        top.pack(fill='x')
        ttk.Label(top, text='Модель:').pack(side='left')
        ttk.Entry(top, textvariable=self.model_dir, width=40).pack(side='left')
        ttk.Button(top, text='Выбрать', command=self.choose_model_dir).pack(side='left', padx=5)
        ttk.Button(top, text='Загрузить', command=self.load_model).pack(side='left')
        ttk.Label(top, textvariable=self.status_var, foreground='blue').pack(side='left', padx=10)

        ttk.Label(frm, text='Текст отзыва:').pack(anchor='w', pady=(10,0))
        self.text_input.pack(fill='x')

        btns = ttk.Frame(frm)
        btns.pack(fill='x', pady=5)
        ttk.Button(btns, text='Предсказать', command=self.predict).pack(side='left')
        ttk.Button(btns, text='Очистить', command=self.clear_text).pack(side='left', padx=5)
        ttk.Button(btns, text='Сохранить фидбек', command=self.save_feedback_dialog).pack(side='left', padx=5)
        ttk.Button(btns, text='Переобучить', command=self.retrain_from_feedback).pack(side='left', padx=5)

        ttk.Label(frm, text='Результат:').pack(anchor='w')
        ttk.Label(frm, textvariable=self.result_var, font=('Arial', 12, 'bold')).pack(anchor='w', pady=(0,5))

        cols = ('class','proba')
        self.proba_tree = ttk.Treeview(frm, columns=cols, show='headings', height=4)
        self.proba_tree.heading('class', text='Класс')
        self.proba_tree.heading('proba', text='Вероятность')
        self.proba_tree.pack(fill='x')

    def choose_model_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.model_dir.set(path)

    def load_model(self):
        md = self.model_dir.get()
        if not os.path.exists(md):
            messagebox.showerror('Ошибка', f'Директория {md} не найдена')
            return
        try:
            self.model = SentimentClassifier.load(md)
            self.status_var.set('Модель загружена')
        except Exception as e:
            messagebox.showerror('Ошибка загрузки', str(e))

    def predict(self):
        if not self.model:
            messagebox.showwarning('Нет модели', 'Сначала загрузите модель')
            return
        text = self.text_input.get('1.0', 'end').strip()
        if not text:
            return
        pred = self.model.predict([text])[0]
        self.result_var.set(pred)
        # вероятности
        try:
            clf = self.model.pipeline.named_steps['clf']
            prep = self.model.pipeline.named_steps['prep']
            vec = self.model.pipeline.named_steps['tfidf']
            processed = prep.transform([text])
            Xv = vec.transform(processed)
            probas = clf.predict_proba(Xv)[0]
            classes = clf.classes_
            for i in self.proba_tree.get_children():
                self.proba_tree.delete(i)
            for c,p in sorted(zip(classes, probas), key=lambda x: -x[1]):
                self.proba_tree.insert('', 'end', values=(c, f'{p:.3f}'))
        except Exception:
            pass

    def clear_text(self):
        self.text_input.delete('1.0', 'end')
        self.result_var.set('')
        for i in self.proba_tree.get_children():
            self.proba_tree.delete(i)

    def save_feedback_dialog(self):
        if not self.result_var.get():
            messagebox.showinfo('Нет результата', 'Сначала выполните предсказание')
            return
        dialog = tk.Toplevel(self.root)
        dialog.title('Обратная связь')
        ttk.Label(dialog, text='Подтвердите или измените метку:').pack(pady=5)
        var = tk.StringVar(value=self.result_var.get())
        entry = ttk.Entry(dialog, textvariable=var)
        entry.pack(pady=5)
        def submit():
            self.append_feedback(self.text_input.get('1.0','end').strip(), self.result_var.get(), var.get())
            dialog.destroy()
            messagebox.showinfo('OK', 'Сохранено в feedback')
        ttk.Button(dialog, text='Сохранить', command=submit).pack(pady=5)

    def append_feedback(self, text: str, prediction: str, user_label: str):
        os.makedirs('data', exist_ok=True)
        row = {
            'timestamp': datetime.utcnow().isoformat(),
            'text': text,
            'prediction': prediction,
            'user_label': user_label
        }
        header_needed = not os.path.exists(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
            if header_needed:
                f.write('timestamp\ttext\tprediction\tuser_label\n')
            f.write(f"{row['timestamp']}\t{row['text'].replace('\t',' ')}\t{row['prediction']}\t{row['user_label']}\n")

    def retrain_from_feedback(self):
        # Простое переобучение: собрать feedback -> объединить с labeled -> запустить train
        if not os.path.exists(FEEDBACK_FILE):
            messagebox.showinfo('Нет данных', 'Файл feedback ещё не создан')
            return
        try:
            fb = pd.read_csv(FEEDBACK_FILE, sep='\t')
            if fb.empty:
                messagebox.showinfo('Пусто', 'Нет строк для обучения')
                return
            # фильтруем только те где user_label не пуст
            fb = fb[fb['user_label'].astype(str).str.strip() != '']
            if fb.empty:
                messagebox.showinfo('Нет меток', 'Нет размеченных строк')
                return
            labeled_path = 'data/reviews_labeled.csv'
            if os.path.exists(labeled_path):
                base = pd.read_csv(labeled_path)
            else:
                base = pd.DataFrame(columns=['text','sentiment'])
            # добавляем новые (уникальные по тексту)
            existing = set(base['text'].astype(str))
            new_rows = []
            for _, r in fb.iterrows():
                if r['text'] not in existing:
                    new_rows.append({'text': r['text'], 'sentiment': r['user_label']})
            if not new_rows:
                messagebox.showinfo('Нет новых', 'Нет новых уникальных строк')
                return
            base = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True)
            base.to_csv(labeled_path, index=False)
            # Запускаем обучение (синхронно для простоты)
            import subprocess, sys
            cmd = [sys.executable, '-m', 'src.sentiment.train', '--data', labeled_path, '--model-dir', self.model_dir.get(), '--class-weight', 'balanced', '--char-ngrams']
            self.status_var.set('Обучение...')
            r = subprocess.run(cmd, text=True)
            if r.returncode == 0:
                self.status_var.set('Переобучено')
                self.load_model()
                messagebox.showinfo('Готово', 'Модель переобучена')
            else:
                self.status_var.set('Ошибка обучения')
        except Exception as e:
            messagebox.showerror('Ошибка', str(e))


def run():
    root = tk.Tk()
    app = SentimentGUI(root)
    root.mainloop()

if __name__ == '__main__':
    run()
