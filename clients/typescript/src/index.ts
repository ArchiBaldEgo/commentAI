// Метки классов для сервиса тональности
export type Label = 'pos' | 'neu' | 'neg';

// Минимальное описание ответа fetch, чтобы не тянуть DOM типы
type FetchResponse = {
  ok: boolean;
  status: number;
  headers: { get(name: string): string | null };
  json(): Promise<any>;
  text(): Promise<string>;
};

// Обобщённый интерфейс fetch — подойдёт и для Node 18+, и для браузера
type FetchLike = (url: string, init?: { method?: string; headers?: Record<string, string>; body?: string; signal?: any }) => Promise<FetchResponse>;

// Параметры клиента — базовый URL, ключ API и таймауты
export interface SentimentClientOptions {
  baseUrl: string;
  apiKey?: string;
  timeoutMs?: number;
  fetchFn?: FetchLike;
}

// Форматы запросов/ответов — держим их максимально простыми
export interface PredictRequest { texts: string[] }
export interface FeedbackItem { text: string; label: Label }
export interface FeedbackRequest { items: FeedbackItem[] }
export interface ProductScoreRequest { productId: string; texts: string[] }

export interface PredictResult {
  predictions?: Label[];
  labels?: Label[];
  probabilities?: number[][];
  scores?: number[];
  [k: string]: unknown;
}

export interface FeedbackResult {
  accepted?: number;
  [k: string]: unknown;
}

export interface ProductScoreResult {
  productId?: string;
  score?: number;
  [k: string]: unknown;
}

// Минималистичный клиент для обращения к сервису тональности
export class SentimentClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private readonly doFetch: FetchLike;

  constructor(opts: SentimentClientOptions) {
    // тут специальной магии нет — просто запоминаем настройки
    if (!opts.baseUrl) throw new Error('baseUrl is required');
    this.baseUrl = opts.baseUrl.replace(/\/$/, '');
    this.apiKey = opts.apiKey;
    this.timeoutMs = opts.timeoutMs ?? 10_000;
    this.doFetch = opts.fetchFn ?? (globalThis as any).fetch;
    if (typeof this.doFetch !== 'function') {
      throw new Error('No fetch implementation found. Provide opts.fetchFn or use Node >= 18 / browser.');
    }
  }

  // Пингуем жив ли сервис
  async health(): Promise<unknown> {
    return this.get('/health');
  }

  // Узнаём список поддерживаемых меток
  async labels(): Promise<Label[] | unknown> {
    return this.get('/labels');
  }

  // Классификация батча текстов
  async predict(texts: string[]): Promise<PredictResult> {
    return this.post('/predict', { texts } satisfies PredictRequest);
  }

  // Средняя «оценка товара» по предсказанным меткам
  async productScore(productId: string, texts: string[]): Promise<ProductScoreResult> {
    return this.post('/product/score', { productId, texts } satisfies ProductScoreRequest);
  }

  // Отправляем размеченные пользователем примеры для дообучения
  async feedback(items: FeedbackItem[]): Promise<FeedbackResult> {
    return this.post('/feedback', { items } satisfies FeedbackRequest);
  }

  // Простой GET с таймаутом
  private async get(path: string): Promise<any> {
    const url = `${this.baseUrl}${path}`;
    const res = await this._withTimeout(this.doFetch(url, { method: 'GET' }));
    if (!res.ok) throw new Error(`GET ${path} failed: ${res.status} ${await res.text()}`);
    const ct = res.headers.get('content-type') || '';
    return ct.includes('application/json') ? res.json() : res.text();
  }

  // И POST тоже максимально без сюрпризов
  private async post(path: string, body: object): Promise<any> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.apiKey) headers['X-API-Key'] = this.apiKey;
    const res = await this._withTimeout(this.doFetch(url, { method: 'POST', headers, body: JSON.stringify(body) }));
    if (!res.ok) throw new Error(`POST ${path} failed: ${res.status} ${await res.text()}`);
    return res.json();
  }

  // Софт‑таймаут без AbortController, чтобы не тянуть DOM типы
  private async _withTimeout<T>(p: Promise<T>): Promise<T> {
    const to = this.timeoutMs;
    return await Promise.race([
      p,
      new Promise<T>((_, reject) => setTimeout(() => reject(new Error(`Request timeout after ${to}ms`)), to))
    ]);
  }
}
