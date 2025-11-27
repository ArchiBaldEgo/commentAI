# Sentiment SDK (TypeScript)

Minimal TypeScript client for the sentiment API.

Install and build (optional):

```bash
cd clients/typescript
npm install
npm run build
```

Usage:

```ts
import { SentimentClient } from './dist/index.js';

const client = new SentimentClient({ baseUrl: 'http://localhost:8000', apiKey: 'secret' });

const health = await client.health();
const labels = await client.labels();
const pred = await client.predict(['Отличный товар', 'Плохая упаковка']);
const score = await client.productScore('SKU-1', ['good', 'bad']);
await client.feedback([{ text: 'good', label: 'pos' }]);
```

Notes:
- Requires Node 18+ (built-in `fetch`) or pass a custom `fetchFn`.
- Response shapes can differ by server version; SDK returns parsed JSON.