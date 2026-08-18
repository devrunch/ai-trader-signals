# ai-trader-signals

FastAPI service: market data, the AI chat agent, signal generation, backtesting, and real-time price ticks. Python 3.12.

Part of the [ai-trader](https://github.com/devrunch/ai-trader) monorepo — run via the umbrella repo's `docker compose up`, not standalone in production. Local dev without Docker is supported (see below).

## What's here

| Path | Responsibility |
|---|---|
| `app/market/` | Market data — `providers/` (Kite Connect for NSE/BSE, yfinance fallback), `router.py` and `service.py` (quotes/history/search/news), `kite_ticker.py` + `live_ticks.py` (real-time WebSocket ticks + the Redis pub/sub bridge to `ai-trader-api`), `calendar.py` (NSE trading calendar) |
| `app/signals/` | The chat agent (`agent/` — orchestrator, tool suite, condition DSL), signal generation (`service.py`), backtesting (`backtest/`), the pre-market brief (`brief.py`) |
| `app/worker/` | Celery tasks — screener runs, daily Zerodha session refresh, intraday square-off |
| `app/llm/` | Bedrock (DeepSeek v3.2 / Mistral / Qwen3) client, OpenAI-compatible |
| `main.py` | FastAPI app, lifespan (executor, Redis, Kite ticker), `/health` (liveness) vs `/ready` (readiness — probes market data + SQS, cached) |

## Real-time price ticks

`KiteTickerClient` (`app/market/kite_ticker.py`) owns the one persistent Kite WebSocket connection for NSE/BSE. `LiveTicks` (`app/market/live_ticks.py`) is the exchange router: NSE/BSE go to the Kite ticker, everything else gets a 5s poll over the same `market_data_router` every other quote call uses. Both publish to Redis (`market:ticks`); `ai-trader-api`'s `SignalsGateway` relays to the browser over the existing Socket.IO connection. NestJS tells this service which symbols to (un)subscribe via `POST /market/internal/live-ticks/{subscribe,unsubscribe}` on the first/last watcher of a room; on process restart, this service asks NestJS for currently-active symbols and resubscribes.

## Local development

```bash
cp .env.example .env      # fill in real values
python -m venv .venv
.venv/Scripts/activate     # .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

Needs a reachable `ai-trader-api` (for the Zerodha session token and internal active-symbols lookups) and Redis (for live ticks) — running the full stack via the umbrella repo's `docker compose up` is simpler than wiring these by hand.

## Testing

```bash
pytest -q                          # 347 tests
ruff check app tests main.py       # lint
mypy app main.py                   # types
```

No test opens a real Kite WebSocket connection or hits a real broker — those paths are proven live (manual verification against the real API), matching how `kite_auth.py`'s scripted login flow was verified.

## Environment

See `.env.example` for the full list. Notable ones: `INTERNAL_API_KEY` (shared secret for service-to-service calls to `ai-trader-api`), `API_SERVICE_URL`, `REDIS_URL`, `ZERODHA_API_KEY`/`ZERODHA_API_SECRET` (Kite Connect), `BEDROCK_*` (the chat agent's LLM), `NEWS_API_KEY` (FinBERT sentiment input).
