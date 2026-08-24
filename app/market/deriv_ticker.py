"""
DerivTickerClient — the one persistent Deriv WebSocket connection for this
process, real-time forex/metals ticks (frxEURUSD, frxXAUUSD, ...). Public,
unauthenticated endpoint -- confirmed live: no account, no API key, and 29
concurrent subscriptions on one connection with zero errors. Mirrors
kite_ticker.py's shape: live_ticks.py is the only thing that imports this
module.

Unlike KiteTickerClient (a threaded SDK callback bridged back onto the
asyncio loop via run_coroutine_threadsafe), this is natively async -- the
whole client runs on this process's own event loop already, so on_tick can
just be awaited/scheduled directly, no thread-bridging needed.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import websockets

from app.market.providers.deriv_provider import WS_URL, DerivProvider, deriv_symbol_for

logger = logging.getLogger(__name__)

# Deriv closes an idle connection after ~2 minutes with no traffic. Ticks
# already arrive every ~1s once anything is subscribed, but a ping keeps
# the connection alive in the gap right after connect/reconnect before the
# first subscribe.
_PING_INTERVAL_SECONDS = 60
_RECONNECT_DELAY_SECONDS = 2


class _Baseline:
    """open/high/low/prev_close for one symbol's current day, fetched once
    via DerivProvider.get_quote() when a subscription starts. A raw Deriv
    tick carries only the latest price (quote/bid/ask) -- change/high/low
    have to be derived against something, and the frontend replaces its
    quote object wholesale per tick (never merges), so every published
    tick needs these fields filled in, not left None. high/low are kept
    fresh in place as new ticks exceed them; open/prev_close are pinned to
    this fetch for the life of the subscription (both are end-of-day-scoped
    values that don't need per-tick freshness)."""

    __slots__ = ("open", "high", "low", "prev_close")

    def __init__(self, open_: float, high: float, low: float, prev_close: float):
        self.open = open_
        self.high = high
        self.low = low
        self.prev_close = prev_close


def _quote_from_tick(symbol: str, exchange: str, tick: dict[str, Any], baseline: _Baseline) -> dict[str, Any]:
    ltp = float(tick["quote"])
    baseline.high = max(baseline.high, ltp)
    baseline.low = min(baseline.low, ltp)
    change = ltp - baseline.prev_close
    change_pct = (change / baseline.prev_close * 100) if baseline.prev_close else 0.0
    return {
        "symbol": symbol,
        "exchange": exchange,
        "ltp": round(ltp, 5),
        "change": round(change, 5),
        "change_percent": round(change_pct, 4),
        "open": baseline.open,
        "high": baseline.high,
        "low": baseline.low,
        "prev_close": baseline.prev_close,
        "volume": None,
        "bid": round(float(tick["bid"]), 5) if "bid" in tick else None,
        "ask": round(float(tick["ask"]), 5) if "ask" in tick else None,
    }


class DerivTickerClient:
    def __init__(self, on_tick: Callable[[dict[str, Any]], None], provider: DerivProvider | None = None):
        self._on_tick = on_tick
        self._provider = provider or DerivProvider()
        self._ws: Any = None
        # (symbol, exchange) -> Deriv's own per-subscription id, needed to
        # unsubscribe (Deriv's "forget" call takes the id, not the symbol).
        self._subscription_ids: dict[tuple[str, str], str] = {}
        # Deriv's own symbol ("frxXAUUSD") -> our (symbol, exchange) pair,
        # for routing an incoming tick back to the caller who asked for it.
        self._pair_by_deriv_symbol: dict[str, tuple[str, str]] = {}
        self._baseline_by_pair: dict[tuple[str, str], _Baseline] = {}
        self._run_task: asyncio.Task | None = None

    async def connect(self) -> None:
        self._run_task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._run_task is not None:
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            await self._ws.close()

    async def subscribe(self, symbol: str, exchange: str) -> bool:
        deriv_sym = deriv_symbol_for(symbol)
        if deriv_sym is None:
            logger.warning("Deriv ticker: no known symbol for %s/%s", symbol, exchange)
            return False
        if self._ws is None:
            logger.warning("Deriv ticker: not connected yet, cannot subscribe %s/%s", symbol, exchange)
            return False

        key = (symbol.upper(), exchange.upper())
        baseline = await self._fetch_baseline(symbol, exchange)
        if baseline is None:
            logger.warning("Deriv ticker: no quote baseline available for %s/%s", symbol, exchange)
            return False

        try:
            await self._ws.send(json.dumps({"ticks": deriv_sym, "subscribe": 1}))
        except Exception:
            logger.exception("Deriv ticker: subscribe failed for %s/%s", symbol, exchange)
            return False
        self._baseline_by_pair[key] = baseline
        self._pair_by_deriv_symbol[deriv_sym] = key
        return True

    async def _fetch_baseline(self, symbol: str, exchange: str) -> _Baseline | None:
        quote = await self._provider.get_quote(symbol, exchange)
        if quote is None:
            return None
        return _Baseline(
            open_=quote["open"], high=quote["high"], low=quote["low"], prev_close=quote["prev_close"],
        )

    async def unsubscribe(self, symbol: str, exchange: str) -> None:
        key = (symbol.upper(), exchange.upper())
        sub_id = self._subscription_ids.pop(key, None)
        self._baseline_by_pair.pop(key, None)
        deriv_sym = deriv_symbol_for(symbol)
        if deriv_sym is not None:
            self._pair_by_deriv_symbol.pop(deriv_sym, None)
        if sub_id is None or self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"forget": sub_id}))
        except Exception:
            logger.exception("Deriv ticker: unsubscribe failed for %s/%s", symbol, exchange)

    async def _run(self) -> None:
        """Owns the connection for the process's whole lifetime -- reconnects
        on any drop. A reconnect loses Deriv's own subscription state, so
        whatever this process still thinks is subscribed gets resubscribed
        fresh on the new connection (a fresh baseline fetch too -- cheap,
        and correctness matters more here than saving one request)."""
        while True:
            try:
                async with websockets.connect(WS_URL, ping_interval=_PING_INTERVAL_SECONDS) as ws:
                    self._ws = ws
                    await self._resubscribe_all()
                    async for raw in ws:
                        self._handle_message(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Deriv ticker connection dropped, reconnecting")
                self._ws = None
                await asyncio.sleep(_RECONNECT_DELAY_SECONDS)

    async def _resubscribe_all(self) -> None:
        pairs = list(self._pair_by_deriv_symbol.values())
        self._subscription_ids.clear()
        self._pair_by_deriv_symbol.clear()
        self._baseline_by_pair.clear()
        for symbol, exchange in pairs:
            await self.subscribe(symbol, exchange)

    def _handle_message(self, msg: dict[str, Any]) -> None:
        if msg.get("msg_type") != "tick" or "tick" not in msg:
            return
        tick = msg["tick"]
        deriv_sym = tick.get("symbol")
        pair = self._pair_by_deriv_symbol.get(deriv_sym)
        if pair is None:
            return
        baseline = self._baseline_by_pair.get(pair)
        if baseline is None:
            return
        sub_id = msg.get("subscription", {}).get("id")
        if sub_id:
            self._subscription_ids[pair] = sub_id
        symbol, exchange = pair
        self._on_tick(_quote_from_tick(symbol, exchange, tick, baseline))
