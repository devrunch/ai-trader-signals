"""
The one persistent Kite Connect WebSocket connection for this process —
real-time ticks for NSE/BSE. Everything non-Kite lives in live_ticks.py,
which is the only thing that imports this module.

KiteTicker resubscribes every currently-subscribed token on its own
whenever the underlying WebSocket reconnects (its _on_open calls
self.resubscribe() from its own in-memory state) — nothing here
duplicates that. That in-memory state does not survive this whole
process restarting; live_ticks.py's resubscribe_from() is what covers
that case, by calling subscribe() again for whatever was active.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from kiteconnect import KiteTicker

logger = logging.getLogger(__name__)


def _quote_from_tick(symbol: str, exchange: str, tick: dict[str, Any]) -> dict[str, Any]:
    ohlc = tick.get("ohlc") or {}
    ltp = float(tick["last_price"])
    prev_close = float(ohlc.get("close", ltp))
    change = ltp - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0
    return {
        "symbol": symbol,
        "exchange": exchange,
        "ltp": round(ltp, 4),
        "change": round(change, 4),
        "change_percent": round(change_pct, 4),
        "open": ohlc.get("open"),
        "high": ohlc.get("high"),
        "low": ohlc.get("low"),
        "prev_close": round(prev_close, 4),
        "volume": tick.get("volume_traded"),
    }


class KiteTickerClient:
    def __init__(self, api_key: str, access_token: str, kite_provider,
                 on_tick: Callable[[dict[str, Any]], None]):
        self._provider = kite_provider
        self._on_tick = on_tick
        self._token_map: dict[int, tuple[str, str]] = {}
        self._ticker = KiteTicker(api_key, access_token)
        self._ticker.on_ticks = self._on_ticks

    def connect(self) -> None:
        self._ticker.connect(threaded=True)

    def close(self) -> None:
        self._ticker.close()

    def _resolve(self, symbol: str, exchange: str) -> int | None:
        self._provider._ensure_instruments()
        row = self._provider._instruments.get(exchange.upper(), {}).get(symbol.upper())
        return row["instrument_token"] if row else None

    def subscribe(self, symbol: str, exchange: str) -> bool:
        token = self._resolve(symbol, exchange)
        if token is None:
            logger.warning("Kite ticker: no instrument token for %s/%s", symbol, exchange)
            return False
        # connect() returns before the WebSocket handshake finishes, so an
        # early subscribe() (e.g. resubscribe_from at startup) can race it —
        # fail closed rather than let the SDK exception abort the caller.
        try:
            self._ticker.subscribe([token])
        except Exception:
            logger.exception("Kite ticker: subscribe failed for %s/%s", symbol, exchange)
            return False
        self._token_map[token] = (symbol.upper(), exchange.upper())
        return True

    def unsubscribe(self, symbol: str, exchange: str) -> None:
        token = self._resolve(symbol, exchange)
        if token is None:
            return
        self._token_map.pop(token, None)
        try:
            self._ticker.unsubscribe([token])
        except Exception:
            logger.exception("Kite ticker: unsubscribe failed for %s/%s", symbol, exchange)

    def _on_ticks(self, ws, ticks: list[dict[str, Any]]) -> None:
        for tick in ticks:
            token = tick.get("instrument_token")
            if not isinstance(token, int):
                continue
            pair = self._token_map.get(token)
            if pair is None:
                continue
            symbol, exchange = pair
            self._on_tick(_quote_from_tick(symbol, exchange, tick))
