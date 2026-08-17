"""
KiteTicker wrapper — owns the one persistent Kite WebSocket connection.
Nothing here knows about non-Kite exchanges; that decision lives in
live_ticks.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.market.kite_ticker import KiteTickerClient


def _client(on_tick=None):
    provider = MagicMock()
    provider._instruments = {
        "NSE": {"RELIANCE": {"tradingsymbol": "RELIANCE", "instrument_token": 738561}},
        "BSE": {},
    }
    provider._ensure_instruments = MagicMock()
    ws = MagicMock()
    client = KiteTickerClient(api_key="key", access_token="tok",
                               kite_provider=provider, on_tick=on_tick or MagicMock())
    client._ticker = ws
    return client, provider, ws


def test_subscribe_resolves_the_token_and_calls_the_sdk():
    client, provider, ws = _client()

    ok = client.subscribe("RELIANCE", "NSE")

    assert ok is True
    ws.subscribe.assert_called_once_with([738561])
    provider._ensure_instruments.assert_called_once()


def test_subscribe_to_an_unresolvable_symbol_fails_closed():
    client, provider, ws = _client()

    ok = client.subscribe("NOTREAL", "NSE")

    assert ok is False
    ws.subscribe.assert_not_called()


def test_unsubscribe_resolves_the_same_token():
    client, provider, ws = _client()
    client.subscribe("RELIANCE", "NSE")

    client.unsubscribe("RELIANCE", "NSE")

    ws.unsubscribe.assert_called_once_with([738561])


def test_on_ticks_resolves_token_back_to_symbol_and_shapes_a_quote():
    seen = []
    client, provider, ws = _client(on_tick=seen.append)
    client.subscribe("RELIANCE", "NSE")

    client._on_ticks(ws, [{
        "instrument_token": 738561, "last_price": 1310.0,
        "ohlc": {"open": 1300.0, "high": 1315.0, "low": 1295.0, "close": 1300.0},
        "volume_traded": 500000,
    }])

    assert len(seen) == 1
    quote = seen[0]
    assert quote["symbol"] == "RELIANCE"
    assert quote["exchange"] == "NSE"
    assert quote["ltp"] == 1310.0
    assert quote["prev_close"] == 1300.0
    assert quote["volume"] == 500000


def test_a_tick_for_an_unknown_token_is_dropped_not_raised():
    seen = []
    client, provider, ws = _client(on_tick=seen.append)

    client._on_ticks(ws, [{"instrument_token": 999999, "last_price": 1.0,
                            "ohlc": {"open": 1, "high": 1, "low": 1, "close": 1},
                            "volume_traded": 0}])

    assert seen == []
