"""
DerivProvider — forex/metals via Deriv's public WebSocket. Mocks
websockets.connect the same way other providers mock their own network
layer -- nothing here touches the real vendor.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from app.market.providers.deriv_provider import (
    _MAX_COUNT,
    KNOWN_PAIRS,
    DerivProvider,
    deriv_symbol_for,
)


def _mock_connect(response: dict, pages: list[dict] | None = None):
    """A stand-in for the shared Deriv connection.

    The real one is a single socket whose answers are matched to their
    requests by `req_id` and read by a background task, so this echoes that
    id back the way Deriv does and yields messages as an async iterator.
    `pages` hands out one response per request in order, then repeats the
    last; `sent` records the payloads, for asserting what was asked for.
    """
    queue = list(pages) if pages else []
    sent: list[dict] = []
    outbox: asyncio.Queue = asyncio.Queue()

    async def _send(raw):
        payload = json.loads(raw)
        sent.append(payload)
        body = queue.pop(0) if queue else response
        await outbox.put(json.dumps(dict(body, req_id=payload.get("req_id"))))

    async def _messages():
        while True:
            yield await outbox.get()

    ws = MagicMock()
    ws.send = AsyncMock(side_effect=_send)
    ws.close = AsyncMock()
    ws.closed = False
    ws.__aiter__ = lambda _self=None: _messages()
    ws.sent = sent

    async def _connect(*_args, **_kwargs):
        return ws

    return _connect, ws


class TestDerivSymbolFor:
    def test_known_pair_resolves_to_deriv_convention(self):
        assert deriv_symbol_for("XAUUSD") == "frxXAUUSD"
        assert deriv_symbol_for("xauusd") == "frxXAUUSD"  # case-insensitive
        assert deriv_symbol_for("EURUSD") == "frxEURUSD"

    def test_unknown_symbol_resolves_to_none(self):
        assert deriv_symbol_for("BTCUSD") is None

    def test_all_29_real_instruments_are_known(self):
        # Confirmed live against Deriv's own active_symbols -- not a
        # tautology, a tripwire against silently losing an entry.
        assert len(KNOWN_PAIRS) == 29


class TestGetQuote:
    @pytest.mark.asyncio
    async def test_returns_none_for_an_unknown_symbol(self):
        provider = DerivProvider()
        assert await provider.get_quote("BTCUSD", "FOREX") is None

    @pytest.mark.asyncio
    async def test_a_real_quote_derives_change_from_the_two_candles(self):
        cm, ws = _mock_connect({
            "candles": [
                {"epoch": 1000, "open": 2600.0, "high": 2610.0, "low": 2595.0, "close": 2605.0},
                {"epoch": 1086400, "open": 2605.0, "high": 2650.5, "low": 2600.0, "close": 2650.5},
            ],
        })
        with patch("app.market.providers.deriv_socket.websockets.connect", cm):
            provider = DerivProvider()
            result = await provider.get_quote("XAUUSD", "FOREX")

        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["ticks_history"] == "frxXAUUSD"
        assert sent["granularity"] == 86400

        assert result["symbol"] == "XAUUSD"
        assert result["exchange"] == "FOREX"
        assert result["ltp"] == 2650.5
        assert result["prev_close"] == 2605.0
        assert result["change"] == pytest.approx(45.5)
        assert result["open"] == 2605.0
        assert result["high"] == 2650.5
        assert result["low"] == 2600.0
        # Never fabricated -- spot forex/metals carry no real trade volume
        # or order-book depth from this vendor.
        assert result["volume"] is None
        assert result["bid"] is None
        assert result["ask"] is None

    @pytest.mark.asyncio
    async def test_a_single_candle_uses_its_own_open_as_prev_close(self):
        cm, ws = _mock_connect({
            "candles": [{"epoch": 1000, "open": 2600.0, "high": 2610.0, "low": 2595.0, "close": 2605.0}],
        })
        with patch("app.market.providers.deriv_socket.websockets.connect", cm):
            result = await DerivProvider().get_quote("XAUUSD", "FOREX")

        assert result["prev_close"] == 2600.0

    @pytest.mark.asyncio
    async def test_a_vendor_error_response_degrades_to_none_not_a_crash(self):
        cm, _ = _mock_connect({"error": {"code": "InvalidSymbol", "message": "bad symbol"}})
        with patch("app.market.providers.deriv_socket.websockets.connect", cm):
            assert await DerivProvider().get_quote("XAUUSD", "FOREX") is None

    @pytest.mark.asyncio
    async def test_a_network_error_degrades_to_none_not_a_crash(self):
        with patch("app.market.providers.deriv_socket.websockets.connect", side_effect=OSError("no route")):
            assert await DerivProvider().get_quote("XAUUSD", "FOREX") is None


class TestGetHistoricalDf:
    @pytest.mark.asyncio
    async def test_returns_none_for_an_unknown_symbol(self):
        assert await DerivProvider().get_historical_df("BTCUSD", "FOREX", "1d", 30) is None

    @pytest.mark.asyncio
    async def test_a_real_candle_series_becomes_a_real_ohlcv_dataframe(self):
        cm, ws = _mock_connect({
            "candles": [
                {"epoch": 1786752000, "open": 2600.0, "high": 2610.0, "low": 2595.0, "close": 2605.0},
                {"epoch": 1786838400, "open": 2605.0, "high": 2620.0, "low": 2600.0, "close": 2615.0},
            ],
        })
        with patch("app.market.providers.deriv_socket.websockets.connect", cm):
            df = await DerivProvider().get_historical_df("XAUUSD", "FOREX", "1d", 10)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2
        assert df["close"].iloc[-1] == 2615.0
        assert isinstance(df.index, pd.DatetimeIndex)

    @pytest.mark.asyncio
    async def test_an_empty_candles_list_returns_none(self):
        cm, _ = _mock_connect({"candles": []})
        with patch("app.market.providers.deriv_socket.websockets.connect", cm):
            assert await DerivProvider().get_historical_df("XAUUSD", "FOREX", "1d", 10) is None

    @pytest.mark.asyncio
    async def test_a_vendor_error_response_degrades_to_none_not_a_crash(self):
        cm, _ = _mock_connect({"error": {"code": "InvalidSymbol", "message": "bad symbol"}})
        with patch("app.market.providers.deriv_socket.websockets.connect", cm):
            assert await DerivProvider().get_historical_df("XAUUSD", "FOREX", "1d", 10) is None


class TestSearch:
    @pytest.mark.asyncio
    async def test_gold_matches_xauusd(self):
        results = await DerivProvider().search("gold", limit=8)
        assert {"symbol": "XAUUSD", "name": "Gold/USD", "exchange": "FOREX"} in results

    @pytest.mark.asyncio
    async def test_eurusd_matches_itself(self):
        results = await DerivProvider().search("eurusd", limit=8)
        assert len(results) == 1
        assert results[0]["symbol"] == "EURUSD"

    @pytest.mark.asyncio
    async def test_limit_is_respected_across_many_matches(self):
        # "usd" matches most of the table -- confirms the cap actually caps.
        results = await DerivProvider().search("usd", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_unrelated_query_matches_nothing(self):
        assert await DerivProvider().search("reliance", limit=8) == []

    @pytest.mark.asyncio
    async def test_empty_query_matches_nothing(self):
        assert await DerivProvider().search("", limit=8) == []


class TestBackwardPaging:
    """One vendor response reaches back 1000 candles from its own `end` and no
    further, and a closed market answers with nothing at all -- which used to
    read as "history ends here" and 404 every weekend 1m chart."""

    @pytest.mark.asyncio
    async def test_an_empty_window_does_not_stop_the_walk(self):
        # The first windows are the weekend; the session is further back.
        connect, ws = _mock_connect(
            {"candles": []},
            pages=[
                {"candles": []},
                {"candles": [{"epoch": 1786752000, "open": 1.0, "high": 2.0,
                             "low": 0.5, "close": 1.5}]},
            ],
        )
        with patch("app.market.providers.deriv_socket.websockets.connect", connect):
            df = await DerivProvider().get_historical_df("XAUUSD", "FOREX", "1m", 5)

        assert df is not None and len(df) == 1
        assert len({r["end"] for r in ws.sent}) > 1

    @pytest.mark.asyncio
    async def test_the_windows_cover_the_span_a_page_at_a_time(self):
        connect, ws = _mock_connect({"candles": []})
        with patch("app.market.providers.deriv_socket.websockets.connect", connect):
            await DerivProvider().get_historical_df("XAUUSD", "FOREX", "1m", 5)

        ends = sorted({int(r["end"]) for r in ws.sent}, reverse=True)
        assert len(ends) > 1, "a 5-day 1m span cannot fit in one 1000-candle window"
        # Each window sits exactly one page further back than the last, so
        # nothing between them is skipped.
        steps = {ends[i] - ends[i + 1] for i in range(len(ends) - 1)}
        assert steps == {_MAX_COUNT * 60}

    @pytest.mark.asyncio
    async def test_a_span_that_fits_in_one_window_is_one_request(self):
        connect, ws = _mock_connect(
            {"candles": [{"epoch": 1786752000, "open": 1.0, "high": 2.0,
                          "low": 0.5, "close": 1.5}]},
        )
        with patch("app.market.providers.deriv_socket.websockets.connect", connect):
            await DerivProvider().get_historical_df("XAUUSD", "FOREX", "1d", 30)

        assert len(ws.sent) == 1

    @pytest.mark.asyncio
    async def test_one_connection_serves_every_window(self):
        # The reason this exists: a connection per window cost 1.48s of
        # handshake each, and eight of them put a 1m chart past the API's
        # upstream timeout.
        connect, ws = _mock_connect({"candles": []})
        opened = 0

        async def counting_connect(*args, **kwargs):
            nonlocal opened
            opened += 1
            return await connect(*args, **kwargs)

        with patch("app.market.providers.deriv_socket.websockets.connect",
                   counting_connect):
            await DerivProvider().get_historical_df("XAUUSD", "FOREX", "1m", 5)

        assert len(ws.sent) > 1
        assert opened == 1

