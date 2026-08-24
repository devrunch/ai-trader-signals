"""
DerivProvider — forex/metals via Deriv's public WebSocket. Mocks
websockets.connect the same way other providers mock their own network
layer -- nothing here touches the real vendor.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from app.market.providers.deriv_provider import DerivProvider, KNOWN_PAIRS, deriv_symbol_for


def _mock_connect(response: dict):
    """websockets.connect(...) used as `async with ... as ws`. Returns a
    context manager whose ws.recv() yields one JSON response."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock(return_value=json.dumps(response))
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, ws


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
        with patch("app.market.providers.deriv_provider.websockets.connect", return_value=cm):
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
        with patch("app.market.providers.deriv_provider.websockets.connect", return_value=cm):
            result = await DerivProvider().get_quote("XAUUSD", "FOREX")

        assert result["prev_close"] == 2600.0

    @pytest.mark.asyncio
    async def test_a_vendor_error_response_degrades_to_none_not_a_crash(self):
        cm, _ = _mock_connect({"error": {"code": "InvalidSymbol", "message": "bad symbol"}})
        with patch("app.market.providers.deriv_provider.websockets.connect", return_value=cm):
            assert await DerivProvider().get_quote("XAUUSD", "FOREX") is None

    @pytest.mark.asyncio
    async def test_a_network_error_degrades_to_none_not_a_crash(self):
        with patch("app.market.providers.deriv_provider.websockets.connect", side_effect=OSError("no route")):
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
        with patch("app.market.providers.deriv_provider.websockets.connect", return_value=cm):
            df = await DerivProvider().get_historical_df("XAUUSD", "FOREX", "1d", 10)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2
        assert df["close"].iloc[-1] == 2615.0
        assert isinstance(df.index, pd.DatetimeIndex)

    @pytest.mark.asyncio
    async def test_an_empty_candles_list_returns_none(self):
        cm, _ = _mock_connect({"candles": []})
        with patch("app.market.providers.deriv_provider.websockets.connect", return_value=cm):
            assert await DerivProvider().get_historical_df("XAUUSD", "FOREX", "1d", 10) is None

    @pytest.mark.asyncio
    async def test_a_vendor_error_response_degrades_to_none_not_a_crash(self):
        cm, _ = _mock_connect({"error": {"code": "InvalidSymbol", "message": "bad symbol"}})
        with patch("app.market.providers.deriv_provider.websockets.connect", return_value=cm):
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
