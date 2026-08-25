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

from app.market.providers.deriv_provider import (
    DerivProvider,
    KNOWN_PAIRS,
    deriv_symbol_for,
    tick_volume_since,
)


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


class TestDukascopyTickVolume:
    """Deriv gives no real volume for spot/CFD forex in either style --
    get_historical_df's volume column instead comes from Dukascopy (see
    dukascopy_bridge.py and deriv_provider.py's own module docstring).
    Mocks dukascopy_bridge.fetch_tick_timestamps directly -- the subprocess
    call itself is covered by test_dukascopy_bridge.py."""

    @pytest.mark.asyncio
    async def test_ticks_are_bucketed_into_the_right_candle(self):
        bucket_starts = [1786752000, 1786752060, 1786752120]  # 1m candles
        ticks_ms = [
            1786752000_000, 1786752030_000,  # 2 in candle 0
            1786752065_000,                   # 1 in candle 1
            # candle 2: none
        ]
        with patch(
            "app.market.providers.deriv_provider.dukascopy_bridge.fetch_tick_timestamps",
            return_value=ticks_ms,
        ) as fake:
            from app.market.providers.deriv_provider import _dukascopy_tick_volume
            counts = await _dukascopy_tick_volume("XAUUSD", bucket_starts[0], bucket_starts[-1] + 60, bucket_starts)

        assert counts == [2.0, 1.0, 0.0]
        fake.assert_called_once_with("xauusd", bucket_starts[0] * 1000, (bucket_starts[-1] + 60) * 1000)

    @pytest.mark.asyncio
    async def test_a_vendor_gap_falls_back_to_zero_for_every_candle(self):
        # Live finding: Dukascopy's publish lag (~15-20 min, confirmed live)
        # means a range reaching up to "now" routinely comes back with no
        # ticks at all for its most recent stretch -- same honest 0.0 as
        # any other vendor gap, not a special case.
        bucket_starts = [1786752000, 1786752060]
        with patch("app.market.providers.deriv_provider.dukascopy_bridge.fetch_tick_timestamps", return_value=None):
            from app.market.providers.deriv_provider import _dukascopy_tick_volume
            counts = await _dukascopy_tick_volume("XAUUSD", bucket_starts[0], bucket_starts[-1] + 60, bucket_starts)

        assert counts == [0.0, 0.0]

    @pytest.mark.asyncio
    async def test_an_empty_tick_list_also_falls_back_to_zero(self):
        bucket_starts = [1786752000, 1786752060]
        with patch("app.market.providers.deriv_provider.dukascopy_bridge.fetch_tick_timestamps", return_value=[]):
            from app.market.providers.deriv_provider import _dukascopy_tick_volume
            counts = await _dukascopy_tick_volume("XAUUSD", bucket_starts[0], bucket_starts[-1] + 60, bucket_starts)

        assert counts == [0.0, 0.0]


class TestGetHistoricalDfVolumeWiring:
    """get_historical_df's own use of the Dukascopy result -- patches
    _dukascopy_tick_volume directly rather than the subprocess, since that
    orchestration is already covered above."""

    CANDLES = [
        {"epoch": 1786752000, "open": 2600.0, "high": 2610.0, "low": 2595.0, "close": 2605.0},
        {"epoch": 1786755600, "open": 2605.0, "high": 2620.0, "low": 2600.0, "close": 2615.0},
    ]

    @pytest.mark.asyncio
    async def test_the_tick_volume_lands_in_the_volume_column(self):
        cm, _ = _mock_connect({"candles": self.CANDLES})
        with patch("app.market.providers.deriv_provider.websockets.connect", return_value=cm), \
             patch("app.market.providers.deriv_provider._dukascopy_tick_volume", return_value=[7.0, 12.0]) as fake:
            df = await DerivProvider().get_historical_df("XAUUSD", "FOREX", "1h", 1)

        assert list(df["volume"]) == [7.0, 12.0]
        fake.assert_called_once_with("XAUUSD", 1786752000, 1786755600 + 3600, [1786752000, 1786755600])


class TestTickVolumeSince:
    """Powers the terminal's 5-second live-volume poll for the still-forming
    candle -- see market/router.py's /tick-volume/{symbol}."""

    @pytest.mark.asyncio
    async def test_counts_real_ticks_since_the_given_epoch(self):
        with patch(
            "app.market.providers.deriv_provider.dukascopy_bridge.fetch_tick_timestamps",
            return_value=[1_000, 2_000, 3_000],
        ) as fake:
            count = await tick_volume_since("XAUUSD", 1786752000)

        assert count == 3
        args = fake.call_args[0]
        assert args[0] == "xauusd"
        assert args[1] == 1786752000 * 1000

    @pytest.mark.asyncio
    async def test_a_symbol_not_covered_by_deriv_returns_none_without_calling_dukascopy(self):
        with patch(
            "app.market.providers.deriv_provider.dukascopy_bridge.fetch_tick_timestamps",
        ) as fake:
            count = await tick_volume_since("RELIANCE", 1786752000)

        assert count is None
        fake.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_vendor_gap_returns_none_not_zero(self):
        # None (couldn't ask) and 0 (asked, got nothing) are different
        # answers -- collapsing them would show a confidently wrong "no
        # trading activity" instead of an honest "couldn't check right now".
        with patch(
            "app.market.providers.deriv_provider.dukascopy_bridge.fetch_tick_timestamps",
            return_value=None,
        ):
            assert await tick_volume_since("XAUUSD", 1786752000) is None

    @pytest.mark.asyncio
    async def test_a_genuinely_empty_tick_list_is_a_real_zero(self):
        with patch(
            "app.market.providers.deriv_provider.dukascopy_bridge.fetch_tick_timestamps",
            return_value=[],
        ):
            assert await tick_volume_since("XAUUSD", 1786752000) == 0


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
