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
    DerivProvider, KNOWN_PAIRS, _TICK_BACKFILL_CONCURRENCY, _TICK_BACKFILL_SECONDS,
    _TICK_CHUNK_SECONDS, _backfill_tick_volume, _fetch_ticks_chunk, deriv_symbol_for,
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


def _mock_connect_sequence(responses: list[dict]):
    """Like _mock_connect, but each successive `websockets.connect()` call
    (a fresh connection per _request(), by this module's own design) yields
    the next response in order -- needed to test get_historical_df's two
    real calls (candles, then ticks) with two different payloads."""
    cms, wss = [], []
    for response in responses:
        cm, ws = _mock_connect(response)
        cms.append(cm)
        wss.append(ws)
    return MagicMock(side_effect=cms), wss


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


class TestFetchTicksChunk:
    """Deriv gives no real volume for spot/CFD forex in either style --
    counting ticks per candle is the stand-in, the same convention MT4/MT5
    use for the same reason (see _fetch_ticks_chunk's own docstring). These
    test the single-chunk request in isolation, at the transport level."""

    @pytest.mark.asyncio
    async def test_a_trustworthy_chunk_returns_sorted_tick_epochs(self):
        cm, ws = _mock_connect({"history": {"times": [1000, 1000 + 900, 1000 + 300]}})
        with patch("app.market.providers.deriv_provider.websockets.connect", return_value=cm):
            result = await _fetch_ticks_chunk("frxXAUUSD", 1000, 2000)

        assert result == [1000, 1300, 1900]  # sorted
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["ticks_history"] == "frxXAUUSD"
        assert sent["style"] == "ticks"
        assert sent["start"] == 1000
        assert sent["end"] == 2000

    @pytest.mark.asyncio
    async def test_a_silently_narrowed_chunk_is_untrusted(self):
        # Live bug: Deriv can silently ignore an earlier `start` and return
        # ticks covering only its recent cache, with no error -- confirmed
        # live, a 1-hour request for XAUUSD came back with real ticks for
        # only the last ~17 minutes. Trusting that as "this chunk is
        # covered" would silently undercount every candle before the real
        # coverage started.
        cm, _ = _mock_connect({"history": {"times": [1000 + 2 * _TICK_CHUNK_SECONDS]}})
        with patch("app.market.providers.deriv_provider.websockets.connect", return_value=cm):
            assert await _fetch_ticks_chunk("frxXAUUSD", 1000, 1000 + _TICK_CHUNK_SECONDS) is None

    @pytest.mark.asyncio
    async def test_an_empty_response_is_untrusted(self):
        cm, _ = _mock_connect({"history": {"times": []}})
        with patch("app.market.providers.deriv_provider.websockets.connect", return_value=cm):
            assert await _fetch_ticks_chunk("frxXAUUSD", 1000, 2000) is None

    @pytest.mark.asyncio
    async def test_a_vendor_error_is_untrusted_not_a_crash(self):
        cm, _ = _mock_connect({"error": {"code": "InvalidSymbol", "message": "bad symbol"}})
        with patch("app.market.providers.deriv_provider.websockets.connect", return_value=cm):
            assert await _fetch_ticks_chunk("frxXAUUSD", 1000, 2000) is None

    @pytest.mark.asyncio
    async def test_a_network_error_is_untrusted_not_a_crash(self):
        with patch("app.market.providers.deriv_provider.websockets.connect", side_effect=OSError("no route")):
            assert await _fetch_ticks_chunk("frxXAUUSD", 1000, 2000) is None


class TestBackfillTickVolume:
    """Orchestration: chunking, bounded-concurrent fan-out, and merging --
    mocks _fetch_ticks_chunk directly rather than the transport, since these
    are about the walk/merge logic, not any one chunk's own request."""

    @pytest.mark.asyncio
    async def test_ticks_from_every_chunk_are_bucketed_into_the_right_candle(self):
        bucket_starts = [1000, 1000 + _TICK_CHUNK_SECONDS, 1000 + 2 * _TICK_CHUNK_SECONDS]

        async def fake_chunk(pair, chunk_start, chunk_end):
            # One real tick per chunk, right at its own start.
            return [chunk_start]

        with patch("app.market.providers.deriv_provider._fetch_ticks_chunk", side_effect=fake_chunk):
            counts = await _backfill_tick_volume(
                "frxXAUUSD", bucket_starts[0], bucket_starts[-1] + _TICK_CHUNK_SECONDS, bucket_starts,
            )

        assert counts == [1.0, 1.0, 1.0]

    @pytest.mark.asyncio
    async def test_an_untrusted_chunk_contributes_zero_without_affecting_others(self):
        bucket_starts = [1000, 1000 + _TICK_CHUNK_SECONDS]

        async def fake_chunk(pair, chunk_start, chunk_end):
            return None if chunk_start == 1000 else [chunk_start, chunk_start + 10]

        with patch("app.market.providers.deriv_provider._fetch_ticks_chunk", side_effect=fake_chunk):
            counts = await _backfill_tick_volume(
                "frxXAUUSD", bucket_starts[0], bucket_starts[-1] + _TICK_CHUNK_SECONDS, bucket_starts,
            )

        assert counts == [0.0, 2.0]

    @pytest.mark.asyncio
    async def test_only_the_trailing_window_is_backfilled_not_the_whole_range(self):
        # Live finding: backfilling the WHOLE fetched candle range hit
        # Deriv's real rate limit hard (21-51s chart loads, mass HTTP 429s,
        # and still-incomplete coverage anyway, since the limiter kept
        # knocking chunks out regardless) -- this only ever attempts the
        # most recent _TICK_BACKFILL_SECONDS, no matter how much wider the
        # full requested candle range is.
        start_epoch = 1000
        end_epoch = start_epoch + 10 * _TICK_BACKFILL_SECONDS  # far wider than the trailing window
        bucket_starts = [start_epoch]  # bucketing itself is covered elsewhere; this is about request scoping

        async def fake_chunk(pair, chunk_start, chunk_end):
            return [chunk_start]

        with patch("app.market.providers.deriv_provider._fetch_ticks_chunk", side_effect=fake_chunk) as fake:
            await _backfill_tick_volume("frxXAUUSD", start_epoch, end_epoch, bucket_starts)

        earliest_requested = min(call.args[1] for call in fake.call_args_list)
        assert earliest_requested >= end_epoch - _TICK_BACKFILL_SECONDS
        assert earliest_requested > start_epoch  # confirms the cutoff actually bit -- not just a wide window

    @pytest.mark.asyncio
    async def test_concurrency_stays_within_the_bound(self):
        in_flight = 0
        peak = 0

        async def fake_chunk(pair, chunk_start, chunk_end):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)  # yield, so overlapping calls actually overlap
            in_flight -= 1
            return None

        bucket_starts = [1000]
        end = 1000 + _TICK_CHUNK_SECONDS * (_TICK_BACKFILL_CONCURRENCY * 3)
        with patch("app.market.providers.deriv_provider._fetch_ticks_chunk", side_effect=fake_chunk):
            await _backfill_tick_volume("frxXAUUSD", 1000, end, bucket_starts)

        assert peak <= _TICK_BACKFILL_CONCURRENCY


class TestGetHistoricalDfVolumeWiring:
    """get_historical_df's own use of the backfill result -- patches
    _backfill_tick_volume directly rather than simulating N real chunk
    requests, since that orchestration is already covered above."""

    CANDLES = [
        {"epoch": 1786752000, "open": 2600.0, "high": 2610.0, "low": 2595.0, "close": 2605.0},
        {"epoch": 1786755600, "open": 2605.0, "high": 2620.0, "low": 2600.0, "close": 2615.0},
    ]

    @pytest.mark.asyncio
    async def test_the_backfilled_counts_land_in_the_volume_column(self):
        cm, _ = _mock_connect({"candles": self.CANDLES})
        with patch("app.market.providers.deriv_provider.websockets.connect", return_value=cm), \
             patch("app.market.providers.deriv_provider._backfill_tick_volume", return_value=[7.0, 12.0]) as fake:
            df = await DerivProvider().get_historical_df("XAUUSD", "FOREX", "1h", 1)

        assert list(df["volume"]) == [7.0, 12.0]
        fake.assert_called_once_with("frxXAUUSD", 1786752000, 1786755600 + 3600, [1786752000, 1786755600])

    @pytest.mark.asyncio
    async def test_a_real_short_span_gets_real_end_to_end_tick_volume(self):
        # One true end-to-end pass with the real transport mock, kept to a
        # single chunk (span == _TICK_CHUNK_SECONDS) so it stays simple.
        candles = [
            {"epoch": 1786752000, "open": 2600.0, "high": 2610.0, "low": 2595.0, "close": 2605.0},
        ]
        connect, wss = _mock_connect_sequence([
            {"candles": candles},
            {"history": {"times": [1786752000, 1786752010, 1786752020]}},  # within the 60s (1m) candle window
        ])
        with patch("app.market.providers.deriv_provider.websockets.connect", connect):
            df = await DerivProvider().get_historical_df("XAUUSD", "FOREX", "1m", 1)

        assert list(df["volume"]) == [3.0]
        ticks_sent = json.loads(wss[1].send.call_args[0][0])
        assert ticks_sent["style"] == "ticks"


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
