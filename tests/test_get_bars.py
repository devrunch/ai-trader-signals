"""`get_bars`, where an empty chart finally says why it is empty.

Every one of these is a production incident in miniature: gold routed to an
equity exchange, a weekend read as the end of history, a window silently
shortened to another vendor's limit.
"""
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.market.contract import BarsStatus, VolumeSource
from app.market.providers.registry import MarketDataRouter


def _frame(rows=2):
    idx = pd.to_datetime([1786752000 + i * 3600 for i in range(rows)], unit="s")
    return pd.DataFrame(
        {"open": [1.0] * rows, "high": [2.0] * rows, "low": [0.5] * rows,
         "close": [1.5] * rows, "volume": [10.0] * rows},
        index=idx,
    )


@pytest.fixture
def router():
    return MarketDataRouter()


def _with_history(router, value):
    return patch.object(router, "get_historical_df", AsyncMock(return_value=value))


class TestRouting:
    @pytest.mark.asyncio
    async def test_gold_is_served_as_forex_whatever_exchange_was_passed(self, router):
        with _with_history(router, _frame()):
            result = await router.get_bars("XAUUSD", "1h", 5, exchange="NSE")

        assert result.status is BarsStatus.OK
        assert result.symbol.exchange == "FOREX"
        assert result.symbol.provider == "deriv"

    @pytest.mark.asyncio
    async def test_an_unknown_symbol_is_its_own_answer(self, router):
        result = await router.get_bars("", "1h", 5)
        assert result.status is BarsStatus.UNKNOWN_SYMBOL
        assert result.bars == []

    @pytest.mark.asyncio
    async def test_forex_is_labelled_as_having_no_volume(self, router):
        # Nothing to compare against an equity's volume: this vendor has none,
        # and the delayed tick count that stood in for it is gone.
        with _with_history(router, _frame()):
            result = await router.get_bars("EURUSD", "1h", 5)
        assert result.volume_source is VolumeSource.NONE


class TestLimits:
    @pytest.mark.asyncio
    async def test_an_unsupported_interval_says_so_instead_of_returning_nothing(self, router):
        result = await router.get_bars("RELIANCE", "2s", 5, exchange="NSE")
        assert result.status is BarsStatus.UNSUPPORTED_INTERVAL
        assert "2s" in result.reason

    @pytest.mark.asyncio
    async def test_a_window_past_the_vendor_limit_is_clamped_and_declared(self, router):
        # Kite serves 60 days of 1m bars. Asking for a year used to return six
        # days with nothing saying so.
        with _with_history(router, _frame()) as history:
            result = await router.get_bars("RELIANCE", "1m", 365, exchange="NSE")

        assert result.truncated_to_days == 60
        assert history.await_args.args[3] == 60

    @pytest.mark.asyncio
    async def test_a_window_inside_the_limit_is_not_flagged(self, router):
        with _with_history(router, _frame()):
            result = await router.get_bars("RELIANCE", "1m", 10, exchange="NSE")
        assert result.truncated_to_days is None

    @pytest.mark.asyncio
    async def test_each_vendor_is_clamped_by_its_own_limit(self, router):
        # The bug this replaces: one global table of yfinance's limits governed
        # Kite too, clamping hourly bars to 720 days where Kite serves 400.
        with _with_history(router, _frame()):
            kite = await router.get_bars("RELIANCE", "1h", 1000, exchange="NSE")
            yahoo = await router.get_bars("AAPL", "1h", 1000)

        assert kite.truncated_to_days == 400
        assert yahoo.truncated_to_days == 720


class TestEmptyResults:
    @pytest.mark.asyncio
    async def test_a_closed_market_is_not_an_error(self, router):
        # Saturday. Forex is shut; that is not a vendor problem and must not
        # surface as one.
        with _with_history(router, None), \
             patch("app.market.providers.registry.datetime") as clock:
            clock.now.return_value = pd.Timestamp("2026-09-12 12:00", tz="UTC").to_pydatetime()
            result = await router.get_bars("XAUUSD", "1m", 1)

        assert result.status is BarsStatus.CLOSED_MARKET
        assert "closed" in result.reason

    @pytest.mark.asyncio
    async def test_a_closed_market_says_when_it_opens(self, router):
        with _with_history(router, None), \
             patch("app.market.providers.registry.datetime") as clock:
            clock.now.return_value = pd.Timestamp("2026-09-12 12:00", tz="UTC").to_pydatetime()
            result = await router.get_bars("XAUUSD", "1m", 1)

        assert "2026-09-13 21:00 UTC" in result.reason

    @pytest.mark.asyncio
    async def test_nothing_during_an_open_session_is_reported_as_a_failure(self, router):
        # Wednesday midday in London: forex is open, so an empty answer is the
        # vendor's problem and should be looked at, not shrugged off.
        with _with_history(router, None), \
             patch("app.market.providers.registry.datetime") as clock:
            clock.now.return_value = pd.Timestamp("2026-09-09 12:00", tz="UTC").to_pydatetime()
            result = await router.get_bars("XAUUSD", "1m", 1)

        assert result.status is BarsStatus.VENDOR_ERROR

    @pytest.mark.asyncio
    async def test_an_unlisted_symbol_is_not_told_the_market_is_closed(self, router):
        # The confident lie this guards against: NOTAREALSYMBOL resolved to the
        # fallback vendor, the session happened to be shut, and the answer was
        # "NASDAQ is closed" about a symbol that does not exist anywhere.
        with _with_history(router, None), \
             patch("app.market.providers.registry.datetime") as clock:
            clock.now.return_value = pd.Timestamp("2026-09-12 12:00", tz="UTC").to_pydatetime()
            result = await router.get_bars("NOTAREALSYMBOL", "1d", 5)

        assert result.status is BarsStatus.NO_DATA
        assert "may not be listed" in result.reason

    @pytest.mark.asyncio
    async def test_a_listed_symbol_is_told_the_market_is_closed(self, router):
        # Same empty answer, but the vendor confirms the listing, so the closed
        # session is the real explanation and worth giving.
        kite = router.by_provider["kite"]
        with _with_history(router, None), \
             patch.object(kite, "knows_symbol", return_value=True, create=True), \
             patch("app.market.providers.registry.datetime") as clock:
            clock.now.return_value = pd.Timestamp("2026-09-12 12:00", tz="UTC").to_pydatetime()
            result = await router.get_bars("RELIANCE", "1d", 5, exchange="NSE")

        assert result.status is BarsStatus.CLOSED_MARKET
