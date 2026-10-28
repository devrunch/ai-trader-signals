"""Which Yahoo ticker a symbol becomes.

This provider is the fallback the router reaches for when Kite returns
nothing. An index went to Yahoo as "NIFTY 50.NS" -- Kite's own name with the
equity suffix bolted on -- which Yahoo cannot resolve, so the fallback failed
for exactly the symbols it most needed to cover. Seen live: Kite blipped
mid-session on a normal trading day, the chat agent got no data for Nifty on
any timeframe, and told the user to check their data connectivity.
"""
from __future__ import annotations

import pytest

from app.market.providers.yfinance_provider import YFinanceProvider


@pytest.fixture
def ticker_for():
    provider = YFinanceProvider.__new__(YFinanceProvider)
    return provider._ticker_for


class TestIndianIndices:
    @pytest.mark.parametrize("symbol,exchange,expected", [
        ("NIFTY 50", "NSE", "^NSEI"),
        ("NIFTY BANK", "NSE", "^NSEBANK"),
        ("INDIA VIX", "NSE", "^INDIAVIX"),
        ("SENSEX", "BSE", "^BSESN"),
    ])
    def test_an_index_becomes_its_own_yahoo_ticker(self, ticker_for, symbol, exchange, expected):
        assert ticker_for(symbol, exchange) == expected

    def test_the_lookup_does_not_care_about_case(self, ticker_for):
        assert ticker_for("nifty 50", "NSE") == "^NSEI"


class TestEverythingElseIsUnchanged:
    def test_an_nse_equity_still_takes_the_ns_suffix(self, ticker_for):
        assert ticker_for("RELIANCE", "NSE") == "RELIANCE.NS"

    def test_a_bse_equity_still_takes_the_bo_suffix(self, ticker_for):
        assert ticker_for("RELIANCE", "BSE") == "RELIANCE.BO"

    def test_an_already_qualified_yahoo_symbol_is_left_alone(self, ticker_for):
        """Adding a suffix to one 404s -- "^NSEI" became "^NSEI.NS" once and
        returned a null Nifty for as long as it stood."""
        assert ticker_for("^NSEI", "NSE") == "^NSEI"
        assert ticker_for("USDINR=X", "NSE") == "USDINR=X"
        assert ticker_for("BZ=F", "NSE") == "BZ=F"

    def test_a_us_ticker_carries_no_suffix(self, ticker_for):
        assert ticker_for("AAPL", "NASDAQ") == "AAPL"

    def test_an_unmapped_indian_symbol_keeps_the_old_behaviour(self, ticker_for):
        """Only indices whose Yahoo ticker is known for certain are mapped.
        Guessing one produces the same silent 404 the map exists to fix."""
        assert ticker_for("NIFTY MIDCAP 100", "NSE") == "NIFTY MIDCAP 100.NS"
