"""
MarketDataRouter's fallback behaviour: Kite is the real NSE/BSE vendor now,
but yfinance is still there as a safety net. Anything that would otherwise
show up as "no data" instead rides the fallback provider, invisibly.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.market.providers.registry import MarketDataRouter


class _FakeProvider:
    def __init__(self, quote=None, df=None, search_results=None, raises=False):
        self._quote = quote
        self._df = df
        self._search_results = search_results or []
        self._raises = raises
        self.calls: list[str] = []

    async def get_quote(self, symbol, exchange):
        self.calls.append("get_quote")
        if self._raises:
            return None
        return self._quote

    async def get_historical_df(self, symbol, exchange, interval, days):
        self.calls.append("get_historical_df")
        if self._raises:
            return None
        return self._df

    async def search(self, query, limit):
        self.calls.append("search")
        return self._search_results[:limit]


def _router_with(kite: _FakeProvider, fallback: _FakeProvider) -> MarketDataRouter:
    router = MarketDataRouter()
    router.fallback = fallback
    router.providers["NSE"] = kite
    router.providers["BSE"] = kite
    return router


@pytest.mark.asyncio
async def test_a_kite_quote_is_used_when_it_succeeds():
    kite = _FakeProvider(quote={"symbol": "RELIANCE", "ltp": 1290})
    fallback = _FakeProvider(quote={"symbol": "RELIANCE", "ltp": 1111})
    router = _router_with(kite, fallback)

    result = await router.get_quote("RELIANCE", "NSE", bypass_cache=True)

    assert result["ltp"] == 1290
    assert "get_quote" not in fallback.calls


@pytest.mark.asyncio
async def test_falls_back_to_yfinance_when_kite_returns_none():
    kite = _FakeProvider(raises=True)
    fallback = _FakeProvider(quote={"symbol": "RELIANCE", "ltp": 1111})
    router = _router_with(kite, fallback)

    result = await router.get_quote("RELIANCE", "NSE", bypass_cache=True)

    assert result["ltp"] == 1111


@pytest.mark.asyncio
async def test_historical_data_falls_back_the_same_way():
    kite = _FakeProvider(raises=True)
    df = pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]})
    fallback = _FakeProvider(df=df)
    router = _router_with(kite, fallback)

    result = await router.get_historical_df("RELIANCE", "NSE", "1d", 30, bypass_cache=True)

    assert result is not None
    assert len(result) == 1


@pytest.mark.asyncio
async def test_nasdaq_never_touches_kite():
    """NSE/BSE-only vendor — an exchange with no entry in `providers` should
    never even attempt the Kite path."""
    kite = _FakeProvider(quote={"symbol": "AAPL", "ltp": 999})  # would be wrong if ever used
    fallback = _FakeProvider(quote={"symbol": "AAPL", "ltp": 230})
    router = _router_with(kite, fallback)

    result = await router.get_quote("AAPL", "NASDAQ", bypass_cache=True)

    assert result["ltp"] == 230
    assert kite.calls == []


@pytest.mark.asyncio
async def test_search_merges_kite_and_yfinance_results():
    kite = _FakeProvider(search_results=[{"symbol": "RELIANCE", "name": "Reliance", "exchange": "NSE"}])
    fallback = _FakeProvider(search_results=[{"symbol": "AAPL", "name": "Apple", "exchange": "NASDAQ"}])
    router = _router_with(kite, fallback)

    results = await router.search("re", limit=8)

    symbols = {r["symbol"] for r in results}
    assert symbols == {"RELIANCE", "AAPL"}


@pytest.mark.asyncio
async def test_search_respects_the_combined_limit():
    kite = _FakeProvider(search_results=[{"symbol": f"K{i}", "name": "x", "exchange": "NSE"} for i in range(5)])
    fallback = _FakeProvider(search_results=[{"symbol": f"Y{i}", "name": "x", "exchange": "NASDAQ"} for i in range(5)])
    router = _router_with(kite, fallback)

    results = await router.search("x", limit=6)

    assert len(results) == 6
