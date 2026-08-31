"""
macro_events -- real macro/news context for the brief's narrative. Mocks
httpx (FRED, yfinance's own network calls go through yf.Ticker, mocked
directly) and a fake Redis client the same shape chart_layouts' own fake
model uses elsewhere in this codebase: minimal, in-memory, just enough
surface for the code under test.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.market import macro_events


def _fred_response(observations: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"observations": observations})
    return resp


class _FakeRedis:
    """In-memory stand-in for redis.asyncio's client -- get/set/aclose only,
    the only surface fred_releases actually uses."""

    def __init__(self):
        self.store: dict[str, bytes] = {}
        self.closed = False

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value.encode() if isinstance(value, str) else value

    async def aclose(self):
        self.closed = True


def _settings(fred_api_key="test-fred-key", redis_url="redis://fake", tavily_api_key="test-tavily-key"):
    s = MagicMock()
    s.fred_api_key = fred_api_key
    s.redis_url = redis_url
    s.tavily_api_key = tavily_api_key
    return s


class TestFredReleases:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_none_not_empty(self):
        with patch("app.market.macro_events.get_settings", return_value=_settings(fred_api_key="")):
            assert await macro_events.fred_releases() is None

    @pytest.mark.asyncio
    async def test_a_fresh_print_is_reported_once_then_suppressed(self):
        fake_redis = _FakeRedis()
        rows = [
            {"date": "2026-07-01", "value": "3.2"},
            {"date": "2026-06-01", "value": "3.0"},
        ]
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(return_value=_fred_response(rows))

        with patch("app.market.macro_events.get_settings", return_value=_settings()), \
             patch("app.market.macro_events.redis.from_url", return_value=fake_redis), \
             patch("app.market.macro_events.httpx.AsyncClient", return_value=client):
            first = await macro_events.fred_releases()

        assert len(first) == len(macro_events._FRED_SERIES)  # every series is "new" on first-ever check
        cpi = next(r for r in first if r["series_id"] == "CPIAUCSL")
        assert cpi == {"name": "CPI (all items)", "series_id": "CPIAUCSL", "actual": 3.2, "prior": 3.0, "date": "2026-07-01"}
        assert fake_redis.closed

        # Same data, second check -- nothing new to report.
        fake_redis2 = _FakeRedis()
        fake_redis2.store = dict(fake_redis.store)
        with patch("app.market.macro_events.get_settings", return_value=_settings()), \
             patch("app.market.macro_events.redis.from_url", return_value=fake_redis2), \
             patch("app.market.macro_events.httpx.AsyncClient", return_value=client):
            second = await macro_events.fred_releases()
        assert second == []

    @pytest.mark.asyncio
    async def test_an_unpublished_point_marked_dot_is_skipped(self):
        fake_redis = _FakeRedis()
        rows = [{"date": "2026-07-01", "value": "."}, {"date": "2026-06-01", "value": "3.0"}]
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(return_value=_fred_response(rows))

        with patch("app.market.macro_events.get_settings", return_value=_settings()), \
             patch("app.market.macro_events.redis.from_url", return_value=fake_redis), \
             patch("app.market.macro_events.httpx.AsyncClient", return_value=client):
            result = await macro_events.fred_releases()
        assert result == []

    @pytest.mark.asyncio
    async def test_a_vendor_error_degrades_to_none_for_that_series_not_a_crash(self):
        fake_redis = _FakeRedis()
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.get = AsyncMock(side_effect=OSError("network down"))

        with patch("app.market.macro_events.get_settings", return_value=_settings()), \
             patch("app.market.macro_events.redis.from_url", return_value=fake_redis):
            result = await macro_events.fred_releases()
        assert result == []


class TestYfinanceHeadlines:
    @pytest.mark.asyncio
    async def test_real_headlines_are_deduplicated_by_url_across_tickers(self):
        shared = {"title": "Fed signals rate path", "content": {"provider": {"displayName": "Reuters"}, "canonicalUrl": {"url": "https://example.com/a"}}}
        unique = {"title": "Gold rallies", "content": {"provider": {"displayName": "Bloomberg"}, "canonicalUrl": {"url": "https://example.com/b"}}}

        with patch("app.market.macro_events.yf.Ticker") as MockTicker:
            def ticker_factory(sym):
                m = MagicMock()
                m.ticker = sym
                m.news = [shared, unique] if sym == "GC=F" else [shared]
                return m
            MockTicker.side_effect = ticker_factory
            result = await macro_events.yfinance_headlines()

        urls = [h["url"] for h in result]
        assert urls.count("https://example.com/a") == 1  # deduplicated even though every ticker returned it
        assert "https://example.com/b" in urls

    @pytest.mark.asyncio
    async def test_a_vendor_error_for_one_ticker_does_not_drop_the_others(self):
        def ticker_factory(sym):
            m = MagicMock()
            if sym == "GC=F":
                type(m).news = property(lambda self: (_ for _ in ()).throw(OSError("rate limited")))
            else:
                m.news = [{"title": f"Headline for {sym}", "content": {}}]
            return m

        with patch("app.market.macro_events.yf.Ticker", side_effect=ticker_factory):
            result = await macro_events.yfinance_headlines()

        assert len(result) == len(macro_events._MACRO_NEWS_TICKERS) - 1


class TestRedditChatter:
    @pytest.mark.asyncio
    async def test_query_is_scoped_to_reddit_via_tavily(self):
        with patch("app.market.macro_events.get_settings", return_value=_settings()), \
             patch("app.market.macro_events.tavily_search", new=AsyncMock(return_value={"results": [], "count": 0})) as fake:
            await macro_events.reddit_chatter("XAUUSD sentiment")

        fake.assert_called_once_with("test-tavily-key", "site:reddit.com XAUUSD sentiment", 5)

    @pytest.mark.asyncio
    async def test_no_tavily_key_surfaces_as_an_error_not_empty_chatter(self):
        with patch("app.market.macro_events.get_settings", return_value=_settings(tavily_api_key="")):
            result = await macro_events.reddit_chatter("gold")
        assert "error" in result
