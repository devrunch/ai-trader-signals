"""
macro_events -- real macro/news context for the brief's narrative. Mocks
httpx (FRED, yfinance's own network calls go through yf.Ticker, mocked
directly) and a fake Redis client the same shape chart_layouts' own fake
model uses elsewhere in this codebase: minimal, in-memory, just enough
surface for the code under test.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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


def _rss(*items: tuple[str, str]) -> bytes:
    """A minimal Yahoo-shaped RSS body: (title, url) per item."""
    body = "".join(
        f"<item><title>{t}</title><link>{u}</link>"
        f"<description>Why it matters</description>"
        f"<pubDate>Sat, 12 Sep 2026 13:48:26 +0000</pubDate></item>"
        for t, u in items
    )
    return f"<rss version='2.0'><channel>{body}</channel></rss>".encode()


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


class _FakeClient:
    """Stands in for httpx.AsyncClient: one canned RSS body per ticker."""

    def __init__(self, by_ticker: dict, fail: set[str] | None = None):
        self.by_ticker, self.fail = by_ticker, fail or set()
        self.requested: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, headers=None):
        ticker = (params or {}).get("s")
        self.requested.append(ticker)
        if ticker in self.fail:
            raise httpx.ConnectError("rate limited")
        return _FakeResponse(self.by_ticker.get(ticker, _rss()))


class TestYahooHeadlines:
    @pytest.mark.asyncio
    async def test_real_headlines_are_deduplicated_by_url_across_tickers(self):
        shared = ("Fed signals rate path", "https://example.com/a")
        unique = ("Gold rallies", "https://example.com/b")
        client = _FakeClient({
            "GC=F": _rss(shared, unique),
            "DX-Y.NYB": _rss(shared),
            "^TNX": _rss(shared),
        })

        with patch("app.market.macro_events.httpx.AsyncClient", return_value=client):
            result = await macro_events.yfinance_headlines()

        urls = [h["url"] for h in result]
        assert urls.count("https://example.com/a") == 1  # deduplicated across tickers
        assert "https://example.com/b" in urls

    @pytest.mark.asyncio
    async def test_a_vendor_error_for_one_ticker_does_not_drop_the_others(self):
        client = _FakeClient(
            {t: _rss((f"Headline for {t}", f"https://example.com/{t}"))
             for t in macro_events._MACRO_NEWS_TICKERS},
            fail={"GC=F"},
        )

        with patch("app.market.macro_events.httpx.AsyncClient", return_value=client):
            result = await macro_events.yfinance_headlines()

        assert len(result) == len(macro_events._MACRO_NEWS_TICKERS) - 1

    @pytest.mark.asyncio
    async def test_the_description_is_carried_through_for_the_impact_analysis(self):
        client = _FakeClient({"GC=F": _rss(("Gold rallies", "https://example.com/b"))})

        with patch("app.market.macro_events.httpx.AsyncClient", return_value=client):
            result = await macro_events.yfinance_headlines(["GC=F"])

        assert result[0]["summary"] == "Why it matters"
        assert result[0]["published_at"] == "Sat, 12 Sep 2026 13:48:26 +0000"

    @pytest.mark.asyncio
    async def test_an_unparseable_body_is_skipped_not_raised(self):
        client = _FakeClient({"GC=F": b"<rss><channel><item>"})

        with patch("app.market.macro_events.httpx.AsyncClient", return_value=client):
            result = await macro_events.yfinance_headlines(["GC=F"])

        assert result == []

class TestRedditChatter:
    @pytest.mark.asyncio
    async def test_query_is_scoped_to_reddit_via_tavily(self):
        with patch("app.market.macro_events.get_settings", return_value=_settings()), \
             patch("app.signals.agent.tools.web.tavily_search", new=AsyncMock(return_value={"results": [], "count": 0})) as fake:
            await macro_events.reddit_chatter("XAUUSD sentiment")

        fake.assert_called_once_with("test-tavily-key", "site:reddit.com XAUUSD sentiment", 5)

    @pytest.mark.asyncio
    async def test_no_tavily_key_surfaces_as_an_error_not_empty_chatter(self):
        with patch("app.market.macro_events.get_settings", return_value=_settings(tavily_api_key="")):
            result = await macro_events.reddit_chatter("gold")
        assert "error" in result
