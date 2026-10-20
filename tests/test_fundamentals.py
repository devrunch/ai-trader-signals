"""Fundamentals off screener.in, and the desk tools the agent was missing.

The fixture is real markup, trimmed from a live screener.in company page.
A hand-written fixture would prove the parser matches my idea of their HTML,
which is not the thing that breaks.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.market import fundamentals as fund

FIXTURE = (Path(__file__).parent / "fixtures_screener_in.html").read_text(encoding="utf-8")


class TestParsing:
    def test_it_reads_the_headline_ratios_off_real_markup(self):
        data = fund.parse("RELIANCE", FIXTURE)
        assert data.ratios["Market Cap"] == "16,26,611"
        assert data.ratios["Stock P/E"] == "41.5"
        assert data.ratios["ROCE"] == "7.78"
        assert data.ratios["ROE"] == "7.71"
        assert data.ratios["Dividend Yield"] == "0.50"

    def test_it_reads_the_growth_tables(self):
        data = fund.parse("RELIANCE", FIXTURE)
        assert "Compounded Sales Growth" in data.growth
        sales = data.growth["Compounded Sales Growth"]
        assert sales["10 Years"] == "8%"
        assert sales["3 Years"] == "-2%"

    def test_a_page_with_no_ratios_is_an_error_not_an_empty_company(self):
        """Their markup will change. A tool that answers "no earnings, no
        debt, no growth" when it actually failed to read the page is worse
        than one that says it is broken."""
        with pytest.raises(fund.FundamentalsError, match="markup"):
            fund.parse("RELIANCE", "<html><body>Please log in</body></html>")

    def test_a_missing_ratio_is_absent_not_zero(self):
        data = fund.parse("RELIANCE", FIXTURE)
        assert "Debt to equity" not in data.ratios
        assert data.ratios.get("Debt to equity") is None


class TestFetching:
    @pytest.mark.asyncio
    async def test_a_cache_hit_never_touches_the_network(self):
        cached = fund.Fundamentals(symbol="TCS", ratios={"Stock P/E": "25"}).to_dict()
        with patch.object(fund, "_cached", AsyncMock(return_value=cached)), \
             patch.object(fund, "_throttled_get",
                         AsyncMock(side_effect=AssertionError("fetched on a cache hit"))):
            data = await fund.fetch("TCS")
        assert data.ratios["Stock P/E"] == "25"

    @pytest.mark.asyncio
    async def test_refresh_bypasses_the_cache(self):
        cached = fund.Fundamentals(symbol="TCS", ratios={"Stock P/E": "1"}).to_dict()
        with patch.object(fund, "_cached", AsyncMock(return_value=cached)), \
             patch.object(fund, "_store", AsyncMock()), \
             patch.object(fund, "_throttled_get", AsyncMock(return_value=FIXTURE)):
            data = await fund.fetch("TCS", refresh=True)
        assert data.ratios["Stock P/E"] == "41.5"

    @pytest.mark.asyncio
    async def test_a_broken_cache_is_slower_not_wrong(self):
        """Redis being down must not take the lookup down with it."""
        with patch.object(fund, "_client", side_effect=RuntimeError("redis gone")), \
             patch.object(fund, "_store", AsyncMock()), \
             patch.object(fund, "_throttled_get", AsyncMock(return_value=FIXTURE)):
            data = await fund.fetch("TCS")
        assert data.ratios["Stock P/E"] == "41.5"

    @pytest.mark.asyncio
    async def test_requests_are_spaced_out(self):
        """screener.in is a free service doing us a favour. Four lookups in
        one answer arriving at once is how an IP gets blocked."""
        assert fund._MIN_INTERVAL_SECONDS >= 1.0

    @pytest.mark.asyncio
    async def test_robots_disallowed_shapes_are_never_requested(self):
        """Their robots.txt disallows ?page=, ?sort=, ?limit= and ?q=. The URL
        template must not be able to grow one."""
        url = fund.BASE_URL.format(symbol="RELIANCE")
        assert url == "https://www.screener.in/company/RELIANCE/"
        for banned in ("?page=", "?sort=", "?limit=", "?q="):
            assert banned not in url


class TestTheTool:
    @staticmethod
    def _ctx(symbol="RELIANCE", exchange="NSE"):
        ctx = MagicMock()
        ctx.symbol, ctx.exchange = symbol, exchange
        return ctx

    @pytest.mark.asyncio
    async def test_it_returns_the_ratios_for_an_indian_symbol(self):
        from app.signals.agent.tools.market import get_fundamentals
        with patch.object(fund, "fetch",
                          AsyncMock(return_value=fund.parse("RELIANCE", FIXTURE))):
            out = await get_fundamentals(self._ctx(), {})
        assert out["ratios"]["Stock P/E"] == "41.5"
        assert "error" not in out

    @pytest.mark.asyncio
    async def test_forex_is_refused_with_a_reason(self):
        """Gold has no balance sheet. Returning an empty record would read as
        a company with no earnings."""
        from app.signals.agent.tools.market import get_fundamentals
        out = await get_fundamentals(self._ctx("XAUUSD", "FOREX"), {})
        assert "error" in out and "balance sheet" in out["error"]

    @pytest.mark.asyncio
    async def test_a_parse_failure_surfaces_as_an_error(self):
        from app.signals.agent.tools.market import get_fundamentals
        with patch.object(fund, "fetch",
                          AsyncMock(side_effect=fund.FundamentalsError("markup changed"))):
            out = await get_fundamentals(self._ctx(), {})
        assert out["error"] == "markup changed"
