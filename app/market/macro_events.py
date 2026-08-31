"""
Real macro/news CONTEXT for the morning brief's narrative.

global_cues.py knows THAT NASDAQ/USD/crude moved overnight (real numbers,
pure statistics). It has never known WHY -- brief.py's own narrative used to
be limited to "NASDAQ moved +1.2%, your beta to NASDAQ is 0.25, implying
+0.3%", with no notion of a CPI print, a Fed statement, or any other real
event behind that move. This module is the "why": official US macro
releases from FRED, general market headlines from yfinance, and Reddit-
flavored chatter via Tavily (see reddit_chatter's own docstring for why
Tavily and not Reddit's own API).

Same division of labour as everywhere else in this app: every value here is
fetched, never invented. The LLM in brief.py only writes the prose
connecting these facts to the price moves global_cues.py already measured.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import httpx
import redis.asyncio as redis
import yfinance as yf

from app.config import get_settings
from app.signals.agent.tools.web import tavily_search

logger = logging.getLogger(__name__)

_VENDOR_ERRORS = (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError, OSError)

FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# FRED series most likely to move USD -- and therefore XAUUSD, which trades
# roughly inverse to the dollar and real US yields. Curated, not FRED's full
# 800,000+ series. name -> series_id.
_FRED_SERIES: dict[str, str] = {
    "CPI (all items)": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "PCE Price Index": "PCEPI",
    "Fed Funds Rate": "FEDFUNDS",
    "Unemployment Rate": "UNRATE",
    "Nonfarm Payrolls": "PAYEMS",
    "Real GDP": "GDP",
}

# How long a "last seen" marker survives in Redis with no fresh check to
# refresh it. Generous on purpose -- this only needs to outlive the longest
# real gap between two runs of the least-frequent caller (the twice-daily
# brief), not model how often each series itself actually updates.
_LAST_SEEN_TTL_SECONDS = 14 * 24 * 60 * 60


async def _fred_get(path: str, api_key: str, **params) -> dict | None:
    url = f"{FRED_BASE_URL}/{path}"
    query = {"api_key": api_key, "file_type": "json", **params}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=query)
            resp.raise_for_status()
            return resp.json()
    except _VENDOR_ERRORS as e:
        logger.warning("FRED request failed (%s): %s", path, e)
        return None


async def fred_releases() -> list[dict] | None:
    """Real US macro releases that are NEW since the last time this ran --
    release name, the actual printed value, and the prior period's value.

    "New" is tracked in Redis (one key per series, `macro:fred:{id}:last_date`),
    not FRED's own release calendar: FRED's `releases/dates` endpoint groups
    series under release IDs that would need to be looked up and verified one
    by one, and getting that mapping wrong silently drops a real release
    instead of just being redundant. Comparing each series' own latest
    observation date against what was last seen is self-correcting even if
    tuned wrong -- worst case a stale idea of "new" makes this run one cycle
    later than it could have, never wrong about what actually changed.

    FRED has no forecast/consensus field (it publishes what happened, not
    what was expected) -- this gives actual-vs-prior, not actual-vs-forecast.
    Still the real print, just without a paid vendor's "beat/miss" framing.

    None (not empty) means "couldn't check" -- no FRED_API_KEY configured, or
    every request failed -- so a caller doesn't read that as a genuinely
    quiet macro window. Empty means "checked, nothing new."
    """
    settings = get_settings()
    api_key = settings.fred_api_key
    if not api_key:
        return None

    r = redis.from_url(settings.redis_url)
    try:
        async def one(name: str, series_id: str) -> dict | None:
            obs = await _fred_get(
                "series/observations", api_key,
                series_id=series_id, sort_order="desc", limit=2,
            )
            if obs is None:
                return None
            rows = obs.get("observations", [])
            # FRED marks a not-yet-published point as "." -- both rows have
            # to be real numbers, or there's no real prior to compare against.
            if len(rows) < 2 or rows[0]["value"] == "." or rows[1]["value"] == ".":
                return None

            latest_date = rows[0]["date"]
            cache_key = f"macro:fred:{series_id}:last_date"
            last_seen_raw = await r.get(cache_key)
            last_seen = last_seen_raw.decode() if last_seen_raw else None
            await r.set(cache_key, latest_date, ex=_LAST_SEEN_TTL_SECONDS)
            if last_seen == latest_date:
                return None  # already reported this print

            return {
                "name": name,
                "series_id": series_id,
                "actual": float(rows[0]["value"]),
                "prior": float(rows[1]["value"]),
                "date": latest_date,
            }

        results = await asyncio.gather(*(one(name, sid) for name, sid in _FRED_SERIES.items()))
        return [r2 for r2 in results if r2]
    finally:
        await r.aclose()


def _fetch_yf_news(ticker: str) -> list[dict]:
    try:
        items = yf.Ticker(ticker).news or []
    except _VENDOR_ERRORS as e:
        logger.warning("yfinance news fetch failed for %s: %s", ticker, e)
        return []

    out: list[dict] = []
    for item in items[:8]:
        # yfinance has changed this payload's shape before (a flat dict, then
        # nested under "content") -- reading both rather than trusting
        # whichever shape was true when this was written.
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = content.get("title") or item.get("title")
        if not title:
            continue
        publisher = ((content.get("provider") or {}).get("displayName")
                     if isinstance(content.get("provider"), dict) else None) or item.get("publisher")
        url = ((content.get("canonicalUrl") or {}).get("url")
               if isinstance(content.get("canonicalUrl"), dict) else None) or item.get("link")
        out.append({"title": title, "publisher": publisher, "url": url})
    return out


# Tickers whose news feed leans macro/USD/gold rather than single-stock --
# yfinance attaches news to a symbol, there's no "general macro" feed to ask
# for directly. Gold futures, the US dollar index, the 10-year Treasury yield.
_MACRO_NEWS_TICKERS = ["GC=F", "DX-Y.NYB", "^TNX"]


async def yfinance_headlines() -> list[dict]:
    """Free, no key, real headlines -- no invented summary."""
    lists = await asyncio.gather(*(asyncio.to_thread(_fetch_yf_news, t) for t in _MACRO_NEWS_TICKERS))
    seen_urls: set[str] = set()
    out: list[dict] = []
    for lst in lists:
        for item in lst:
            url = item.get("url")
            if url:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
            out.append(item)
    return out


async def reddit_chatter(query: str, max_results: int = 5) -> dict:
    """Real Reddit threads, via Tavily's own crawl of reddit.com -- not
    Reddit's own API. Every free path to Reddit directly is closed as of
    2026: the official API's free tier now needs manual approval even for
    non-commercial use (its own docs call it "unsuitable for most
    production applications"), the commercial tier is $12,000/month, and
    Reddit killed the unauthenticated `.json`-suffix trick on 2026-05-30.
    Tavily -- already used for the chat agent's web_search tool -- is the
    only door still open; see brief.py's caller for how its free-credit
    budget is rationed across callers so this doesn't exhaust it.

    Returns `{"error": ...}` when Tavily isn't configured or the call
    fails -- callers must treat that as "couldn't check", not "no chatter".
    """
    settings = get_settings()
    return await tavily_search(settings.tavily_api_key, f"site:reddit.com {query}", max_results)
