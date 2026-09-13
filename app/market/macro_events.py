"""
Real macro/news CONTEXT for the morning brief's narrative.

global_cues.py knows THAT NASDAQ/USD/crude moved overnight (real numbers,
pure statistics). It has never known WHY -- brief.py's own narrative used to
be limited to "NASDAQ moved +1.2%, your beta to NASDAQ is 0.25, implying
+0.3%", with no notion of a CPI print, a Fed statement, or any other real
event behind that move. This module is the "why": official US macro
releases from FRED, general market headlines from Yahoo's RSS feeds, and Reddit-
flavored chatter via Tavily (see reddit_chatter's own docstring for why
Tavily and not Reddit's own API).

Same division of labour as everywhere else in this app: every value here is
fetched, never invented. The LLM in brief.py only writes the prose
connecting these facts to the price moves global_cues.py already measured.
"""
from __future__ import annotations

import asyncio
import logging
from xml.etree import ElementTree

import httpx
import redis.asyncio as redis

from app.config import get_settings

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


# Yahoo's per-symbol RSS feed. The same headlines the yfinance package
# returns, plus a real description -- and no pandas: importing yfinance for
# this one HTTP call cost 65 MB of resident memory, measured.
_YF_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline"
# Yahoo returns an empty body to an unset/robot User-Agent.
_YF_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ai-trader/1.0)"}
# One ticker's feed is a few KB; this only bounds a pathological response.
_YF_MAX_BYTES = 2_000_000


def _rss_text(item: ElementTree.Element, tag: str) -> str:
    node = item.find(tag)
    return (node.text or "").strip() if node is not None else ""


async def _fetch_yf_news(ticker: str, client: httpx.AsyncClient) -> list[dict]:
    """Real headlines for one symbol. Never raises -- one ticker's feed
    failing must not take the others down with it."""
    try:
        resp = await client.get(
            _YF_RSS, params={"s": ticker, "region": "US", "lang": "en-US"},
            headers=_YF_HEADERS,
        )
        resp.raise_for_status()
        if len(resp.content) > _YF_MAX_BYTES:
            logger.warning("Yahoo RSS for %s was unexpectedly large -- skipping", ticker)
            return []
        root = ElementTree.fromstring(resp.content)
    except (httpx.HTTPError, ElementTree.ParseError, OSError) as e:
        logger.warning("Yahoo news fetch failed for %s: %s", ticker, e)
        return []

    out: list[dict] = []
    for item in root.iter("item"):
        title = _rss_text(item, "title")
        if not title:
            continue
        out.append({
            "title": title,
            # The feed carries no publisher field of its own; the source name
            # belongs to Yahoo's aggregation either way.
            "publisher": "Yahoo Finance",
            "url": _rss_text(item, "link"),
            # RFC 822 ("Sat, 12 Sep 2026 13:48:26 +0000"), parsed by the caller
            # in news.py the same way every other source's timestamp is.
            "published_at": _rss_text(item, "pubDate"),
            "summary": _rss_text(item, "description"),
        })
        if len(out) >= 8:
            break
    return out


# Tickers whose news feed leans macro/USD/gold rather than single-stock --
# Yahoo attaches news to a symbol, there's no "general macro" feed to ask
# for directly. Gold futures, the US dollar index, the 10-year Treasury yield.
_MACRO_NEWS_TICKERS = ["GC=F", "DX-Y.NYB", "^TNX"]


async def yfinance_headlines(tickers: list[str] | None = None) -> list[dict]:
    """Free, no key, real headlines from Yahoo's RSS feeds -- no invented summary.

    `tickers` defaults to the macro trio the brief wants; news.py passes a
    wider, asset-class-spanning set for the market feed. Yahoo attaches news
    to a symbol, so the ticker list IS the topic selection.
    """
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        lists = await asyncio.gather(
            *(_fetch_yf_news(t, client) for t in (tickers or _MACRO_NEWS_TICKERS))
        )
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
    # Imported here, not at module scope: the tool package pulls the agent
    # stack (and pandas) in with it, and the news pipeline imports this module.
    from app.signals.agent.tools.web import tavily_search

    settings = get_settings()
    return await tavily_search(settings.tavily_api_key, f"site:reddit.com {query}", max_results)
