"""Company fundamentals for Indian equities, read from screener.in.

The one thing this platform cannot compute from price. Every indicator it
carries is arithmetic over OHLCV; a P/E, a return on equity or a debt ratio
comes off a balance sheet, and no price feed in the stack carries one.

Scope is deliberately one company at a time, fetched when something asks for
it. screener.in's own screens are login-gated, and their robots.txt disallows
`?page=`, `?sort=`, `?limit=` and `?q=` — so walking a result set is out,
whatever credentials one has. `/company/<SYMBOL>/` is allowed and serves the
whole page unauthenticated, which is exactly the shape a per-symbol lookup
needs.

A parse that finds no ratios returns an error rather than an empty record.
Their markup will change eventually, and a tool that answers "no debt, no
earnings, no growth" when it actually failed to read the page is worse than
one that says it is broken.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
import time
from dataclasses import asdict, dataclass, field

import httpx
import redis.asyncio as redis

from app.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://www.screener.in/company/{symbol}/"
CACHE_PREFIX = "fundamentals:"
# A balance sheet changes once a quarter. Six hours means a conversation never
# refetches the same company and a daily reader still sees current numbers,
# while screener.in is asked once per company per session rather than per
# question.
CACHE_TTL_SECONDS = 6 * 3600
_TIMEOUT = 20.0
# Asking about four companies in one answer must not arrive as four
# simultaneous requests. Serialised with a floor between them instead.
_MIN_INTERVAL_SECONDS = 1.5
_USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

_lock = asyncio.Lock()
_last_request_at = 0.0


@dataclass
class Fundamentals:
    symbol: str
    name: str | None = None
    ratios: dict[str, str] = field(default_factory=dict)
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    growth: dict[str, dict[str, str]] = field(default_factory=dict)
    source: str = "screener.in"

    def to_dict(self) -> dict:
        return asdict(self)


class FundamentalsError(Exception):
    """The page could not be fetched or could not be read. Distinct from
    "this company has no such ratio", which is a missing key."""


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse(symbol: str, raw: str) -> Fundamentals:
    """Pull the readable numbers off a company page.

    Raises when the headline ratios are absent: that is the signal their
    markup moved, and it must not be reported as a company without earnings.
    """
    out = Fundamentals(symbol=symbol.upper())

    title = re.search(r"<title>([^<]*)</title>", raw)
    if title:
        # "Reliance Industries Ltd share price | About … - Screener"
        out.name = _text(title.group(1)).split("|")[0].replace("share price", "").strip() or None

    for label, blob in re.findall(
        r"<li[^>]*>\s*<span class=\"name\">(.*?)</span>(.*?)</li>", raw, re.S
    ):
        key = _text(label)
        if not key:
            continue
        number = re.search(r"<span class=\"number\">([^<]*)</span>", blob)
        value = _text(number.group(1)) if number else _text(blob)
        if value:
            out.ratios[key] = value

    for kind in ("pros", "cons"):
        block = re.search(r"<div class=\"%s\"[^>]*>.*?<ul>(.*?)</ul>" % kind, raw, re.S)
        if not block:
            continue
        points = [_text(li) for li in re.findall(r"<li[^>]*>(.*?)</li>", block.group(1), re.S)]
        setattr(out, kind, [p for p in points if p])

    for table in re.findall(r"<table class=\"ranges-table\">(.*?)</table>", raw, re.S):
        rows = re.findall(r"<tr>(.*?)</tr>", table, re.S)
        if not rows:
            continue
        heading = _text(rows[0])
        periods: dict[str, str] = {}
        for row in rows[1:]:
            cells = [_text(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
            if len(cells) >= 2:
                periods[cells[0].rstrip(":")] = cells[1]
        if heading and periods:
            out.growth[heading] = periods

    if not out.ratios:
        raise FundamentalsError(
            "screener.in returned a page with no readable ratios — their markup "
            "has probably changed. Reporting this rather than an empty company."
        )
    return out


def _client() -> redis.Redis:
    return redis.from_url(get_settings().redis_url, decode_responses=True)


async def _cached(symbol: str) -> dict | None:
    try:
        client = _client()
        try:
            import json
            raw = await client.get(CACHE_PREFIX + symbol.upper())
            return json.loads(raw) if raw else None
        finally:
            await client.aclose()
    except Exception:
        # A cold cache is slower, never wrong. A broken Redis must not take
        # the lookup down with it.
        logger.warning("Fundamentals cache read failed for %s", symbol, exc_info=True)
        return None


async def _store(symbol: str, payload: dict) -> None:
    try:
        client = _client()
        try:
            import json
            await client.set(CACHE_PREFIX + symbol.upper(), json.dumps(payload),
                             ex=CACHE_TTL_SECONDS)
        finally:
            await client.aclose()
    except Exception:
        logger.warning("Fundamentals cache write failed for %s", symbol, exc_info=True)


async def _throttled_get(url: str) -> str:
    """One request at a time, with a floor between them.

    screener.in is a free service doing us a favour. Four lookups in one
    answer arriving as four simultaneous requests is how an IP gets blocked.
    """
    global _last_request_at
    async with _lock:
        wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
        finally:
            _last_request_at = time.monotonic()

    if resp.status_code == 404:
        raise FundamentalsError("No such company on screener.in")
    if resp.status_code != 200:
        raise FundamentalsError(f"screener.in returned HTTP {resp.status_code}")
    return resp.text


async def fetch(symbol: str, *, refresh: bool = False) -> Fundamentals:
    """Fundamentals for one NSE/BSE symbol, from cache or from the page."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise FundamentalsError("No symbol given")

    if not refresh:
        hit = await _cached(symbol)
        if hit:
            return Fundamentals(**hit)

    raw = await _throttled_get(BASE_URL.format(symbol=symbol))
    parsed = parse(symbol, raw)
    await _store(symbol, parsed.to_dict())
    logger.info("Fundamentals fetched for %s (%d ratios)", symbol, len(parsed.ratios))
    return parsed
