"""News the calendar cannot know about.

The calendar is a schedule: CPI at 12:30, payrolls first Friday. It has
nothing to say about a tariff announcement, a strike on a shipping lane, or a
central banker saying something unplanned. Those move the same instruments and
arrive without warning, which is exactly why they are worth a push.

Selection is deterministic and free. The hourly news pipeline has already
scored every headline for sentiment and per-instrument impact, so deciding
which ones matter here is a filter over work that was already paid for — no
model call, no second fetch. That is the whole reason this is a sink on that
pipeline rather than a pipeline of its own.

Summarising is the opposite: it costs a model call, so it happens only when
someone taps the button. A feed that summarised every item would spend all day
writing paragraphs about headlines nobody opened.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import redis.asyncio as redis

from app.config import get_settings

logger = logging.getLogger(__name__)

SEEN_PREFIX = "events.shock.seen:"
OFFER_PREFIX = "events.shock.offer:"
# Long enough that a story surviving several hourly ticks is pushed once, short
# enough that a genuinely recurring theme can be pushed again next week.
SEEN_TTL_SECONDS = 5 * 24 * 3600
OFFER_TTL_SECONDS = 3 * 24 * 3600
# A phone that buzzes six times an hour gets muted, and a muted channel is
# worth nothing. The cap is per pipeline tick.
MAX_PER_RUN = 3

# What this trader is actually exposed to: gold, the dollar, and the majors.
# An impact on an Indian mid-cap is real analysis and not his problem.
WATCHED_ASSET_CLASSES = frozenset({"FOREX", "MCX"})
WATCHED_SYMBOLS = frozenset({
    "XAUUSD", "XAGUSD", "GOLD", "SILVER", "DXY", "USD",
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "WTI", "BRENT", "CL", "USOIL",
})

# Events with no scheduled time that reprice the dollar or the metal. Matched
# on whole words so "warehouse" is not a war and "strike" survives as a verb
# without matching "strikeout".
_SHOCK_WORDS = (
    r"war", r"invasion", r"invades?", r"missile", r"airstrikes?", r"ceasefire",
    r"tariffs?", r"sanctions?", r"embargo", r"export controls?",
    r"strikes?", r"shutdown", r"default", r"downgrades?", r"bailout",
    r"emergency", r"intervention", r"devalu\w+", r"opec\+?",
    r"rate cut", r"rate hike", r"unscheduled", r"resigns?", r"ousted",
    r"escalat\w+", r"retaliat\w+", r"blockade", r"coup",
)
_SHOCK_RE = re.compile(r"\b(?:" + "|".join(_SHOCK_WORDS) + r")\b", re.IGNORECASE)


@dataclass(frozen=True)
class Shock:
    article_id: str
    headline: str
    source: str
    url: str
    published_at: str
    sentiment: str
    symbols: tuple[str, ...]
    reason: str
    """Why this one was selected — "hits XAUUSD" or "tariffs". Carried so the
    message can say what it matched on instead of asserting importance."""


def _client() -> redis.Redis:
    return redis.from_url(get_settings().redis_url, decode_responses=True)


def _watched_impacts(article: dict) -> list[dict]:
    """Impacts on instruments this desk covers.

    `impacts: None` means the analysis could not run at all — different from
    an empty list, which means it ran and honestly found nothing. Neither is a
    hit, but only the first is worth a log line.
    """
    impacts = article.get("impacts")
    if impacts is None:
        return []
    out = []
    for impact in impacts:
        symbol = str(impact.get("symbol") or "").upper()
        asset_class = str(impact.get("assetClass") or "").upper()
        if asset_class in WATCHED_ASSET_CLASSES or symbol in WATCHED_SYMBOLS:
            out.append(impact)
    return out


def select(article: dict) -> Shock | None:
    """Is this headline worth waking someone for?

    Two independent ways in, because they catch different failures. The impact
    analysis catches a story whose wording is dull but whose effect is not;
    the vocabulary catches a story the analyser scored as OTHER because it is
    geopolitics rather than markets, which is most of what this exists for.
    """
    headline = str(article.get("headline") or "").strip()
    if not headline:
        return None

    hits = _watched_impacts(article)
    matched = _SHOCK_RE.search(f"{headline} {article.get('description') or ''}")
    if not hits and not matched:
        return None

    if hits:
        symbols = tuple(dict.fromkeys(
            str(i.get("symbol") or "").upper() for i in hits if i.get("symbol")))
        reason = "hits " + ", ".join(symbols[:3]) if symbols else "affects this desk"
    else:
        symbols = ()
        reason = matched.group(0).lower()

    return Shock(
        article_id=str(article.get("id") or article.get("url") or headline)[-40:],
        headline=headline,
        source=str(article.get("source") or ""),
        url=str(article.get("url") or ""),
        published_at=str(article.get("publishedAt") or ""),
        sentiment=str(article.get("sentiment") or "NEUTRAL"),
        symbols=symbols,
        reason=reason,
    )


def rank(shocks: list[Shock]) -> list[Shock]:
    """Most worth reading first, then cut to the cap.

    A story with a named instrument beats one that only matched a word: the
    first says what moves, the second says something happened. Stable within
    each tier, so the feed's own newest-first order survives.
    """
    return sorted(shocks, key=lambda s: (not s.symbols, s.sentiment == "NEUTRAL"))


async def unseen(shocks: list[Shock]) -> list[Shock]:
    """Drop the ones already pushed. The pipeline runs hourly and a story
    survives several ticks; without this the same headline arrives all day."""
    if not shocks:
        return []
    client = _client()
    try:
        fresh = []
        for shock in shocks:
            key = SEEN_PREFIX + shock.article_id
            # SET NX is the claim: two ticks racing cannot both win it.
            if await client.set(key, "1", ex=SEEN_TTL_SECONDS, nx=True):
                fresh.append(shock)
        return fresh
    finally:
        await client.aclose()


async def offer(shock: Shock) -> str:
    """Store what a [Brief me] button will summarise, and return its token."""
    token = shock.article_id[-24:].replace(":", "_")
    client = _client()
    try:
        await client.set(OFFER_PREFIX + token, json.dumps({
            "headline": shock.headline, "source": shock.source, "url": shock.url,
            "symbols": list(shock.symbols), "sentiment": shock.sentiment,
        }), ex=OFFER_TTL_SECONDS)
    finally:
        await client.aclose()
    return token


async def offered(token: str) -> dict | None:
    client = _client()
    try:
        raw = await client.get(OFFER_PREFIX + token)
        return json.loads(raw) if raw else None
    finally:
        await client.aclose()


def render(shock: Shock) -> str:
    """The push itself: the headline, where it came from, and what it hit.

    No interpretation. Interpretation costs a model call and is what the
    button is for — this message has to be worth reading without one.
    """
    lines = [f"*{shock.headline}*"]
    detail = " · ".join(x for x in (shock.source, shock.reason) if x)
    if detail:
        lines.append(f"_{detail}_")
    if shock.url:
        lines.append(shock.url)
    return "\n".join(lines)
