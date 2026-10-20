"""The event desk and the news feed, as tools the agent can call.

Both already existed and neither was reachable. Asked what could move gold
this week, the agent ran two web searches and answered "sources suggest a
relatively light week" — while this process held the calendar with forecast
and previous for every release, a glossary saying which way a beat points,
and a measured study of what gold did after past prints of that release.

It was guessing at what it could read exactly. These three tools close that,
and they add no new logic: the calendar, the reaction study and the scored
news feed are all here already.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from app.signals.agent.tools.base import Handler, ToolContext

logger = logging.getLogger(__name__)

# Beyond this a phone-sized answer turns into a calendar dump.
MAX_EVENTS = 12
MAX_HEADLINES = 10
# What each instrument actually reacts to. Gold and the dollar move on US
# data; a euro pair cares about both sides of the quote.
_DRIVERS = {
    "XAUUSD": ("USD",), "XAGUSD": ("USD",), "XPTUSD": ("USD",), "XPDUSD": ("USD",),
}
_MAJORS = {"EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "USD"}


def drivers_for(symbol: str) -> tuple[str, ...]:
    """Which currencies' releases move this instrument.

    A pair is driven by both its legs — EURUSD reacts to a German print and a
    US one — so the symbol is split rather than mapped.
    """
    symbol = (symbol or "").upper()
    if symbol in _DRIVERS:
        return _DRIVERS[symbol]
    if len(symbol) == 6:
        base, quote = symbol[:3], symbol[3:]
        legs = tuple(leg for leg in (base, quote) if leg in _MAJORS)
        if legs:
            return legs
    # An equity or anything unrecognised: US data still sets global risk.
    return ("USD",)


async def get_events(ctx: ToolContext, args: dict) -> Any:
    """Scheduled releases that can move this instrument, with both branches.

    `beat_is_bullish` comes from each release's own published spec, not from
    the model — 29 of the glossary's events read backwards from the obvious
    (an unemployment beat is bad news), which is exactly the kind of thing a
    plausible-sounding guess gets wrong.
    """
    from app.events.calendar import Impact, fetch_week, upcoming
    from app.events.specs import spec_for

    symbol = str(args.get("symbol") or ctx.symbol).upper()
    currencies = set(args.get("currencies") or drivers_for(symbol))
    wanted = {Impact.HIGH} if args.get("high_impact_only") else {Impact.HIGH, Impact.MEDIUM}

    events = await fetch_week()
    if events is None:
        # None, not [] — an empty calendar is a real answer and the caller
        # must not report "nothing scheduled" on the morning of a Fed day.
        return {"error": "The economic calendar feed is unavailable right now."}

    relevant = upcoming(events, currencies=currencies, impacts=wanted)
    now = datetime.now(UTC)
    rows = []
    for event in relevant[:MAX_EVENTS]:
        spec = spec_for(event.currency, event.title)
        hours = (event.when - now).total_seconds() / 3600
        row = {
            "currency": event.currency,
            "title": event.title,
            "when_utc": event.when.isoformat(),
            "in_hours": round(hours, 1),
            "impact": event.impact.value if hasattr(event.impact, "value") else str(event.impact),
            "forecast": event.forecast,
            "previous": event.previous,
        }
        if spec is not None:
            row["what_it_is"] = spec.notes or None
            # None is a real state: some releases have no published direction,
            # and inventing one is how a brief becomes confidently wrong.
            row["beat_means"] = (
                None if spec.beat_is_bullish is None
                else (f"stronger {event.currency}" if spec.beat_is_bullish
                      else f"weaker {event.currency}")
            )
        rows.append(row)

    return {
        "symbol": symbol,
        "drivers": sorted(currencies),
        "events": rows,
        "total_upcoming": len(relevant),
        "note": ("`beat_means` is the release's own published direction, for the "
                 "currency — not for this symbol. Gold moves opposite a stronger "
                 "dollar."),
    }


async def get_event_reactions(ctx: ToolContext, args: dict) -> Any:
    """What this market actually did after past prints of one release.

    Measured from minute bars, not remembered. Returns "sample too thin" by
    name when it is: a median over three prints spanning two rate regimes
    looks like evidence and is not.
    """
    from app.events import history, reaction
    from app.events.brief import GOLD

    currency = str(args.get("currency") or "USD").upper()
    title = str(args.get("title") or "").strip()
    if not title:
        return {"error": "Which release? Pass the title as the calendar gives it."}
    symbol = str(args.get("symbol") or GOLD).upper()

    rows = history.past_prints(currency, title)
    if not rows:
        return {"currency": currency, "title": title,
                "error": "No past prints of that release on file."}

    instances = reaction.instances_from_history(list(rows), limit=10)
    measured = await reaction.measure(symbol, instances)
    stats = reaction.aggregate(f"{currency}:{title}", symbol, measured)
    return {
        "currency": currency,
        "title": title,
        "symbol": symbol,
        "sample": stats.sample,
        "enough": stats.enough,
        "summary": stats.summary() or "sample too thin to characterise",
        "beats": stats.beats,
        "misses": stats.misses,
        "note": ("Moves are percent, 60 minutes after the print. A thin sample "
                 "is reported as thin rather than averaged into a number."),
    }


async def get_news(ctx: ToolContext, args: dict) -> Any:
    """The scored news feed, filtered to what moves something tradeable.

    Each headline already carries its sentiment and the instruments it
    affects — computed hourly by the news pipeline, so reading it here costs
    nothing. Preferred over web_search, which returns prose about the market
    rather than a list of what is moving in it.
    """
    from app.market import news as news_mod

    symbol = str(args.get("symbol") or "").upper()
    limit = max(1, min(int(args.get("limit") or 6), MAX_HEADLINES))

    try:
        result = await news_mod.get_market_news_result(page_size=25)
    except Exception as e:
        logger.warning("News lookup failed: %s", e)
        return {"error": f"News feed unavailable: {type(e).__name__}"}

    articles = result.get("articles") or []
    if symbol:
        # Keep anything the analysis tied to this symbol, plus anything it
        # could not analyse at all — `impacts: None` means "unread", and
        # dropping those would quietly hide stories during an LLM outage.
        def touches(a: dict) -> bool:
            impacts = a.get("impacts")
            if impacts is None:
                return True
            return any(symbol in str(i.get("symbol") or "").upper() for i in impacts)
        articles = [a for a in articles if touches(a)]

    rows = [{
        "headline": a.get("headline"),
        "source": a.get("source"),
        "published": a.get("publishedAt"),
        "sentiment": a.get("sentiment") if a.get("sentimentAvailable") else None,
        "moves": [
            {"symbol": i.get("symbol"), "direction": i.get("direction"),
             "why": i.get("reason")}
            for i in (a.get("impacts") or [])
        ] or None,
        "url": a.get("url"),
    } for a in articles[:limit]]

    return {
        "symbol": symbol or None,
        "headlines": rows,
        "matched": len(articles),
        "degraded": result.get("degraded"),
        "note": ("Free news tiers run one to twenty-four hours behind. A story "
                 "here may already be priced in."),
    }


TOOLS: dict[str, Handler] = {
    "get_events": get_events,
    "get_event_reactions": get_event_reactions,
    "get_news": get_news,
}
