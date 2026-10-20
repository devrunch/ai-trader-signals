"""Rank a list of instruments by whether any of them is actually tradeable.

`scan_watchlist` already screens, but it ranks on raw indicator readings and
stops there — it never costs a trade, so "RELIANCE looks oversold" arrives
with no entry, no stop and no idea whether the reward is worth the risk.

This runs the real thing on each name: the full confluence panel both ways,
then `propose_trade` on whichever side the panel favours, then the same gates
the signal engine enforces. The output is the table a trader would build by
hand, with the rejections kept and labelled rather than hidden — "28 of 29 are
not worth taking today" is the answer most days, and a screener that only ever
shows winners is lying about the market.

Bounded hard. Each name costs a bar fetch plus 27 indicators, and this runs
inside a chat request against free data feeds.
"""
from __future__ import annotations

import logging
from typing import Any

from app.signals import confluence
from app.signals import indicators as ind_mod
from app.signals.agent.tools.base import BASE_INTERVAL, Handler, ToolContext
from app.signals.agent.tools.trade_plan import propose_trade

logger = logging.getLogger(__name__)

# A chat turn cannot wait on a hundred instruments, and the free providers
# rate-limit. Beyond this the answer arrives after the user has moved on.
MAX_SYMBOLS = 30
# Liquid, widely held, and spread across sectors — so a sector-wide move shows
# up as several names rather than being missed entirely. Used when the caller
# names nothing and has no watchlist.
NIFTY_LIQUID = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE",
    "ASIANPAINT", "MARUTI", "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "WIPRO", "NESTLEIND", "ONGC", "NTPC", "POWERGRID", "TATASTEEL",
    "JSWSTEEL", "ADANIENT", "COALINDIA", "GRASIM", "TECHM",
]


async def _one(ctx: ToolContext, symbol: str, exchange: str, interval: str,
               entry_mode: str) -> dict | None:
    """One instrument, scored and costed. None when it cannot be read."""
    frame = await ctx.frame(symbol, interval)
    if frame is None or len(frame) < 60:
        return None

    indicators = dict(ind_mod.compute(frame))
    buy = confluence.measure("BUY", indicators)
    sell = confluence.measure("SELL", indicators)
    if not buy.votes:
        return None
    # The panel picks the side, not the model. Asking for a direction it
    # already disagrees with produces a rejection that tells nobody anything.
    side = "BUY" if buy.ratio >= sell.ratio else "SELL"
    report = buy if side == "BUY" else sell

    trade = await propose_trade(ctx, {
        "symbol": symbol, "exchange": exchange, "direction": side,
        "interval": interval, "entry_mode": entry_mode,
    })
    if "error" in trade:
        return {"symbol": symbol, "error": trade["error"]}

    return {
        "symbol": symbol,
        "direction": side,
        "confluence_pct": round(report.ratio, 1),
        "families": {k: round(v) for k, v in report.families.items()},
        "verdict": trade.get("verdict"),
        "entry": trade.get("entry"),
        "stop": trade.get("stop"),
        "target": trade.get("target"),
        "reward_risk": trade.get("reward_risk"),
        "order_type": trade.get("order_type"),
        "away_pct": trade.get("distance_from_price_pct"),
        "regime": (trade.get("regime") or {}).get("kind"),
        "stretched": bool(trade.get("stretched")),
        "blocked_by": [g["gate"] for g in trade.get("gates", []) if not g["passed"]],
    }


async def screen_market(ctx: ToolContext, args: dict) -> Any:
    """Score and cost a list of instruments, best first."""
    exchange = str(args.get("exchange") or ctx.exchange).upper()
    interval = str(args.get("interval") or BASE_INTERVAL)
    entry_mode = str(args.get("entry_mode") or "level").lower()
    min_confluence = float(args.get("min_confluence") or 0)
    tradeable_only = bool(args.get("tradeable_only"))

    symbols = [str(s).upper() for s in (args.get("symbols") or []) if s]
    source = "the list you gave me"
    if not symbols:
        book = await ctx.book()
        watch = book.get("watchlist") if isinstance(book, dict) else None
        symbols = [str(i.get("symbol") if isinstance(i, dict) else i).upper()
                   for i in (watch or [])]
        symbols = [s for s in symbols if s]
        source = "your watchlist"
    if not symbols and exchange in ("NSE", "BSE"):
        symbols, source = list(NIFTY_LIQUID), "a liquid NSE default list"
    if not symbols:
        return {"error": "Nothing to screen. Name some symbols or add a watchlist."}

    over_cap = len(symbols) > MAX_SYMBOLS
    symbols = symbols[:MAX_SYMBOLS]

    rows = [r for r in await ctx.gather_bounded(
        symbols, lambda s: _one(ctx, s, exchange, interval, entry_mode)) if r]
    scored = [r for r in rows if "error" not in r]
    unreadable = [r["symbol"] for r in rows if "error" in r]

    passing = [r for r in scored if r["verdict"] == "passes_every_gate"]
    # Passing trades first, then by how strongly the panel agreed. A rejection
    # at 95% agreement is more interesting than one at 52% and should not sort
    # alphabetically into the middle.
    scored.sort(key=lambda r: (r["verdict"] != "passes_every_gate",
                               -(r["confluence_pct"] or 0)))
    shown = passing if tradeable_only else [
        r for r in scored if (r["confluence_pct"] or 0) >= min_confluence]

    return {
        "universe": source,
        "exchange": exchange,
        "interval": interval,
        "entry_mode": entry_mode,
        "screened": len(scored),
        "tradeable": len(passing),
        "results": shown,
        "unreadable": unreadable or None,
        "truncated_to": MAX_SYMBOLS if over_cap else None,
        "note": ("`verdict` is the same gate set the signal engine uses. Most names "
                 "are rejected most days — report that honestly rather than "
                 "presenting the least-bad one as a trade. `blocked_by` names which "
                 "gate failed."),
    }


TOOLS: dict[str, Handler] = {"screen_market": screen_market}
