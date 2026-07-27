"""
Market-data tools: what the chart is doing.

Read-only. Nothing here touches the user's account or places anything on the
chart — those are `account.py` and `chart.py`.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from app.signals import analysis
from app.signals import indicators as ind_mod
from app.signals.agent.tools.base import BASE_INTERVAL, Handler, ToolContext
from app.signals.patterns import detect_patterns as _detect_patterns

# Beyond this many bars the model is not reading rows, it is paying for them —
# each candle costs ~25 tokens, and every row is resent on every later round.
MAX_CANDLES = 30


async def get_candles(ctx: ToolContext, args: dict) -> Any:
    interval = args.get("interval", BASE_INTERVAL)
    asked = max(1, int(args.get("count") or 20))
    count = min(asked, MAX_CANDLES)
    df = await ctx.frame(args.get("symbol"), interval)
    if df is None or df.empty:
        return {"error": f"No {interval} data available"}

    tail = df.tail(count)
    out: dict[str, Any] = {
        "symbol": ctx.resolve(args),
        "interval": interval,
        "candles": [
            {"time": str(i), "open": round(float(r.open), 2), "high": round(float(r.high), 2),
             "low": round(float(r.low), 2), "close": round(float(r.close), 2),
             "volume": int(r.volume) if pd.notna(r.volume) else 0}
            for i, r in tail.iterrows()
        ],
    }

    if asked > count:
        # The model asked for a longer window, so answer the question it was
        # actually asking — the shape of that window — rather than silently
        # returning a third of it and letting it reason as though it had the lot.
        out["range"] = _range_summary(df.tail(asked))
        out["note"] = (
            f"Showing the most recent {count} of {asked} bars in full; `range` "
            "summarises the whole window. Use get_indicators or get_levels for "
            "anything the summary does not answer."
        )
    return out


def _range_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """What a longer window looked like, without sending every row of it."""
    close = frame["close"]
    first, last = float(close.iloc[0]), float(close.iloc[-1])
    return {
        "bars": int(len(frame)),
        "from": str(frame.index[0]),
        "to": str(frame.index[-1]),
        "high": round(float(frame["high"].max()), 2),
        "low": round(float(frame["low"].min()), 2),
        "open": round(float(frame["open"].iloc[0]), 2),
        "close": round(last, 2),
        "change_pct": round((last - first) / first * 100, 2) if first else None,
    }


async def get_indicators(ctx: ToolContext, args: dict) -> Any:
    interval = args.get("interval", BASE_INTERVAL)
    df = await ctx.frame(args.get("symbol"), interval)
    if df is None or len(df) < 50:
        return {"error": "Not enough data to compute indicators"}
    out = dict(ind_mod.compute(df, args.get("names")))
    out["interval"] = interval
    out["symbol"] = ctx.resolve(args)
    out["last_price"] = round(float(df["close"].iloc[-1]), 2)
    return out


async def detect_patterns(ctx: ToolContext, args: dict) -> Any:
    interval = args.get("interval", BASE_INTERVAL)
    lookback = max(3, min(int(args.get("lookback") or 10), 40))
    df = await ctx.frame(args.get("symbol"), interval)
    if df is None or len(df) < 20:
        return {"error": "Not enough data for pattern detection"}
    res = _detect_patterns(df, lookback)
    res["symbol"] = ctx.resolve(args)
    res["interval"] = interval
    return res


async def get_levels(ctx: ToolContext, args: dict) -> Any:
    df = await ctx.frame(args.get("symbol"), BASE_INTERVAL)
    if df is None or len(df) < 30:
        return {"error": "Not enough data to compute levels"}
    tl = analysis.trendline(df)
    fib = analysis.fibonacci(df)
    return {
        "symbol": ctx.resolve(args),
        "last_price": round(float(df["close"].iloc[-1]), 2),
        "support_resistance": analysis.support_resistance(df),
        "trend": {"direction": tl["direction"]} if tl else None,
        "fib_swing": {"high": fib["high"], "low": fib["low"]} if fib else None,
    }


async def read_chart(ctx: ToolContext, args: dict) -> Any:
    """Indicators and levels together, in one round.

    The model asked for these two separately on almost every turn, which cost a
    whole extra LLM round — and a round is the expensive unit here, since each
    one resends the entire transcript and every tool schema. They are combined
    rather than merged: both halves keep their own shape, so a caller that wants
    only one can still ask for it.
    """
    interval = args.get("interval", BASE_INTERVAL)
    indicators, levels = await asyncio.gather(
        get_indicators(ctx, {**args, "interval": interval}),
        get_levels(ctx, args),
    )

    # Either half may legitimately fail — levels need fewer bars than indicators
    # do — so a partial answer is returned rather than nothing.
    out: dict[str, Any] = {"symbol": ctx.resolve(args), "interval": interval}
    if isinstance(indicators, dict) and "error" not in indicators:
        out["indicators"] = {k: v for k, v in indicators.items()
                             if k not in ("symbol", "interval")}
    else:
        out["indicators_error"] = (indicators or {}).get("error")

    if isinstance(levels, dict) and "error" not in levels:
        out["levels"] = {k: v for k, v in levels.items() if k not in ("symbol",)}
    else:
        out["levels_error"] = (levels or {}).get("error")

    if "indicators" not in out and "levels" not in out:
        return {"error": out.get("indicators_error") or out.get("levels_error")
                or "Not enough data to read this chart"}
    return out


async def compare_symbols(ctx: ToolContext, args: dict) -> Any:
    syms = [str(s).upper() for s in (args.get("symbols") or [])][:4]
    if len(syms) < 2:
        return {"error": "Give at least two symbols to compare"}
    rows = [r for r in await snapshots(ctx, syms) if r]
    if not rows:
        return {"error": "Could not load data for those symbols"}
    return {
        "compared": len(rows),
        "symbols": rows,
        "note": "Higher ADX means a stronger trend; RSI near 50 is neutral. "
                "Ranking is descriptive, not a recommendation.",
    }


# Screen criteria, each a predicate over one snapshot row. A dict rather than a
# chain of ifs so adding a criterion is one line and the set is enumerable.
_CRITERIA = {
    "near_support":    lambda r: r.get("dist_to_support_pct") is not None and abs(r["dist_to_support_pct"]) <= 1.5,
    "near_resistance": lambda r: r.get("dist_to_resistance_pct") is not None and abs(r["dist_to_resistance_pct"]) <= 1.5,
    "trending":        lambda r: (r.get("adx") or 0) >= 20,
    "oversold":        lambda r: (r.get("rsi") or 50) <= 35,
    "overbought":      lambda r: (r.get("rsi") or 50) >= 65,
    "volume_spike":    lambda r: (r.get("volume_ratio") or 0) >= 1.5,
    "all":             lambda r: True,
}


async def scan_watchlist(ctx: ToolContext, args: dict) -> Any:
    book = await ctx.book()
    watch = book.get("watchlist") if isinstance(book, dict) else None
    if not watch:
        return {"error": "Your watchlist is empty or unavailable — add symbols in the Terminal first."}

    limit = getattr(ctx.settings, "watchlist_scan_limit", 15)
    syms = [
        str(item.get("symbol") or "" if isinstance(item, dict) else item).upper()
        for item in watch[:limit]
    ]
    syms = [s for s in syms if s]

    rows = [r for r in await snapshots(ctx, syms) if r]
    if not rows:
        return {"error": "Could not load data for your watchlist symbols"}

    criteria = str(args.get("criteria") or "all").lower()
    keep = _CRITERIA.get(criteria, _CRITERIA["all"])
    matched = sorted([r for r in rows if keep(r)], key=lambda r: -(r.get("adx") or 0))
    return {
        "criteria": criteria,
        "scanned": len(rows),
        "matched": len(matched),
        "results": matched,
        "note": "Screening is descriptive. Confirm each candidate on its own chart before acting.",
    }


# ---------------------------------------------------------------------------
# Snapshots — shared by comparison and screening
# ---------------------------------------------------------------------------

async def snapshots(ctx: ToolContext, symbols: list[str]) -> list[dict | None]:
    return await ctx.gather_bounded(symbols, lambda s: snapshot(ctx, s))


async def snapshot(ctx: ToolContext, symbol: str) -> dict | None:
    """Compact one-symbol summary: price, regime, and distance to the levels."""
    df = await ctx.frame(symbol, BASE_INTERVAL)
    if df is None or len(df) < 60:
        return None

    vals = ind_mod.compute(df, ["rsi", "adx", "atr", "ema", "supertrend", "volume"])
    last = round(float(df["close"].iloc[-1]), 2)
    levels = analysis.support_resistance(df)
    nearest_sup = max((x["value"] for x in levels if x["kind"] == "support"), default=None)
    nearest_res = min((x["value"] for x in levels if x["kind"] == "resistance"), default=None)
    ema20, ema50 = vals.get("ema20"), vals.get("ema50")
    atr = vals.get("atr")

    return {
        "symbol": symbol,
        "last_price": last,
        "rsi": vals.get("rsi"),
        "adx": vals.get("adx"),
        "atr": atr,
        "atr_pct": round(atr / last * 100, 2) if atr and last else None,
        "trend": ("up" if ema20 and ema50 and ema20 > ema50 else "down" if ema20 and ema50 else None),
        "supertrend_dir": vals.get("supertrend_dir"),
        "volume_ratio": vals.get("volume_ratio"),
        "nearest_support": nearest_sup,
        "nearest_resistance": nearest_res,
        "dist_to_support_pct": round((last - nearest_sup) / last * 100, 2) if nearest_sup else None,
        "dist_to_resistance_pct": round((nearest_res - last) / last * 100, 2) if nearest_res else None,
    }


TOOLS: dict[str, Handler] = {
    "read_chart": read_chart,
    "get_candles": get_candles,
    "get_indicators": get_indicators,
    "detect_patterns": detect_patterns,
    "get_levels": get_levels,
    "compare_symbols": compare_symbols,
    "scan_watchlist": scan_watchlist,
}
