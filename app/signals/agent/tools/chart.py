"""
Chart tools: what the user sees on the chart.

These are the only tools that write to `ctx.drawings` and the indicator part of
`ctx.results`. Nothing here is rendered server-side — a drawing is data, and
the browser draws it. Coordinates always come from `analysis`, never from the
model.
"""
from __future__ import annotations

from typing import Any

from app.signals import analysis
from app.signals.agent.tools.base import Handler, ToolContext

SUPPORT = "#16c784"
RESIST = "#f0525d"
TREND = "#6c5ce7"

# KLineChart's built-in indicator names. Anything outside this set would be
# accepted here and then silently fail to render.
CHART_INDICATORS = {"EMA", "MA", "SMA", "BOLL", "SAR", "VOL", "MACD", "RSI", "KDJ", "BBI"}


async def draw_on_chart(ctx: ToolContext, args: dict) -> Any:
    what = args.get("what")
    handler = _DRAWINGS.get(str(what))
    if handler is None:
        return {"error": f"Unknown drawing '{what}'. Valid: {sorted(_DRAWINGS)}"}
    return handler(ctx)


def _draw_trendline(ctx: ToolContext) -> dict:
    tl = analysis.trendline(ctx.base_df)
    if not tl:
        return {"error": "No clear trend line found"}
    ctx.drawings.append({
        "kind": "segment", "points": tl["points"],
        "color": TREND, "label": f"{tl['direction']}-trend",
    })
    return {"drawn": "trendline", "direction": tl["direction"]}


def _draw_levels(ctx: ToolContext) -> dict:
    levels = analysis.support_resistance(ctx.base_df)
    for lvl in levels:
        ctx.drawings.append({
            "kind": "priceline", "value": lvl["value"],
            "color": SUPPORT if lvl["kind"] == "support" else RESIST,
            "label": f"{lvl['kind']} ₹{lvl['value']}",
        })
    return {"drawn": "support_resistance", "levels": levels}


def _draw_fibonacci(ctx: ToolContext) -> dict:
    fib = analysis.fibonacci(ctx.base_df)
    if not fib:
        return {"error": "No clear swing for Fibonacci"}
    ctx.drawings.append({"kind": "fibonacci", "points": fib["points"]})
    return {"drawn": "fibonacci", "high": fib["high"], "low": fib["low"]}


_DRAWINGS = {
    "trendline": _draw_trendline,
    "support_resistance": _draw_levels,
    "fibonacci": _draw_fibonacci,
}


async def add_chart_indicator(ctx: ToolContext, args: dict) -> Any:
    add = [str(x).upper() for x in (args.get("add") or []) if str(x).upper() in CHART_INDICATORS]
    remove = [str(x).upper() for x in (args.get("remove") or []) if str(x).upper() in CHART_INDICATORS]
    if not add and not remove:
        return {"error": f"No valid chart indicators given. Valid: {sorted(CHART_INDICATORS)}"}

    current = ctx.results.setdefault("chart_indicators", {"add": [], "remove": []})
    current["add"] = sorted({*current["add"], *add})
    current["remove"] = sorted({*current["remove"], *remove})
    return {"chart_updated": True, "added": add, "removed": remove}


TOOLS: dict[str, Handler] = {
    "draw_on_chart": draw_on_chart,
    "add_chart_indicator": add_chart_indicator,
}
