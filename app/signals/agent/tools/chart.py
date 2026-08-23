"""
Chart tools: what the user sees on the chart.

These are the only tools that write to `ctx.drawings` and the indicator part of
`ctx.results`. Nothing here is rendered server-side — a drawing is data, and
the browser draws it. Coordinates always come from `analysis`, never from the
model.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.signals import analysis, conditions
from app.signals.agent.tools.base import Handler, ToolContext

SUPPORT = "#16c784"
RESIST = "#f0525d"
TREND = "#6c5ce7"
SERIES_LINE = "#e0ab4a"

# A line needs enough points to read as a line, unlike a handful of trade
# markers — but each point becomes its own chart overlay object on the
# frontend (see `applyDrawings`'s "series" case), so this is still a cap, not
# "plot everything ever fetched".
MAX_SERIES_POINTS = 180


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


async def plot_series(ctx: ToolContext, args: dict) -> Any:
    """Draw ANY of the validated series on the chart as a line — the general
    escape hatch for a request that does not match `draw_on_chart`'s three
    fixed shapes or a preset indicator toggle.

    "The 5-bar highest high and lowest low" is exactly `highest`/`lowest` with
    `length: 5` — two calls, two lines. Whatever the user names, if it maps to
    an entry in `conditions.SERIES` this draws it; if it does not, the error
    lists what does, so the model can retry with a real name rather than
    inventing a tool that does not exist.

    Still governed by the same safety rule as everything else here: the model
    picks a NAME and a PARAM, both validated against an allow-list before any
    computation happens. It never supplies a formula, and nothing here evals
    anything — the extension is in the allow-list (`conditions.SERIES`), not in
    what the model is trusted to do.
    """
    name = str(args.get("series") or "")
    params = args.get("params") or {}
    requested_label = str(args.get("label") or name)
    # The model can write ANY free-text label — nothing stops it from naming a
    # plain `close` line "Keltner Upper Band" when it can't actually compute
    # one. The number stays real either way, but a false name on a real number
    # is still a lie the chart would tell. The true series name is appended so
    # the label can never fully detach from what was actually plotted.
    label = requested_label if requested_label.strip().lower() == name.lower() else f"{requested_label} ({name})"

    df = await ctx.frame(args.get("symbol"), args.get("interval") or "15m")
    if df is None or df.empty:
        return {"error": "Not enough data to compute a series"}

    try:
        series = conditions.compute_series(df, name, params)
    except conditions.SpecError as e:
        return {"error": str(e), "available_series": conditions.available_series()}

    values = series.dropna()
    if values.empty:
        return {"error": f"'{name}' produced no values — not enough bars for these parameters yet"}

    points = [
        {"timestamp": int(pd.Timestamp(ts).timestamp() * 1000), "value": round(float(v), 4)}
        for ts, v in values.tail(MAX_SERIES_POINTS).items()
    ]
    ctx.drawings.append({
        "kind": "series", "points": points, "color": SERIES_LINE, "label": label,
    })
    # The model narrating "current SMA" from a SEPARATE get_indicators call risks
    # a different interval than what was just drawn — chart says one number,
    # prose says another. Handing back the point already on the line removes
    # the reason to make that second call at all.
    return {
        "drawn": name, "params": params, "points_plotted": len(points),
        "last_value": points[-1]["value"],
    }


TOOLS: dict[str, Handler] = {
    "draw_on_chart": draw_on_chart,
    "plot_series": plot_series,
}
