"""
Reactive chart-indicator control -- the agent reads and changes what's
actually attached to the user's chart, only when asked in the same turn.

Replaces the old add_chart_indicator/CHART_INDICATORS tool (see chart.py),
which toggled names from a fixed klinecharts-era catalog {EMA, MA, SMA,
BOLL, ...} that stopped existing once the frontend migrated to the Pine
model -- confirmed live: its result landed in ctx.results["chart_indicators"],
which the frontend's applyChartIndicators has read and silently discarded
("Intentionally does nothing") since that migration. These tools operate on
the real, current model: an indicator is Pine source plus id/label/pane and
optional params/style/visibility, exactly what the settings gear itself
edits.

`ctx.chart_indicators` is what the browser last reported over the
chart_state socket event (see ai-trader-api's signals.gateway.ts) --
real-but-possibly-stale data, refreshed on every indicator change and on
connect. Mutations here never touch that list directly (this process holds
no chart state of its own); they append an instruction to
ctx.results["indicator_changes"], which the frontend applies through the
exact same ChartAdapter methods the settings gear itself calls
(setIndicatorPlotStyle, reattach with new params, etc.) -- one mutation
path, whether a human or the agent drives it.
"""
from __future__ import annotations

from typing import Any

from app.signals.agent.tools.base import Handler, ToolContext
from app.signals.agent.tools.pine_validation import check_pine_source, synthetic_bars
from app.signals.pine.sandbox import run_pine_script


def _find(ctx: ToolContext, indicator_id: str) -> dict | None:
    for ind in ctx.chart_indicators:
        if ind.get("id") == indicator_id:
            return ind
    return None


async def list_chart_indicators(ctx: ToolContext, args: dict) -> Any:
    """What's actually attached right now -- real data pushed from the
    browser, not a guess. Empty (not an error) when nothing is attached, or
    when the browser hasn't sent a chart_state event yet (a client too old
    to emit one, or the very first message of a session)."""
    return {
        "indicators": ctx.chart_indicators,
        "interval": ctx.chart_interval,
        "count": len(ctx.chart_indicators),
    }


async def set_indicator_params(ctx: ToolContext, args: dict) -> Any:
    """Override one or more of an attached indicator's own input.*()
    values, keyed by varId (the script's own variable name, e.g. `length`
    for `length = input.int(100, ...)`) -- get the real varIds from
    list_chart_indicators first; titles shown in the settings gear are a
    hint, not the key this takes.

    Verified against the real sandbox before being accepted: a wrong varId
    or an out-of-range value is caught here, with PineTS's own real error,
    rather than silently failing once it reaches the browser.
    """
    indicator_id = str(args.get("id") or "")
    params = args.get("params") or {}
    if not indicator_id or not isinstance(params, dict) or not params:
        return {"error": "Both 'id' and a non-empty 'params' object are required."}

    target = _find(ctx, indicator_id)
    if target is None:
        return {"error": f"No indicator with id '{indicator_id}' is currently attached. "
                          f"Call list_chart_indicators to see what's actually there."}

    source = str(target.get("source") or "")
    result = await run_pine_script(source, synthetic_bars(), mode="indicator", input_overrides=params)
    if not result["ok"]:
        return {"error": f"These settings don't apply cleanly: {result['error']}"}

    ctx.results.setdefault("indicator_changes", {}).setdefault("update", []).append(
        {"id": indicator_id, "params": params},
    )
    return {"updated": indicator_id, "params": params}


async def edit_indicator_source(ctx: ToolContext, args: dict) -> Any:
    """Replace an attached indicator's ENTIRE Pine source in place -- for a
    real logic change (not a settings tweak; that's set_indicator_params).
    Get the current source from list_chart_indicators first so the edit is
    a real diff, not a guess at what's already there. Validated against the
    real sandbox before being accepted, same as generate_custom_indicator.
    """
    indicator_id = str(args.get("id") or "")
    source = str(args.get("source") or "").strip()
    if not indicator_id or not source:
        return {"error": "Both 'id' and 'source' are required."}

    target = _find(ctx, indicator_id)
    if target is None:
        return {"error": f"No indicator with id '{indicator_id}' is currently attached. "
                          f"Call list_chart_indicators to see what's actually there."}

    feedback = await check_pine_source(source, synthetic_bars())
    if feedback:
        return {"error": f"That edit doesn't validate: {feedback}"}

    ctx.results.setdefault("indicator_changes", {}).setdefault("edit_source", []).append(
        {"id": indicator_id, "source": source},
    )
    return {"edited": indicator_id}


async def remove_chart_indicator(ctx: ToolContext, args: dict) -> Any:
    indicator_id = str(args.get("id") or "")
    if not indicator_id:
        return {"error": "'id' is required."}
    target = _find(ctx, indicator_id)
    if target is None:
        return {"error": f"No indicator with id '{indicator_id}' is currently attached."}

    ctx.results.setdefault("indicator_changes", {}).setdefault("remove", []).append(indicator_id)
    return {"removed": indicator_id, "label": target.get("label")}


TOOLS: dict[str, Handler] = {
    "list_chart_indicators": list_chart_indicators,
    "set_indicator_params": set_indicator_params,
    "edit_indicator_source": edit_indicator_source,
    "remove_chart_indicator": remove_chart_indicator,
}
