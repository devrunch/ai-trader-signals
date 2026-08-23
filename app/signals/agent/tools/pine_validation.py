"""
Shared Pine-source validation for any agent tool that writes or edits a
script before it reaches the chart -- generate_custom_indicator and
chart_indicators.edit_indicator_source both need the identical gate, since
both ultimately hand source to the same render pipeline (pine-render.ts)
and the same sandbox (run_pine_script).
"""
from __future__ import annotations

import random

from app.signals.pine.sandbox import run_pine_script

# Valid Pine, but the render pipeline (pine-render.ts) has no support for
# these two yet: bgcolor() has no renderer at all, and plotchar()/plotarrow()
# were never verified against the real boolean-plot marker path the way
# plotshape() was (confirmed live this session) -- left forbidden rather than
# assumed to work. fill() and plotshape() are NOT in this list any more: both
# were built and verified this session (real fill-band rendering, real
# plotshape()-to-marker conversion).
FORBIDDEN_CALLS = ("bgcolor(", "plotchar(", "plotarrow(")
# strategy.*() is a different, separate capability (these tools author
# indicators only) and request.security() needs a live multi-symbol/
# multi-timeframe data feed this sandbox doesn't provide -- both still
# genuinely unsupported. input.*() is NOT in this list any more: the
# settings gear (Inputs tab) is real now, and PineTS's own Indicator class
# resolves input.*() overrides for real (verified live) -- see
# ai-trader-signals/app/pine_sandbox/worker.mjs.
FORBIDDEN_NAMESPACES = ("strategy.", "request.security(")


def forbidden_call_feedback(source: str) -> str | None:
    """None if the source avoids every call these tools don't support
    rendering (or don't wire up at all), otherwise the feedback to retry
    with. A source-text check, not just a prompt rule -- the model has
    ignored the prompt rule before."""
    for call in FORBIDDEN_CALLS:
        if call in source:
            return (f"'{call}' is not supported by the chart renderer yet — "
                     f"use plot() only (a band, via '<Name> Upper'/'<Name> Lower' titles, "
                     f"is the honest way to approximate a filled/shaded look).")
    for ns in FORBIDDEN_NAMESPACES:
        if ns in source:
            return f"'{ns}' is not available to this tool."
    return None


def synthetic_bars(n: int = 80) -> list[dict]:
    """A deterministic, non-degenerate OHLCV series for the dynamic check --
    enough bars for common indicator windows (up to ~50) to settle past
    their warmup period, with real (non-flat) movement so a formula that
    divides by a range/spread doesn't accidentally pass by dividing by zero
    on a dead-flat run. openTime in milliseconds -- PineTS's own bar shape,
    not this app's usual seconds convention (see lib/api/pine.ts's own note
    on the same quirk).
    """
    rng = random.Random(7)
    bars = []
    price = 100.0
    t0 = 1767000900000
    for i in range(n):
        price += rng.uniform(-1.0, 1.2)
        open_ = price
        close = price + rng.uniform(-0.5, 0.5)
        high = max(open_, close) + rng.uniform(0.05, 0.5)
        low = min(open_, close) - rng.uniform(0.05, 0.5)
        bars.append({
            "open": round(open_, 4), "high": round(high, 4), "low": round(low, 4),
            "close": round(close, 4), "volume": rng.randint(1000, 10000),
            "openTime": t0 + i * 60_000,
        })
        price = close
    return bars


async def check_pine_source(source: str, bars: list[dict]) -> str | None:
    """None if the Pine source is fully usable, otherwise the feedback to
    retry with. Two gates, in order: does it avoid calls the render
    pipeline can't show (`forbidden_call_feedback`, free -- no subprocess),
    and -- only if that's clean -- does it actually run against real bars
    and produce at least one real plot() output (the sandbox call). The
    second gate never runs on source that already failed the first, so an
    obvious rule violation never pays for a sandboxed run.
    """
    feedback = forbidden_call_feedback(source)
    if feedback:
        return feedback
    result = await run_pine_script(source, bars, mode="indicator")
    if not result["ok"]:
        return result["error"] or "Pine execution failed"
    if not result["plots"]:
        return "The script ran but produced no plot() output — every indicator needs at least one plot() call with a distinct title."
    return None
