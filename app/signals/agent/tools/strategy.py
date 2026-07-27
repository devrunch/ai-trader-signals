"""
Strategy tools: backtests and trade maths.

Two ways in. `backtest_strategy` runs a named preset; `build_strategy` runs
rules the user described in their own words, which the model turns into a
declarative SPEC — never code. `app/signals/conditions.py` validates that spec
against an allow-list before a single bar is read; a rejected spec comes back
with the exact reason so the model can correct itself instead of guessing.

Both apply costs and a stop by default. Without them the tool reported a
strategy nobody would run — a position could sit at -20% and count as one open
trade — and the agent quoted its win rate.
"""
from __future__ import annotations

from typing import Any

from app.signals import analysis
from app.signals.agent.tools.base import BASE_INTERVAL, Handler, ToolContext

DEFAULT_STOP_PCT = 2.0
MIN_BARS = 100

# A chart carrying several hundred annotations is unreadable, and an older
# marker is less likely to be what the user is looking at — so newest win.
MAX_MARKERS = 60

WIN = "#16c784"
LOSS = "#f0525d"
ENTRY = "#8b8a9e"


async def backtest_strategy(ctx: ToolContext, args: dict) -> Any:
    df = await ctx.frame(args.get("symbol"), BASE_INTERVAL)
    if df is None or len(df) < MIN_BARS:
        return {"error": "Not enough history to backtest"}

    stop_pct = args.get("stop_pct")
    res = analysis.backtest(
        df,
        args.get("strategy", "ma_cross"),
        stop_pct=DEFAULT_STOP_PCT if stop_pct is None else float(stop_pct),
        cost_pct=ctx.settings.cost_pct_round_trip,
    )
    res["symbol"] = ctx.resolve(args)
    res["caveat"] = (
        f"In-sample only. A {res['stop_pct']}% stop and "
        f"{ctx.settings.cost_pct_round_trip}% round-trip costs ARE applied, and entries "
        "fill at the next bar's open rather than the signal bar's close. "
        "Treat small trade counts as unreliable."
    )
    return _publish(ctx, res)


async def build_strategy(ctx: ToolContext, args: dict) -> Any:
    from app.signals import conditions

    interval = args.get("interval", BASE_INTERVAL)
    df = await ctx.frame(args.get("symbol"), interval)
    if df is None or len(df) < MIN_BARS:
        return {"error": "Not enough history to backtest a strategy"}

    spec = {
        "name": args.get("name") or "Custom strategy",
        "entry": args.get("entry"),
        "exit": args.get("exit"),
    }
    try:
        res = conditions.run_strategy(df, spec, cost_pct=ctx.settings.cost_pct_round_trip)
    except conditions.SpecError as e:
        # Surfaced verbatim: this is the model's feedback loop.
        return {"error": f"Invalid strategy specification: {e}",
                "available_indicators": conditions.available_series()}

    if "error" in res:
        return res

    res["symbol"] = ctx.resolve(args)
    res["interval"] = interval
    res["caveat"] = (
        f"In-sample over {res['bars']} bars on one symbol. A {res['stop_pct']}% stop and "
        f"{ctx.settings.cost_pct_round_trip}% round-trip costs are applied, and entries fill "
        "at the next bar's open. Nothing here is out-of-sample, so the live result would be "
        "worse. State the trade count when you report this — under ~30 trades the win rate "
        "is noise."
    )
    return _publish(ctx, res)


async def simulate_trade(ctx: ToolContext, args: dict) -> Any:
    side = str(args.get("side", "BUY")).upper()
    if side not in ("BUY", "SELL"):
        return {"error": "side must be BUY or SELL"}

    entry, target, stop = float(args["entry"]), float(args["target"]), float(args["stop"])
    if entry <= 0:
        return {"error": "Entry must be positive."}

    # Reject a nonsensical scenario rather than returning confident maths for a
    # trade whose target is on the losing side of the entry.
    wants_up = side == "BUY"
    if (target > entry) != wants_up or (stop < entry) != wants_up:
        return {"error": f"For a {side}, target and stop are on the wrong side of entry."}

    res = analysis.simulate_trade(side, entry, target, stop, int(args.get("quantity") or 1))
    ctx.results["simulation"] = res
    return res


# ---------------------------------------------------------------------------
# Shared tail: draw the trades, then drop them from the model's context.
# ---------------------------------------------------------------------------

def _publish(ctx: ToolContext, res: dict) -> dict:
    """Mark the trades on the chart, record the run, return a compact result.

    The user very much wants to see where the rules fired; the model does not
    need 60 trade objects in its context to say so. The full trade list stays
    in `ctx.results["strategy"]`, which is what the strategies tab reads.
    """
    trades = res.get("trades") or []
    res["markers_plotted"] = _plot_trades(ctx, trades)
    ctx.results["strategy"] = dict(res)          # keeps `trades`
    res.pop("trades", None)                      # the model's copy does not
    return res


def _plot_trades(ctx: ToolContext, trades: list[dict], limit: int = MAX_MARKERS) -> int:
    """A marker for every entry and exit a backtest produced.

    Coordinates come from the backtest, which computed them from real bars. The
    model never authors a price or a timestamp here — it only chose the rules
    that produced them.
    """
    shown = trades[-limit:]
    for t in shown:
        ctx.drawings.append({
            "kind": "trade_marker", "side": "BUY",
            "timestamp": t["entry_ts"], "value": t["entry_price"],
            "color": ENTRY, "label": f"entry {t['entry_price']}",
        })
        if t.get("exit_ts") is None:
            continue
        # Green/red by OUTCOME, not by direction — on a long-only strategy every
        # exit is a sell, so colouring by side would make every marker identical
        # and say nothing.
        win = (t.get("pnl_pct") or 0) >= 0
        ctx.drawings.append({
            "kind": "trade_marker", "side": "SELL",
            "timestamp": t["exit_ts"], "value": t["exit_price"],
            "color": WIN if win else LOSS,
            "label": f"{t.get('exit_reason', 'exit')} {t.get('pnl_pct'):+.2f}%",
        })
    return len(shown)


TOOLS: dict[str, Handler] = {
    "backtest_strategy": backtest_strategy,
    "build_strategy": build_strategy,
    "simulate_trade": simulate_trade,
}
