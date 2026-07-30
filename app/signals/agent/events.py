"""
The agent's event stream — one producer, three consumers.

Everything the agent does during a turn is emitted here as a typed event:
which tool it called, with what arguments, what came back, how long it took,
and what it finally concluded. That single stream then serves three purposes
that would otherwise be three separate systems:

  1. **Live progress.** Streamed to the browser over SSE while the turn runs, so
     the user sees "Fetching 15m candles… Computing RSI, ADX… Running backtest"
     instead of a spinner that claims 15 seconds and takes 55.
  2. **The session record.** The same events, persisted, are the transcript of
     what the agent did and why — which is the memory a user means when they ask
     "what did it say about RELIANCE last week".
  3. **The strategies tab.** Filtering the stream for strategy events gives every
     run, its rules, its trades, and the reason each trade opened and closed.

Keeping this as ONE stream is the whole design. If progress, history and strategy
records were built separately they would drift, exactly as the indicator engine
and the signal evaluator did before they were unified.

The emitter is deliberately a plain callback rather than a queue or a bus: the
orchestrator is synchronous within a turn, tests can pass a list's `append`, and
a turn that nobody is watching costs one no-op call per event.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol


class EventKind(str, Enum):
    TURN_STARTED = "turn_started"
    HANDOFF = "handoff"            # triage passed the question to the analyst
    THINKING = "thinking"          # model is composing; no tool running
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TOOL_FAILED = "tool_failed"
    DRAWING = "drawing"            # something was placed on the chart
    STRATEGY_RUN = "strategy_run"  # a backtest completed — feeds the strategies tab
    MESSAGE = "message"            # the final answer
    TURN_FINISHED = "turn_finished"
    ERROR = "error"


@dataclass
class AgentEvent:
    kind: EventKind
    # Monotonic milliseconds since the turn began. Wall-clock timestamps are
    # added at persistence time; within a turn only elapsed time is meaningful,
    # and it is what the progress UI renders.
    at_ms: int
    label: str = ""                       # one human-readable line, already written
    tool: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


class Emitter(Protocol):
    def __call__(self, event: AgentEvent) -> None: ...


def _noop(event: AgentEvent) -> None:
    return None


# ---------------------------------------------------------------------------
# Human labels.
#
# Written HERE, not in the frontend. The backend knows what a tool did and with
# what arguments; the frontend would have to reimplement that mapping and would
# drift from it the first time a tool changed. It also means the same wording
# appears live, in the session transcript, and in any future notification.
# ---------------------------------------------------------------------------

_TOOL_LABELS: dict[str, str] = {
    "read_chart": "Reading the chart",
    "get_candles": "Reading price history",
    "get_indicators": "Computing indicators",
    "detect_patterns": "Scanning for candlestick patterns",
    "compare_symbols": "Comparing symbols",
    "scan_watchlist": "Screening your watchlist",
    "add_chart_indicator": "Updating the chart",
    "get_levels": "Finding support and resistance",
    "get_portfolio": "Reading your account",
    "get_positions": "Reading your open positions",
    "analyse_exposure": "Checking your concentration",
    "position_size": "Sizing the position",
    "portfolio_risk": "Measuring your open risk",
    "risk_limits": "Checking your risk limits",
    "backtest_strategy": "Backtesting a preset strategy",
    "build_strategy": "Backtesting your rules",
    "simulate_trade": "Working through the trade maths",
    "draw_on_chart": "Drawing on the chart",
    "plot_series": "Plotting on the chart",
}


def label_for(tool: str, args: dict[str, Any] | None = None) -> str:
    """A sentence a non-trader can read, specific enough to be worth showing."""
    base = _TOOL_LABELS.get(tool, f"Running {tool.replace('_', ' ')}")
    args = args or {}
    symbol = args.get("symbol")
    interval = args.get("interval")
    if tool == "get_candles" and interval:
        return f"{base} ({interval}{f' · {symbol}' if symbol else ''})"
    if tool == "get_indicators":
        names = args.get("names")
        if isinstance(names, list) and names:
            shown = ", ".join(str(n).upper() for n in names[:4])
            more = f" +{len(names) - 4}" if len(names) > 4 else ""
            return f"{base}: {shown}{more}"
    if tool == "compare_symbols":
        syms = args.get("symbols")
        if isinstance(syms, list) and syms:
            return f"{base}: {', '.join(str(s).upper() for s in syms[:4])}"
    if symbol:
        return f"{base} · {str(symbol).upper()}"
    return base


class TurnRecorder:
    """Collects a turn's events and forwards them to a live emitter.

    Holds the full list because the turn record is persisted at the end, and
    forwards each event as it happens because the browser wants it now. One
    object does both so the two can never disagree about what occurred.
    """

    def __init__(self, emit: Emitter | None = None):
        self._emit: Emitter = emit or _noop
        self._t0 = time.monotonic()
        self.events: list[AgentEvent] = []

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def emit(self, kind: EventKind, label: str = "", tool: str | None = None, **detail: Any) -> AgentEvent:
        event = AgentEvent(kind=kind, at_ms=self.elapsed_ms(), label=label, tool=tool, detail=detail)
        self.events.append(event)
        try:
            self._emit(event)
        except Exception:
            # A broken listener (a disconnected browser, most likely) must never
            # take down the turn it is watching.
            pass
        return event

    # -- convenience wrappers, so call sites read as intent -----------------

    def tool_started(self, tool: str, args: dict[str, Any]) -> None:
        self.emit(EventKind.TOOL_STARTED, label_for(tool, args), tool=tool, args=_safe_args(args))

    def tool_finished(self, tool: str, took_ms: int, summary: str = "") -> None:
        self.emit(EventKind.TOOL_FINISHED, summary or label_for(tool), tool=tool, took_ms=took_ms)

    def tool_failed(self, tool: str, reason: str) -> None:
        self.emit(EventKind.TOOL_FAILED, f"{label_for(tool)} — failed", tool=tool, reason=reason)

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]


def _safe_args(args: dict[str, Any]) -> dict[str, Any]:
    """Arguments trimmed to what is safe and useful to show.

    Condition trees and history arrays are unbounded and belong in the strategy
    record, not in a progress line. Nothing here is secret, but a progress feed
    should stay a progress feed.
    """
    out: dict[str, Any] = {}
    for key, value in list(args.items())[:8]:
        if isinstance(value, (str, int, float, bool)):
            out[key] = value if not isinstance(value, str) else value[:80]
        elif isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
            out[key] = value[:8]
        else:
            out[key] = f"<{type(value).__name__}>"
    return out
