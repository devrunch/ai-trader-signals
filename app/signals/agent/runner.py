"""
Tool execution: dispatch, timing, error policy, recording.

Split from the toolbox because they are different jobs. The tool modules under
`tools/` decide *what a tool does*; this decides *what happens around every tool
call* — how long it took, how a failure is classified, and what the user is told
while it runs. Those rules must be identical for all seventeen tools, which is
only guaranteed if they live in one place.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.signals.agent import tools as tool_registry
from app.signals.agent.events import TurnRecorder, label_for
from app.signals.agent.tools.base import ToolContext

logger = logging.getLogger(__name__)

# How many times one tool may run in a single turn. A model that asks for the
# same series a fourth time with slightly different arguments is looping, not
# researching — and every call costs a market-data round-trip inside a
# user-facing request, plus its result in the context of every later round.
MAX_CALLS_PER_TOOL = 3


class ToolRunner:
    """Runs one turn's tool calls against the registry.

    Also enforces the two limits that only make sense across a whole turn:
    a repeat of an identical call is served from memory, and a tool that has
    already run its budget is refused with a reason the model can act on.
    """

    def __init__(self, ctx: ToolContext, recorder: TurnRecorder | None = None,
                 registry: dict | None = None, max_calls_per_tool: int = MAX_CALLS_PER_TOOL):
        self.ctx = ctx
        # A turn nobody is watching still records — into a recorder whose
        # emitter is a no-op. That keeps the turn document identical whether or
        # not a browser was attached.
        self.recorder = recorder or TurnRecorder()
        self.registry = registry if registry is not None else tool_registry.REGISTRY
        self.max_calls_per_tool = max_calls_per_tool
        self._counts: dict[str, int] = {}
        self._memo: dict[tuple[str, str], Any] = {}

    def calls_made(self, name: str) -> int:
        return self._counts.get(name, 0)

    def exhausted(self) -> frozenset[str]:
        """Tools that have used their per-turn budget.

        Read by the orchestrator, which stops advertising them: offering a tool
        that can only be refused costs tokens on every remaining round and
        invites the model to spend a round finding out.
        """
        return frozenset(
            name for name, used in self._counts.items() if used >= self.max_calls_per_tool
        )

    async def run(self, name: str, args: dict) -> Any:
        handler = self.registry.get(name)
        if handler is None:
            return {"error": f"Unknown tool '{name}'"}

        key = (name, _fingerprint(args))
        if key in self._memo:
            # Not work, so not a progress line — but it stays in the transcript,
            # because "the model asked twice" is worth knowing when reading back
            # what a turn actually did.
            cached = self._memo[key]
            self.recorder.tool_finished(name, 0, f"{label_for(name, args)} — already known")
            return dict(cached) if isinstance(cached, dict) else cached

        if self._counts.get(name, 0) >= self.max_calls_per_tool:
            # Told, not silently dropped: the model can only stop looping if it
            # learns that it is looping.
            self.recorder.tool_failed(name, "per-turn call limit reached")
            return {"error": f"'{name}' has already run {self.max_calls_per_tool} times this turn. "
                             "Use what you have and answer, or call a different tool."}

        self._counts[name] = self._counts.get(name, 0) + 1
        self.recorder.tool_started(name, args)
        started = time.monotonic()

        try:
            result = await handler(self.ctx, args)
        except (KeyError, ValueError, TypeError) as e:
            # Bad or missing arguments from the model — recoverable, and the
            # model can usually correct itself once told what went wrong.
            logger.info("Tool %s called with bad arguments: %s", name, e)
            self.recorder.tool_failed(name, str(e))
            return {"error": f"{name} failed: {e}"}
        except Exception as e:
            # A genuine bug. Still return an error object rather than killing
            # the turn, but log it with a traceback — a blanket handler turns
            # real defects into friendly sentences the model narrates around,
            # and nothing ever surfaces them.
            logger.exception("Tool %s raised unexpectedly", name)
            self.recorder.tool_failed(name, str(e))
            return {"error": f"{name} failed unexpectedly: {e}"}

        took_ms = int((time.monotonic() - started) * 1000)
        if isinstance(result, dict) and "error" in result:
            # Failures are not memoised: a data fetch that failed once may well
            # succeed on the retry the model chooses to make.
            self.recorder.tool_failed(name, str(result["error"])[:200])
            return result

        self.recorder.tool_finished(name, took_ms, summarise(name, result))
        self._memo[key] = dict(result) if isinstance(result, dict) else result
        return result


def _fingerprint(args: dict) -> str:
    """A stable key for an argument dict, whatever order the model emitted it in."""
    try:
        return json.dumps(args or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Unserialisable arguments cannot be compared safely, so treat every
        # such call as unique rather than risking a wrong cache hit.
        return repr(object())


def summarise(tool: str, result: Any) -> str:
    """One line describing what came back, for the progress feed.

    Written server-side because only here is the shape of each tool result
    known; a frontend reimplementation would drift the first time a tool
    changed, and the same wording is needed live, in the stored transcript and
    in any future notification.
    """
    base = label_for(tool)
    if not isinstance(result, dict):
        return base

    formatter = _SUMMARIES.get(tool)
    if formatter is None:
        return base
    try:
        return formatter(base, result)
    except (KeyError, TypeError, ValueError):
        return base


_SUMMARIES = {
    "backtest_strategy": lambda b, r: (
        f"{b} — {r['num_trades']} trades, {r.get('win_rate')}% win rate"
        if "num_trades" in r else b
    ),
    "build_strategy": lambda b, r: (
        f"{b} — {r['num_trades']} trades, {r.get('win_rate')}% win rate"
        if "num_trades" in r else b
    ),
    "get_candles": lambda b, r: (
        f"{b} — {len(r['candles'])} bars" if r.get("candles") else b
    ),
    "scan_watchlist": lambda b, r: (
        f"{b} — {r.get('matched', 0)} of {r.get('scanned', 0)} match"
    ),
    "get_levels": lambda b, r: (
        f"{b} — {len(r['support_resistance'])} levels"
        if r.get("support_resistance") is not None else b
    ),
    "list_chart_indicators": lambda b, r: (
        f"{b} — {r['count']} attached" if "count" in r else b
    ),
    "set_indicator_params": lambda b, r: (
        f"{b} — {', '.join(f'{k}: {v}' for k, v in r['params'].items())}"
        if r.get("params") else b
    ),
    "remove_chart_indicator": lambda b, r: (
        f"{b} — {r['label']}" if r.get("label") else b
    ),
    "position_size": lambda b, r: (
        f"{b} — {r['recommended_shares']} shares" if "recommended_shares" in r else b
    ),
}
