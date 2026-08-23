"""
The chat agent's toolbox — composition only.

This file used to be 417 lines holding seventeen tools, a dispatch table, a
concurrency helper, a chart-marker plotter and a result summariser. It is now
the seam that wires three things together:

  * `tools/`   — what each tool does, four modules by group
  * `runner.py`— what happens around every call: timing, errors, recording
  * `events.py`— the typed stream those recordings go into

Design rules, unchanged and still load-bearing:
  * The agent supplies INTENT; deterministic maths supplies numbers. No price
    level, size, or statistic in a tool result is authored by the model.
  * Tools are read-only in Phase 1. Nothing here places an order or mutates
    state — execution tools are a later phase and must be confirmation-gated.
  * Every tool result is plain JSON-serialisable data.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import get_settings
from app.market.providers.registry import market_data_router
from app.signals.agent.budget import Budget
from app.signals.agent.context import TradingContextClient
from app.signals.agent.events import TurnRecorder
from app.signals.agent.runner import ToolRunner
from app.signals.agent.tools.base import ToolContext


class AgentToolbox:
    """One turn's tools, ready to execute.

    Kept as a class because a turn has state — the frame cache, the drawings and
    the results buffer — and because the orchestrator wants one object to hand
    around rather than four.
    """

    def __init__(
        self,
        symbol: str,
        exchange: str,
        base_df: pd.DataFrame,
        user_id: str | None = None,
        context=None,
        market=market_data_router,
        settings=None,
        recorder: TurnRecorder | None = None,
        llm=None,
        budget: Budget | None = None,
        chart_indicators: list[dict] | None = None,
        chart_interval: str | None = None,
    ):
        settings = settings or get_settings()
        self.ctx = ToolContext(
            symbol=symbol,
            exchange=exchange,
            base_df=base_df,
            user_id=user_id,
            account=context or TradingContextClient(user_id, settings),
            market=market,
            settings=settings,
            llm=llm,
            # The orchestrator passes the turn's own Budget so a tool's LLM
            # spend lands in the same total as triage/loop/wrap-up. Built fresh
            # here only for callers (mostly tests) that construct a toolbox
            # standalone, so ctx.budget is never None for a tool that needs it.
            budget=budget or Budget.from_settings(settings),
            chart_indicators=chart_indicators,
            chart_interval=chart_interval,
        )
        self.runner = ToolRunner(
            self.ctx, recorder,
            max_calls_per_tool=getattr(settings, "max_calls_per_tool", 3),
        )

    # -- what the orchestrator reads at the end of a turn -------------------

    @property
    def drawings(self) -> list[dict]:
        return self.ctx.drawings

    @property
    def results(self) -> dict:
        return self.ctx.results

    @property
    def recorder(self) -> TurnRecorder:
        return self.runner.recorder

    def exhausted_tools(self) -> frozenset[str]:
        """Tools that have used their per-turn budget and should stop being offered."""
        return self.runner.exhausted()

    async def execute(self, name: str, args: dict) -> Any:
        return await self.runner.run(name, args)
