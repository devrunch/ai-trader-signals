"""
Shared state for one turn's tools.

Every tool receives the same `ToolContext` and its own argument dict. That is
the whole interface — a tool is a plain async function, not a method on a
417-line class, so a tool module can be read, tested and changed without
loading the other three.

The context owns the two things tools legitimately share: the frame cache
(several tools want the same 15m data and must not each re-fetch it) and the
two output buffers (`drawings`, `results`) that the orchestrator hands back to
the browser.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import pandas as pd

from app.market.intervals import default_days

logger = logging.getLogger(__name__)

# A tool: given the turn's context and the model's arguments, return JSON data.
Handler = Callable[["ToolContext", dict], Awaitable[Any]]

BASE_INTERVAL = "15m"


class ToolContext:
    """Per-turn state every tool can read, and the two buffers they write to."""

    def __init__(
        self,
        symbol: str,
        exchange: str,
        base_df: pd.DataFrame,
        user_id: str | None = None,
        account=None,
        market=None,
        settings=None,
        llm=None,
        budget=None,
        chart_indicators: list[dict] | None = None,
        chart_interval: str | None = None,
    ):
        self.symbol = symbol.upper()
        self.exchange = exchange.upper()
        self.base_df = base_df
        self.user_id = user_id
        self.settings = settings
        self.account = account          # TradingContextClient — the user's book
        self.market = market            # provider router
        self.llm = llm                  # the turn's real LlmClient — for tools that write text/formulas themselves
        self.budget = budget            # the turn's Budget — record() any ctx.llm call into it, same as orchestrator.py does
        # What is actually attached to the user's chart RIGHT NOW, pushed from
        # the browser over the chart_state socket event (see
        # ai-trader-api/src/signals/signals.gateway.ts) and forwarded through
        # ChatBody.chart_state. Each entry mirrors the frontend's own
        # AttachedIndicator shape: id/source/label/pane/params/style/visibility.
        # Empty when the browser hasn't sent one yet (first load, or a client
        # too old to emit it) — tools.chart_indicators.list_chart_indicators
        # reports that honestly rather than pretending the chart is empty.
        self.chart_indicators: list[dict] = chart_indicators or []
        self.chart_interval = chart_interval

        # Written by tools, read by the orchestrator when the turn ends.
        self.drawings: list[dict] = []
        self.results: dict = {}

        # (symbol, interval) -> frame. One turn asking for the same series from
        # three tools used to mean three network round-trips inside a
        # user-facing request.
        self._frames: dict[tuple[str, str], pd.DataFrame | None] = {}

    # -- data ---------------------------------------------------------------

    async def frame(self, symbol: str | None, interval: str) -> pd.DataFrame | None:
        """OHLCV for a symbol/interval, fetched at most once per turn."""
        sym = (symbol or self.symbol).upper()
        if sym == self.symbol and interval == BASE_INTERVAL:
            return self.base_df

        key = (sym, interval)
        if key not in self._frames:
            self._frames[key] = await self.market.get_historical_df(
                sym, self.exchange, interval=interval, days=default_days(interval)
            )
        return self._frames[key]

    def resolve(self, args: dict) -> str:
        """The symbol a tool call is about — its argument, or the chart's."""
        return str(args.get("symbol") or self.symbol).upper()

    async def book(self) -> dict:
        """The user's portfolio, positions and watchlist. Memoised upstream."""
        if self.account is None:
            return {"error": "No user context available."}
        return await self.account.get()

    # -- concurrency --------------------------------------------------------

    async def gather_bounded(self, items: list[str], one: Callable[[str], Awaitable[Any]]) -> list[Any]:
        """Run `one` over `items` with bounded concurrency, never raising.

        Bounded rather than unbounded because the free market-data providers
        rate-limit aggressively, and this runs inside a chat request.
        """
        sem = asyncio.Semaphore(getattr(self.settings, "screener_concurrency", 4))

        async def guarded(item: str) -> Any:
            async with sem:
                try:
                    return await one(item)
                except Exception:
                    logger.exception("Concurrent tool step failed for %s", item)
                    return None

        return list(await asyncio.gather(*(guarded(i) for i in items)))
