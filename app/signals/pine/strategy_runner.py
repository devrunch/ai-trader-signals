"""Confirmed-bar-only Pine strategy execution.

Never feeds a forming bar to the sandbox -- process_confirmed_bar is only
ever called once a bar has actually closed (the caller's job, matching how
CandlestickChart.tsx already treats live ticks as provisional until the
period rolls over). Order translation goes through PaperTradingService's
existing placeOrder() contract -- this file computes WHAT to place, never
HOW an order is priced, risk-checked, or filled.
"""
from __future__ import annotations

from typing import Any

from app.signals.pine.sandbox import run_pine_script


class PineStrategyRunner:
    def __init__(self, strategy_id: str, source: str, symbol: str, exchange: str):
        self.strategy_id = strategy_id
        self.source = source
        self.symbol = symbol
        self.exchange = exchange
        self._bars: list[dict[str, Any]] = []
        self._seen_entry_ids: set[str] = set()

    async def process_confirmed_bar(self, bar: dict[str, Any]) -> list[dict[str, Any]]:
        self._bars.append(bar)
        result = await run_pine_script(self.source, self._bars, mode="strategy")
        if not result["ok"]:
            return []

        orders: list[dict[str, Any]] = []
        for trade in result["strategy"]["opentrades"] + result["strategy"]["closedtrades"]:
            key = f"{self.strategy_id}:{trade['entry_bar_index']}:{trade['entry_id']}"
            if key in self._seen_entry_ids:
                continue
            self._seen_entry_ids.add(key)
            orders.append({
                "symbol": self.symbol,
                "exchange": self.exchange,
                "side": "BUY" if trade["size"] > 0 else "SELL",
                "type": "MARKET",
                "quantity": abs(trade["size"]),
                "clientOrderId": key,
                "decisionTurnId": f"pine-strategy:{self.strategy_id}",
            })
        return orders
