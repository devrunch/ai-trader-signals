import pytest

from app.signals.pine.strategy_runner import PineStrategyRunner

STRATEGY_SOURCE = """
//@version=5
strategy("t", overlay=true)
if (close > open)
    strategy.entry("long", strategy.long, qty=1)
"""

BARS = [
    {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000, "openTime": 1767000900000},
    {"open": 101, "high": 103, "low": 100, "close": 102, "volume": 1000, "openTime": 1767000960000},
]


@pytest.mark.asyncio
async def test_process_confirmed_bar_emits_a_place_order_dto_for_a_new_entry():
    runner = PineStrategyRunner(strategy_id="s1", source=STRATEGY_SOURCE, symbol="RELIANCE", exchange="NSE")
    orders = []
    for bar in BARS:
        orders += await runner.process_confirmed_bar(bar)
    assert len(orders) >= 1
    order = orders[0]
    assert order["symbol"] == "RELIANCE"
    assert order["side"] == "BUY"
    assert order["clientOrderId"]  # deterministic, non-empty


@pytest.mark.asyncio
async def test_process_confirmed_bar_is_idempotent_on_replay():
    runner = PineStrategyRunner(strategy_id="s1", source=STRATEGY_SOURCE, symbol="RELIANCE", exchange="NSE")
    first_pass = []
    for bar in BARS:
        first_pass += await runner.process_confirmed_bar(bar)

    runner2 = PineStrategyRunner(strategy_id="s1", source=STRATEGY_SOURCE, symbol="RELIANCE", exchange="NSE")
    second_pass = []
    for bar in BARS:
        second_pass += await runner2.process_confirmed_bar(bar)

    first_ids = {o["clientOrderId"] for o in first_pass}
    second_ids = {o["clientOrderId"] for o in second_pass}
    assert first_ids == second_ids  # same strategy_id + same bars -> same clientOrderIds, every time
