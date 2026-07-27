"""
The tool runner — what happens around every tool call.

The rules here apply identically to all seventeen tools, which is the reason
they live in one place rather than in each tool.
"""
from __future__ import annotations

import pytest

from app.signals.agent.events import EventKind, TurnRecorder
from app.signals.agent.runner import ToolRunner, summarise
from app.signals.agent.tools.base import ToolContext


def _ctx() -> ToolContext:
    return ToolContext("RELIANCE", "NSE", base_df=None, settings=object())


def _runner(registry, recorder=None, **kw) -> ToolRunner:
    return ToolRunner(_ctx(), recorder or TurnRecorder(), registry=registry, **kw)


# ---------------------------------------------------------------------------
# Dispatch and error policy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_unknown_tool_is_an_error_not_a_crash():
    out = await _runner({}).run("no_such_tool", {})
    assert "error" in out


@pytest.mark.asyncio
async def test_bad_arguments_come_back_as_a_correctable_error():
    """The model can usually fix itself once told what went wrong."""
    async def needs_entry(ctx, args):
        return {"ok": float(args["entry"])}

    seen = []
    out = await _runner({"t": needs_entry}, TurnRecorder(seen.append)).run("t", {})
    assert "error" in out
    assert EventKind.TOOL_FAILED in [e.kind for e in seen]


@pytest.mark.asyncio
async def test_an_unexpected_exception_does_not_kill_the_turn():
    async def broken(ctx, args):
        raise RuntimeError("boom")

    out = await _runner({"t": broken}).run("t", {})
    assert "error" in out


# ---------------------------------------------------------------------------
# Memoisation — the model re-asks for the same thing across rounds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_identical_repeat_call_is_served_from_memory():
    calls = {"n": 0}

    async def counted(ctx, args):
        calls["n"] += 1
        return {"value": calls["n"]}

    runner = _runner({"t": counted})
    first = await runner.run("t", {"symbol": "RELIANCE"})
    second = await runner.run("t", {"symbol": "RELIANCE"})

    assert calls["n"] == 1, "the tool ran twice for the same question"
    assert first["value"] == second["value"] == 1


@pytest.mark.asyncio
async def test_argument_order_does_not_defeat_the_cache():
    calls = {"n": 0}

    async def counted(ctx, args):
        calls["n"] += 1
        return {"n": calls["n"]}

    runner = _runner({"t": counted})
    await runner.run("t", {"a": 1, "b": 2})
    await runner.run("t", {"b": 2, "a": 1})
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_different_arguments_are_a_different_call():
    calls = {"n": 0}

    async def counted(ctx, args):
        calls["n"] += 1
        return {"n": calls["n"]}

    runner = _runner({"t": counted})
    await runner.run("t", {"interval": "5m"})
    await runner.run("t", {"interval": "15m"})
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_a_cache_hit_is_not_reported_as_fresh_work():
    """A repeat is not progress — but it stays in the transcript, because
    'the model asked twice' is worth knowing when reading a turn back."""
    async def ok(ctx, args):
        return {"ok": True}

    seen = []
    runner = _runner({"t": ok}, TurnRecorder(seen.append))
    await runner.run("t", {})
    seen.clear()
    await runner.run("t", {})

    kinds = [e.kind for e in seen]
    assert EventKind.TOOL_STARTED not in kinds
    assert "already known" in seen[-1].label


@pytest.mark.asyncio
async def test_a_failure_is_never_cached():
    """A data fetch that failed once may succeed on the retry."""
    attempts = {"n": 0}

    async def flaky(ctx, args):
        attempts["n"] += 1
        return {"error": "upstream down"} if attempts["n"] == 1 else {"ok": True}

    runner = _runner({"t": flaky})
    assert "error" in await runner.run("t", {})
    assert (await runner.run("t", {}))["ok"] is True


@pytest.mark.asyncio
async def test_a_cached_result_cannot_be_mutated_by_its_caller():
    async def ok(ctx, args):
        return {"bars": [1, 2, 3]}

    runner = _runner({"t": ok})
    first = await runner.run("t", {})
    first["injected"] = True
    assert "injected" not in await runner.run("t", {})


# ---------------------------------------------------------------------------
# Per-turn call caps
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_tool_cannot_run_unbounded_in_one_turn():
    calls = {"n": 0}

    async def counted(ctx, args):
        calls["n"] += 1
        return {"n": calls["n"]}

    runner = _runner({"t": counted}, max_calls_per_tool=2)
    for i in range(5):
        out = await runner.run("t", {"i": i})       # distinct args defeat the cache

    assert calls["n"] == 2
    assert "error" in out


@pytest.mark.asyncio
async def test_hitting_the_cap_tells_the_model_what_to_do_instead():
    """A silently dropped call teaches the model nothing; it will just loop."""
    async def ok(ctx, args):
        return {"ok": True}

    runner = _runner({"t": ok}, max_calls_per_tool=1)
    await runner.run("t", {"i": 0})
    out = await runner.run("t", {"i": 1})
    assert "answer" in out["error"] or "different tool" in out["error"]


@pytest.mark.asyncio
async def test_the_cap_is_per_tool_not_per_turn():
    async def ok(ctx, args):
        return {"ok": True}

    runner = _runner({"a": ok, "b": ok}, max_calls_per_tool=1)
    assert "error" not in await runner.run("a", {})
    assert "error" not in await runner.run("b", {})


@pytest.mark.asyncio
async def test_cached_repeats_do_not_burn_the_budget():
    """Otherwise a model that re-reads its own answer gets punished for it."""
    async def ok(ctx, args):
        return {"ok": True}

    runner = _runner({"t": ok}, max_calls_per_tool=2)
    for _ in range(5):
        out = await runner.run("t", {})
    assert out["ok"] is True
    assert runner.calls_made("t") == 1


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def test_a_summary_reads_as_a_finding_not_a_function_name():
    assert summarise("build_strategy", {"num_trades": 8, "win_rate": 50.0}) == \
        "Backtesting your rules — 8 trades, 50.0% win rate"


def test_a_summary_never_breaks_on_an_unexpected_result_shape():
    assert summarise("get_candles", {"candles": None})
    assert summarise("scan_watchlist", {})
    assert summarise("get_levels", "not a dict")
