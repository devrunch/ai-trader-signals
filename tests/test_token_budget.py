"""
What a turn costs.

The tool schemas serialise to ~3,400 tokens and are resent on every round, and
every tool result joins the transcript and is resent with them. Nothing counted
any of it. These tests pin the three controls that now bound it.
"""
from __future__ import annotations

import json

import pytest

from app.config import get_settings
from app.signals.agent.offers import NEEDS_ACCOUNT, schemas_for
from app.signals.agent.schemas import TOOL_SCHEMAS
from app.signals.agent.transcript import STALE_NOTE, Transcript


def names(schemas: list[dict]) -> set[str]:
    return {s["function"]["name"] for s in schemas}


# ---------------------------------------------------------------------------
# What gets advertised
# ---------------------------------------------------------------------------

def test_the_full_schema_set_is_expensive_enough_to_be_worth_gating():
    """If this ever gets cheap, the gating below stops being worth its own module."""
    assert len(json.dumps(TOOL_SCHEMAS)) > 8_000


def test_an_authenticated_user_is_offered_everything():
    assert names(schemas_for(TOOL_SCHEMAS, user_id="u1")) == names(TOOL_SCHEMAS)


def test_account_tools_are_not_offered_without_a_user():
    """They can only return an error, and the model spends a whole round
    discovering that."""
    offered = names(schemas_for(TOOL_SCHEMAS, user_id=None))
    assert not (offered & NEEDS_ACCOUNT)
    assert "get_candles" in offered


def test_gating_on_a_missing_user_saves_real_tokens():
    """Not a fixed ratio forever — every non-account tool added since dilutes
    it further, since the gated-out set (account tools) stays the same size
    while the whole grows. 0.8 still means real savings, with headroom."""
    full = len(json.dumps(TOOL_SCHEMAS))
    gated = len(json.dumps(schemas_for(TOOL_SCHEMAS, user_id=None)))
    assert gated < full * 0.8


def test_a_tool_that_has_run_out_of_budget_stops_being_offered():
    offered = names(schemas_for(TOOL_SCHEMAS, user_id="u1", exhausted={"get_candles"}))
    assert "get_candles" not in offered
    assert "get_levels" in offered


def test_the_order_of_the_remaining_schemas_is_preserved():
    gated = schemas_for(TOOL_SCHEMAS, user_id="u1", exhausted={"get_candles"})
    original = [n for n in [s["function"]["name"] for s in TOOL_SCHEMAS] if n != "get_candles"]
    assert [s["function"]["name"] for s in gated] == original


def test_we_never_advertise_an_empty_tool_list():
    """`tool_choice="auto"` with no tools is not a meaningful request, and some
    endpoints reject it outright."""
    everything = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert schemas_for(TOOL_SCHEMAS, user_id=None, exhausted=everything) == TOOL_SCHEMAS


# ---------------------------------------------------------------------------
# The transcript ceiling
# ---------------------------------------------------------------------------

def _big_result(kb: int) -> dict:
    return {"rows": ["x" * 1000 for _ in range(kb)]}


def test_a_transcript_under_the_ceiling_is_left_alone():
    t = Transcript("system", max_total_tokens=10_000)
    t.add_tool_result("c1", {"rsi": 55})
    assert t.dropped_results == 0
    assert '"rsi": 55' in t.messages[-1]["content"]


def test_the_oldest_tool_results_are_shed_first():
    """The model has already reasoned over the old ones; the newest is the one
    it is about to use."""
    t = Transcript("system", max_tool_result_chars=10_000_000, max_total_tokens=2_000)
    for i in range(6):
        t.add_tool_result(f"c{i}", _big_result(3))

    tools = [m for m in t.messages if m["role"] == "tool"]
    assert tools[0]["content"] == STALE_NOTE
    assert tools[-1]["content"] != STALE_NOTE, "the newest result must survive"
    assert t.dropped_results > 0


def test_shedding_stops_as_soon_as_the_transcript_fits():
    t = Transcript("system", max_tool_result_chars=10_000_000, max_total_tokens=2_000)
    for i in range(6):
        t.add_tool_result(f"c{i}", _big_result(3))

    kept = [m for m in t.messages if m["role"] == "tool" and m["content"] != STALE_NOTE]
    assert kept, "shedding everything would leave the model nothing to answer from"


def test_the_question_and_the_system_prompt_are_never_shed():
    """Dropping those would change what was asked."""
    t = Transcript("system prompt here", max_tool_result_chars=10_000_000, max_total_tokens=500)
    t.add_user("where is support?")
    for i in range(5):
        t.add_tool_result(f"c{i}", _big_result(3))

    assert t.messages[0]["content"] == "system prompt here"
    assert any(m.get("content") == "where is support?" for m in t.messages)


def test_a_dropped_result_tells_the_model_not_to_re_request_it():
    t = Transcript("system", max_tool_result_chars=10_000_000, max_total_tokens=1_000)
    for i in range(5):
        t.add_tool_result(f"c{i}", _big_result(3))

    assert "do not re-request" in STALE_NOTE
    assert any(m["content"] == STALE_NOTE for m in t.messages if m["role"] == "tool")


def test_the_ceiling_is_optional():
    t = Transcript("system", max_tool_result_chars=10_000_000)
    for i in range(5):
        t.add_tool_result(f"c{i}", _big_result(3))
    assert t.dropped_results == 0


# ---------------------------------------------------------------------------
# Candles at the source
# ---------------------------------------------------------------------------

class _Market:
    def __init__(self, df): self.df = df
    async def get_historical_df(self, symbol, exchange="NSE", interval="15m", days=30):
        return self.df


def _ctx(frame):
    from app.signals.agent.tools.base import ToolContext
    return ToolContext("RELIANCE", "NSE", frame, market=_Market(frame), settings=get_settings())


@pytest.mark.asyncio
async def test_a_modest_request_is_answered_in_full(trending_frame):
    from app.signals.agent.tools.market import get_candles

    out = await get_candles(_ctx(trending_frame), {"interval": "15m", "count": 10})
    assert len(out["candles"]) == 10
    assert "range" not in out


@pytest.mark.asyncio
async def test_a_large_request_is_capped_but_still_answered(trending_frame):
    """The model asked about a longer window; returning a third of it silently
    would let it reason as though it had the lot."""
    from app.signals.agent.tools.market import MAX_CANDLES, get_candles

    out = await get_candles(_ctx(trending_frame), {"interval": "15m", "count": 200})

    assert len(out["candles"]) == MAX_CANDLES
    assert out["range"]["bars"] > MAX_CANDLES
    assert out["range"]["high"] >= max(c["high"] for c in out["candles"])
    assert "note" in out


# ---------------------------------------------------------------------------
# One round instead of two
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reading_the_chart_returns_indicators_and_levels_together(trending_frame):
    """These two were asked for separately on almost every turn, and a round is
    the expensive unit — each one resends the whole transcript and every schema."""
    from app.signals.agent.tools.market import read_chart

    out = await read_chart(_ctx(trending_frame), {})

    assert out["indicators"]
    assert out["levels"]["support_resistance"] is not None
    # The shared keys are lifted, not duplicated into both halves.
    assert "symbol" not in out["indicators"]
    assert "symbol" not in out["levels"]


@pytest.mark.asyncio
async def test_a_half_that_fails_does_not_take_the_other_half_with_it(trending_frame):
    """Levels need fewer bars than indicators do, so one half can legitimately
    fail while the other is fine."""
    from app.signals.agent.tools.market import read_chart

    short = trending_frame.tail(40)          # enough for levels, not for indicators
    out = await read_chart(_ctx(short), {})

    assert "levels" in out
    assert out["indicators_error"]


@pytest.mark.asyncio
async def test_both_halves_failing_is_reported_as_one_error(trending_frame):
    from app.signals.agent.tools.market import read_chart

    out = await read_chart(_ctx(trending_frame.tail(5)), {})
    assert "error" in out


@pytest.mark.asyncio
async def test_the_summary_describes_the_window_that_was_asked_for(trending_frame):
    from app.signals.agent.tools.market import get_candles

    out = await get_candles(_ctx(trending_frame), {"interval": "15m", "count": 120})
    r = out["range"]
    assert r["low"] <= r["close"] <= r["high"]
    assert isinstance(r["change_pct"], float)
