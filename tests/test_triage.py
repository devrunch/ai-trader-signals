"""
The front desk.

Asked "hi", the analyst ran four rounds and spent 14,299 tokens before saying
hello. A prompt telling it not to is a suggestion; an agent that is never handed
the tool schemas cannot call a tool however it feels about the question.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.signals.agent import orchestrator
from app.signals.agent.events import EventKind, TurnRecorder
from app.signals.agent.triage import triage

from test_orchestrator import FakeBox, FakeLlm, _response


def _llm(text: str) -> FakeLlm:
    return FakeLlm(_response(content=text))


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def test_a_greeting_is_answered_without_the_analyst():
    llm = _llm("Hello — ask me anything about the chart you're looking at.")
    assert triage(llm, "RELIANCE", "NSE", "hi").handled


def test_a_market_question_is_handed_off():
    assert not triage(_llm("HANDOFF"), "RELIANCE", "NSE", "where is support?").handled


def test_the_handoff_word_is_accepted_however_the_model_dresses_it():
    for reply in ["HANDOFF", "handoff", "HANDOFF.", '"HANDOFF"', "  HANDOFF  "]:
        assert not triage(_llm(reply), "RELIANCE", "NSE", "q").handled, reply


def test_the_word_inside_a_sentence_is_an_answer_not_a_handoff():
    """The model talking ABOUT handing off is not the same as doing it — and
    silently swallowing that sentence would lose a real reply."""
    assert triage(_llm("I would normally HANDOFF that, but I can help."), "R", "NSE", "q").handled


def test_an_empty_reply_hands_off():
    assert not triage(_llm(""), "RELIANCE", "NSE", "hi").handled


def test_a_fabricated_backtest_is_rejected_not_shown():
    """Live bug: triage — which has NO tools and cannot run a backtest — was
    asked to build and test an RSI/EMA strategy, ignored its own instruction
    to hand that off, and answered with a full fake report ("Total Trades: 7,
    Win Rate: 42.86%...") that no code ever computed. Its own system prompt
    already said this needs the analyst; this is the fallback for when that
    instruction does not hold."""
    fake_report = (
        "**Backtest Results:**\n- **Total Trades**: 7\n- **Win Rate**: 42.86%\n"
        "- **Average Win**: 3.02%"
    )
    assert not triage(_llm(fake_report), "RELIANCE", "NSE", "backtest my RSI strategy").handled


def test_a_fabricated_chart_drawing_claim_is_rejected_not_shown():
    """Triage has no chart tools -- any claim that something was marked,
    drawn, or added to the chart has to be invented, the same class of
    problem a fabricated backtest report is."""
    fake_report = (
        "**Smart Money Concepts Setup Applied:**\n"
        "- **3 blocks marked** (green rectangles on chart)\n"
        "- **Bearish BOS marked** (red arrows on chart)"
    )
    assert not triage(
        _llm(fake_report), "RELIANCE", "NSE", "add a Smart Money Concepts entry setup"
    ).handled


def test_a_fabricated_indicator_removal_claim_is_rejected_not_shown():
    """Live bug: asked to "remove the SMA", triage ignored its own handoff
    instruction and answered "Done. I've removed the 9-period SMA signal
    line from your RSI indicator" -- triage has no tools (set_indicator_params/
    edit_indicator_source/remove_chart_indicator all live on the analyst),
    so nothing was actually removed, and it even named the wrong indicator.
    Same class of problem as a fabricated backtest or a fabricated chart
    drawing claim -- this is the REMOVE/CHANGE side of it, which the
    original pattern (add/draw/mark only) did not cover."""
    fake_report = "Done. I've removed the 9-period SMA signal line from your RSI indicator."
    assert not triage(_llm(fake_report), "RELIANCE", "NSE", "remove the SMA").handled


def test_a_fabricated_settings_change_claim_is_rejected_not_shown():
    fake_report = "Done. I've changed the SMA length from 20 to 50 periods."
    assert not triage(_llm(fake_report), "RELIANCE", "NSE", "change the SMA length to 50").handled


def test_triage_system_prompt_requires_handoff_for_anything_drawn_on_the_chart():
    """The handoff criteria named "indicators" generically, which a live
    request phrased as "add a Smart Money Concepts entry setup" did not
    match -- triage answered it directly with a fabricated result instead
    of handing off. The prompt must say plainly that ANYTHING drawn,
    marked, or added to the chart needs the analyst, regardless of what
    the user calls it."""
    from app.signals.agent.triage import TRIAGE_SYSTEM

    assert "adding, plotting, marking, or changing ANYTHING on the chart" in TRIAGE_SYSTEM
    assert "Smart Money Concepts" in TRIAGE_SYSTEM


def test_ordinary_small_talk_is_not_caught_by_the_fabrication_check():
    """The check is for backtest-report vocabulary specifically, not any
    number — the product-description answer triage is allowed to give
    mentions NSE/BSE and paper trading without tripping it."""
    reply = "This is an AI analyst for NSE/BSE stocks. All trading is simulated with paper money."
    assert triage(_llm(reply), "RELIANCE", "NSE", "what does this do?").handled


def test_triage_failing_hands_off_rather_than_breaking_the_turn():
    """It is an optimisation. A failing optimisation must fall through to the
    thing it was optimising."""
    class Broken:
        def chat(self, **kwargs): raise RuntimeError("bedrock is down")

    assert not triage(Broken(), "RELIANCE", "NSE", "hi").handled


def test_triage_is_never_given_tools():
    """The whole mechanism. If a schema ever reaches this call, the front desk
    can start a full analysis and the cost saving disappears."""
    llm = _llm("Hello.")
    triage(llm, "RELIANCE", "NSE", "hi")

    assert "tools" not in llm.calls[0]
    assert "tool_choice" not in llm.calls[0]


def test_only_the_last_exchange_is_considered():
    """Otherwise an earlier analysis pulls a plain thank-you back into the
    analyst, which is the case triage exists to prevent."""
    llm = _llm("You're welcome.")
    history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    triage(llm, "RELIANCE", "NSE", "thanks!", history)

    # system + at most two history turns + the message
    assert len(llm.calls[0]["messages"]) <= 4


# ---------------------------------------------------------------------------
# What the turn looks like
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_triaged_turn_never_reaches_the_analyst(trending_frame):
    llm = _llm("Hello — what would you like to know about RELIANCE?")
    box = FakeBox()

    out = await orchestrator.run_chat(
        llm, "RELIANCE", "NSE", "hi", trending_frame,
        toolbox=box, settings=get_settings(),
    )

    assert out["message"].startswith("Hello")
    assert out["stop_reason"] == "answered_by_triage"
    assert box.executed == [], "the analyst's tools must never have run"
    assert len(llm.calls) == 1, "one call, not five"
    # Cheap, not free: a turn triage handles must still reach the daily budget,
    # or it becomes a way to spend without being charged.
    assert out["usage"]["total_tokens"] > 0
    assert out["usage"]["llm_calls"] == 1


@pytest.mark.asyncio
async def test_a_triaged_turn_is_still_a_recorded_turn(trending_frame):
    """It cost money and it is part of the conversation, so it gets an id, an
    event stream and a place in the session like any other turn."""
    saved: list[dict] = []

    class Store:
        def save(self, turn): saved.append(turn)

    seen: list = []
    out = await orchestrator.run_chat(
        _llm("Hello."), "RELIANCE", "NSE", "hi", trending_frame,
        toolbox=FakeBox(), settings=get_settings(),
        recorder=TurnRecorder(seen.append), store=Store(),
    )

    assert out["turn_id"]
    assert [e.kind for e in seen][-1] == EventKind.TURN_FINISHED
    assert saved and saved[0]["turn_id"] == out["turn_id"]


@pytest.mark.asyncio
async def test_a_handoff_is_visible_in_the_turn(trending_frame):
    """The session record should say which agent answered — otherwise a short
    reply and a researched one look identical afterwards."""
    llm = FakeLlm(_response(content="HANDOFF"), _response(content="Support sits at 1380."))
    seen: list = []

    out = await orchestrator.run_chat(
        llm, "RELIANCE", "NSE", "where is support?", trending_frame,
        toolbox=FakeBox(), settings=get_settings(), recorder=TurnRecorder(seen.append),
    )

    assert out["message"] == "Support sits at 1380."
    assert EventKind.HANDOFF in [e.kind for e in seen]


@pytest.mark.asyncio
async def test_triage_can_be_switched_off(trending_frame):
    settings = get_settings().model_copy(update={"triage_enabled": False})
    llm = FakeLlm(_response(content="Straight to the analyst."))
    seen: list = []

    out = await orchestrator.run_chat(
        llm, "RELIANCE", "NSE", "hi", trending_frame,
        toolbox=FakeBox(), settings=settings, recorder=TurnRecorder(seen.append),
    )

    assert out["stop_reason"] is None
    assert EventKind.HANDOFF not in [e.kind for e in seen]
    assert "tools" in llm.calls[0], "with triage off the analyst answers directly"

