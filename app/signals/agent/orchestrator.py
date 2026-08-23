"""
Chat agent orchestration — three phases, not one function.

The model is given a toolbox (market data, indicators, levels, the user's
portfolio, risk/sizing maths, backtests, chart drawing) and decides for itself
what to look up before answering. All numbers come from the tools; the model
only interprets them.

  prepare  -> build the turn's state: transcript, toolbox, budgets
  loop     -> alternate model calls and tool calls until an answer or a ceiling
  finalise -> emit the closing events and hand back one result object

Each phase is separately testable, and `finalise` is the single place a turn
ends — which is where persistence will attach.
"""
from __future__ import annotations

import asyncio
import json
import logging

import pandas as pd

from app.config import get_settings
from app.llm.client import LlmClient
from app.signals import prompts
from app.signals.agent.budget import Budget
from app.signals.agent.events import EventKind, TurnRecorder
from app.signals.agent.offers import schemas_for
from app.signals.agent.schemas import TOOL_SCHEMAS
from app.signals.agent.store import TurnStore, save_quietly
from app.signals.agent.toolbox import AgentToolbox
from app.signals.agent.transcript import Transcript
from app.signals.agent.triage import triage
from app.signals.agent.turn import TurnState, new_turn_id

logger = logging.getLogger(__name__)

# Said to the model when a ceiling is hit, so a bounded turn still answers with
# what it has rather than returning nothing.
WRAP_UP = "Summarise your findings now, without calling more tools."

# Why the turn stopped, in words the user could be shown.
STOP_REASONS = {
    "rounds": "reached its research limit",
    "time": "ran out of time",
    "tokens": "reached its cost limit",
}


async def run_chat(
    llm: LlmClient,
    symbol: str,
    exchange: str,
    message: str,
    df: pd.DataFrame,
    history: list[dict] | None = None,
    user_id: str | None = None,
    toolbox: AgentToolbox | None = None,
    settings=None,
    recorder: TurnRecorder | None = None,
    store: TurnStore | None = None,
    chart_state: dict | None = None,
) -> dict:
    state = prepare(llm, symbol, exchange, message, df, history, user_id, toolbox, settings, recorder, chart_state)
    try:
        # The front desk first. It holds no tool schemas, so a greeting costs a
        # few hundred tokens instead of the ~2,400 of schemas that the analyst
        # resends on every one of its rounds.
        if getattr(state.settings, "triage_enabled", True):
            decision = await asyncio.to_thread(triage, llm, symbol, exchange, message, history)
            # Counted whichever way it went. A turn triage handles is cheap, not
            # free, and one that never reached the daily budget would be a way
            # to spend without being charged.
            if decision.response is not None:
                state.budget.record(decision.response)

            if decision.handled:
                state.final_text = decision.answer or ""
                state.stop_reason = "answered_by_triage"
                return finalise(state, store)
            state.recorder.emit(EventKind.HANDOFF, "Passing this to the analyst")

        await loop(llm, state)
    except Exception as e:
        logger.exception("Chat agent failed for %s: %s", symbol, e)
        state.failed = True
        state.stop_reason = "error"
        state.recorder.emit(EventKind.ERROR, "The analysis could not be completed",
                            reason=str(e)[:200])
    return finalise(state, store)


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def prepare(
    llm: LlmClient,
    symbol: str,
    exchange: str,
    message: str,
    df: pd.DataFrame,
    history: list[dict] | None = None,
    user_id: str | None = None,
    toolbox: AgentToolbox | None = None,
    settings=None,
    recorder: TurnRecorder | None = None,
    chart_state: dict | None = None,
) -> TurnState:
    settings = settings or get_settings()
    recorder = recorder or TurnRecorder()
    # Built before the toolbox, and handed to it, so a tool's own LLM call
    # (generate_custom_indicator writes formulas with one) records into the
    # exact same Budget the loop below is also recording into — not a second,
    # invisible one that never reaches the turn's usage total.
    budget = Budget.from_settings(settings)
    # What the browser last reported over the chart_state socket event —
    # real but possibly slightly stale (see chart_indicators.py's own note).
    # Absent entirely for a client too old to emit one; the tools handle
    # that as "nothing attached", not an error.
    chart_state = chart_state or {}
    box = toolbox or AgentToolbox(symbol, exchange, df, user_id,
                                  settings=settings, recorder=recorder,
                                  llm=llm, budget=budget,
                                  chart_indicators=chart_state.get("indicators"),
                                  chart_interval=chart_state.get("interval"))
    last_price = round(float(df["close"].iloc[-1]), 2)
    turn_id = new_turn_id()

    transcript = Transcript(
        prompts.chat_system_prompt(symbol, exchange, last_price),
        max_tool_result_chars=getattr(settings, "max_tool_result_chars", 6_000),
        # A fraction of the turn's whole token budget: the transcript is resent
        # on every round, so letting it grow to the full budget would spend the
        # budget on one round.
        max_total_tokens=getattr(settings, "chat_token_budget", 60_000) // 4,
    )
    transcript.add_history(history, settings.chat_history_turns)
    transcript.add_user(message)

    # The id rides on the first event, so a client watching the stream knows
    # which turn it is watching before the turn has finished.
    recorder.emit(EventKind.TURN_STARTED, f"Analysing {symbol}",
                  turn_id=turn_id, symbol=symbol, exchange=exchange, last_price=last_price)

    return TurnState(
        turn_id=turn_id, symbol=symbol, exchange=exchange, last_price=last_price,
        transcript=transcript, box=box, recorder=recorder,
        budget=budget, settings=settings, user_id=user_id,
    )


# ---------------------------------------------------------------------------
# loop
# ---------------------------------------------------------------------------

async def loop(llm: LlmClient, state: TurnState) -> None:
    """Alternate model calls and tool calls until an answer or a ceiling."""
    while True:
        reason = state.budget.exhausted()
        if reason:
            logger.info("Chat turn for %s stopped — %s (%s)",
                        state.symbol, reason, state.budget.to_dict())
            state.stop_reason = reason
            break

        state.budget.start_round()
        state.recorder.emit(EventKind.THINKING, "Thinking", round=state.budget.rounds_used)

        response = await asyncio.to_thread(
            llm.chat,
            temperature=0, max_tokens=900,
            messages=state.transcript.messages,
            # Not the whole set: schemas are resent every round, and a tool that
            # cannot run is not worth ~200 tokens to advertise.
            tools=schemas_for(
                TOOL_SCHEMAS,
                user_id=state.user_id,
                exhausted=_exhausted(state),
            ),
            tool_choice="auto",
        )
        state.budget.record(response)

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            state.final_text = (msg.content or "").strip()
            return

        if (msg.content or "").strip():
            state.partial_text = (msg.content or "").strip()

        state.transcript.add_tool_calls(msg.content, tool_calls)
        await _run_tools(state, tool_calls)

    if not state.final_text:
        state.final_text = await _wrap_up(llm, state)


def _exhausted(state: TurnState) -> frozenset[str]:
    """Tools that have run out of budget. Tolerant of a toolbox double that
    does not implement it — several tests supply one."""
    getter = getattr(state.box, "exhausted_tools", None)
    return getter() if callable(getter) else frozenset()


async def _run_tools(state: TurnState, tool_calls) -> None:
    for tc in tool_calls:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        result = await state.box.execute(tc.function.name, args)
        state.tools_called.append(tc.function.name)
        state.transcript.add_tool_result(tc.id, result)


async def _wrap_up(llm: LlmClient, state: TurnState) -> str:
    """One final tool-free call, so a budget-limited turn still answers.

    Not counted against the round budget — it is the turn's conclusion, not more
    research, and refusing it would trade a small cost for no answer at all.
    """
    state.transcript.add_instruction(WRAP_UP)
    response = await asyncio.to_thread(
        llm.chat, temperature=0, max_tokens=600, messages=state.transcript.messages,
    )
    state.budget.record(response)
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# finalise
# ---------------------------------------------------------------------------

def finalise(state: TurnState, store: TurnStore | None = None) -> dict:
    """Close the turn: emit what it produced, then return one result object.

    The single place a turn ends, for every caller — buffered, streamed or
    scheduled — which is why persistence attaches here rather than in whichever
    HTTP handler happened to start it.
    """
    if not state.failed:
        _emit_outputs(state)

    state.recorder.emit(EventKind.MESSAGE, "Answer ready", chars=len(state.answer))
    state.recorder.emit(EventKind.TURN_FINISHED, "Done",
                        took_ms=state.recorder.elapsed_ms(),
                        stop_reason=state.stop_reason,
                        usage=state.budget.usage.to_dict())

    result = state.to_result()
    # After the events are complete, so what is stored is the whole turn.
    save_quietly(store, result)
    return result


def _emit_outputs(state: TurnState) -> None:
    if state.box.drawings:
        state.recorder.emit(EventKind.DRAWING,
                            f"Placed {len(state.box.drawings)} marks on the chart",
                            count=len(state.box.drawings))

    # Turning an indicator on changes the user's chart just as much as drawing a
    # line does, but it lands in `results` rather than `drawings` — so it left no
    # trace at all. Reading a turn back, the chart had changed and the record did
    # not say why.
    chart = state.box.results.get("chart_indicators")
    if isinstance(chart, dict):
        added, removed = chart.get("add") or [], chart.get("remove") or []
        if added or removed:
            parts = []
            if added:
                parts.append(f"showed {', '.join(added)}")
            if removed:
                parts.append(f"hid {', '.join(removed)}")
            state.recorder.emit(EventKind.DRAWING, f"Chart: {' and '.join(parts)}",
                                add=added, remove=removed)

    # Same gap, same fix: a custom indicator changes the chart just as much as
    # a built-in one being toggled on, but it lands in `custom_indicators`
    # inside `results`, not `drawings` — so without this it left no trace either.
    custom = state.box.results.get("custom_indicators")
    if isinstance(custom, list) and custom:
        names = [str(c.get("name")) for c in custom if isinstance(c, dict)]
        labels = [str(c.get("displayLabel") or c.get("name")) for c in custom if isinstance(c, dict)]
        noun = "indicator" if len(labels) == 1 else "indicators"
        state.recorder.emit(EventKind.DRAWING, f"Chart: added custom {noun} {', '.join(labels)}",
                            names=names, labels=labels)

    strategy = state.box.results.get("strategy")
    if isinstance(strategy, dict) and "error" not in strategy:
        # The strategies tab reads this event, not the chat transcript.
        state.recorder.emit(
            EventKind.STRATEGY_RUN, f"Backtested {strategy.get('strategy') or strategy.get('name')}",
            **{k: strategy.get(k) for k in
               ("strategy", "name", "symbol", "interval", "num_trades", "win_rate",
                "total_return_pct", "stopped_out", "stop_pct", "spec", "bars", "trades")},
        )
