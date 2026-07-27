"""
The agent event stream.

One producer, three consumers — live progress, the session record, and the
strategies tab. These tests pin the contract all three depend on.
"""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.signals.agent.events import AgentEvent, EventKind, TurnRecorder, label_for


# ---------------------------------------------------------------------------
# Labels — written server-side on purpose
# ---------------------------------------------------------------------------

def test_labels_are_written_for_a_non_trader():
    assert label_for("get_candles", {"interval": "15m"}) == "Reading price history (15m)"
    assert label_for("scan_watchlist") == "Screening your watchlist"
    assert "RSI, ADX" in label_for("get_indicators", {"names": ["rsi", "adx"]})


def test_a_long_indicator_list_is_summarised_not_dumped():
    out = label_for("get_indicators", {"names": ["rsi", "adx", "atr", "macd", "ema", "vwap"]})
    assert "+2" in out


def test_an_unknown_tool_still_gets_a_readable_label():
    """A new tool must never render as a raw identifier in the UI."""
    assert label_for("some_new_tool") == "Running some new tool"


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

def test_events_are_both_collected_and_forwarded():
    """One object does both so the live feed and the stored transcript can never
    disagree about what happened."""
    seen: list[AgentEvent] = []
    rec = TurnRecorder(seen.append)
    rec.tool_started("get_candles", {"interval": "5m"})
    rec.tool_finished("get_candles", 120, "Read 200 bars")
    assert len(seen) == 2
    assert len(rec.events) == 2
    assert [e.kind for e in seen] == [EventKind.TOOL_STARTED, EventKind.TOOL_FINISHED]


def test_a_broken_listener_cannot_kill_the_turn():
    """The listener is usually a browser, which can vanish mid-turn."""
    def explode(event):
        raise RuntimeError("client went away")
    rec = TurnRecorder(explode)
    rec.tool_started("get_candles", {})          # must not raise
    assert len(rec.events) == 1


def test_no_listener_is_a_valid_state():
    rec = TurnRecorder()
    rec.emit(EventKind.THINKING, "Thinking")
    assert rec.to_list()[0]["kind"] == "thinking"


def test_events_serialise_to_plain_json():
    rec = TurnRecorder()
    rec.emit(EventKind.STRATEGY_RUN, "Backtested x", num_trades=8)
    payload = json.dumps(rec.to_list())          # must not raise
    assert "strategy_run" in payload


def test_elapsed_time_is_monotonic_and_non_negative():
    rec = TurnRecorder()
    a = rec.emit(EventKind.THINKING, "one")
    b = rec.emit(EventKind.THINKING, "two")
    assert 0 <= a.at_ms <= b.at_ms


def test_unbounded_arguments_are_trimmed_out_of_the_progress_feed():
    """A condition tree belongs in the strategy record, not in a progress line."""
    rec = TurnRecorder()
    rec.tool_started("build_strategy", {
        "entry": {"all": [{"indicator": "rsi", "op": "<", "value": 30}]},
        "name": "x" * 500,
    })
    args = rec.events[0].detail["args"]
    assert args["entry"] == "<dict>"
    assert len(args["name"]) <= 80


# ---------------------------------------------------------------------------
# The toolbox emits without being asked
# ---------------------------------------------------------------------------

class _Market:
    def __init__(self, df): self.df = df
    async def get_historical_df(self, symbol, exchange="NSE", interval="15m", days=30):
        return self.df


@pytest.mark.asyncio
async def test_every_tool_call_is_recorded(trending_frame):
    from app.config import get_settings
    from app.signals.agent.context import StaticTradingContext
    from app.signals.agent.toolbox import AgentToolbox

    seen: list[AgentEvent] = []
    rec = TurnRecorder(seen.append)
    box = AgentToolbox("RELIANCE", "NSE", trending_frame, context=StaticTradingContext({}),
                       market=_Market(trending_frame), settings=get_settings(), recorder=rec)

    await box.execute("get_levels", {})
    kinds = [e.kind for e in seen]
    assert EventKind.TOOL_STARTED in kinds
    assert EventKind.TOOL_FINISHED in kinds


@pytest.mark.asyncio
async def test_a_failing_tool_is_recorded_as_failed(trending_frame):
    from app.config import get_settings
    from app.signals.agent.context import StaticTradingContext
    from app.signals.agent.toolbox import AgentToolbox

    seen: list[AgentEvent] = []
    box = AgentToolbox("RELIANCE", "NSE", trending_frame, context=StaticTradingContext({}),
                       market=_Market(trending_frame), settings=get_settings(),
                       recorder=TurnRecorder(seen.append))

    # compare_symbols with one symbol returns an error dict rather than raising.
    await box.execute("compare_symbols", {"symbols": ["RELIANCE"]})
    assert EventKind.TOOL_FAILED in [e.kind for e in seen]


@pytest.mark.asyncio
async def test_a_backtest_summary_carries_the_trade_count(trending_frame):
    from app.config import get_settings
    from app.signals.agent.context import StaticTradingContext
    from app.signals.agent.toolbox import AgentToolbox

    seen: list[AgentEvent] = []
    box = AgentToolbox("RELIANCE", "NSE", trending_frame, context=StaticTradingContext({}),
                       market=_Market(trending_frame), settings=get_settings(),
                       recorder=TurnRecorder(seen.append))
    await box.execute("backtest_strategy", {"strategy": "ma_cross"})
    finished = [e for e in seen if e.kind == EventKind.TOOL_FINISHED]
    assert finished and "trades" in finished[-1].label


# ---------------------------------------------------------------------------
# The SSE endpoint
# ---------------------------------------------------------------------------

class _FakeService:
    """Stands in for SignalService: emits a couple of events, then answers."""
    market = None

    async def chat(self, symbol, exchange, message, history=None, user_id=None, recorder=None):
        if recorder is not None:
            recorder.emit(EventKind.TURN_STARTED, f"Analysing {symbol}")
            recorder.emit(EventKind.TOOL_STARTED, "Reading price history", tool="get_candles")
            recorder.emit(EventKind.TOOL_FINISHED, "Read 200 bars", tool="get_candles")
        return {"message": "done", "drawings": [], "results": {},
                "events": recorder.to_list() if recorder else []}


def _sse_client(service):
    from app.signals.router import get_service, router
    app = FastAPI()
    app.include_router(router, prefix="/signals")
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def _events_from(text: str) -> list[dict]:
    return [json.loads(line[len("data: "):])
            for line in text.splitlines() if line.startswith("data: ")]


def test_the_stream_emits_progress_then_a_final_result():
    with _sse_client(_FakeService()) as client:
        r = client.post("/signals/chat/stream",
                        json={"symbol": "RELIANCE", "message": "what do you see?"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

    events = _events_from(r.text)
    kinds = [e["kind"] for e in events]
    assert "turn_started" in kinds
    assert "tool_started" in kinds
    # The last event must be the full answer, so a client can ignore progress
    # entirely and still be correct.
    assert kinds[-1] == "result"
    assert events[-1]["detail"]["message"] == "done"


def test_buffering_is_disabled_or_proxies_turn_the_stream_back_into_one_blob():
    with _sse_client(_FakeService()) as client:
        r = client.post("/signals/chat/stream", json={"symbol": "RELIANCE", "message": "hi"})
    assert r.headers.get("x-accel-buffering") == "no"
    assert r.headers.get("cache-control") == "no-cache"


def test_a_quiet_turn_still_sends_something_so_proxies_do_not_close_it():
    """One LLM round can take tens of seconds with nothing to report, and idle
    proxies close a silent connection at 30–60s — inside a turn's budget."""
    import asyncio

    from app.signals import router as router_mod

    class Slow:
        async def chat(self, symbol, exchange, message, history=None, user_id=None, recorder=None):
            await asyncio.sleep(0.05)
            return {"message": "done", "drawings": [], "results": {}, "events": []}

    original = router_mod.SSE_HEARTBEAT_SECONDS
    router_mod.SSE_HEARTBEAT_SECONDS = 0.01
    try:
        with _sse_client(Slow()) as client:
            r = client.post("/signals/chat/stream", json={"symbol": "RELIANCE", "message": "hi"})
    finally:
        router_mod.SSE_HEARTBEAT_SECONDS = original

    assert ": keep-alive" in r.text
    # A comment frame is ignored by every SSE client, so the payload is unaffected.
    assert _events_from(r.text)[-1]["kind"] == "result"


def test_a_slow_reader_cannot_make_the_turn_hold_every_event_in_memory():
    """The turn must never wait on its reader: a slow client would otherwise
    slow the analysis it is watching."""
    from app.signals import router as router_mod

    class Chatty:
        async def chat(self, symbol, exchange, message, history=None, user_id=None, recorder=None):
            for i in range(50):
                recorder.emit(EventKind.THINKING, f"step {i}")
            return {"message": "done", "drawings": [], "results": {},
                    "events": recorder.to_list()}

    original = router_mod.SSE_MAX_PENDING
    router_mod.SSE_MAX_PENDING = 5
    try:
        with _sse_client(Chatty()) as client:
            r = client.post("/signals/chat/stream", json={"symbol": "RELIANCE", "message": "hi"})
    finally:
        router_mod.SSE_MAX_PENDING = original

    events = _events_from(r.text)
    assert len(events) < 50, "progress should have been shed under backpressure"
    # The answer is never shed — it does not go through the same path.
    assert events[-1]["kind"] == "result"
    assert events[-1]["detail"]["message"] == "done"


def test_a_failing_turn_streams_an_error_rather_than_dying_silently():
    class Boom:
        async def chat(self, *a, **k):
            raise RuntimeError("bedrock is down")

    with _sse_client(Boom()) as client:
        r = client.post("/signals/chat/stream", json={"symbol": "RELIANCE", "message": "hi"})

    events = _events_from(r.text)
    assert events[-1]["kind"] == "error"
    # The user-facing label must not carry the internal reason.
    assert "bedrock" not in events[-1]["label"].lower()
