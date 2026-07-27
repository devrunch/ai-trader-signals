"""
`SignalService` end-to-end against fakes.

The point of this file is that it exists at all. The old `SignalService.__init__`
constructed a boto3 SQS client and an OpenAI client unconditionally, so none of
the orchestration — including the regime filter and the confidence threshold —
could be reached from a test without a live AWS account and a live LLM endpoint.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.signals.service import SignalService


class FakeLlm:
    """Returns a canned JSON body, records the prompts it was given."""

    def __init__(self, payload: dict | str):
        self.payload = payload
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        body = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        message = SimpleNamespace(content=body, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ExplodingLlm:
    def chat(self, **kwargs):
        raise RuntimeError("bedrock is down")


class RecordingPublisher:
    def __init__(self):
        self.published = []

    async def publish(self, signal):
        self.published.append(signal)


class FakeMarket:
    def __init__(self, frame):
        self.frame = frame

    async def get_historical_df(self, symbol, exchange="NSE", interval="15m", days=30):
        return self.frame


def buy_payload(entry, target, stop, confidence=0.8):
    return {"signal_type": "BUY", "confidence": confidence, "entry_price": entry,
            "target_price": target, "stop_loss": stop, "reasoning": "trend continuation"}


def make_service(frame, llm, publisher=None):
    from app.config import get_settings
    return SignalService(llm=llm, publisher=publisher or RecordingPublisher(),
                         market=FakeMarket(frame), settings=get_settings())


@pytest.mark.asyncio
async def test_insufficient_data_returns_a_reason_not_a_bare_none(trending_frame):
    svc = make_service(trending_frame.head(10), FakeLlm(buy_payload(100, 110, 96)))
    result = await svc.generate("RELIANCE")
    assert result.signal is None
    assert result.reason == "insufficient_data"


@pytest.mark.asyncio
async def test_missing_frame_is_distinguishable_from_a_quiet_market():
    svc = make_service(None, FakeLlm(buy_payload(100, 110, 96)))
    result = await svc.generate("RELIANCE")
    assert result.reason == "insufficient_data"


@pytest.mark.asyncio
async def test_llm_failure_is_reported_as_llm_error_not_as_no_signal(trending_frame, monkeypatch):
    """'0 signals' used to be indistinguishable between a quiet market and a
    dead LLM endpoint."""
    import app.signals.sentiment as sent

    async def neutral(symbol):
        return dict(sent.NEUTRAL)
    monkeypatch.setattr(sent, "symbol_sentiment", neutral)

    svc = make_service(trending_frame, ExplodingLlm())
    result = await svc.generate("RELIANCE")
    assert result.signal is None
    assert result.reason in ("llm_error", "llm_no_answer")


@pytest.mark.asyncio
async def test_a_valid_signal_is_published(trending_frame, monkeypatch):
    import app.signals.sentiment as sent

    async def neutral(symbol):
        return dict(sent.NEUTRAL)
    monkeypatch.setattr(sent, "symbol_sentiment", neutral)

    from app.signals import indicators as ind
    vals = ind.compute(trending_frame, ind.SIGNAL_SET)
    ltp, atr = vals["ltp"], vals["atr"]
    # Entry at LTP, stop 2x ATR below, target 4x ATR above -> R:R 2.0, inside
    # every gate. Direction matches the synthetic uptrend.
    payload = buy_payload(ltp, round(ltp + 4 * atr, 2), round(ltp - 2 * atr, 2))

    publisher = RecordingPublisher()
    svc = make_service(trending_frame, FakeLlm(payload), publisher)
    result = await svc.generate("RELIANCE")

    assert result.signal is not None, f"unexpected rejection: {result.reason}"
    assert result.signal.symbol == "RELIANCE"
    assert publisher.published == [result.signal]


@pytest.mark.asyncio
async def test_publish_false_keeps_it_off_the_live_feed(trending_frame, monkeypatch):
    import app.signals.sentiment as sent

    async def neutral(symbol):
        return dict(sent.NEUTRAL)
    monkeypatch.setattr(sent, "symbol_sentiment", neutral)

    from app.signals import indicators as ind
    vals = ind.compute(trending_frame, ind.SIGNAL_SET)
    payload = buy_payload(vals["ltp"], round(vals["ltp"] + 4 * vals["atr"], 2),
                          round(vals["ltp"] - 2 * vals["atr"], 2))

    publisher = RecordingPublisher()
    svc = make_service(trending_frame, FakeLlm(payload), publisher)
    result = await svc.generate("RELIANCE", publish=False)

    assert result.signal is not None
    assert publisher.published == []


@pytest.mark.asyncio
async def test_low_confidence_is_rejected_with_its_own_reason(trending_frame, monkeypatch):
    import app.signals.sentiment as sent

    async def neutral(symbol):
        return dict(sent.NEUTRAL)
    monkeypatch.setattr(sent, "symbol_sentiment", neutral)

    from app.signals import indicators as ind
    vals = ind.compute(trending_frame, ind.SIGNAL_SET)
    payload = buy_payload(vals["ltp"], round(vals["ltp"] + 4 * vals["atr"], 2),
                          round(vals["ltp"] - 2 * vals["atr"], 2), confidence=0.2)

    svc = make_service(trending_frame, FakeLlm(payload))
    result = await svc.generate("RELIANCE")
    assert result.signal is None
    assert result.reason == "low_confidence"


@pytest.mark.asyncio
async def test_a_markdown_fenced_response_is_still_parsed(trending_frame, monkeypatch):
    import app.signals.sentiment as sent

    async def neutral(symbol):
        return dict(sent.NEUTRAL)
    monkeypatch.setattr(sent, "symbol_sentiment", neutral)

    from app.signals import indicators as ind
    vals = ind.compute(trending_frame, ind.SIGNAL_SET)
    payload = buy_payload(vals["ltp"], round(vals["ltp"] + 4 * vals["atr"], 2),
                          round(vals["ltp"] - 2 * vals["atr"], 2))

    svc = make_service(trending_frame, FakeLlm("```json\n" + json.dumps(payload) + "\n```"))
    result = await svc.generate("RELIANCE")
    assert result.signal is not None, result.reason


@pytest.mark.asyncio
async def test_generate_signal_wrapper_still_returns_a_bare_signal(trending_frame):
    svc = make_service(trending_frame.head(10), FakeLlm(buy_payload(100, 110, 96)))
    assert await svc.generate_signal("RELIANCE") is None


# ---------------------------------------------------------------------------
# Forming-bar and staleness guards (checklist 04)
# ---------------------------------------------------------------------------

def test_a_still_forming_bar_is_dropped(bars):
    """Indicators read iloc[-1]. On an intraday feed that is a partially-formed
    candle whose high, low and close will all still change."""
    from app.signals.service import drop_forming_bar
    import pandas as pd

    now = pd.Timestamp.now()
    frame = bars([(1, 2, 0, 1)] * 3)
    # Last bar opened 2 minutes ago — a 15m candle is nowhere near closed.
    frame.index = pd.DatetimeIndex(
        [now - pd.Timedelta(minutes=32), now - pd.Timedelta(minutes=17), now - pd.Timedelta(minutes=2)]
    )
    assert len(drop_forming_bar(frame)) == 2


def test_a_closed_bar_is_kept(bars):
    from app.signals.service import drop_forming_bar
    import pandas as pd

    now = pd.Timestamp.now()
    frame = bars([(1, 2, 0, 1)] * 3)
    frame.index = pd.DatetimeIndex(
        [now - pd.Timedelta(minutes=50), now - pd.Timedelta(minutes=35), now - pd.Timedelta(minutes=20)]
    )
    assert len(drop_forming_bar(frame)) == 3


def test_a_non_datetime_index_drops_the_last_bar_rather_than_guessing(bars):
    """Losing one bar of history is cheap; acting on a forming one is not."""
    from app.signals.service import drop_forming_bar
    frame = bars([(1, 2, 0, 1)] * 5).reset_index(drop=True)
    assert len(drop_forming_bar(frame)) == 4


@pytest.mark.asyncio
async def test_stale_data_is_reported_as_its_own_reason(trending_frame, monkeypatch):
    """Without this the engine computes a fresh-looking signal from prices that
    stopped updating an hour ago."""
    import pandas as pd
    import app.signals.sentiment as sent

    async def neutral(symbol):
        return dict(sent.NEUTRAL)
    monkeypatch.setattr(sent, "symbol_sentiment", neutral)

    stale = trending_frame.copy()
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=6)
    stale.index = pd.date_range(end=end, periods=len(stale), freq="15min")

    svc = make_service(stale, FakeLlm(buy_payload(100, 110, 96)))
    result = await svc.generate("RELIANCE")
    assert result.reason == "stale_data"
