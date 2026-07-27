"""
Contract test for `POST /signals/evaluate`.

This endpoint exists so the NestJS performance view stops running its own
divergent copy of the evaluator. The request and response field names below are
the contract that `ai-trader-api/src/signals/signals-upstream.client.ts` and
`eval.ts` are written against — if they change here, that code breaks silently
(unknown keys are dropped, missing ones become undefined). Hence a test that
asserts the wire shape, not just the arithmetic.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.signals.router import get_service, router
from tests.conftest import make_bars


class FakeMarket:
    """Returns a fixed frame and counts fetches per (symbol, exchange)."""

    def __init__(self, frames: dict[str, pd.DataFrame | None]):
        self.frames = frames
        self.fetches: list[tuple[str, str]] = []

    async def get_historical_df(self, symbol, exchange="NSE", interval="15m", days=30):
        self.fetches.append((symbol, exchange))
        return self.frames.get(symbol)


def utc_bars(rows, start="2026-01-05 09:15"):
    frame = make_bars(rows, start=start, freq="5min")
    frame.index = frame.index.tz_localize("UTC")
    return frame


@pytest.fixture
def client_and_market():
    # A BUY on RELIANCE that runs to target, and a SELL on TCS that gaps
    # through its stop on the second bar.
    market = FakeMarket({
        "RELIANCE": utc_bars([(100, 101, 99.5, 100.5), (100.5, 106, 100, 105)]),
        "TCS": utc_bars([(100, 100.5, 99.5, 100), (110, 112, 109, 111)]),
        "NODATA": None,
    })

    app = FastAPI()
    app.include_router(router, prefix="/signals")
    app.dependency_overrides[get_service] = lambda: type("S", (), {"market": market})()
    with TestClient(app) as client:
        yield client, market


def signal(sid, symbol, direction, entry, target, stop, at="2026-01-05T00:00:00Z"):
    return {"id": sid, "symbol": symbol, "exchange": "NSE", "direction": direction,
            "entry_price": entry, "target_price": target, "stop_loss": stop,
            "generated_at": at}


def test_response_carries_exactly_the_keys_the_api_client_reads(client_and_market):
    client, _ = client_and_market
    resp = client.post("/signals/evaluate", json={
        "signals": [signal("a", "RELIANCE", "BUY", 100, 105, 95)],
    })
    assert resp.status_code == 200
    row = resp.json()["results"][0]
    assert set(row) == {"id", "symbol", "outcome", "exit_price", "pnl_pct"}
    assert row["id"] == "a"


def test_ids_are_echoed_so_the_caller_can_pair_by_id_not_position(client_and_market):
    client, _ = client_and_market
    resp = client.post("/signals/evaluate", json={"signals": [
        signal("first", "RELIANCE", "BUY", 100, 105, 95),
        signal("second", "TCS", "SELL", 100, 95, 105),
    ]})
    rows = {r["id"]: r for r in resp.json()["results"]}
    assert rows["first"]["outcome"] == "TARGET_HIT"
    assert rows["second"]["outcome"] == "STOP_HIT"


def test_costs_and_gap_fills_are_applied_here_unlike_the_old_typescript_copy(client_and_market):
    client, _ = client_and_market
    resp = client.post("/signals/evaluate", json={"signals": [
        signal("gap", "TCS", "SELL", 100, 95, 105),
    ]})
    row = resp.json()["results"][0]
    # Gapped up to 110 on the open — the old evaluator booked this at the 105
    # stop for -5%. Filling at the open and charging costs gives ~-10.12%.
    assert row["exit_price"] == 110.0
    assert row["pnl_pct"] < -10


def test_bars_are_fetched_once_per_symbol_not_once_per_signal(client_and_market):
    client, market = client_and_market
    client.post("/signals/evaluate", json={"signals": [
        signal("a", "RELIANCE", "BUY", 100, 105, 95),
        signal("b", "RELIANCE", "BUY", 100, 106, 94),
        signal("c", "RELIANCE", "SELL", 100, 95, 105),
    ]})
    assert market.fetches == [("RELIANCE", "NSE")]


def test_a_symbol_with_no_bars_reports_no_data(client_and_market):
    client, _ = client_and_market
    resp = client.post("/signals/evaluate", json={
        "signals": [signal("x", "NODATA", "BUY", 100, 105, 95)],
    })
    row = resp.json()["results"][0]
    assert row["outcome"] == "NO_DATA"
    assert row["exit_price"] is None


def test_a_signal_generated_after_every_available_bar_is_open_not_resolved(client_and_market):
    """Guards against evaluating a trade using bars from before it existed."""
    client, _ = client_and_market
    resp = client.post("/signals/evaluate", json={
        "signals": [signal("late", "RELIANCE", "BUY", 100, 105, 95, at="2030-01-01T00:00:00Z")],
    })
    assert resp.json()["results"][0]["outcome"] == "OPEN"


def test_summary_reports_resolved_wins_and_win_rate(client_and_market):
    client, _ = client_and_market
    resp = client.post("/signals/evaluate", json={"signals": [
        signal("a", "RELIANCE", "BUY", 100, 105, 95),
        signal("b", "TCS", "SELL", 100, 95, 105),
        signal("c", "NODATA", "BUY", 100, 105, 95),
    ]})
    summary = resp.json()["summary"]
    assert summary == {
        "evaluated": 3, "resolved": 2, "wins": 1, "losses": 1,
        "win_rate": 50.0, "avg_pnl_pct": pytest.approx(summary["avg_pnl_pct"]),
    }


def test_an_empty_signal_list_is_rejected(client_and_market):
    client, _ = client_and_market
    assert client.post("/signals/evaluate", json={"signals": []}).status_code == 422


def test_an_unparseable_timestamp_is_a_422_not_a_silent_skip(client_and_market):
    client, _ = client_and_market
    resp = client.post("/signals/evaluate", json={
        "signals": [signal("bad", "RELIANCE", "BUY", 100, 105, 95, at="not-a-date")],
    })
    assert resp.status_code == 422
