"""HTTP-layer test for POST /pine/run -- the sandbox itself is covered by
tests/test_pine_sandbox.py; this proves the route wiring (path, body
validation, response passthrough) against the real sandbox subprocess."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.signals.router import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_pine_run_returns_plot_data_for_a_valid_script():
    resp = client.post("/pine/run", json={
        "source": '//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")',
        "bars": [
            {"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i, "volume": 1000, "openTime": 1767000900000 + i * 60000}
            for i in range(10)
        ],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body["plots"]["SMA5"], list)


def test_pine_run_rejects_a_request_with_no_source():
    resp = client.post("/pine/run", json={"bars": []})
    assert resp.status_code == 422
