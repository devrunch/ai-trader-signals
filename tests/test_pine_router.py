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


def test_pine_run_resolves_syminfo_when_symbol_and_exchange_are_given():
    """syminfo.ticker is undefined against a raw bars array (PineTS never
    populates it without a real IProvider) -- confirmed the fix by running
    a script that reads it through the router's real symbol/exchange
    wiring, against the real sandbox subprocess, not a mock."""
    resp = client.post("/pine/run", json={
        "source": (
            '//@version=5\nindicator("t")\nplot(1, "x")\n'
            'if barstate.islast\n'
            '    alert("BUY - " + syminfo.ticker, alert.freq_once_per_bar_close)'
        ),
        "bars": [
            {"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i, "volume": 1000, "openTime": 1767000900000 + i * 60000}
            for i in range(10)
        ],
        "symbol": "RELIANCE",
        "exchange": "NSE",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True, body.get("error")


def test_pine_run_returns_input_metadata_and_applies_real_overrides():
    """Real PineTS Indicator/getInputsMeta() usage (app/pine_sandbox/worker.mjs),
    not the source-text substitution an earlier draft considered -- proves both
    that the metadata a settings form would render from is real, and that an
    override actually changes the computed output through PineTS's own runtime."""
    source = (
        '//@version=5\nindicator("t")\n'
        'length = input.int(20, minval=1, title="Length")\n'
        'plot(ta.sma(close, length), "SMA")'
    )
    bars = [
        {"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i, "volume": 1000, "openTime": 1767000900000 + i * 60000}
        for i in range(30)
    ]

    default_resp = client.post("/pine/run", json={"source": source, "bars": bars})
    assert default_resp.status_code == 200
    default_body = default_resp.json()
    assert default_body["ok"] is True, default_body.get("error")
    assert default_body["inputsMeta"] == [
        {"type": "int", "defval": 20, "varId": "length", "title": "Length", "minval": 1}
    ]

    overridden_resp = client.post("/pine/run", json={"source": source, "bars": bars, "inputOverrides": {"length": 5}})
    assert overridden_resp.status_code == 200
    overridden_body = overridden_resp.json()
    assert overridden_body["ok"] is True, overridden_body.get("error")

    default_last = default_body["plots"]["SMA"][-1]["value"]
    overridden_last = overridden_body["plots"]["SMA"][-1]["value"]
    assert default_last != overridden_last
