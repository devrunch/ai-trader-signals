"""HTTP-layer test for POST /pine/run -- the sandbox itself is covered by
tests/test_pine_sandbox.py; this proves the route wiring (path, body
validation, response passthrough) against the real sandbox subprocess."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.signals.router import MAX_BATCH_SCRIPTS, router

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


class TestPineRunBatch:
    """One request, one copy of the bars, many scripts.

    Every indicator on a chart runs against the same bars, and the single-run
    route made the client upload them once per indicator -- ten indicators on
    a 1m chart sent the same 83 KB ten times. These tests mock the sandbox:
    what matters here is the wiring (bars shared, ids echoed, one bad script
    not taking the others with it), not PineTS itself, which
    tests/test_pine_sandbox.py covers.
    """

    def _body(self, n: int = 3) -> dict:
        return {
            "bars": [{"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                      "volume": 10, "openTime": 1767000900000 + i * 60000}
                     for i in range(5)],
            "scripts": [{"id": f"i{i}", "source": f"//{i}"} for i in range(n)],
        }

    def test_every_script_runs_against_the_same_bars(self, monkeypatch):
        seen = []

        async def fake(source, bars, **kw):
            seen.append((source, len(bars)))
            return {"ok": True, "plots": {"p": [1, 2, 3]}}

        monkeypatch.setattr("app.signals.router.run_pine_script", fake)
        resp = client.post("/pine/run-batch", json=self._body(3))

        assert resp.status_code == 200
        assert [s for s, _ in seen] == ["//0", "//1", "//2"]
        assert {n for _, n in seen} == {5}          # the one copy, every time

    def test_each_result_carries_the_id_it_was_asked_for(self, monkeypatch):
        """The client attaches by id; results come back in a list, and
        matching them up by position would break the moment one is dropped."""
        async def fake(source, bars, **kw):
            return {"ok": True, "plots": {"p": []}}

        monkeypatch.setattr("app.signals.router.run_pine_script", fake)
        resp = client.post("/pine/run-batch", json=self._body(3))

        assert [r["id"] for r in resp.json()["results"]] == ["i0", "i1", "i2"]

    def test_one_failing_script_does_not_take_the_others_down(self, monkeypatch):
        async def fake(source, bars, **kw):
            if source == "//1":
                raise RuntimeError("boom")
            return {"ok": True, "plots": {"p": []}}

        monkeypatch.setattr("app.signals.router.run_pine_script", fake)
        results = client.post("/pine/run-batch", json=self._body(3)).json()["results"]

        assert [r["ok"] for r in results] == [True, False, True]
        assert "boom" in results[1]["error"]

    def test_a_script_the_sandbox_rejects_is_reported_not_raised(self, monkeypatch):
        """A Pine compile error is a normal result, not an exception."""
        async def fake(source, bars, **kw):
            return {"ok": False, "error": "line 1: no such function"}

        monkeypatch.setattr("app.signals.router.run_pine_script", fake)
        results = client.post("/pine/run-batch", json=self._body(1)).json()["results"]

        assert results[0] == {"id": "i0", "ok": False, "error": "line 1: no such function"}

    def test_an_oversized_batch_is_refused(self, monkeypatch):
        """A chart holds a handful of indicators. A request asking for
        hundreds is a bug or an abuse, and each one costs a sandbox run."""
        async def fake(source, bars, **kw):
            return {"ok": True, "plots": {"p": []}}

        monkeypatch.setattr("app.signals.router.run_pine_script", fake)
        resp = client.post("/pine/run-batch", json=self._body(MAX_BATCH_SCRIPTS + 1))

        assert resp.status_code == 422

    def test_an_empty_batch_is_not_an_error(self, monkeypatch):
        async def fake(source, bars, **kw):
            raise AssertionError("must not run anything")

        monkeypatch.setattr("app.signals.router.run_pine_script", fake)
        resp = client.post("/pine/run-batch", json={"bars": [], "scripts": []})

        assert resp.status_code == 200
        assert resp.json()["results"] == []
