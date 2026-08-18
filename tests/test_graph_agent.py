from __future__ import annotations

import asyncio
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config import get_settings
from app.signals.agent.budget import Budget
from app.signals.agent.context import StaticTradingContext
from app.signals.agent.toolbox import AgentToolbox


class _StaticMarket:
    def __init__(self, df): self.df = df
    async def get_historical_df(self, symbol, exchange="NSE", interval="15m", days=30):
        return self.df


def _toolbox(trending_frame, llm=None, budget=None):
    return AgentToolbox("RELIANCE", "NSE", trending_frame, context=StaticTradingContext({}),
                        market=_StaticMarket(trending_frame), settings=get_settings(),
                        llm=llm, budget=budget)


@pytest.mark.asyncio
async def test_valid_on_first_try(trending_frame):
    box = _toolbox(trending_frame)
    with (
        patch("app.signals.agent.tools.graph_agent._write_formula",
              return_value="result = line(ema(close, 20))"),
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "20 EMA", "label": "EMA 20"})

    assert result["created"].startswith("DIA_CUSTOM_")
    assert box.results["custom_indicators"][0]["source"] == "result = line(ema(close, 20))"
    assert box.results["custom_indicators"][0]["outputName"] == "result"
    assert box.results["custom_indicators"][0]["displayLabel"] == "EMA 20"


@pytest.mark.asyncio
async def test_invalid_then_valid_on_retry(trending_frame):
    box = _toolbox(trending_frame)
    write_calls = []

    async def fake_write(ctx, description, feedback=None, source=None):
        write_calls.append(feedback)
        return "result = line(ema(close, 20))" if feedback else "not diascript at all"

    async def fake_validate(source: str, output_name: str) -> dict:
        if source == "not diascript at all":
            return {"valid": False, "error": {"message": "unexpected token"}}
        return {"valid": True, "outputType": "line"}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node", new=fake_validate),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "20 EMA"})

    assert len(write_calls) == 2
    assert write_calls[0] is None
    assert write_calls[1] == "unexpected token"
    assert "created" in result


@pytest.mark.asyncio
async def test_invalid_twice_returns_error_without_touching_results(trending_frame):
    box = _toolbox(trending_frame)

    async def fake_write(ctx, description, feedback=None, source=None):
        return "still not diascript"

    async def fake_validate(source: str, output_name: str) -> dict:
        return {"valid": False, "error": {"message": "unexpected token"}}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node", new=fake_validate),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "bad request"})

    assert "error" in result
    assert "custom_indicators" not in box.results


@pytest.mark.asyncio
async def test_missing_description_short_circuits(trending_frame):
    box = _toolbox(trending_frame)
    result = await box.execute("generate_custom_indicator", {})
    assert "error" in result
    assert "custom_indicators" not in box.results


# ---------------------------------------------------------------------------
# Fix 1: names are derived from formula content, not a per-turn counter.
#
# The counter used to reset fresh with every ToolContext — one per HTTP
# request — so turn 2's first custom indicator was named DIA_CUSTOM_1, exactly
# like turn 1's first one, even though the formulas were unrelated. The
# frontend dedupes on name, so every turn after the first silently lost its
# indicator with no error anywhere.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_different_formulas_in_the_same_turn_get_distinct_names(trending_frame):
    box = _toolbox(trending_frame)

    async def fake_write(ctx, description, feedback=None, source=None):
        return f"result = line(ema(close, {description}))"

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}),
    ):
        first = await box.execute("generate_custom_indicator", {"description": "20"})
        second = await box.execute("generate_custom_indicator", {"description": "50"})

    assert first["created"] != second["created"]
    assert len(box.results["custom_indicators"]) == 2
    # The bookkeeping that tells two calls apart must never leak into `results`
    # as a key of its own — that dict is serialised verbatim as the turn's
    # browser-facing payload (see turn.py).
    assert set(box.results.keys()) == {"custom_indicators"}


@pytest.mark.asyncio
async def test_the_identical_formula_gets_the_same_name_even_across_separate_toolboxes(trending_frame):
    """The name comes from the formula's own content, not a per-turn counter,
    so the SAME formula written in two different turns (two different
    ToolContext instances, each with its own would-be counter starting at
    zero) is idempotent instead of colliding on an unrelated formula."""
    async def fake_write(ctx, description, feedback=None, source=None):
        return "result = line(ema(close, 20))"

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}),
    ):
        turn_one = _toolbox(trending_frame)
        turn_two = _toolbox(trending_frame)
        first = await turn_one.execute("generate_custom_indicator", {"description": "20 EMA"})
        second = await turn_two.execute("generate_custom_indicator", {"description": "the 20 period EMA"})

    assert first["created"] == second["created"]


def test_system_prompt_never_offers_barcolor_as_a_safe_output_wrapper():
    """barcolor(...) parses fine — diascript-validate only checks that SOME
    output wrapper was used, it doesn't know which ones the real klinecharts
    render adapter supports. The adapter has no case for barcolor (or fill)
    and throws, so the prompt must never suggest it as something to wrap
    `result` in — only ever list it among what NOT to use."""
    from app.signals.agent.tools.graph_agent import SYSTEM_PROMPT

    wrap_rule, _, do_not_use = SYSTEM_PROMPT.partition("Do NOT use")
    assert "barcolor" not in wrap_rule
    assert "barcolor" in do_not_use


@pytest.mark.asyncio
async def test_a_hung_validator_is_killed_and_reaped_not_left_running():
    """asyncio.wait_for cancels the AWAIT on timeout, not the OS process —
    without an explicit kill+wait, a genuinely hung diascript-validate (the
    exact case the timeout exists for) would leak as an orphaned process on
    every retry."""
    from app.signals.agent.tools import graph_agent

    calls = {"kill": 0, "waited": 0}

    class _HangingProc:
        returncode = None

        async def communicate(self, data):
            await asyncio.sleep(10)  # never resolves before the short timeout below

        def kill(self):
            calls["kill"] += 1

        async def wait(self):
            calls["waited"] += 1

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _HangingProc()

    with (
        patch.object(graph_agent.asyncio, "create_subprocess_exec", fake_create_subprocess_exec),
        patch.object(graph_agent, "VALIDATE_TIMEOUT_SECONDS", 0.05),
    ):
        result = await graph_agent._validate_via_node("result = line(close)", "result")

    assert result == {"valid": False, "error": {"message": "validator unavailable"}}
    assert calls["kill"] == 1
    assert calls["waited"] == 1


@pytest.mark.asyncio
async def test_a_broken_pipe_is_killed_and_reaped_like_a_timeout():
    """communicate() can raise OSError too (the child dying mid-write), not
    just time out — that path used to escape the timeout-only except clause
    without the same kill+reap treatment."""
    from app.signals.agent.tools import graph_agent

    calls = {"kill": 0, "waited": 0}

    class _BrokenPipeProc:
        returncode = None

        async def communicate(self, data):
            raise OSError("broken pipe")

        def kill(self):
            calls["kill"] += 1

        async def wait(self):
            calls["waited"] += 1

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _BrokenPipeProc()

    with patch.object(graph_agent.asyncio, "create_subprocess_exec", fake_create_subprocess_exec):
        result = await graph_agent._validate_via_node("result = line(close)", "result")

    assert result == {"valid": False, "error": {"message": "validator unavailable"}}
    assert calls["kill"] == 1
    assert calls["waited"] == 1


@pytest.mark.asyncio
async def test_uvloops_bare_runtimeerror_is_caught_like_a_broken_pipe():
    """Found live in production via a real user request ("Gaussian filter
    trend indicator"): uvloop (the real production event loop, not plain
    asyncio) signals "child died before communicate() finished writing
    stdin" as a bare RuntimeError ("unable to perform operation on
    <WriteUnixTransport closed=True...>; the handler is closed"), not an
    OSError. The narrower (TimeoutError, OSError) except clause let this
    escape uncaught and crash the whole tool call."""
    from app.signals.agent.tools import graph_agent

    calls = {"kill": 0, "waited": 0}

    class _DeadTransportProc:
        returncode = None

        async def communicate(self, data):
            raise RuntimeError(
                "unable to perform operation on <WriteUnixTransport closed=True reading=False "
                "0xfbd6feb83e00>; the handler is closed"
            )

        def kill(self):
            calls["kill"] += 1

        async def wait(self):
            calls["waited"] += 1

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _DeadTransportProc()

    with patch.object(graph_agent.asyncio, "create_subprocess_exec", fake_create_subprocess_exec):
        result = await graph_agent._validate_via_node("result = line(close)", "result")

    assert result == {"valid": False, "error": {"message": "validator unavailable"}}
    assert calls["kill"] == 1
    assert calls["waited"] == 1


@pytest.mark.asyncio
async def test_killing_an_already_dead_process_does_not_raise():
    """The crash that gets us into the except block often means the child
    already exited -- proc.kill() on an already-gone PID raises
    ProcessLookupError, which must not escape and mask the real error."""
    from app.signals.agent.tools import graph_agent

    calls = {"waited": 0}

    class _AlreadyGoneProc:
        returncode = None

        async def communicate(self, data):
            raise RuntimeError("transport already closed")

        def kill(self):
            raise ProcessLookupError("no such process")

        async def wait(self):
            calls["waited"] += 1

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _AlreadyGoneProc()

    with patch.object(graph_agent.asyncio, "create_subprocess_exec", fake_create_subprocess_exec):
        result = await graph_agent._validate_via_node("result = line(close)", "result")

    # kill() raising means the process is already gone -- nothing left to
    # reap, so wait() correctly never runs. The thing under test is that
    # ProcessLookupError itself doesn't escape and mask the real error.
    assert result == {"valid": False, "error": {"message": "validator unavailable"}}
    assert calls["waited"] == 0


# ---------------------------------------------------------------------------
# Fix 3: outputType is enforced, not just parsed.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_disallowed_output_type_is_treated_as_a_validation_failure_and_retried(trending_frame):
    """barcolor/fill parse fine — diascript-validate only checks that SOME
    output wrapper was used. The real klinecharts render adapter only
    implements line/band/marker/histogram/background, so an accepted-but-
    unrenderable outputType must be caught here (and be eligible for the one
    retry), not shipped to the frontend to crash on render."""
    box = _toolbox(trending_frame)
    write_calls = []

    async def fake_write(ctx, description, feedback=None, source=None):
        write_calls.append(feedback)
        if feedback:
            return "result = line(ema(close, 20))"
        return 'result = barcolor(close > open, "green", "red")'

    async def fake_validate(source: str, output_name: str) -> dict:
        if "barcolor" in source:
            return {"valid": True, "outputType": "barcolor"}
        return {"valid": True, "outputType": "line"}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node", new=fake_validate),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "color bars"})

    assert len(write_calls) == 2
    assert write_calls[0] is None
    assert "not supported by the chart renderer" in write_calls[1]
    assert "created" in result


@pytest.mark.asyncio
async def test_a_disallowed_output_type_that_survives_the_retry_is_an_error(trending_frame):
    box = _toolbox(trending_frame)

    async def fake_write(ctx, description, feedback=None, source=None):
        return "result = fill(a, b)"

    async def fake_validate(source: str, output_name: str) -> dict:
        return {"valid": True, "outputType": "fill"}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node", new=fake_validate),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "fill area"})

    assert "error" in result
    assert "custom_indicators" not in box.results


# ---------------------------------------------------------------------------
# Fix 4: the retry shows the model its own failed source, and a wrapping
# markdown code fence is stripped before treating a response as source.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_retry_call_is_given_the_failed_source(trending_frame):
    """A parser error like "Unexpected token ')' (line 1, col 26)" is close to
    useless without the line it refers to."""
    box = _toolbox(trending_frame)
    seen_sources = []

    async def fake_write(ctx, description, feedback=None, source=None):
        if feedback:
            seen_sources.append(source)
            return "result = line(ema(close, 20))"
        return "not diascript at all"

    async def fake_validate(source: str, output_name: str) -> dict:
        if source == "not diascript at all":
            return {"valid": False, "error": {"message": "unexpected token"}}
        return {"valid": True, "outputType": "line"}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node", new=fake_validate),
    ):
        await box.execute("generate_custom_indicator", {"description": "20 EMA"})

    assert seen_sources == ["not diascript at all"]


@pytest.mark.asyncio
async def test_write_formula_puts_the_failed_source_in_the_retry_message():
    from app.signals.agent.tools import graph_agent

    class _FakeLlm:
        def __init__(self):
            self.calls: list[dict] = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="result = line(close)"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    llm = _FakeLlm()
    ctx = SimpleNamespace(llm=llm, budget=Budget.from_settings(get_settings()))

    await graph_agent._write_formula(
        ctx, "20 EMA", feedback="Unexpected token ')' (line 1, col 26)", source="result = line(ema(close, ))",
    )

    sent = llm.calls[0]["messages"][-1]["content"]
    assert "result = line(ema(close, ))" in sent
    assert "Unexpected token ')' (line 1, col 26)" in sent


@pytest.mark.asyncio
async def test_write_formula_strips_a_markdown_code_fence():
    """An LLM wrapping its answer in ```diascript ... ``` (ignoring the "no
    markdown fences" rule) is a plausible failure mode — it must not burn the
    one retry on a pure formatting artifact."""
    from app.signals.agent.tools import graph_agent

    class _FencedLlm:
        def chat(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content="```diascript\nresult = line(ema(close, 20))\n```"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    ctx = SimpleNamespace(llm=_FencedLlm(), budget=Budget.from_settings(get_settings()))
    source = await graph_agent._write_formula(ctx, "20 EMA")

    assert source == "result = line(ema(close, 20))"


@pytest.mark.asyncio
async def test_write_formula_strips_a_bare_code_fence_with_no_language_tag():
    from app.signals.agent.tools import graph_agent

    class _FencedLlm:
        def chat(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content="```\nresult = line(close)\n```"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    ctx = SimpleNamespace(llm=_FencedLlm(), budget=Budget.from_settings(get_settings()))
    source = await graph_agent._write_formula(ctx, "close price")

    assert source == "result = line(close)"


# ---------------------------------------------------------------------------
# Fix 2: the writer LLM's spend is counted against the turn's real budget.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_writer_llm_call_uses_the_turns_client_and_records_its_usage(trending_frame):
    """_write_formula used to call the process-wide get_llm() directly, so its
    tokens never touched the turn's Budget — the one thing to_result()["usage"]
    (and a daily budget) is summed from. It must use ctx.llm — the turn's real
    injected client — and record into ctx.budget, exactly like triage, the main
    loop, and wrap-up already do via state.budget.record(response)."""
    class _FakeLlm:
        def __init__(self):
            self.calls: list[dict] = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="result = line(ema(close, 20))"))],
                usage=SimpleNamespace(prompt_tokens=42, completion_tokens=7),
            )

    llm = _FakeLlm()
    budget = Budget.from_settings(get_settings())
    box = _toolbox(trending_frame, llm=llm, budget=budget)

    with patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}):
        result = await box.execute("generate_custom_indicator", {"description": "20 EMA"})

    assert "created" in result
    assert llm.calls, "ctx.llm (the turn's real client), not the process-wide singleton, must be used"
    assert budget.usage.calls == 1
    assert budget.usage.prompt_tokens == 42
    assert budget.usage.completion_tokens == 7


# ---------------------------------------------------------------------------
# Fix 5: one real integration test for the actual cross-process contract.
#
# Every other test here mocks _validate_via_node entirely — nothing in this
# suite actually exercises the real diascript-validate subprocess. Skipped
# when the binary isn't installed; run for real wherever it is.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("diascript-validate") is None, reason="diascript-validate not installed")
@pytest.mark.asyncio
async def test_real_validator_accepts_a_valid_formula_and_rejects_an_invalid_one():
    from app.signals.agent.tools.graph_agent import _validate_via_node

    valid = await _validate_via_node("result = line(ema(close, 20))", "result")
    assert valid["valid"] is True

    invalid = await _validate_via_node("result = line(ema(close, ))", "result")
    assert invalid["valid"] is False
