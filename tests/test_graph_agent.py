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
              return_value=("main", "result = line(ema(close, 20))")),
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}),
        patch("app.signals.agent.tools.graph_agent._dynamic_check_via_node",
              return_value={"valid": True}),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "20 EMA", "label": "EMA 20"})

    assert result["created"].startswith("DIA_CUSTOM_")
    assert result["pane"] == "main"
    assert box.results["custom_indicators"][0]["source"] == "result = line(ema(close, 20))"
    assert box.results["custom_indicators"][0]["outputName"] == "result"
    assert box.results["custom_indicators"][0]["displayLabel"] == "EMA 20"
    assert box.results["custom_indicators"][0]["pane"] == "main"


@pytest.mark.asyncio
async def test_invalid_then_valid_on_retry(trending_frame):
    box = _toolbox(trending_frame)
    write_calls = []

    async def fake_write(ctx, description, feedback=None, source=None):
        write_calls.append(feedback)
        return ("main", "result = line(ema(close, 20))") if feedback else ("main", "not diascript at all")

    async def fake_validate(source: str, output_name: str) -> dict:
        if source == "not diascript at all":
            return {"valid": False, "error": {"message": "unexpected token"}}
        return {"valid": True, "outputType": "line"}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node", new=fake_validate),
        patch("app.signals.agent.tools.graph_agent._dynamic_check_via_node",
              return_value={"valid": True}),
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
        return "main", "still not diascript"

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
        return "main", f"result = line(ema(close, {description}))"

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}),
        patch("app.signals.agent.tools.graph_agent._dynamic_check_via_node",
              return_value={"valid": True}),
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
        return "main", "result = line(ema(close, 20))"

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}),
        patch("app.signals.agent.tools.graph_agent._dynamic_check_via_node",
              return_value={"valid": True}),
    ):
        turn_one = _toolbox(trending_frame)
        turn_two = _toolbox(trending_frame)
        first = await turn_one.execute("generate_custom_indicator", {"description": "20 EMA"})
        second = await turn_two.execute("generate_custom_indicator", {"description": "the 20 period EMA"})

    assert first["created"] == second["created"]


def test_extract_pane_parses_main_and_strips_the_line():
    from app.signals.agent.tools.graph_agent import _extract_pane

    pane, source = _extract_pane("PANE: main\nresult = line(ema(close, 20))")
    assert pane == "main"
    assert source == "result = line(ema(close, 20))"


def test_extract_pane_parses_sub_case_insensitively():
    from app.signals.agent.tools.graph_agent import _extract_pane

    pane, source = _extract_pane("pane: SUB\nresult = line(rsi(close, 14))")
    assert pane == "sub"
    assert source == "result = line(rsi(close, 14))"


def test_extract_pane_defaults_to_sub_when_the_model_drops_the_line():
    """Found live: a mis-paned overlay is just an extra pane, but a
    mis-paned oscillator overlapping candles at the wrong scale is worse --
    "sub" is the safer default when the model didn't say."""
    from app.signals.agent.tools.graph_agent import _extract_pane

    pane, source = _extract_pane("result = line(ema(close, 20))")
    assert pane == "sub"
    assert source == "result = line(ema(close, 20))"


def test_system_prompt_mentions_exp_and_gaussian_guidance():
    """A Gaussian filter needs exp()-based weights to be real, not a plain
    ema()/sma() relabeled as "Gaussian-like" — the prompt must give the exact
    math AND state the general rule against faking a named technique (the
    general rule is checked on its own elsewhere; Gaussian's exp() guidance
    is specific enough to warrant its own assertion here)."""
    from app.signals.agent.tools.graph_agent import SYSTEM_PROMPT

    assert "exp(x)" in SYSTEM_PROMPT
    assert "Gaussian" in SYSTEM_PROMPT
    assert "Never fake sophistication" in SYSTEM_PROMPT


@pytest.mark.skipif(shutil.which("diascript-validate") is None, reason="diascript-validate not installed")
@pytest.mark.asyncio
async def test_the_prompts_own_gaussian_filter_example_is_actually_valid_diascript():
    """The worked example in SYSTEM_PROMPT is what the model pattern-matches
    against — if it doesn't actually validate, every real Gaussian filter
    request built from it would fail too."""
    from app.signals.agent.tools.graph_agent import _validate_via_node

    gaussian_example = (
        "w0 = exp(0)\nw1 = exp(-1/18)\nw2 = exp(-4/18)\nw3 = exp(-9/18)\n"
        "w4 = exp(-16/18)\nw5 = exp(-25/18)\nw6 = exp(-36/18)\nw7 = exp(-49/18)\nw8 = exp(-64/18)\n"
        "wsum = w0+w1+w2+w3+w4+w5+w6+w7+w8\n"
        "result = line((w0*close + w1*ref(close,1) + w2*ref(close,2) + w3*ref(close,3) + "
        "w4*ref(close,4) + w5*ref(close,5) + w6*ref(close,6) + w7*ref(close,7) + w8*ref(close,8)) / wsum)"
    )
    result = await _validate_via_node(gaussian_example, "result")
    assert result == {"valid": True, "outputType": "line"}


def test_system_prompt_never_offers_barcolor_as_a_safe_output_wrapper():
    """barcolor(...) parses fine — diascript-validate only checks that SOME
    output wrapper was used, it doesn't know which ones the real klinecharts
    render adapter supports. The adapter has no case for barcolor (recoloring
    the candles themselves needs a different integration point than the
    per-indicator figures every other output type goes through) and throws,
    so the prompt must never suggest it as something to wrap `result` in —
    only ever list it among what NOT to use."""
    from app.signals.agent.tools.graph_agent import SYSTEM_PROMPT

    wrap_rule, _, do_not_use = SYSTEM_PROMPT.partition("Do NOT use")
    assert "barcolor" not in wrap_rule
    assert "barcolor" in do_not_use


def test_system_prompt_teaches_fill_as_a_real_output_wrapper():
    """fill() now maps onto klinecharts' real polygon figure type — it must
    be taught as usable (with the bare-identifier-argument constraint that
    makes it different from every other output wrapper), not forbidden.
    Checked against the forbidden-items bullet specifically, not everything
    after "Do NOT use" — the worked example a few paragraphs later
    legitimately uses fill() too."""
    from app.signals.agent.tools.graph_agent import SYSTEM_PROMPT

    wrap_rule, _, rest = SYSTEM_PROMPT.partition("Do NOT use")
    forbidden_bullet, _, _ = rest.partition("\n\n")
    assert "fill(...)" in wrap_rule
    assert "fill" not in forbidden_bullet
    assert "bare NAME of an already-declared formula" in SYSTEM_PROMPT
    assert 'result = fill(upper, lower, "#2196F333")' in SYSTEM_PROMPT


def test_renderable_output_types_includes_fill():
    from app.signals.agent.tools.graph_agent import RENDERABLE_OUTPUT_TYPES

    assert "fill" in RENDERABLE_OUTPUT_TYPES
    assert "barcolor" not in RENDERABLE_OUTPUT_TYPES


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
            return "main", "result = line(ema(close, 20))"
        return "sub", 'result = barcolor(close > open, "green", "red")'

    async def fake_validate(source: str, output_name: str) -> dict:
        if "barcolor" in source:
            return {"valid": True, "outputType": "barcolor"}
        return {"valid": True, "outputType": "line"}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node", new=fake_validate),
        patch("app.signals.agent.tools.graph_agent._dynamic_check_via_node",
              return_value={"valid": True}),
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
        return "main", "result = fill(a, b)"

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
            return "main", "result = line(ema(close, 20))"
        return "main", "not diascript at all"

    async def fake_validate(source: str, output_name: str) -> dict:
        if source == "not diascript at all":
            return {"valid": False, "error": {"message": "unexpected token"}}
        return {"valid": True, "outputType": "line"}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node", new=fake_validate),
        patch("app.signals.agent.tools.graph_agent._dynamic_check_via_node",
              return_value={"valid": True}),
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
    # _FakeLlm's response has no PANE: line -- _extract_pane defaults it to
    # "sub", which is fine here since this test is only about the retry
    # message content, not pane classification.

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
    pane, source = await graph_agent._write_formula(ctx, "20 EMA")

    assert source == "result = line(ema(close, 20))"
    assert pane == "sub"  # no PANE: line in the fenced response -- defaults to "sub"


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
    pane, source = await graph_agent._write_formula(ctx, "close price")

    assert source == "result = line(close)"
    assert pane == "sub"


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

    with (
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}),
        patch("app.signals.agent.tools.graph_agent._dynamic_check_via_node",
              return_value={"valid": True}),
    ):
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


# ---------------------------------------------------------------------------
# Fix 6: static parsing is not enough. diascript-validate only checks that a
# formula parses and names a real output wrapper -- it never executes it, so
# it cannot see a formula that crashes on real bars (e.g. `ref()` pointing
# past the edge of loaded history) or one whose condition can mathematically
# never be true (e.g. comparing a bar against a window that already includes
# it). `_dynamic_check_via_node` actually runs the formula against synthetic
# bars through the real diascript engine to catch both classes before they
# reach a user's chart.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dynamic_check_is_skipped_when_static_validation_already_failed(trending_frame):
    """The dynamic gate is a second, more expensive check -- it must never
    run on a formula that already failed the cheap parse check."""
    box = _toolbox(trending_frame)
    dynamic_calls = []

    async def fake_dynamic(source, output_name):
        dynamic_calls.append(source)
        return {"valid": True}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula",
              return_value=("main", "not diascript at all")),
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": False, "error": {"message": "unexpected token"}}),
        patch("app.signals.agent.tools.graph_agent._dynamic_check_via_node", new=fake_dynamic),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "bad request"})

    assert "error" in result
    assert dynamic_calls == []


@pytest.mark.asyncio
async def test_a_dynamic_check_failure_triggers_a_retry_with_its_feedback(trending_frame):
    """A formula that parses fine but fails the real-data check must be
    eligible for the same one retry a parse failure gets -- and the model
    must see the ACTUAL dynamic-check message, not a generic one."""
    box = _toolbox(trending_frame)
    write_calls = []

    async def fake_write(ctx, description, feedback=None, source=None):
        write_calls.append(feedback)
        if feedback:
            return "main", "result = line(ema(close, 20))"
        return "main", "result = line(ref(close, -1))"

    async def fake_dynamic(source, output_name):
        if "ref(close, -1)" in source:
            return {"valid": False, "error": {"message": "crashes on the newest candles"}}
        return {"valid": True}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}),
        patch("app.signals.agent.tools.graph_agent._dynamic_check_via_node", new=fake_dynamic),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "trend filter"})

    assert len(write_calls) == 2
    assert write_calls[0] is None
    assert write_calls[1] == "crashes on the newest candles"
    assert "created" in result


@pytest.mark.asyncio
async def test_a_dynamic_check_failure_that_survives_the_retry_is_an_error(trending_frame):
    box = _toolbox(trending_frame)

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula",
              return_value=("main", "result = line(ref(close, -1))")),
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}),
        patch("app.signals.agent.tools.graph_agent._dynamic_check_via_node",
              return_value={"valid": False, "error": {"message": "crashes on the newest candles"}}),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "trend filter"})

    assert "error" in result
    assert "crashes on the newest candles" in result["error"]
    assert "custom_indicators" not in box.results


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.asyncio
async def test_real_dynamic_check_catches_a_negative_ref_offset_crashing_on_real_bars():
    """A centered/symmetric filter built from ref(close, -1)..ref(close, -4)
    is valid diascript, but a negative offset asks for a bar that hasn't
    happened yet -- it crashes the render adapter on the newest candles,
    where there is no "future" bar to read. The real (non-mocked) checker
    must catch this."""
    from app.signals.agent.tools.graph_agent import _dynamic_check_via_node

    source = (
        "trend = (ref(close, 4) + ref(close, 3) + ref(close, 2) + ref(close, 1) + close + "
        "ref(close, -1) + ref(close, -2) + ref(close, -3) + ref(close, -4)) / 9\n"
        "result = line(trend)"
    )
    result = await _dynamic_check_via_node(source, "result")
    assert result["valid"] is False
    assert "crashes" in result["error"]["message"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.asyncio
async def test_real_dynamic_check_catches_a_condition_that_can_never_fire():
    """Comparing close against highest(high, 10) is a no-op — that window
    already includes the current bar's own high, so close can never exceed
    it. The marker would sit on the chart doing nothing, forever, with no
    error anywhere. The real (non-mocked) checker must catch this."""
    from app.signals.agent.tools.graph_agent import _dynamic_check_via_node

    source = (
        "swing_high = highest(high, 10)\n"
        "break_cond = close > swing_high and ref(close, 1) <= ref(swing_high, 1)\n"
        'result = marker(break_cond, "triangle-up", "#4CAF50")'
    )
    result = await _dynamic_check_via_node(source, "result")
    assert result["valid"] is False
    assert "never true" in result["error"]["message"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.asyncio
async def test_real_dynamic_check_catches_division_by_zero_on_a_flat_market():
    """A Fisher-Transform-style formula dividing by (highest - lowest) is
    correct on a normal market and NaN/Infinity when the market is dead
    flat, since the denominator goes to zero. The "flat" synthetic scenario
    exists specifically to exercise that case."""
    from app.signals.agent.tools.graph_agent import _dynamic_check_via_node

    source = (
        "range_ = highest(high, 10) - lowest(low, 10)\n"
        "value = 2 * (close - lowest(low, 10)) / range_ - 1\n"
        "result = line(0.5 * log((1 + value) / (1 - value)))"
    )
    result = await _dynamic_check_via_node(source, "result")
    assert result["valid"] is False
    assert "non-finite" in result["error"]["message"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.asyncio
async def test_real_dynamic_check_passes_a_correctly_causal_weighted_filter():
    """A real, causal, normalized weighted filter — using only past-bar
    ref() offsets — must not be flagged."""
    from app.signals.agent.tools.graph_agent import _dynamic_check_via_node

    source = (
        "w0 = exp(-0)\nw1 = exp(-1)\nw2 = exp(-4)\nw3 = exp(-9)\n"
        "wsum = w0+w1+w2+w3\n"
        "result = line((w0*close + w1*ref(close,1) + w2*ref(close,2) + w3*ref(close,3)) / wsum)"
    )
    result = await _dynamic_check_via_node(source, "result")
    assert result["valid"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.asyncio
async def test_the_prompts_own_order_block_example_is_real_and_passes_both_checks():
    """held() gives real persistent state, which is what genuine Smart Money
    Concepts / order-block logic needs. This exact worked example must pass
    both the real validator and the real dynamic checker."""
    from app.signals.agent.tools.graph_agent import _dynamic_check_via_node, _validate_via_node

    source = (
        "swing_high_now = high == highest(high, 10)\n"
        "last_swing_high = held(swing_high_now, high)\n"
        "bearish_candle = close < open\n"
        "last_bear_ob_high = held(bearish_candle, high)\n"
        "last_bear_ob_low = held(bearish_candle, low)\n"
        "bos_up = close > ref(last_swing_high, 1) and ref(close, 1) <= ref(last_swing_high, 1)\n"
        'result = marker(bos_up, "triangle-up", "#4CAF50")'
    )
    static = await _validate_via_node(source, "result")
    assert static == {"valid": True, "outputType": "marker"}

    dynamic = await _dynamic_check_via_node(source, "result")
    assert dynamic["valid"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.asyncio
async def test_the_prompts_own_wavelet_example_is_real_and_passes_both_checks():
    """The stationary ("a trous") two-level Haar decomposition is a real
    multi-scale decomposition built entirely from existing primitives —
    variance drops monotonically from raw close to the level-1 to the
    level-2 approximation, which is genuine scale separation, not just
    smoothing. This exact worked example must pass both checks."""
    from app.signals.agent.tools.graph_agent import _dynamic_check_via_node, _validate_via_node

    source = (
        "approx1 = (close + ref(close, 1)) / 2\n"
        "detail1 = (close - ref(close, 1)) / 2\n"
        "approx2 = (approx1 + ref(approx1, 2)) / 2\n"
        "detail2 = (approx1 - ref(approx1, 2)) / 2\n"
        "result = line(approx2)"
    )
    static = await _validate_via_node(source, "result")
    assert static == {"valid": True, "outputType": "line"}

    dynamic = await _dynamic_check_via_node(source, "result")
    assert dynamic["valid"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.asyncio
async def test_the_prompts_own_fill_example_is_real_and_passes_both_checks():
    """fill() now maps onto klinecharts' real polygon figure type. This
    exact worked example must pass both the real validator and the real
    dynamic checker."""
    from app.signals.agent.tools.graph_agent import _dynamic_check_via_node, _validate_via_node

    source = (
        "upper = ema(close, 20)\n"
        "lower = ema(close, 50)\n"
        'result = fill(upper, lower, "#2196F333")'
    )
    static = await _validate_via_node(source, "result")
    assert static == {"valid": True, "outputType": "fill"}

    dynamic = await _dynamic_check_via_node(source, "result")
    assert dynamic["valid"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.asyncio
async def test_real_dynamic_check_catches_a_fill_dividing_by_a_zero_range():
    """fill()'s output object carries no series of its own (outputs.ts wraps
    it as just {between, color}) — the dynamic checker has to pull the two
    named series' OWN values out of result.values instead of the output
    itself. This confirms that path actually inspects the real numbers,
    rather than silently passing on an empty placeholder array."""
    from app.signals.agent.tools.graph_agent import _dynamic_check_via_node

    source = (
        "spread = highest(high, 10) - lowest(low, 10)\n"
        "upper = close + 1 / spread\n"
        "lower = close - 1 / spread\n"
        'result = fill(upper, lower, "#2196F333")'
    )
    result = await _dynamic_check_via_node(source, "result")
    assert result["valid"] is False
    assert "non-finite" in result["error"]["message"]


def test_system_prompt_forbids_negative_ref_offsets():
    """A negative ref() offset asks for a bar that hasn't happened yet, which
    crashes the renderer on the newest candles. The prompt must forbid this
    outright rather than relying on the dynamic gate alone to catch it after
    the fact."""
    from app.signals.agent.tools.graph_agent import SYSTEM_PROMPT

    assert "n must be zero or a positive whole number" in SYSTEM_PROMPT
    assert "bars forward" in SYSTEM_PROMPT


def test_system_prompt_warns_about_self_inclusive_window_comparisons():
    """close > highest(high, length) can never be true, since that window
    already includes the current bar. The prompt must show the correct
    ref(..., 1)-shifted pattern, and a worked example must demonstrate it
    end to end."""
    from app.signals.agent.tools.graph_agent import SYSTEM_PROMPT

    assert "always include the CURRENT bar" in SYSTEM_PROMPT
    assert "ref(highest(high, length), 1)" in SYSTEM_PROMPT
    assert "prior_high = ref(highest(high, 10), 1)" in SYSTEM_PROMPT


def test_system_prompt_never_fakes_sophistication_for_any_named_technique():
    """A request naming a specific technique must get the real underlying
    math, never a plain weighted filter or a generic breakout condition
    relabeled with that technique's name. The honesty rule must be general —
    it names Gaussian, wavelet AND Smart Money Concepts as techniques that
    must be built for real, not approximated."""
    from app.signals.agent.tools.graph_agent import SYSTEM_PROMPT

    assert "Never fake sophistication" in SYSTEM_PROMPT
    assert "wavelet" in SYSTEM_PROMPT.lower()
    assert "smart money concepts" in SYSTEM_PROMPT.lower()


def test_system_prompt_teaches_held_as_real_persistent_state_not_forbidden():
    """held() is fully supported by diascript's own parser and validator, and
    it is the primitive real Smart Money Concepts / order-block / break-of-
    structure logic needs. The prompt must teach it, not forbid it, and a
    worked example must use it."""
    from app.signals.agent.tools.graph_agent import SYSTEM_PROMPT

    do_not_use = SYSTEM_PROMPT.split("Do NOT use:", 1)[1]
    assert "held()" not in do_not_use
    assert "real persistent state" in SYSTEM_PROMPT
    assert "last_swing_high = held(swing_high_now, high)" in SYSTEM_PROMPT


def test_system_prompt_teaches_a_real_wavelet_decomposition():
    """A stationary ("a trous") two-level Haar-style decomposition is
    genuinely expressible with existing primitives, with no engine change
    needed — it separates scales for real, rather than being a relabeled
    moving average. The prompt must teach the actual construction, and a
    worked example must use it."""
    from app.signals.agent.tools.graph_agent import SYSTEM_PROMPT

    assert "à trous" in SYSTEM_PROMPT or "a trous" in SYSTEM_PROMPT.lower()
    assert "approx2 = (approx1 + ref(approx1, 2)) / 2" in SYSTEM_PROMPT
