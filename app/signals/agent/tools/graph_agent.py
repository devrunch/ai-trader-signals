"""
Custom-indicator authoring: the model writes diascript, a real parser
validates it, one retry with the real error if it fails. Nothing here
executes diascript — only ever a subprocess parse check. Rendering happens
entirely client-side, via the same registerDiascriptIndicator pipeline that
already draws DIA_EMA20/DIA_RSI14.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
from typing import Any

from app.signals.agent.tools.base import Handler, ToolContext

logger = logging.getLogger(__name__)

VALIDATE_TIMEOUT_SECONDS = 5

# The real klinecharts render adapter this feature feeds only implements these
# five (see D:\adizx\diascript\src\engine\outputs.ts's buildOutput). barcolor
# and fill parse fine — diascript-validate only checks that SOME output
# wrapper was used, it has no idea which ones the render adapter supports — but
# both throw at render time, client-side, well after this tool has returned.
RENDERABLE_OUTPUT_TYPES = {"line", "band", "marker", "histogram", "background"}

_CODE_FENCE_RE = re.compile(r"^```[^\n]*\n?(.*?)\n?```$", re.DOTALL)

SYSTEM_PROMPT = """You write diascript — a small, safe formula language for \
technical indicators. Output ONLY diascript source, nothing else: no prose, \
no markdown fences, no explanation.

Rules:
- Exactly one formula, always named `result`.
- It must be wrapped in an output function: line(...), band(...), \
marker(...), histogram(...), or background(...).
- Available series: open, high, low, close, volume.
- Available functions: sma(series, length), ema(series, length), \
wma(series, length), stdev(series, length), highest(series, length), \
lowest(series, length), sum(series, length), rsi(series, length), \
true_range(), typical_price(), abs(x), min(a, b), max(a, b), \
ref(series, n) (value n bars back, null before history starts), \
prev(n) (this same formula's own value n bars back).
- Point-wise math (+, -, *, /), comparisons, and and/or/not are allowed \
between series and numbers.
- band(upper, lower) takes the UPPER bound first, the LOWER bound second. \
marker(condition, shape, color) takes a boolean condition, then a shape name \
and a color, both quoted strings. background(condition, color) takes a \
boolean condition, then a quoted color string.
- Do NOT use: held(), series(), input, fill(...), barcolor(...), or the \
time/session/symbol namespaces. fill needs two already-declared formulas to \
fill between, which a single-formula file can never have. barcolor is not \
supported by the klinecharts render adapter this feature feeds. The rest \
are not needed for a single indicator formula, and using them raises the \
chance of a subtle scoping mistake.

Examples:

Input: the 20-EMA minus the 50-EMA
Output:
result = line(ema(close, 20) - ema(close, 50))

Input: RSI with a 21-period length
Output:
result = line(rsi(close, 21))

Input: a band around price at 2 standard deviations
Output:
result = band(close + stdev(close, 20) * 2, close - stdev(close, 20) * 2)

Input: mark bars where RSI crosses above 70 with a red down-triangle
Output:
result = marker(rsi(close, 14) > 70, "triangle-down", "#F44336")

Input: highlight the background red when price is below its 50-SMA
Output:
result = background(close < sma(close, 50), "#F44336")
"""


def _strip_code_fence(text: str) -> str:
    """Drop a wrapping ``` fence, if the model ignored the "no markdown" rule.

    Not a markdown parser — just enough to turn the one obvious formatting
    artifact (```diascript ... ``` or ``` ... ```) back into the source it
    was hiding, so a stray fence doesn't burn the one retry on a formatting
    mistake instead of a real content error.
    """
    text = text.strip()
    match = _CODE_FENCE_RE.match(text)
    return match.group(1).strip() if match else text


async def _write_formula(
    ctx: ToolContext, description: str, feedback: str | None = None, source: str | None = None,
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if feedback:
        content = f"{description}\n\nYou wrote:\n{source}\n\nThat failed to validate: {feedback}\nFix it."
    else:
        content = description
    messages.append({"role": "user", "content": content})

    # ctx.llm, not the process-wide get_llm() singleton — this is the turn's
    # real injected client, and recording into ctx.budget is what makes this
    # call's tokens show up in the turn's usage total (see budget.py), exactly
    # like every other LLM call the turn makes.
    response = await asyncio.to_thread(ctx.llm.chat, temperature=0, max_tokens=300, messages=messages)
    ctx.budget.record(response)
    return _strip_code_fence((response.choices[0].message.content or "").strip())


async def _validate_via_node(source: str, output_name: str) -> dict:
    # shutil.which resolves an npm-installed .cmd shim on Windows local dev;
    # the bare name still works fine in the real Docker deployment target.
    executable = shutil.which("diascript-validate") or "diascript-validate"
    try:
        proc = await asyncio.create_subprocess_exec(
            executable, "--output", output_name,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError) as e:
        logger.warning("diascript-validate unavailable: %s", e)
        return {"valid": False, "error": {"message": "validator unavailable"}}

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(source.encode()), timeout=VALIDATE_TIMEOUT_SECONDS,
        )
    except (asyncio.TimeoutError, OSError) as e:
        # wait_for only cancels the await — the OS process is still running
        # and nothing else is tracking it, so it has to be reaped here. A
        # broken pipe (OSError) leaves the same orphan risk as a timeout, so
        # it gets the same kill+reap treatment.
        proc.kill()
        await proc.wait()
        logger.warning("diascript-validate did not complete: %s", e)
        return {"valid": False, "error": {"message": "validator unavailable"}}

    if proc.returncode != 0:
        return {"valid": False, "error": {"message": stderr.decode().strip() or "validator crashed"}}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"valid": False, "error": {"message": "validator returned malformed output"}}


def _validation_feedback(result: dict) -> str | None:
    """None if the result is fully usable, otherwise the message to retry with.

    "Usable" is stricter than the validator's own `valid` flag: diascript-validate
    only checks that the formula parses and names a real output wrapper — it has
    no idea which of those wrappers the klinecharts render adapter actually
    implements. Rejecting an unsupported outputType here, the same way a parse
    failure is rejected, closes that whole class of bug rather than trusting the
    prompt alone to keep the model away from barcolor/fill.
    """
    if not result.get("valid"):
        return result.get("error", {}).get("message") or "invalid formula"
    output_type = result.get("outputType")
    if output_type not in RENDERABLE_OUTPUT_TYPES:
        return (f"Output type '{output_type}' is not supported by the chart renderer — "
                f"use one of: {', '.join(sorted(RENDERABLE_OUTPUT_TYPES))}")
    return None


async def generate_custom_indicator(ctx: ToolContext, args: dict) -> Any:
    description = str(args.get("description") or "").strip()
    if not description:
        return {"error": "A description of the indicator is required."}

    output_name = "result"
    source = await _write_formula(ctx, description)
    result = await _validate_via_node(source, output_name)
    feedback = _validation_feedback(result)

    if feedback:
        failed_source = source
        source = await _write_formula(ctx, description, feedback=feedback, source=failed_source)
        result = await _validate_via_node(source, output_name)
        feedback = _validation_feedback(result)

    if feedback:
        return {"error": f"Could not build a valid indicator: {feedback}"}

    display_label = str(args.get("label") or description)[:60]
    # Named from the formula's own content, not a per-turn counter: a counter
    # reset fresh on every HTTP request meant turn 2's first indicator reused
    # turn 1's name, and the frontend dedupes on name — so every turn after the
    # first silently lost its custom indicator. Content hashing makes identical
    # formulas idempotent and different formulas always distinct, regardless of
    # which turn produced them.
    indicator_name = f"DIA_CUSTOM_{hashlib.sha1(source.encode()).hexdigest()[:8]}"

    ctx.results.setdefault("custom_indicators", []).append({
        "name": indicator_name, "source": source,
        "outputName": output_name, "displayLabel": display_label,
    })
    return {"created": indicator_name, "label": display_label}


TOOLS: dict[str, Handler] = {
    "generate_custom_indicator": generate_custom_indicator,
}
