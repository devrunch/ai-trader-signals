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
_PANE_LINE_RE = re.compile(r"^PANE:\s*(main|sub)\s*$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = """You write diascript — a small, safe formula language for \
technical indicators. Output exactly two things, nothing else — no prose, \
no extra markdown fences, no explanation:

1. A first line stating where this indicator belongs on the chart: \
`PANE: main` if it overlays directly on the price chart, same scale as price \
(moving averages, bands, trend/smoothing filters, price channels) — or \
`PANE: sub` if it needs its own separate pane below the price chart, a \
different scale than price (oscillators, momentum indicators, anything \
volume-based, anything bounded like 0-100).
2. Then the diascript formula on the following line(s).

Rules:
- Exactly one formula, always named `result`.
- It must be wrapped in an output function: line(...), band(...), \
marker(...), histogram(...), or background(...).
- Available series: open, high, low, close, volume.
- Available functions: sma(series, length), ema(series, length), \
wma(series, length), stdev(series, length), highest(series, length), \
lowest(series, length), sum(series, length), highestbars(series, length) \
(bar-offset of the highest value in the window, 0=most recent), \
lowestbars(series, length), rsi(series, length), \
true_range(), typical_price(), abs(x), min(a, b), max(a, b), \
log(x), sqrt(x), exp(x), \
ref(series, n) (value n bars back, null before history starts), \
prev(n) (this same formula's own value n bars back).
- For anything needing weighted-average-style math over a FIXED small \
window (a Gaussian filter, ALMA, a custom weighted moving average), \
write the weights out explicitly as literal numbers for each `ref(x, k)` \
term (k = 0 to window-1) rather than looking for a loop — there is no \
loop construct, every window is unrolled by hand at authoring time. \
exp(x) is what makes a real Gaussian-shaped weight kernel possible: \
weight(k) = exp(-(k*k) / (2*sigma*sigma)), normalized so the weights sum \
to 1. Never substitute a plain ema()/sma() and call it "Gaussian-like" — \
if the request needs real Gaussian weights, compute them with exp() for \
real, using the primitives above.
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
PANE: main
result = line(ema(close, 20) - ema(close, 50))

Input: RSI with a 21-period length
Output:
PANE: sub
result = line(rsi(close, 21))

Input: a Gaussian filter trend indicator
Output:
PANE: main
w0 = exp(0)
w1 = exp(-1/18)
w2 = exp(-4/18)
w3 = exp(-9/18)
w4 = exp(-16/18)
w5 = exp(-25/18)
w6 = exp(-36/18)
w7 = exp(-49/18)
w8 = exp(-64/18)
wsum = w0+w1+w2+w3+w4+w5+w6+w7+w8
result = line((w0*close + w1*ref(close,1) + w2*ref(close,2) + w3*ref(close,3) + w4*ref(close,4) + w5*ref(close,5) + w6*ref(close,6) + w7*ref(close,7) + w8*ref(close,8)) / wsum)

Input: a band around price at 2 standard deviations
Output:
PANE: main
result = band(close + stdev(close, 20) * 2, close - stdev(close, 20) * 2)

Input: mark bars where RSI crosses above 70 with a red down-triangle
Output:
PANE: sub
result = marker(rsi(close, 14) > 70, "triangle-down", "#F44336")

Input: highlight the background red when price is below its 50-SMA
Output:
PANE: main
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


def _extract_pane(text: str) -> tuple[str, str]:
    """Pulls the leading `PANE: main`/`PANE: sub` line back out, returning
    (pane, remaining_source). Defaults to "sub" if the model dropped the
    line entirely — the safer default, since a wrongly-sub-paned overlay is
    just an extra pane, but a wrongly-main-paned oscillator overlaps candles
    at the wrong scale."""
    match = _PANE_LINE_RE.search(text)
    if not match:
        return "sub", text
    pane = match.group(1).lower()
    remaining = (text[:match.start()] + text[match.end():]).strip()
    return pane, remaining


async def _write_formula(
    ctx: ToolContext, description: str, feedback: str | None = None, source: str | None = None,
) -> tuple[str, str]:
    """Returns (pane, source) — see _extract_pane."""
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
    raw = _strip_code_fence((response.choices[0].message.content or "").strip())
    return _extract_pane(raw)


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
    except Exception as e:
        # wait_for only cancels the await — the OS process is still running
        # and nothing else is tracking it, so it has to be reaped here.
        # Deliberately broad: a subprocess pipe is an OS/event-loop boundary,
        # and different loop implementations signal the same underlying
        # failure with different exception types — a broken pipe is OSError
        # under plain asyncio, but uvloop (the production event loop; this
        # was found live via a real "Gaussian filter" request) raises a bare
        # RuntimeError ("unable to perform operation on <WriteUnixTransport
        # closed=True...>; the handler is closed") when the child dies before
        # communicate() finishes writing its stdin. Narrowing this to named
        # exception types is exactly what let that RuntimeError escape
        # uncaught last time — the contract here is "never let a subprocess
        # problem crash the tool call," not "catch every type we've seen so far."
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass  # already gone -- the crash that got us here often means it already exited
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
    pane, source = await _write_formula(ctx, description)
    result = await _validate_via_node(source, output_name)
    feedback = _validation_feedback(result)

    if feedback:
        failed_source = source
        pane, source = await _write_formula(ctx, description, feedback=feedback, source=failed_source)
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
        "outputName": output_name, "displayLabel": display_label, "pane": pane,
    })
    return {"created": indicator_name, "label": display_label, "pane": pane}


TOOLS: dict[str, Handler] = {
    "generate_custom_indicator": generate_custom_indicator,
}
