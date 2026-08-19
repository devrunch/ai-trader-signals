"""
Custom-indicator authoring: the model writes diascript, a real parser
validates it, then a dynamic check actually runs it against synthetic bars
through the real engine — a formula can be grammatically perfect and still
crash on real data or be silently dead, and only running it catches that.
One retry with the real feedback if either check fails. Rendering happens
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
from pathlib import Path
from typing import Any

from app.signals.agent.tools.base import Handler, ToolContext

logger = logging.getLogger(__name__)

VALIDATE_TIMEOUT_SECONDS = 5

# Five synthetic-market scenarios per formula, each a fresh Node process --
# still well under a chat turn's budget.
DYNAMIC_CHECK_TIMEOUT_SECONDS = 8
DYNAMIC_CHECK_SCRIPT = Path(__file__).parent / "diascript_dynamic_check.mjs"

# The real klinecharts render adapter this feature feeds implements these six
# (see D:\adizx\diascript\src\adapters\render\klinecharts\adapter.ts's
# figuresFor/calcResultFor — fill() maps onto klinecharts' own polygon figure
# type). barcolor parses fine — diascript-validate only checks that SOME
# output wrapper was used, it has no idea which ones the render adapter
# supports — but it still throws at render time, client-side, well after this
# tool has returned: recoloring the candles themselves needs a different
# klinecharts integration point than the per-indicator figures every other
# output type here goes through.
RENDERABLE_OUTPUT_TYPES = {"line", "band", "marker", "histogram", "background", "fill"}

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
marker(...), histogram(...), background(...), or fill(...).
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
- ref(series, n): n must be zero or a positive whole number — bars BACK \
only. A negative n would mean bars forward, which do not exist yet on the \
newest candles — it passes every check but crashes the chart exactly there, \
which is exactly where a trader is looking. There is no way to look at a \
future bar, ever.
- highest/lowest/highestbars/lowestbars/sum/sma/wma/stdev over `length` \
bars always include the CURRENT bar in that window. So a comparison like \
`close > highest(high, length)` can never be true — the current bar's own \
high is already part of what it is being compared against. To ask "did \
price just break above its recent high," compare against the PRIOR window \
instead: shift it back with ref(..., 1), e.g. `ref(highest(high, length), 1)`.
- Guard any division whose denominator could be zero (e.g. \
`highest(x, n) - lowest(x, n)` on a dead-flat run) with \
max(denominator, 0.0001) rather than dividing by a raw range or spread.
- held(condition, value): real persistent state, not a fixed window — \
while condition is false it keeps returning whatever it last returned when \
condition was true (0 before the first true bar ever happens). This is what \
makes genuine STRUCTURAL indicators possible: the last confirmed swing \
high/low, the last bearish or bullish candle before a break, and any other \
"remember this until something better replaces it" pattern — including \
real Smart Money Concepts order-block and break-of-structure logic. Do not \
write these off as impossible; held() is exactly the primitive for them. \
See the order-block worked example below.
- held()'s pre-trigger 0 is a real price-scale value, not a "no data yet" \
marker — if you ever plot a held() value directly with line(), the chart's \
y-axis is real prices (hundreds or thousands), so every bar before the \
first trigger renders as 0, and the line rockets from 0 up to the real \
price the instant it fires. That is a chart-breaking spike, not a flat \
lead-in. ALWAYS gate a held()-tracked LINE with a companion held() boolean \
flag and divide by it — both are 0 before the first trigger, so 0/0 is a \
real NaN, which the chart correctly skips instead of drawing: \
`has_swing = held(condition, 1)` then `line(tracked_value / has_swing)`. \
marker()/background() conditions built from held() do not need this — a \
boolean condition that is merely false pre-trigger never draws anything, \
there is no spike to gate.
- For anything needing weighted-average-style math over a FIXED small \
window (a Gaussian filter, ALMA, a custom weighted moving average), \
write the weights out explicitly as literal numbers for each `ref(x, k)` \
term (k = 0 to window-1) rather than looking for a loop — there is no \
loop construct, every window is unrolled by hand at authoring time. \
exp(x) is what makes a real Gaussian-shaped weight kernel possible: \
weight(k) = exp(-(k*k) / (2*sigma*sigma)), normalized so the weights sum \
to 1.
- A real multi-scale decomposition (a wavelet-style trend/cycle split) is \
also buildable, without any loop, as a small cascade of named formulas: \
compute a 2-tap moving average and a 2-tap difference (the Haar scaling \
and wavelet filters) at spacing 1, then repeat the SAME pair of filters on \
the first pair's average, but at spacing 2 instead of 1 (ref(prior, 2) — \
this is the standard "à trous"/stationary wavelet construction: each level \
doubles the tap spacing instead of halving the data, so every level still \
produces one value per bar). Chain as many levels as needed by doubling \
the spacing again each time. See the wavelet worked example below.
- Never fake sophistication: if a request names a specific technique \
(Gaussian, wavelet, Smart Money Concepts, or anything else), build the \
REAL underlying math with the primitives above — never substitute a plain \
ema()/sma() or a generic breakout condition and label it with the \
requested technique's name. The only honest exception is a request that \
genuinely needs data this tool has no access to (another symbol, another \
timeframe, a user-tunable parameter) — say so plainly instead of faking it.
- Point-wise math (+, -, *, /), comparisons, and and/or/not are allowed \
between series and numbers.
- band(upper, lower) takes the UPPER bound first, the LOWER bound second. \
marker(condition, shape, color) takes a boolean condition, then a shape name \
and a color, both quoted strings. background(condition, color) takes a \
boolean condition, then a quoted color string.
- fill(a, b, color): shades the area between two SERIES, for a band-fill or \
channel-fill look. a and b must each be the bare NAME of an already-declared \
formula earlier in the source (e.g. `upper`, `lower`) — not an inline \
expression like `close + 1`. Declare the two series as their own named \
formulas first, then fill(...) between their names. color is a quoted \
string, and can carry alpha for a translucent fill (e.g. "#4CAF3320").
- Do NOT use: series(), input, barcolor(...), or the time/session/symbol \
namespaces. series() needs another symbol's or timeframe's data prefetched \
by the caller, which this tool does not wire up. input needs a UI to let a \
user tune it later, which does not exist here. barcolor recolors the \
candles themselves, which the klinecharts render adapter this feature feeds \
has no integration point for — it parses fine but throws at render time, \
client-side.

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

Input: shade the area between the 20-EMA and the 50-EMA
Output:
PANE: main
upper = ema(close, 20)
lower = ema(close, 50)
result = fill(upper, lower, "#2196F333")

Input: mark bars where RSI crosses above 70 with a red down-triangle
Output:
PANE: sub
result = marker(rsi(close, 14) > 70, "triangle-down", "#F44336")

Input: mark a bullish break of structure when price closes above its recent 10-bar swing high
Output:
PANE: main
prior_high = ref(highest(high, 10), 1)
result = marker(close > prior_high, "triangle-up", "#4CAF50")

Input: a Smart Money Concepts setup — mark a bullish order block break of structure
Output:
PANE: main
swing_high_now = high == highest(high, 10)
last_swing_high = held(swing_high_now, high)
bearish_candle = close < open
last_bear_ob_high = held(bearish_candle, high)
last_bear_ob_low = held(bearish_candle, low)
bos_up = close > ref(last_swing_high, 1) and ref(close, 1) <= ref(last_swing_high, 1)
result = marker(bos_up, "triangle-up", "#4CAF50")

Input: plot the last confirmed swing high as a line on the chart
Output:
PANE: main
swing_high_now = high == highest(high, 10)
has_swing = held(swing_high_now, 1)
last_swing_high = held(swing_high_now, high)
result = line(last_swing_high / has_swing)

Input: a wavelet transform indicator that decomposes price into trend and cycle components
Output:
PANE: main
approx1 = (close + ref(close, 1)) / 2
detail1 = (close - ref(close, 1)) / 2
approx2 = (approx1 + ref(approx1, 2)) / 2
detail2 = (approx1 - ref(approx1, 2)) / 2
result = line(approx2)

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


async def _dynamic_check_via_node(source: str, output_name: str) -> dict:
    """Actually runs the formula against synthetic bar data through the real
    diascript engine — the same one the frontend renders with — catching the
    class of bug `_validate_via_node`'s parse-only check cannot: a formula
    that is grammatically valid but crashes on real bars (a `ref()` with a
    negative, future-looking offset), or is silently dead (a comparison
    against a window that already includes the bar being compared, which can
    never be satisfied). See diascript_dynamic_check.mjs for the scenarios.

    Fails OPEN, unlike `_validate_via_node`: it spawns a second Node
    subprocess to actually execute a formula, a heavier and less certain
    operation than a pure parse. If that tooling is unavailable or misbehaves,
    that must never block indicator generation — only an actual finding from
    a clean run does that.
    """
    executable = shutil.which("node") or "node"
    try:
        proc = await asyncio.create_subprocess_exec(
            executable, str(DYNAMIC_CHECK_SCRIPT), output_name,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError) as e:
        logger.warning("diascript dynamic check unavailable: %s", e)
        return {"valid": True}

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(source.encode()), timeout=DYNAMIC_CHECK_TIMEOUT_SECONDS,
        )
    except Exception as e:
        # Same deliberately-broad catch as _validate_via_node, and the same
        # reason — a subprocess pipe is an OS/event-loop boundary, and uvloop
        # signals a dead child mid-write as a bare RuntimeError, not OSError.
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        logger.warning("diascript dynamic check did not complete: %s", e)
        return {"valid": True}

    if proc.returncode != 0:
        logger.warning("diascript dynamic check crashed: %s", stderr.decode().strip())
        return {"valid": True}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning("diascript dynamic check returned malformed output: %s", stdout.decode()[:200])
        return {"valid": True}


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


async def _check(source: str, output_name: str) -> str | None:
    """None if the formula is fully usable, otherwise the feedback to retry
    with. Two independent gates, in order: does it even parse and name a
    renderable output (`_validate_via_node`), and — only if that's clean —
    does it actually survive being run against real numbers
    (`_dynamic_check_via_node`). The second gate never runs on a formula that
    already failed the first, so a plain syntax mistake never pays for a
    wasted subprocess spawn.
    """
    feedback = _validation_feedback(await _validate_via_node(source, output_name))
    if feedback:
        return feedback
    dynamic = await _dynamic_check_via_node(source, output_name)
    if not dynamic.get("valid"):
        return dynamic.get("error", {}).get("message") or "formula failed a real-data check"
    return None


async def generate_custom_indicator(ctx: ToolContext, args: dict) -> Any:
    description = str(args.get("description") or "").strip()
    if not description:
        return {"error": "A description of the indicator is required."}

    output_name = "result"
    pane, source = await _write_formula(ctx, description)
    feedback = await _check(source, output_name)

    if feedback:
        failed_source = source
        pane, source = await _write_formula(ctx, description, feedback=feedback, source=failed_source)
        feedback = await _check(source, output_name)

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
