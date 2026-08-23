"""
Custom-indicator authoring: the model writes Pine Script, a source-text gate
rejects calls the render pipeline can't show (see pine_validation.py), then
the real sandbox (app/pine_sandbox/) actually runs it against synthetic bars
through the real PineTS engine — a script can be syntactically valid Pine
and still crash on real data or produce no plot() output at all, and only
running it catches that. One retry with the real feedback if either gate
fails. Rendering happens client-side, via pine-render.ts's
attachPinePlotsToPane, once LightweightChartsAdapter.attachPineIndicator
calls the same /api/pine/run endpoint this validation already exercises.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any

from app.signals.agent.tools.base import Handler, ToolContext
from app.signals.agent.tools.pine_validation import check_pine_source, synthetic_bars

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"^```[^\n]*\n?(.*?)\n?```$", re.DOTALL)
_PANE_LINE_RE = re.compile(r"^PANE:\s*(main|sub)\s*$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = """You write Pine Script (v5) source for a single technical \
indicator. Output exactly two things, nothing else — no prose, no extra \
markdown fences, no explanation:

1. A first line stating where this indicator belongs on the chart: \
`PANE: main` if it overlays directly on the price chart, same scale as price \
(moving averages, bands, trend/smoothing filters, price channels) — or \
`PANE: sub` if it needs its own separate pane below the price chart, a \
different scale than price (oscillators, momentum indicators, anything \
volume-based, anything bounded like 0-100).
2. Then the Pine source on the following line(s), starting with //@version=5 \
and an indicator(...) declaration.

Rules:
- Every plot() call needs a distinct title string as its second argument — \
the render pipeline keys output series by that title, not by variable name. \
Two or more plot() calls are fine (e.g. a band's two edges).
- For a band or channel (an upper and a lower line meant to be shaded \
between), name the two plot() titles "<Name> Upper" and "<Name> Lower" \
exactly — that naming is what the render pipeline uses to draw a filled \
band instead of two independent lines. This is the ONLY way to get a \
band/channel look; there is no other mechanism today.
- plot() is renderable, and so are fill() and plotshape() — both render for \
real now. fill(p1, p2, color) shades the region between two plot()s you've \
already made (reference them by the variable each plot() call returns, e.g. \
`p1 = plot(...)`, then `fill(p1, p2, color=...)`). plotshape() marks bars \
matching a boolean condition (e.g. a crossover) with a marker; give it a \
clear title like "Buy" or "Sell" so it reads correctly on the chart. Do NOT \
use bgcolor(), plotchar(), or plotarrow() — those still have no renderer.
- input.*() is renderable now too — the chart has a real settings panel \
(a gear icon on the indicator's legend row) built from exactly the \
input.*() calls in your script, letting the user tune a value after you've \
written it. Use input.int()/input.float()/input.bool() for anything a user \
might reasonably want to adjust (a length, a threshold, a toggle) instead \
of hardcoding it — give each one a clear title=. Values you assign a \
variable from input.*() work exactly like a plain number/bool everywhere \
else in the script.
- Do NOT use: request.security() (needs another symbol's/timeframe's data \
this tool does not prefetch), strategy.*() (this tool authors INDICATORS, \
never strategies — a different, separate capability).
- Pine has real for loops and real mutable state (var, :=) — unlike some \
formula languages, a fixed-window weighted average (a Gaussian filter, \
ALMA) does not need its taps unrolled by hand; write a real for loop over \
the window and accumulate the weighted sum, exactly like the Gaussian \
worked example below.
- Never fake sophistication: if a request names a specific technique \
(Gaussian, wavelet, Smart Money Concepts, or anything else), build the \
REAL underlying math — never substitute a plain sma()/ema() or a generic \
breakout condition and label it with the requested technique's name. The \
only honest exception is a request that genuinely needs data this tool has \
no access to (another symbol, another timeframe, a user-tunable parameter) \
— say so plainly instead of faking it.

Examples:

Input: the 20-EMA minus the 50-EMA
Output:
PANE: sub
//@version=5
indicator("EMA Diff", overlay=false)
plot(ta.ema(close, 20) - ta.ema(close, 50), "EMA Diff")

Input: RSI with a 21-period length
Output:
PANE: sub
//@version=5
indicator("RSI 21", overlay=false)
plot(ta.rsi(close, 21), "RSI")

Input: a Gaussian filter trend indicator
Output:
PANE: main
//@version=5
indicator("Gaussian Filter", overlay=true)
length = 9
sigma = 3.0
sum_w = 0.0
sum_wv = 0.0
for k = 0 to length - 1
    w = math.exp(-(k * k) / (2 * sigma * sigma))
    sum_w += w
    sum_wv += w * close[k]
plot(sum_wv / sum_w, "Gaussian")

Input: a band around price at 2 standard deviations
Output:
PANE: main
//@version=5
indicator("StdDev Band", overlay=true)
dev = ta.stdev(close, 20) * 2
plot(close + dev, "Band Upper")
plot(close - dev, "Band Lower")

Input: a wavelet transform indicator that decomposes price into trend and cycle components
Output:
PANE: main
//@version=5
indicator("Wavelet Trend", overlay=true)
// à trous / stationary wavelet construction: a 2-tap moving average and a
// 2-tap difference (the Haar scaling and wavelet filters) at spacing 1,
// then the SAME pair of filters repeated on the first pair's average, at
// spacing 2 instead of 1 -- each level doubles the tap spacing rather than
// halving the data, so every level still produces one value per bar.
approx1 = (close + close[1]) / 2
approx2 = (approx1 + approx1[2]) / 2
plot(approx2, "Trend")

Input: plot the last confirmed swing high as a line on the chart
Output:
PANE: main
//@version=5
indicator("Swing High", overlay=true)
isSwingHigh = high == ta.highest(high, 10)
var float lastSwingHigh = na
if isSwingHigh
    lastSwingHigh := high
plot(lastSwingHigh, "Last Swing High")
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


async def generate_custom_indicator(ctx: ToolContext, args: dict) -> Any:
    description = str(args.get("description") or "").strip()
    if not description:
        return {"error": "A description of the indicator is required."}

    # Live investigation gap, found this session: a real failure (3 calls,
    # all rejected) had no way to diagnose after the fact -- neither the
    # analyst's own `description` argument (its paraphrase of what the user
    # actually asked, which can drift from the original request) nor the
    # rejected source/feedback were logged anywhere. A clean re-run of the
    # exact same request succeeded 5/5 times in isolation, which points at
    # the paraphrase (or something else about the live turn) rather than
    # the model's raw Pine-writing ability -- but that's inference, not
    # evidence, without this logging.
    logger.info("generate_custom_indicator: description=%r", description[:300])
    bars = synthetic_bars()
    pane, source = await _write_formula(ctx, description)
    feedback = await check_pine_source(source, bars)

    if feedback:
        logger.warning("generate_custom_indicator attempt 1 rejected: %s\nsource:\n%s", feedback, source)
        failed_source = source
        pane, source = await _write_formula(ctx, description, feedback=feedback, source=failed_source)
        feedback = await check_pine_source(source, bars)

    if feedback:
        logger.warning("generate_custom_indicator attempt 2 rejected: %s\nsource:\n%s", feedback, source)
        return {"error": f"Could not build a valid indicator: {feedback}"}

    display_label = str(args.get("label") or description)[:60]
    # Named from the source's own content, not a per-turn counter: a counter
    # reset fresh on every HTTP request meant turn 2's first indicator reused
    # turn 1's id, and the frontend dedupes on id — so every turn after the
    # first silently lost its custom indicator. Content hashing makes
    # identical scripts idempotent and different scripts always distinct,
    # regardless of which turn produced them.
    indicator_id = f"pine_{hashlib.sha1(source.encode()).hexdigest()[:8]}"

    ctx.results.setdefault("custom_indicators", []).append({
        "id": indicator_id, "source": source, "label": display_label, "pane": pane,
    })
    return {"created": indicator_id, "label": display_label, "pane": pane}


TOOLS: dict[str, Handler] = {
    "generate_custom_indicator": generate_custom_indicator,
}
