"""
Custom-indicator authoring: the model writes diascript, a real parser
validates it, one retry with the real error if it fails. Nothing here
executes diascript — only ever a subprocess parse check. Rendering happens
entirely client-side, via the same registerDiascriptIndicator pipeline that
already draws DIA_EMA20/DIA_RSI14.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.signals.agent.tools.base import Handler, ToolContext

logger = logging.getLogger(__name__)

VALIDATE_TIMEOUT_SECONDS = 5

SYSTEM_PROMPT = """You write diascript — a small, safe formula language for \
technical indicators. Output ONLY diascript source, nothing else: no prose, \
no markdown fences, no explanation.

Rules:
- Exactly one formula, always named `result`.
- It must be wrapped in an output function: line(...), band(...), \
marker(...), histogram(...), barcolor(...), or background(...).
- Available series: open, high, low, close, volume.
- Available functions: sma(series, length), ema(series, length), \
wma(series, length), stdev(series, length), highest(series, length), \
lowest(series, length), sum(series, length), rsi(series, length), \
true_range(), typical_price(), abs(x), min(a, b), max(a, b), \
ref(series, n) (value n bars back, null before history starts), \
prev(n) (this same formula's own value n bars back).
- Point-wise math (+, -, *, /), comparisons, and and/or/not are allowed \
between series and numbers.
- Do NOT use: held(), series(), input, fill(...), or the time/session/symbol \
namespaces — none of these are needed for a single indicator formula (fill \
needs two already-declared formulas to fill between, which a single-formula \
file can never have), and using them raises the chance of a subtle scoping \
mistake.

Examples:

Input: the 20-EMA minus the 50-EMA
Output:
result = line(ema(close, 20) - ema(close, 50))

Input: RSI with a 21-period length
Output:
result = line(rsi(close, 21))

Input: a band around price at 2 standard deviations
Output:
result = band(close - stdev(close, 20) * 2, close + stdev(close, 20) * 2)
"""


async def _write_formula(description: str, feedback: str | None = None) -> str:
    from app.llm.client import get_llm

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "user",
        "content": description if not feedback else f"{description}\n\nThat failed to validate: {feedback}\nFix it.",
    })

    llm = get_llm()
    response = await asyncio.to_thread(llm.chat, temperature=0, max_tokens=300, messages=messages)
    return (response.choices[0].message.content or "").strip()


async def _validate_via_node(source: str, output_name: str) -> dict:
    try:
        proc = await asyncio.create_subprocess_exec(
            "diascript-validate", "--output", output_name,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(source.encode()), timeout=VALIDATE_TIMEOUT_SECONDS,
        )
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
        logger.warning("diascript-validate unavailable: %s", e)
        return {"valid": False, "error": {"message": "validator unavailable"}}

    if proc.returncode != 0:
        return {"valid": False, "error": {"message": stderr.decode().strip() or "validator crashed"}}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"valid": False, "error": {"message": "validator returned malformed output"}}


async def generate_custom_indicator(ctx: ToolContext, args: dict) -> Any:
    description = str(args.get("description") or "").strip()
    if not description:
        return {"error": "A description of the indicator is required."}

    output_name = "result"
    source = await _write_formula(description)
    result = await _validate_via_node(source, output_name)

    if not result.get("valid"):
        first_error = result.get("error", {}).get("message", "invalid formula")
        source = await _write_formula(description, feedback=first_error)
        result = await _validate_via_node(source, output_name)

    if not result.get("valid"):
        final_error = result.get("error", {}).get("message", "invalid formula")
        return {"error": f"Could not build a valid indicator: {final_error}"}

    seq = ctx.results.get("_custom_indicator_seq", 0) + 1
    ctx.results["_custom_indicator_seq"] = seq
    display_label = str(args.get("label") or description)[:60]
    indicator_name = f"DIA_CUSTOM_{seq}"

    ctx.results.setdefault("custom_indicators", []).append({
        "name": indicator_name, "source": source,
        "outputName": output_name, "displayLabel": display_label,
    })
    return {"created": indicator_name, "label": display_label}


TOOLS: dict[str, Handler] = {
    "generate_custom_indicator": generate_custom_indicator,
}
