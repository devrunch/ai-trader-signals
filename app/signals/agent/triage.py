"""
The front desk.

A prompt telling the analyst "don't use tools for a greeting" is a suggestion,
and the model is free to ignore it — asked "hi", it ran four rounds and spent
14,299 tokens before saying hello back. Telling it harder does not fix that,
because the tools are still on the table.

So this is a separate, smaller agent that is handed **no tool schemas at all**.
It cannot call a tool, however it feels about the question. It either answers in
a sentence or hands off to the analyst, and that is the whole of its authority.

Two agents rather than one prompt, for a reason worth stating: the cost of the
analyst is not its reasoning, it is the ~2,400 tokens of tool schemas resent on
every round. An agent that is never offered them costs a few hundred tokens per
turn, so triage pays for itself the first time someone says thanks.

Deliberately NOT a keyword rule. "How are you" and "how is RELIANCE" differ by
one word, and a rule that misroutes a real question into a chat reply is far
worse than one wasted turn — the user asked about their money and got small
talk. A model reads intent; a regex reads spelling.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The analyst is the default. Triage only diverts what it is confident about.
HANDOFF = "HANDOFF"

# Triage is handed no tools, ever — that is the entire security property this
# module exists for. Its own prompt already says a backtest, a strategy, a
# specific trade, or anything drawn on the chart needs the analyst; this
# catches the turns where it answers anyway instead of complying. Anything
# matching HAS to be invented, because nothing on this path computed or drew
# it. Same "validate before trusting" pattern as an unknown series name in
# plot_series — the model got clear instructions, this is what happens if
# they were not followed.
_FABRICATED_RESULT = re.compile(
    r"\b(win rate|total trades|winning trades|losing trades|average win|average loss|"
    r"total return|backtest results?|trade count|num_trades|entry signals|exit signals|"
    r"blocks? marked|arrows? (on|plotted)|rectangles? (on|marked)|zones? marked|"
    r"marked on (the )?chart|(added|drawn) to (the )?chart)\b",
    re.IGNORECASE,
)

TRIAGE_SYSTEM = (
    "You are the front desk of a trading terminal. The user is looking at a chart of "
    "{symbol} ({exchange}). Decide who should answer.\n\n"
    "Reply with exactly the word HANDOFF, and nothing else, if answering would need any of:\n"
    "  - market data, prices, candles, indicators, support/resistance, trends\n"
    "  - adding, plotting, marking, or changing ANYTHING on the chart itself — an indicator, "
    "a drawing, a marker, a zone, a line, by any name the user gives it (Smart Money Concepts, "
    "order blocks, Gaussian filter, or anything else) — even if you think you know what it would "
    "look like. You have no chart tools; only the analyst can actually draw something\n"
    "  - the user's portfolio, positions, cash, risk or position sizing\n"
    "  - a backtest, a strategy, or the maths of a specific trade\n"
    "  - an opinion on what {symbol} or any other instrument is doing or might do\n\n"
    "Otherwise answer it yourself, in one to three sentences. You handle greetings, "
    "thanks, small talk, and questions about what this product can do.\n\n"
    "What this product does: it is an AI analyst on Indian equities (NSE/BSE). It reads "
    "charts, computes indicators, finds levels, draws on the chart, backtests rules the "
    "user describes, and sizes positions against their paper account. All trading is "
    "simulated with paper money. It cannot place real orders and does not predict prices.\n\n"
    "If you are unsure, reply HANDOFF. Being unhelpful is a smaller failure than "
    "answering a question about the user's money without looking anything up."
)


@dataclass
class TriageResult:
    """What the front desk decided, and what deciding it cost.

    The response comes back with the answer because triage is a real LLM call
    and has to be paid for: a turn it handles is cheap, not free, and one that
    did not count against the daily budget would be a way to spend without
    being charged.
    """
    answer: str | None
    response: Any = None

    @property
    def handled(self) -> bool:
        return self.answer is not None


def triage(llm, symbol: str, exchange: str, message: str,
           history: list[dict] | None = None) -> TriageResult:
    """A direct answer, or a handoff to the analyst.

    Never raises: triage is an optimisation, and a failing optimisation must
    fall through to the thing it was optimising rather than break the turn.
    """
    messages: list[dict] = [
        {"role": "system", "content": TRIAGE_SYSTEM.format(symbol=symbol, exchange=exchange)}
    ]
    # The last exchange only. Triage decides about THIS message; more history
    # costs tokens and lets an earlier analysis pull a plain thank-you back into
    # the analyst.
    for item in (history or [])[-2:]:
        role = "assistant" if item.get("role") == "assistant" else "user"
        messages.append({"role": role, "content": str(item.get("content", ""))})
    messages.append({"role": "user", "content": message})

    try:
        # No `tools` argument. That is the entire mechanism — this agent cannot
        # call a tool because it was never told any exist.
        response = llm.chat(temperature=0, max_tokens=160, messages=messages)
        text = (response.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("Triage failed for %s — handing off to the analyst", symbol)
        return TriageResult(None)

    if not text or _is_handoff(text):
        # Still carries the response: the handoff cost tokens too, and the
        # analyst's usage should be added to it rather than replace it.
        return TriageResult(None, response)
    if _FABRICATED_RESULT.search(text):
        logger.warning("Triage answered with fabricated-result-shaped text — forcing handoff instead")
        return TriageResult(None, response)
    return TriageResult(text, response)


def _is_handoff(text: str) -> bool:
    """Whether the model asked to hand off.

    Tolerant of the model dressing the word up ("HANDOFF." / "handoff") but not
    of it appearing mid-sentence, which would be the model talking *about*
    handing off rather than doing it.
    """
    stripped = text.strip().strip(".").strip('"').strip()
    return stripped.upper() == HANDOFF
