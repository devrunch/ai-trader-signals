"""
One chat turn, as an object.

The orchestrator used to be a single function holding the messages, both
budgets, the toolbox, the recorder and two fallback strings in local variables.
Nothing could be inspected, asserted on, or persisted without running the whole
loop against a live model.

`TurnState` is that state, named. The loop reads and updates it; `finalise`
turns it into the response and, later, into the stored turn document — the two
are deliberately the same object, so what the user was shown and what was
recorded cannot diverge.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from app.signals.agent.budget import Budget
from app.signals.agent.events import TurnRecorder
from app.signals.agent.transcript import Transcript

# What the user sees when the model produced nothing usable at all. Never a
# stack trace, never an apology for a failure that did not happen.
NO_CONCLUSION = "I wasn't able to reach a conclusion on that one."
FAILED = "I couldn't complete that analysis - please try rephrasing."


def new_turn_id() -> str:
    """A sortable, unguessable id for one turn.

    Minted here rather than by whichever caller happens to store the turn, so
    the id exists inside the event stream from the first event — the streamed
    path, the buffered path and the stored record all name the same turn.

    Time-prefixed so a plain lexical sort is chronological (listing a session's
    turns needs no secondary index), and random-suffixed so an id cannot be
    guessed from a neighbouring one — a turn id will later be the handle for
    "why was this trade taken", which is another user's data if guessable.
    """
    return f"{int(time.time() * 1000):013x}{secrets.token_hex(8)}"


@dataclass
class TurnState:
    turn_id: str
    symbol: str
    exchange: str
    last_price: float
    transcript: Transcript
    box: Any                       # AgentToolbox
    recorder: TurnRecorder
    budget: Budget
    settings: Any
    # Whether there is an authenticated user decides whether the account tools
    # are worth advertising at all — without one they can only fail.
    user_id: str | None = None

    # A clean final message from the model, if it gave one.
    final_text: str = ""
    # Some models emit prose alongside tool calls. Kept as a fallback so an
    # answer is never lost entirely, but a tool-free final message wins.
    partial_text: str = ""
    stop_reason: str | None = None
    failed: bool = False
    tools_called: list[str] = field(default_factory=list)

    @property
    def answer(self) -> str:
        if self.failed:
            return FAILED
        return self.final_text or self.partial_text or NO_CONCLUSION

    def to_result(self) -> dict[str, Any]:
        """The response body, and the turn record. One shape, one source."""
        return {
            "turn_id": self.turn_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "message": self.answer,
            "drawings": self.box.drawings,
            "results": self.box.results,
            "events": self.recorder.to_list(),
            "usage": self.budget.to_dict(),
            "stop_reason": self.stop_reason,
        }
