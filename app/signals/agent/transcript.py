"""
The message list sent to the model, and what is allowed into it.

Its own module because the transcript is where a turn's cost actually
accumulates. Every tool result appended here is resent on every subsequent
round — a 60-bar candle payload added in round 1 is paid for again in rounds
2, 3, 4, 5 and 6. Nothing used to bound it.

Two rules, both enforced here rather than at each call site:
  * conversation history is capped by turn count before the turn starts;
  * an oversized tool result is replaced by a compact form that keeps the
    findings and drops the rows.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Scalars are the findings; long lists are the raw material the tool already
# reduced for us. When a result must shrink, the lists go first.
_KEEP_LIST_ITEMS = 3


STALE_NOTE = ('{"note": "An earlier tool result was dropped to stay within the context '
              'budget. Its findings are in the conversation above; do not re-request it."}')

# Rough enough. A real tokeniser would be exact and would cost a dependency and
# a call per message; this is used to decide when to shed, not to bill anyone.
CHARS_PER_TOKEN = 4


class Transcript:
    """The messages for one turn, bounded."""

    def __init__(self, system: str, max_tool_result_chars: int = 6_000,
                 max_total_tokens: int | None = None):
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self.max_tool_result_chars = max_tool_result_chars
        # None disables the ceiling — used where the caller has its own budget.
        self.max_total_tokens = max_total_tokens
        self.trimmed_results = 0
        self.dropped_results = 0

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    def __len__(self) -> int:
        return len(self._messages)

    # -- building -----------------------------------------------------------

    def add_history(self, history: list[dict] | None, max_turns: int) -> None:
        """Prior conversation, oldest first, capped to the most recent turns."""
        for item in (history or [])[-max_turns:]:
            role = "assistant" if item.get("role") == "assistant" else "user"
            self._messages.append({"role": role, "content": str(item.get("content", ""))})

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_tool_calls(self, content: str | None, tool_calls) -> None:
        """The assistant turn that requested tools, in the shape the API expects."""
        self._messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })

    def add_tool_result(self, tool_call_id: str, result: Any) -> None:
        payload = json.dumps(result, default=str)
        if len(payload) > self.max_tool_result_chars:
            payload = json.dumps(_shrink(result), default=str)
            self.trimmed_results += 1
            logger.info("Tool result trimmed to fit the transcript budget")
        self._messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": payload})
        self._enforce_ceiling()

    # -- bounding ------------------------------------------------------------

    def approx_tokens(self) -> int:
        """Roughly what this transcript costs to send, once."""
        return sum(len(str(m.get("content") or "")) for m in self._messages) // CHARS_PER_TOKEN

    def _enforce_ceiling(self) -> None:
        """Shed the oldest tool results until the transcript fits.

        Per-result trimming bounds ONE payload; it does nothing about twenty of
        them, and the whole transcript is resent on every round. Oldest first
        because the model has already reasoned over them — its later messages
        carry what it concluded — whereas the newest result is the one it is
        about to use.

        Only tool results are shed. The system prompt, the user's question and
        the model's own turns stay: dropping those would change what was asked.
        """
        if not self.max_total_tokens:
            return

        for message in self._messages:
            if self.approx_tokens() <= self.max_total_tokens:
                return
            if message.get("role") != "tool" or message.get("content") == STALE_NOTE:
                continue
            message["content"] = STALE_NOTE
            self.dropped_results += 1
            logger.info("Dropped an earlier tool result to stay within the context budget")

    def add_instruction(self, text: str) -> None:
        """A steering message from us, not from the user."""
        self._messages.append({"role": "user", "content": text})


def _shrink(result: Any) -> Any:
    """A large tool result reduced to its findings.

    Keeps every scalar — those are the numbers the model reasons with — and
    replaces long lists with their first few items plus a count, so the model
    can still see the shape of what it asked for and knows it was truncated.
    """
    if not isinstance(result, dict):
        return {"note": "result too large to include in full"}

    out: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, list) and len(value) > _KEEP_LIST_ITEMS:
            out[key] = value[:_KEEP_LIST_ITEMS]
            out[f"{key}_omitted"] = len(value) - _KEEP_LIST_ITEMS
        elif isinstance(value, (dict, list)):
            out[key] = value
        else:
            out[key] = value
    out["note"] = ("Trimmed to fit the context budget — the omitted rows are raw data, "
                   "not findings. Do not re-request them; work with what is here.")
    return out
