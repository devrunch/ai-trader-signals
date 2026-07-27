"""
Which tools to advertise on a given round.

Every schema sent is paid for on every round: the full set serialises to ~3,400
tokens, so a turn running its round cap spends ~20,000 tokens on schemas alone,
before a single candle is fetched.

The rules here gate on FACTS, never on a guess about what the user meant. A
keyword rule that hides `build_strategy` because the question did not say
"backtest" would silently disable a tool the model was about to need, and the
failure would look like the model being unhelpful. Two facts qualify:

  * there is no authenticated user, so the account tools can only fail;
  * a tool has used its per-turn budget, so it can no longer run.

Both are things we know, and in both cases advertising the tool costs tokens to
offer something that cannot work.
"""
from __future__ import annotations

from app.signals.agent.tools import account
from app.signals.agent.tools import market as market_tools

# Tools that read the user's book. Without a user id they return an error, and
# the model spends a whole round discovering that.
NEEDS_ACCOUNT: frozenset[str] = frozenset(account.TOOLS) | {"scan_watchlist"}

assert "scan_watchlist" in market_tools.TOOLS, "scan_watchlist moved — update NEEDS_ACCOUNT"


def schemas_for(
    schemas: list[dict],
    *,
    user_id: str | None,
    exhausted: frozenset[str] | set[str] = frozenset(),
) -> list[dict]:
    """The subset worth offering, in the original order."""
    blocked = set(exhausted)
    if not user_id:
        blocked |= NEEDS_ACCOUNT

    if not blocked:
        return schemas

    kept = [s for s in schemas if _name(s) not in blocked]
    # Never advertise nothing: a tool_choice of "auto" with an empty tool list
    # is not a meaningful request, and some endpoints reject it outright.
    return kept or schemas


def _name(schema: dict) -> str:
    return str(schema.get("function", {}).get("name", ""))
