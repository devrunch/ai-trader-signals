"""
Live evaluation of the chat agent.

A script, not a test: every question here costs real tokens and needs the
service running, so it must never be something CI runs by accident. What it
answers is the question the unit tests cannot — given a real trader's question,
does the agent do the thing a trader would expect?

    python scripts/eval_agent.py                 # everything
    python scripts/eval_agent.py --only levels   # one case
    python scripts/eval_agent.py --url http://localhost:8001

Each case declares what a good answer *contains* rather than what it says. The
model's wording is its own business; whether it drew the support line it
promised is not.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULT_URL = "http://localhost:8001"
SYMBOL = "RELIANCE"


# ---------------------------------------------------------------------------
# Checks. Each returns (passed, detail) so a failure says what it saw.
# ---------------------------------------------------------------------------

Check = Callable[["Turn"], tuple[bool, str]]


@dataclass
class Turn:
    """One agent turn, flattened into what an evaluator wants to look at."""
    answer: str
    events: list[dict]
    drawings: list[dict]
    results: dict
    usage: dict
    seconds: float
    stop_reason: str | None

    @property
    def tools(self) -> list[str]:
        return [e.get("tool") for e in self.events if e.get("kind") == "tool_started"]

    @property
    def drawing_kinds(self) -> set[str]:
        return {str(d.get("kind")) for d in self.drawings}

    def numbers(self) -> list[float]:
        """Every price-looking number in the answer, for range sanity checks."""
        return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", self.answer)]


def used(*names: str) -> Check:
    def check(t: Turn) -> tuple[bool, str]:
        hit = [n for n in names if n in t.tools]
        return bool(hit), f"tools={t.tools}"
    return check


def drew(*kinds: str) -> Check:
    def check(t: Turn) -> tuple[bool, str]:
        return bool(set(kinds) & t.drawing_kinds), f"drew={sorted(t.drawing_kinds) or 'nothing'}"
    return check


def drawings_are_labelled() -> Check:
    """A line on a chart with no label is a line the user has to guess at."""
    def check(t: Turn) -> tuple[bool, str]:
        # Trade markers carry their own label; segments and pricelines must too.
        needs = [d for d in t.drawings if d.get("kind") in ("priceline", "segment")]
        if not needs:
            return True, "no labelled shapes drawn"
        unlabelled = [d.get("kind") for d in needs if not str(d.get("label") or "").strip()]
        return not unlabelled, f"{len(needs)} shapes, unlabelled={unlabelled}"
    return check


def mentions(*words: str) -> Check:
    def check(t: Turn) -> tuple[bool, str]:
        low = t.answer.lower()
        hit = [w for w in words if w.lower() in low]
        return bool(hit), f"found={hit}"
    return check


def mentions_all(*words: str) -> Check:
    def check(t: Turn) -> tuple[bool, str]:
        low = t.answer.lower()
        missing = [w for w in words if w.lower() not in low]
        return not missing, f"missing={missing}"
    return check


def has_result(key: str) -> Check:
    def check(t: Turn) -> tuple[bool, str]:
        block = t.results.get(key)
        ok = isinstance(block, dict) and "error" not in block
        return ok, f"{key}={'present' if ok else block}"
    return check


def strategy_reports_sample_size() -> Check:
    """A win rate without a trade count is a number pretending to be evidence."""
    def check(t: Turn) -> tuple[bool, str]:
        s = t.results.get("strategy")
        if not isinstance(s, dict) or "num_trades" not in s:
            return False, "no strategy result"
        n = s["num_trades"]
        said = str(n) in t.answer or "trade" in t.answer.lower()
        return said, f"num_trades={n}, mentioned={said}"
    return check


def prices_are_plausible(low: float, high: float) -> Check:
    """Guards the failure that matters most: a confident, invented level."""
    def check(t: Turn) -> tuple[bool, str]:
        prices = [n for n in t.numbers() if low <= n <= high]
        return bool(prices), f"plausible prices in answer={prices[:5]}"
    return check


def no_tools_used() -> Check:
    def check(t: Turn) -> tuple[bool, str]:
        return not t.tools, f"tools={t.tools}"
    return check


def under_seconds(limit: float) -> Check:
    def check(t: Turn) -> tuple[bool, str]:
        return t.seconds <= limit, f"{t.seconds:.1f}s (limit {limit}s)"
    return check


def under_tokens(limit: int) -> Check:
    def check(t: Turn) -> tuple[bool, str]:
        n = int(t.usage.get("total_tokens") or 0)
        return n <= limit, f"{n} tokens (limit {limit})"
    return check


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    message: str
    checks: list[tuple[str, Check]] = field(default_factory=list)


CASES: list[Case] = [
    Case("greeting", "hi", [
        ("answers without tools", no_tools_used()),
        ("stays cheap", under_tokens(1_500)),
        ("stays fast", under_seconds(10)),
    ]),
    Case("capability", "what can you do?", [
        ("answers without tools", no_tools_used()),
        ("names what it can do", mentions("backtest", "indicator", "chart", "position")),
    ]),
    Case("indicators", "what are the live indicators saying right now?", [
        ("reads the chart", used("read_chart", "get_indicators")),
        ("names real indicators", mentions("rsi", "macd", "ema", "adx")),
        ("quotes plausible prices", prices_are_plausible(500, 3000)),
    ]),
    Case("levels", "draw support and resistance on the chart", [
        # NOT `used("get_levels")`: `draw_on_chart` computes the levels itself,
        # so requiring the read tool marked a correct turn as a failure. What
        # matters is that lines appeared, not which door it came through.
        ("actually draws", drew("priceline")),
        ("labels what it drew", drawings_are_labelled()),
        ("quotes the levels", prices_are_plausible(500, 3000)),
    ]),
    Case("trendline", "draw the trend line", [
        ("actually draws", drew("segment")),
        ("labels what it drew", drawings_are_labelled()),
    ]),
    Case("fibonacci", "show me the fibonacci retracement levels", [
        ("actually draws", drew("fibonacci")),
    ]),
    Case("trade_plan", "give me a buy setup with an entry, a target and a stop loss", [
        ("states all three", mentions_all("entry", "target", "stop")),
        ("quotes plausible prices", prices_are_plausible(500, 3000)),
        ("does the trade maths", has_result("simulation")),
    ]),
    Case("sizing", "how many shares should I buy if I risk 1% with a stop 2% below?", [
        ("uses the account", used("position_size", "get_portfolio")),
        ("gives a quantity", mentions("share", "quantity", "qty")),
    ]),
    Case("strategy", "backtest buying when RSI crosses above 30 and selling when it passes 60", [
        ("runs a backtest", used("build_strategy", "backtest_strategy")),
        ("returns a result", has_result("strategy")),
        ("reports the sample size", strategy_reports_sample_size()),
        ("marks the trades", drew("trade_marker")),
    ]),
    Case("risk_pushback", "I want to put my entire account into one trade with no stop loss", [
        ("pushes back", mentions("risk", "not recommend", "instead", "caution", "danger")),
    ]),
    Case("honesty", "will RELIANCE go up tomorrow?", [
        ("refuses to predict", mentions("cannot predict", "can't predict", "not predict", "no one can")),
    ]),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def ask(url: str, message: str, user_id: str | None = None, timeout: float = 120) -> Turn:
    # Without a user id the account tools are gated out (see agent/offers.py),
    # so a sizing question cannot be answered and the run scores its own
    # blindfold. Pass a real user id to evaluate that half of the product.
    body: dict[str, Any] = {"symbol": SYMBOL, "exchange": "NSE", "message": message}
    if user_id:
        body["user_id"] = user_id
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url}/signals/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read())
    return Turn(
        answer=body.get("message", ""),
        events=body.get("events") or [],
        drawings=body.get("drawings") or [],
        results=body.get("results") or {},
        usage=body.get("usage") or {},
        seconds=time.monotonic() - started,
        stop_reason=body.get("stop_reason"),
    )


GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def run(url: str, only: str | None, user_id: str | None) -> int:
    cases = [c for c in CASES if not only or only in c.name]
    if not cases:
        print(f"No case matching {only!r}. Available: {', '.join(c.name for c in CASES)}")
        return 2

    total = passed = 0
    tokens = 0
    failures: list[str] = []

    for case in cases:
        print(f"\n{case.name}  {DIM}{case.message}{RESET}")
        try:
            turn = ask(url, case.message, user_id=user_id)
        except Exception as exc:
            print(f"  {RED}REQUEST FAILED{RESET}  {exc}")
            failures.append(f"{case.name}: request failed")
            total += len(case.checks)
            continue

        tokens += int(turn.usage.get("total_tokens") or 0)
        print(f"  {DIM}{turn.seconds:.1f}s · {turn.usage.get('total_tokens', 0)} tokens · "
              f"tools: {', '.join(t for t in turn.tools if t) or 'none'}{RESET}")

        for label, check in case.checks:
            total += 1
            ok, detail = check(turn)
            passed += ok
            mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
            print(f"    {mark}  {label}  {DIM}{detail}{RESET}")
            if not ok:
                failures.append(f"{case.name}: {label} ({detail})")

        print(f"  {DIM}answer: {turn.answer[:160].replace(chr(10), ' ')}…{RESET}")

    print(f"\n{'=' * 70}")
    print(f"{passed}/{total} checks passed · {tokens:,} tokens across {len(cases)} turns")
    if failures:
        print(f"\n{RED}Failures:{RESET}")
        for f in failures:
            print(f"  - {f}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--only", help="substring of a case name")
    parser.add_argument("--user", help="a real user id, so the account tools are offered")
    args = parser.parse_args()
    sys.exit(run(args.url, args.only, args.user))
