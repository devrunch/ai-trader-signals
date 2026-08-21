"""Single source of truth for interval→history-window questions.

Three divergent tables existed (service.py, agent_tools.py, yfinance_provider.py)
answering two different questions and disagreeing on three of four entries.

The two questions are genuinely different and are kept apart here:

  * ``DEFAULT_DAYS``    — how much history we *want* for an interval. A product
                          decision: enough bars for a 200-period EMA to warm up
                          without paying for history nobody looks at.
  * ``VENDOR_MAX_DAYS`` — how far back the vendor will actually go. A vendor
                          fact, not a choice. Exceeding it is an error, not a
                          truncation, so callers must clamp before requesting.

Callers that ask "how many days?" want :func:`default_days`. Providers that ask
"is this range legal?" want :func:`clamp_days`. Nothing should re-declare either
table locally.
"""
from __future__ import annotations

# How much history to REQUEST for a given interval (product decision)
DEFAULT_DAYS: dict[str, int] = {"1m": 5, "5m": 10, "15m": 20, "30m": 30, "1h": 60, "1d": 400}

# How far back the vendor actually allows, with a safety margin — yfinance
# rejects a range that touches the exact documented boundary.
# "30m" sits between 15m and 1h below, same as the interval itself does --
# not vendor-confirmed the way the others were (they came from empirical
# tuning against real requests); revisit if real usage ever hits this ceiling.
VENDOR_MAX_DAYS: dict[str, int] = {"1m": 6, "5m": 58, "15m": 58, "30m": 100, "1h": 720}

# Used for intervals we have no entry for. The default window is deliberately
# modest (an unknown interval is more likely intraday than daily); the vendor
# ceiling is deliberately huge so an unknown interval is never silently clipped
# to a few days — daily/weekly bars have no practical vendor limit.
_FALLBACK_DEFAULT_DAYS = 20
_NO_VENDOR_LIMIT = 100_000


def default_days(interval: str) -> int:
    """How many days of history to request for `interval`."""
    return DEFAULT_DAYS.get(interval, _FALLBACK_DEFAULT_DAYS)


def clamp_days(interval: str, days: int) -> int:
    """Clamp a requested window to what the vendor will serve for `interval`.

    Always returns at least 1 — a zero or negative window is a caller bug that
    would otherwise become an empty, silently-wrong DataFrame.
    """
    return max(1, min(days, VENDOR_MAX_DAYS.get(interval, _NO_VENDOR_LIMIT)))
