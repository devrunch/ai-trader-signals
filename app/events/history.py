"""Past prints of a given release, from the calendar history on disk.

A CSV rather than a database because it is read a few times a day, never
written, and shipping it as a file means the reaction study works on a fresh
box with no migration. The file is the published ForexFactory calendar export
(2007 onward); rows are matched on currency and event name, which is how the
same release is identified across years.
"""
from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

csv.field_size_limit(10 ** 7)

# Outside the repo: 66MB of history is data, not source. deploy.sh puts it
# here; a box without it degrades to briefs with no measured reaction rather
# than failing.
HISTORY_PATH = Path.home() / "ai-trader" / "data" / "ff_history.csv"


@lru_cache(maxsize=64)
def past_prints(currency: str, title: str, limit: int = 24) -> tuple[dict, ...]:
    """The most recent rows for one release, newest first.

    Empty when the file is missing -- the caller reports "no measured history"
    rather than treating absence as a flat reaction.
    """
    if not HISTORY_PATH.exists():
        logger.warning("Calendar history not on this box (%s)", HISTORY_PATH)
        return ()

    rows = []
    try:
        with HISTORY_PATH.open(encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                if row.get("Currency") == currency and row.get("Event") == title:
                    rows.append(row)
    except OSError as e:
        logger.warning("Could not read calendar history: %s", e)
        return ()

    rows.sort(key=lambda r: r.get("DateTime") or "", reverse=True)
    return tuple(rows[:limit])
