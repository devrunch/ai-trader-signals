"""Walking a vendor's history backwards, once, for every vendor that needs it.

Some vendors cannot be asked for a range. Deriv serves whatever exists in
``[end - count*granularity, end]``, clamps ``count`` to 1000 and ignores
``start`` entirely, so "five days of 1m bars" is eight requests, not one.

Two things make that loop worth writing once rather than per provider:

* It must step on wall-clock time, not on what came back. A closed market
  answers with nothing, and reading that as "history ends here" is what left
  every 1m forex chart blank all weekend.
* A page that fails and a page that is legitimately empty are different, and
  the caller needs to be able to tell the difference afterwards.
"""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import pandas as pd

# A page is one vendor round trip. Ten covers five days of 1m bars against a
# 1000-bar page size, which is the heaviest window the terminal offers.
DEFAULT_MAX_PAGES = 10

FetchPage = Callable[[int, int], Awaitable["pd.DataFrame | None"]]


@dataclass(frozen=True)
class Walk:
    """What the walk found, and whether the vendor misbehaved while finding it.

    ``df`` empty with ``vendor_failed`` False is a real answer: the range holds
    no bars. Empty with ``vendor_failed`` True is the vendor letting us down,
    and the two must never be reported to a user as the same thing.
    """

    df: pd.DataFrame
    vendor_failed: bool


async def walk_back(
    fetch: FetchPage,
    *,
    granularity_seconds: int,
    span_days: int,
    page_bars: int,
    max_pages: int = DEFAULT_MAX_PAGES,
    now: int | None = None,
) -> Walk:
    """Page backwards from now until `span_days` is covered.

    `fetch(end_epoch, count)` returns that page's bars, or None if the vendor
    failed. Frames are concatenated and deduplicated on their index, so an
    overlap between pages costs nothing.
    """
    now = int(time.time()) if now is None else now
    oldest_wanted = now - span_days * 86400
    window = page_bars * granularity_seconds
    end = now
    frames: list[pd.DataFrame] = []
    failed = False

    for _ in range(max_pages):
        page = await fetch(end, page_bars)
        if page is None:
            failed = True
            break
        if not page.empty:
            frames.append(page)
        end -= window
        # Keep going past the requested span while nothing has been found at
        # all: a day of 1m bars asked for on a Sunday is entirely closed
        # market, and the last session is one window further back.
        if end <= oldest_wanted and frames:
            break

    if not frames:
        return Walk(df=pd.DataFrame(), vendor_failed=failed)

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return Walk(df=df, vendor_failed=failed)
