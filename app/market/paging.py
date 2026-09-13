"""Walking a vendor's history backwards, once, for every vendor that needs it.

Some vendors cannot be asked for a range. Deriv serves whatever exists in
``[end - count*granularity, end]``, clamps ``count`` to 1000 and ignores
``start`` entirely, so "five days of 1m bars" is eight requests, not one.

Three things make this worth writing once rather than per provider:

* The windows are independent, so they are fetched **concurrently**. Measured
  against Deriv from the production box: eight pages sequentially cost 13.8s
  and the same eight concurrently cost 2.3s, against a 10s upstream timeout in
  the API. Sequential paging was the reason 1m gold charts failed while 15m
  and 1d worked.
* The walk must continue past a stretch with no data. A closed market answers
  with nothing, and reading that as "history ends here" is what left every 1m
  forex chart blank over a weekend.
* A page that failed and a page that is legitimately empty are different, and
  the caller has to be able to tell afterwards.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import pandas as pd

# A page is one vendor round trip. Ten covers five days of 1m bars against a
# 1000-bar page size, which is the heaviest window the terminal offers.
DEFAULT_MAX_PAGES = 10

# Concurrent requests to one vendor. Enough to collapse the whole walk into
# roughly two round trips; low enough not to look like a burst to a free
# endpoint that documents no rate limit and could start refusing us.
DEFAULT_MAX_CONCURRENCY = 4

FetchPage = Callable[[int, int], Awaitable["pd.DataFrame | None"]]


@dataclass(frozen=True)
class Walk:
    """What the walk found, and whether the vendor misbehaved while finding it.

    ``df`` empty with ``vendor_failed`` False is a real answer: the range holds
    no bars. Empty with ``vendor_failed`` True is the vendor letting us down,
    and the two must never reach a user as the same thing.
    """

    df: pd.DataFrame
    vendor_failed: bool


async def _fetch_windows(
    fetch: FetchPage, ends: list[int], page_bars: int, max_concurrency: int,
) -> tuple[list[pd.DataFrame], bool]:
    limit = asyncio.Semaphore(max_concurrency)

    async def one(end: int):
        async with limit:
            return await fetch(end, page_bars)

    pages = await asyncio.gather(*(one(end) for end in ends), return_exceptions=True)

    frames: list[pd.DataFrame] = []
    failed = False
    for page in pages:
        # An exception from one window is that window's failure, not the
        # walk's: the other pages still hold real bars worth returning.
        if isinstance(page, BaseException) or page is None:
            failed = True
            continue
        if not page.empty:
            frames.append(page)
    return frames, failed


async def walk_back(
    fetch: FetchPage,
    *,
    granularity_seconds: int,
    span_days: int,
    page_bars: int,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    now: int | None = None,
) -> Walk:
    """Every bar the vendor will hand over for `span_days`, back from now.

    `fetch(end_epoch, count)` returns that window's bars, or None if the vendor
    failed. Frames are concatenated and deduplicated on their index, so an
    overlap between windows costs nothing.
    """
    now = int(time.time()) if now is None else now
    window = page_bars * granularity_seconds
    planned = min(max_pages, max(1, math.ceil(span_days * 86400 / window)))

    ends = [now - i * window for i in range(planned)]
    frames, failed = await _fetch_windows(fetch, ends, page_bars, max_concurrency)

    # Nothing at all, and budget to spare: keep reaching back. A day of 1m bars
    # asked for on a Sunday covers only closed market, and the last session is
    # further back than the requested span.
    if not frames and planned < max_pages:
        older = [now - i * window for i in range(planned, max_pages)]
        more, failed_again = await _fetch_windows(fetch, older, page_bars, max_concurrency)
        frames += more
        failed = failed or failed_again

    if not frames:
        return Walk(df=pd.DataFrame(), vendor_failed=failed)

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return Walk(df=df, vendor_failed=failed)
