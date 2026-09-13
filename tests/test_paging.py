"""The backward walk, which every vendor that cannot be asked for a range uses.

Written once because the two rules it encodes are the ones that broke in
production: step on wall-clock time rather than on what came back, and keep a
failed page distinguishable from an empty one.
"""
import pandas as pd
import pytest

from app.market.paging import walk_back

NOW = 1_789_000_000        # fixed, so these assertions do not drift with the clock
HOUR = 3600


def _page(*epochs: int) -> pd.DataFrame:
    df = pd.DataFrame({"epoch": list(epochs), "close": [1.0] * len(epochs)})
    df["date"] = pd.to_datetime(df["epoch"], unit="s")
    return df.set_index("date").sort_index()


def _recorder(pages):
    """Returns (fetch, calls) where `pages` are handed out in order."""
    calls: list[tuple[int, int]] = []
    queue = list(pages)

    async def fetch(end_epoch: int, count: int):
        calls.append((end_epoch, count))
        return queue.pop(0) if queue else _page()

    return fetch, calls


@pytest.mark.asyncio
async def test_a_span_that_fits_in_one_window_is_one_request():
    fetch, calls = _recorder([_page(NOW - HOUR)])
    walk = await walk_back(fetch, granularity_seconds=HOUR, span_days=1,
                           page_bars=1000, now=NOW)

    assert len(calls) == 1
    assert len(walk.df) == 1
    assert walk.vendor_failed is False


@pytest.mark.asyncio
async def test_a_longer_span_walks_backwards_a_whole_window_at_a_time():
    # 5 days of 1m bars against a 1000-bar page is 8 windows, not one request.
    fetch, calls = _recorder([_page(NOW - 60), _page(NOW - 60_000), _page(NOW - 120_000)])
    await walk_back(fetch, granularity_seconds=60, span_days=5, page_bars=1000, now=NOW)

    ends = [end for end, _ in calls]
    assert len(ends) > 1
    assert ends == sorted(ends, reverse=True)
    assert ends[0] - ends[1] == 1000 * 60


@pytest.mark.asyncio
async def test_an_empty_page_does_not_end_the_walk():
    # The weekend case: the recent windows are closed market, and the last
    # session is further back. Stopping at the first empty page is what left
    # 1m forex charts blank.
    fetch, calls = _recorder([_page(), _page(), _page(NOW - 300_000)])
    walk = await walk_back(fetch, granularity_seconds=60, span_days=1,
                           page_bars=1000, now=NOW)

    assert len(calls) == 3
    assert len(walk.df) == 1


@pytest.mark.asyncio
async def test_it_gives_up_after_the_page_budget():
    fetch, calls = _recorder([])          # every page empty
    walk = await walk_back(fetch, granularity_seconds=60, span_days=3650,
                           page_bars=1000, max_pages=4, now=NOW)

    assert len(calls) == 4
    assert walk.df.empty
    assert walk.vendor_failed is False    # empty is not failure


@pytest.mark.asyncio
async def test_a_vendor_failure_is_reported_not_silently_empty():
    async def fetch(end_epoch, count):
        return None

    walk = await walk_back(fetch, granularity_seconds=60, span_days=1,
                           page_bars=1000, now=NOW)

    assert walk.df.empty
    assert walk.vendor_failed is True


@pytest.mark.asyncio
async def test_a_failure_partway_keeps_what_was_already_collected():
    # Bars in hand are still worth returning; the flag says the answer is short.
    pages = [_page(NOW - 60)]

    async def fetch(end_epoch, count):
        return pages.pop(0) if pages else None

    walk = await walk_back(fetch, granularity_seconds=60, span_days=30,
                           page_bars=1000, now=NOW)

    assert len(walk.df) == 1
    assert walk.vendor_failed is True


@pytest.mark.asyncio
async def test_overlapping_pages_are_deduplicated_and_sorted():
    fetch, _ = _recorder([_page(NOW - 60, NOW - 120), _page(NOW - 120, NOW - 180)])
    walk = await walk_back(fetch, granularity_seconds=60, span_days=5,
                           page_bars=1, now=NOW)

    epochs = [int(e) for e in walk.df.index.astype("datetime64[s]").astype("int64")]
    assert epochs == sorted(epochs)
    assert len(epochs) == len(set(epochs))
