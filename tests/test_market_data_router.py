"""
MarketDataRouter's per-exchange quote cache bucketing. Construction is
side-effect-free (KiteProvider/TwelveDataProvider just store config, no
network calls at __init__) so a real router instance is safe to build here.
"""
from __future__ import annotations

from app.market.providers.registry import (
    FOREX_QUOTE_TTL_SECONDS,
    QUOTE_TTL_SECONDS,
    MarketDataRouter,
)


def test_forex_gets_its_own_quote_cache_bucket():
    router = MarketDataRouter()
    assert router._quote_cache_for("FOREX") is router._forex_quote_cache
    assert router._quote_cache_for("forex") is router._forex_quote_cache  # case-insensitive


def test_every_other_exchange_shares_the_main_quote_cache():
    router = MarketDataRouter()
    for exch in ("NSE", "BSE", "MCX", "NASDAQ", "NYSE"):
        assert router._quote_cache_for(exch) is router._quote_cache


def test_forex_ttl_stays_within_twelve_data_free_tier_headroom():
    """Twelve Data's free tier is 8 credits/min; the poll loop's own 5s
    interval means a shorter TTL directly raises real vendor call volume
    (see registry.py's own comment on FOREX_QUOTE_TTL_SECONDS) -- this is a
    tripwire, not a tautology: it fails loudly if someone lowers the TTL
    without re-checking the math."""
    calls_per_minute = 60 / FOREX_QUOTE_TTL_SECONDS
    assert calls_per_minute <= 8

    # The whole point of a dedicated bucket: FOREX must actually be shorter
    # than the blanket TTL, or this is dead code.
    assert FOREX_QUOTE_TTL_SECONDS < QUOTE_TTL_SECONDS


def test_invalidate_clears_the_correct_bucket_for_forex():
    router = MarketDataRouter()
    key = ("quote", "XAUUSD", "FOREX")
    router._forex_quote_cache[key] = {"ltp": 2650.0}

    router.invalidate("XAUUSD", "FOREX")

    assert key not in router._forex_quote_cache


def test_cache_stats_reports_the_forex_bucket():
    router = MarketDataRouter()
    stats = router.cache_stats()
    assert "forex_quotes" in stats
    assert stats["forex_quotes"] == 0
