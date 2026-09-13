"""
DerivProvider — forex majors/minors and precious metals (XAU/XAG/XPD/XPT
vs USD) via Deriv's public WebSocket API. Free, no account, no API key. No
rate-limit wall for ordinary use -- confirmed live: all 29 instruments
below stream concurrently on one connection with zero errors, and
quote/historical requests here work the same way against the same public
endpoint.

Deriv has no separate plain-REST surface -- every request (even a one-off
quote) goes over WebSocket, so unlike every other provider in this app
(which wrap a synchronous SDK/httpx call in a thread executor), this one is
natively async: each call opens a short-lived connection, sends one
request, reads the matching response, and closes. The always-on connection
for live ticks is a separate, persistent client (deriv_ticker.py) -- same
split this app already has for Kite (KiteProvider's REST calls vs
KiteTickerClient's one persistent socket), not a new pattern.

Volume, though, does NOT come from Deriv -- neither its candles nor its
ticks_history carry a real size field (spot/CFD forex has none to report),
and Deriv's own tick-history rate limit turned out far tighter than "no
rate-limit wall" above suggests: backfilling a whole multi-day range in
Deriv ticks (~70-85 chunked requests) triggered heavy HTTP 429s, 21-51s
chart loads, and still-incomplete coverage anyway. get_historical_df's
volume column instead comes from Dukascopy (see dukascopy_bridge.py) --
counting real ticks from a real ECN feed, one request for the WHOLE
candle range (confirmed live: 33,000+ ticks for a 3-hour XAUUSD window in
under 500ms, no chunking needed), free, no account. Its one real
limitation is a ~15-20 minute publish lag, so the most recent few candles
keep the same honest 0.0 until it catches up.
"""
from __future__ import annotations

import bisect
import logging
import time

import pandas as pd
from websockets.exceptions import WebSocketException

from app.market.contract import ProviderCapabilities, VolumeSource
from app.market.intervals import clamp_days
from app.market.paging import walk_back
from app.market.providers import dukascopy_bridge
from app.market.providers.deriv_socket import socket_for

logger = logging.getLogger(__name__)

WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"

# Same rationale as every other provider's own _VENDOR_ERRORS: these degrade
# to None/[], the caller's "no data" path. Anything else is a bug in our own
# code and is re-raised through logger.exception.
_VENDOR_ERRORS = (WebSocketException, OSError, TimeoutError,
                  KeyError, ValueError, TypeError, IndexError)

_GRANULARITY_MAP = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "1d": 86400}
_BARS_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "1d": 1}
# Vendor-confirmed by probing it live (2026-09-13): `count` is clamped to
# 1000 server-side, `start` is ignored, and what you actually get back is the
# candles that exist in [end - count*granularity, end]. So "the last N"
# cannot be asked for: a 1m request anchored at now reaches back 1000 minutes
# and no further, which over a weekend is entirely closed market and comes
# back empty -- with no error. That was a 404 on every 1m forex chart.
_MAX_COUNT = 1000

# Enough backward pages for 5 days of 1m bars, the heaviest window the
# terminal offers. Each page is one WebSocket round trip.
_MAX_PAGES = 10

# Tick volume costs by wall-clock range, not by candle count -- Dukascopy
# publishes one tick file per hour, so a 5-day window is 120 file fetches
# through a Node subprocess for millions of ticks. Past this the volume
# column is null: "not measured", which is a different answer from zero.
_TICK_VOLUME_MAX_SPAN_SECONDS = 48 * 3600

# App-facing symbol (no slash, matches this app's SYMBOL_RE) -> (Deriv's own
# symbol, display name). All 29 real instruments Deriv's public API lists
# under market "forex"/"commodities" -- confirmed live via active_symbols,
# not guessed.
KNOWN_PAIRS: dict[str, tuple[str, str]] = {
    "AUDCAD": ("frxAUDCAD", "AUD/CAD"), "AUDCHF": ("frxAUDCHF", "AUD/CHF"),
    "AUDJPY": ("frxAUDJPY", "AUD/JPY"), "AUDNZD": ("frxAUDNZD", "AUD/NZD"),
    "AUDUSD": ("frxAUDUSD", "AUD/USD"),
    "EURAUD": ("frxEURAUD", "EUR/AUD"), "EURCAD": ("frxEURCAD", "EUR/CAD"),
    "EURCHF": ("frxEURCHF", "EUR/CHF"), "EURGBP": ("frxEURGBP", "EUR/GBP"),
    "EURJPY": ("frxEURJPY", "EUR/JPY"), "EURNZD": ("frxEURNZD", "EUR/NZD"),
    "EURUSD": ("frxEURUSD", "EUR/USD"),
    "GBPAUD": ("frxGBPAUD", "GBP/AUD"), "GBPCAD": ("frxGBPCAD", "GBP/CAD"),
    "GBPCHF": ("frxGBPCHF", "GBP/CHF"), "GBPJPY": ("frxGBPJPY", "GBP/JPY"),
    "GBPNZD": ("frxGBPNZD", "GBP/NZD"), "GBPUSD": ("frxGBPUSD", "GBP/USD"),
    "NZDJPY": ("frxNZDJPY", "NZD/JPY"), "NZDUSD": ("frxNZDUSD", "NZD/USD"),
    "USDCAD": ("frxUSDCAD", "USD/CAD"), "USDCHF": ("frxUSDCHF", "USD/CHF"),
    "USDJPY": ("frxUSDJPY", "USD/JPY"), "USDMXN": ("frxUSDMXN", "USD/MXN"),
    "USDPLN": ("frxUSDPLN", "USD/PLN"),
    "XAGUSD": ("frxXAGUSD", "Silver/USD"), "XAUUSD": ("frxXAUUSD", "Gold/USD"),
    "XPDUSD": ("frxXPDUSD", "Palladium/USD"), "XPTUSD": ("frxXPTUSD", "Platinum/USD"),
}


def deriv_symbol_for(app_symbol: str) -> str | None:
    entry = KNOWN_PAIRS.get(app_symbol.upper())
    return entry[0] if entry else None


async def _request(payload: dict) -> dict:
    """One request over the connection this loop already holds.

    It used to open a connection per call, which cost 1.48s of handshake for
    a 50ms answer -- eight of those in a row is what pushed a 1m forex chart
    past the API's upstream timeout. See deriv_socket.py.
    """
    return await socket_for(WS_URL).request(payload)


async def _dukascopy_tick_volume(app_symbol: str, start_epoch: int, end_epoch: int, bucket_starts: list[int]) -> list[float]:
    """Tick count per candle, as a volume stand-in -- see this module's
    top-of-file rationale for why this is sourced from Dukascopy, not
    Deriv. Dukascopy's own instrument codes are this app's symbols
    lowercased for every pair/metal in KNOWN_PAIRS (confirmed live for
    XAUUSD, EURUSD, XAGUSD) -- no separate mapping table needed. A vendor
    gap (an unlisted instrument, the subprocess failing, or -- routinely --
    the most recent ~15-20 minutes the publish lag hasn't caught up to
    yet) all fall back to the same honest 0.0 per candle; the caller has
    no way to tell those apart and does not need to. `bucket_starts` must
    be sorted ascending -- the caller's `candles` list already is.
    """
    counts = await dukascopy_bridge.fetch_tick_counts(
        app_symbol.lower(), start_epoch * 1000, end_epoch * 1000, bucket_starts,
    )
    # A length mismatch means the bridge and this side disagree about the
    # candles; pandas would otherwise raise deep in the frame build, far from
    # the cause. Volume is enrichment -- drop it, keep the bars.
    if not counts or len(counts) != len(bucket_starts):
        return [0.0] * len(bucket_starts)
    return counts


async def tick_volume_since(app_symbol: str, since_epoch: int) -> int | None:
    """Real ECN tick count in [since_epoch, now) -- what the terminal polls
    every few seconds to keep the still-forming candle's volume live.
    get_historical_df's own bucketed volume only refreshes on a fresh fetch
    (a new chart load, a pan-back), so without this the current candle sat
    at whatever it was when that fetch ran, same "not live" gap this exists
    to close.

    None means "not a Dukascopy-covered instrument, or the vendor call
    itself failed" -- the caller must not treat that as 0 ticks, a real and
    different answer (see fetch_tick_timestamps' own docstring for why a gap
    and a genuine empty range are indistinguishable one level down, but
    "not covered at all" is knowable here before ever calling it)."""
    if deriv_symbol_for(app_symbol) is None:
        return None
    now_ms = int(time.time() * 1000)
    ticks_ms = await dukascopy_bridge.fetch_tick_timestamps(app_symbol.lower(), since_epoch * 1000, now_ms)
    if ticks_ms is None:
        return None
    return len(ticks_ms)


class DerivProvider:
    capabilities = ProviderCapabilities(
        intervals=frozenset(_GRANULARITY_MAP),
        max_bars_per_request=_MAX_COUNT,
        # Bounded by the paging budget (_MAX_PAGES windows) rather than by a
        # published retention -- Deriv documents none, and these are what the
        # walk can actually reach.
        max_days_by_interval={"1m": 7, "5m": 30, "15m": 90, "30m": 180, "1h": 365, "1d": 3650},
        volume_source=VolumeSource.TICKS,
        supports_ticks=False,           # tick counts come from Dukascopy, not here
        anchors_window_on_end=True,
    )

    def _pair_for(self, symbol: str) -> str | None:
        return deriv_symbol_for(symbol)

    # ------------------------------------------------------------------
    # get_quote
    # ------------------------------------------------------------------
    async def get_quote(self, symbol: str, exchange: str) -> dict | None:
        pair = self._pair_for(symbol)
        if pair is None:
            return None
        try:
            # Two daily candles: [yesterday (completed), today (forming)].
            # The forming candle's own close is the latest tick -- Deriv's
            # own semantics, confirmed live -- so this one request covers
            # ltp/open/high/low/prev_close together, no second call needed.
            resp = await _request({
                "ticks_history": pair, "style": "candles",
                "granularity": 86400, "count": 2, "end": "latest",
            })
            if "error" in resp:
                logger.warning("Deriv quote error for %s: %s", pair, resp["error"])
                return None
            candles = resp.get("candles") or []
            if not candles:
                return None

            today = candles[-1]
            prev_close = float(candles[-2]["close"]) if len(candles) > 1 else float(today["open"])
            ltp = float(today["close"])
            change = ltp - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0.0

            return {
                "symbol": symbol.upper(),
                # FOREX, not whatever was asked for: this provider serves no other
                # exchange, and the caller may have guessed (see resolve_exchange).
                "exchange": "FOREX",
                "ltp": round(ltp, 5),
                "change": round(change, 5),
                "change_percent": round(change_pct, 4),
                "open": float(today["open"]),
                "high": float(today["high"]),
                "low": float(today["low"]),
                "prev_close": round(prev_close, 5),
                # No real trade volume on a spot/CFD forex feed -- never
                # fabricated, same "absent not guessed" rule every other
                # provider here follows for a field the vendor doesn't give.
                "volume": None,
                "bid": None,
                "ask": None,
                "spread": None,
            }
        except _VENDOR_ERRORS as e:
            logger.warning("Deriv quote fetch failed for %s/%s: %s", symbol, exchange, e)
            return None
        except Exception:
            logger.exception("Unexpected error in Deriv quote for %s/%s", symbol, exchange)
            return None

    # ------------------------------------------------------------------
    # get_historical_df
    # ------------------------------------------------------------------
    async def fetch_page(self, pair: str, granularity: int, end_epoch: int,
                         count: int) -> pd.DataFrame | None:
        """One vendor request: the candles in [end - count*granularity, end].

        None means the vendor failed; an empty frame means it answered with
        nothing. That distinction is the whole point -- collapsing it is what
        turned a closed market into a 404. Walking across pages belongs to
        app/market/paging.py, not here."""
        resp = await _request({
            "ticks_history": pair, "style": "candles",
            "granularity": granularity, "count": count, "end": str(end_epoch),
        })
        if "error" in resp:
            logger.warning("Deriv time_series error for %s: %s", pair, resp["error"])
            return None
        candles = resp.get("candles") or []
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        df["date"] = pd.to_datetime(df["epoch"], unit="s")
        return df.set_index("date").sort_index()

    async def _tick_volume(self, symbol: str, bucket_starts: list[int],
                           granularity: int) -> list[float | None] | None:
        """Tick-count volume, or None when it would cost more than it is worth
        (see _TICK_VOLUME_MAX_SPAN_SECONDS) or when the vendor call fails.

        Enrichment, never a precondition: losing volume must not cost the caller
        its bars, which is how a broken bridge subprocess used to 404 a chart."""
        # Measure the most recent stretch only, rather than all-or-nothing: a
        # chart that paged back 40 days still gets volume on the part anyone is
        # looking at, and older bars carry null -- not measured, not zero.
        end = bucket_starts[-1] + granularity
        first = bisect.bisect_left(bucket_starts, end - _TICK_VOLUME_MAX_SPAN_SECONDS)
        recent = bucket_starts[first:]
        if not recent:
            return None
        try:
            counts = await _dukascopy_tick_volume(symbol, recent[0], end, recent)
        except Exception:
            logger.exception("Tick volume failed for %s -- bars are unaffected", symbol)
            return None
        return [None] * first + list(counts)

    async def get_historical_df(
        self, symbol: str, exchange: str, interval: str, days: int
    ) -> pd.DataFrame | None:
        pair = self._pair_for(symbol)
        if pair is None:
            return None
        try:
            granularity = _GRANULARITY_MAP.get(interval, 86400)
            walk = await walk_back(
                lambda end, count: self.fetch_page(pair, granularity, end, count),
                granularity_seconds=granularity,
                span_days=clamp_days(interval, days),
                page_bars=self.capabilities.max_bars_per_request,
                max_pages=_MAX_PAGES,
            )
            if walk.df.empty:
                return None

            df = walk.df
            # astype("datetime64[s]") first: pandas keeps the resolution it was
            # built with, so a bare astype("int64") is seconds here and
            # nanoseconds elsewhere depending on how the frame was made.
            bucket_starts = [int(e) for e in df.index.astype("datetime64[s]").astype("int64")]
            tick_volume = await self._tick_volume(symbol, bucket_starts, granularity)
            for col in ("open", "high", "low", "close"):
                df[col] = df[col].astype(float)
            df["volume"] = tick_volume
            return df[["open", "high", "low", "close", "volume"]]
        except _VENDOR_ERRORS as e:
            logger.warning("Deriv historical fetch failed for %s/%s: %s", symbol, exchange, e)
            return None
        except Exception:
            logger.exception("Unexpected error in Deriv history for %s/%s", symbol, exchange)
            return None

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    async def search(self, query: str, limit: int) -> list[dict]:
        q = query.strip().lower()
        if not q:
            return []
        out = []
        for app_symbol, (deriv_sym, name) in KNOWN_PAIRS.items():
            haystacks = (app_symbol.lower(), name.lower(), deriv_sym.lower())
            if any(q in h for h in haystacks):
                out.append({"symbol": app_symbol, "name": name, "exchange": "FOREX"})
        return out[:limit]
