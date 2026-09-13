"""What a market-data provider promises, and what a bars request answers with.

The old contract (app/market/providers/base.py) was three methods returning
``dict | None`` / ``DataFrame | None``. It said nothing about a vendor's limits,
which intervals it serves, where its volume comes from, or *why* a result was
empty -- so the router guessed, and each guess became an incident: a forex pair
routed to an equity exchange, a weekend read as the end of history, an
enrichment failure read as no data at all.

Everything here is declared per provider and carried on every result, so those
questions have answers instead of guesses. See
docs/superpowers/specs/2026-09-13-market-data-contract-design.md.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from app.market.sessions import SessionSpec


class AssetClass(StrEnum):
    EQUITY = "equity"
    INDEX = "index"
    FX = "fx"
    METAL = "metal"
    COMMODITY = "commodity"


class VolumeSource(StrEnum):
    """Where a bar's volume comes from, which decides what a missing value
    means. EXCHANGE: real traded quantity. TICKS: a tick count standing in for
    volume (Dukascopy behind Deriv). NONE: this vendor has none, and a zero
    would be a fabrication."""

    EXCHANGE = "exchange"
    TICKS = "ticks"
    NONE = "none"


class BarsStatus(StrEnum):
    """Why a result looks the way it does.

    NO_DATA and VENDOR_ERROR are the pair that matters: collapsing them into a
    single ``None`` is what turned a closed market into a 404 on every forex
    chart, and a dead subprocess into "this symbol has no history".
    """

    OK = "ok"
    NO_DATA = "no_data"                        # the vendor answered; the range is genuinely empty
    CLOSED_MARKET = "closed_market"            # empty because nothing trades then
    OUT_OF_RETENTION = "out_of_retention"      # older than this vendor keeps
    UNSUPPORTED_INTERVAL = "unsupported_interval"
    UNKNOWN_SYMBOL = "unknown_symbol"
    VENDOR_ERROR = "vendor_error"              # the vendor, or the transport, failed


@dataclass(frozen=True)
class ProviderCapabilities:
    """A vendor's limits, as measured facts rather than folklore.

    These used to live wherever someone first needed them -- most damagingly in
    a single global `VENDOR_MAX_DAYS` table holding *yfinance's* limits and
    applied to every provider, which clamps Kite to windows it would happily
    serve and never described Deriv's real ceiling at all.
    """

    intervals: frozenset[str]
    # Most bars one request can return. Deriv clamps to 1000 server-side; the
    # router pages around it rather than each provider inventing its own loop.
    max_bars_per_request: int
    # How far back each interval reaches, per this vendor.
    max_days_by_interval: Mapping[str, int]
    volume_source: VolumeSource
    supports_ticks: bool = False
    # True when the vendor serves [end - count*granularity, end] and ignores a
    # `start` (Deriv). The router then walks `end` backwards in whole windows
    # instead of asking for a range it would silently not honour.
    anchors_window_on_end: bool = False

    def max_days(self, interval: str, default: int = 100_000) -> int:
        return self.max_days_by_interval.get(interval, default)


@dataclass(frozen=True)
class SymbolInfo:
    """A resolved symbol: which vendor answers for it, and what it is.

    Requests reach a provider only through one of these, which is what makes
    "XAUUSD on NSE" unrepresentable rather than a 404.
    """

    symbol: str                 # app-facing, e.g. XAUUSD, RELIANCE
    vendor_symbol: str          # what the vendor calls it, e.g. frxXAUUSD, RELIANCE.NS
    provider: str               # registry key of the provider that owns it
    asset_class: AssetClass
    exchange: str               # what we report back, never what a caller guessed
    session: SessionSpec
    volume_source: VolumeSource
    intervals: frozenset[str] = field(default_factory=frozenset)
    # True when a real listing table says this symbol exists (Deriv's pair
    # table). False when we merely assumed a vendor could serve it, which is
    # every unrecognised string: the fallback vendor accepts anything and
    # answers with nothing, and "NASDAQ is closed" is a confident lie to tell
    # about a symbol that does not exist.
    authoritative: bool = True


@dataclass(frozen=True)
class BarsResult:
    """Bars plus the reason they are what they are.

    An empty ``bars`` is never self-explanatory: closed market, exhausted
    retention and a broken vendor look identical in the data and are three
    different things to a user, to the chat agent, and to monitoring.
    """

    bars: list[dict]
    status: BarsStatus
    symbol: SymbolInfo | None = None
    volume_source: VolumeSource = VolumeSource.NONE
    reason: str | None = None            # human-readable, safe to show a user
    # Set when the request was clamped to what the vendor allows, so a caller
    # asking for a year of 1m bars can tell it did not get one.
    truncated_to_days: int | None = None

    @property
    def ok(self) -> bool:
        return self.status is BarsStatus.OK
