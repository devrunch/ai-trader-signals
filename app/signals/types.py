"""
Shared signal types.

Lives in its own module so `validation`, `publisher`, `backtest` and `service`
can all speak the same vocabulary without importing each other.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypedDict


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Indicators(TypedDict, total=False):
    """The de-facto contract between the indicator engine and its consumers.

    Produced by `indicators.compute()`; read by the signal prompt, the quant
    confluence gate, the morning brief, and (via Mongo) the frontend. It was
    typed as bare `dict`, which is how two copies of the indicator engine
    managed to drift apart on VWAP without anything noticing. `total=False`
    because callers request a subset by name.
    """
    rsi: float | None
    macd: float | None
    macd_hist: float | None
    macd_signal: float | None
    ema20: float | None
    ema50: float | None
    ema200: float | None
    adx: float | None
    di_plus: float | None
    di_minus: float | None
    supertrend_dir: int | None
    vwap: float | None
    atr: float | None
    ltp: float | None
    volume: int | None
    volume_avg20: int | None
    volume_ratio: float | None


@dataclass
class GeneratedSignal:
    symbol: str
    exchange: str
    signal_type: SignalType
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    reasoning: str
    indicators: dict
