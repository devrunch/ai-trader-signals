from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def settings():
    from app.config import get_settings
    return get_settings()


def make_bars(rows: list[tuple[float, float, float, float]], start: str = "2026-01-01 09:15",
              freq: str = "15min", volume: float = 1000.0) -> pd.DataFrame:
    """Build an OHLCV frame from (open, high, low, close) tuples."""
    idx = pd.date_range(start=start, periods=len(rows), freq=freq)
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [volume] * len(rows),
        },
        index=idx,
    )


@pytest.fixture
def bars():
    return make_bars


@pytest.fixture
def trending_frame() -> pd.DataFrame:
    """300 bars of a noisy uptrend across three sessions — enough warm-up for
    every indicator in SIGNAL_SET except EMA200."""
    rng = np.random.default_rng(7)
    n = 300
    drift = np.linspace(100.0, 130.0, n)
    noise = rng.normal(0, 0.4, n)
    close = drift + noise
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.5, n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.5, n)
    idx = pd.date_range("2026-01-05 09:15", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(5_000, 50_000, n).astype(float)},
        index=idx,
    )
