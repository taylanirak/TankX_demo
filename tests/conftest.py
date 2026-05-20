from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260520)

@pytest.fixture
def synthetic_ohlcv(rng: np.random.Generator) -> pd.DataFrame:
    n = 7 * 24 * 12
    timestamps = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC')
    log_prices = np.cumsum(rng.normal(0.0, 0.001, n)) + np.log(50000.0)
    close = np.exp(log_prices)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0.0, 5.0, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(rng.normal(100.0, 30.0, n))
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=pd.DatetimeIndex(timestamps, name='timestamp'))
