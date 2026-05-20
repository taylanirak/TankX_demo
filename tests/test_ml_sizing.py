from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tankx.ml.sizing import vol_target_positions


def _frame(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC', name='timestamp')
    return pd.DataFrame({'close': close, 'volume': np.full(n, 100.0)}, index=idx)

def test_low_vol_keeps_size_capped_at_max_leverage() -> None:
    n = 500
    close = 100.0 + np.arange(n) * 1e-06
    df = _frame(close)
    positions = pd.Series([1.0] * n, index=df.index)
    scaled = vol_target_positions(positions, df, daily_target=0.01, lookback=50, max_leverage=1.0)
    tail = scaled.dropna().iloc[50:]
    assert (tail.abs() <= 1.0 + 1e-09).all()
    assert (tail.abs() > 0.99).all()

def test_high_vol_shrinks_position() -> None:
    n = 500
    rng = np.random.default_rng(seed=0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.1, n)))
    df = _frame(close)
    positions = pd.Series([1.0] * n, index=df.index)
    scaled = vol_target_positions(positions, df, daily_target=0.01, lookback=50)
    tail = scaled.dropna().iloc[100:]
    assert (tail.abs() < 0.99).any()
    assert (tail.abs() <= 1.0 + 1e-09).all()

def test_zero_position_stays_zero() -> None:
    n = 300
    rng = np.random.default_rng(seed=1)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = _frame(close)
    positions = pd.Series(0.0, index=df.index)
    scaled = vol_target_positions(positions, df, lookback=50)
    assert (scaled == 0.0).all()

def test_short_position_remains_negative() -> None:
    n = 300
    rng = np.random.default_rng(seed=2)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = _frame(close)
    positions = pd.Series([-1.0] * n, index=df.index)
    scaled = vol_target_positions(positions, df, lookback=50).dropna()
    tail = scaled.iloc[50:]
    assert (tail < 0).all()

def test_rejects_misaligned_index() -> None:
    n = 100
    rng = np.random.default_rng(seed=3)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = _frame(close)
    bad = pd.Series([1.0] * (n - 1), index=df.index[:-1])
    with pytest.raises(ValueError, match='must match'):
        vol_target_positions(bad, df)

def test_rejects_bad_params() -> None:
    n = 100
    df = _frame(np.full(n, 100.0))
    positions = pd.Series([1.0] * n, index=df.index)
    with pytest.raises(ValueError, match='daily_target'):
        vol_target_positions(positions, df, daily_target=0.0)
    with pytest.raises(ValueError, match='lookback'):
        vol_target_positions(positions, df, lookback=2)

def test_no_lookahead_via_causal_shift() -> None:
    n = 400
    rng = np.random.default_rng(seed=4)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df_a = _frame(close)
    close_b = close.copy()
    close_b[n // 2:] *= 2.0
    df_b = _frame(close_b)
    pos = pd.Series([1.0] * n, index=df_a.index)
    sa = vol_target_positions(pos, df_a, lookback=50)
    sb = vol_target_positions(pos.reindex(df_b.index), df_b, lookback=50)
    pd.testing.assert_series_equal(sa.iloc[:n // 2 - 1], sb.iloc[:n // 2 - 1], check_names=False)
