from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tankx.ml.features_mtf import build_mtf_features


def _frame_5m(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC', name='timestamp')
    return pd.DataFrame({'close': close, 'volume': np.full(n, 100.0)}, index=idx)

def test_mtf_features_shape_and_columns() -> None:
    n = 24 * 12 * 7
    rng = np.random.default_rng(seed=0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = _frame_5m(close)
    feats = build_mtf_features(df, timeframes=('1h', '4h'))
    assert feats.index.equals(df.index)
    expected = {'ret_1h_lag1', 'vol_1h_lag1', 'ret_4h_lag1', 'vol_4h_lag1'}
    assert set(feats.columns) == expected

def test_mtf_features_warmup_is_nan() -> None:
    n = 6
    rng = np.random.default_rng(seed=0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = _frame_5m(close)
    feats = build_mtf_features(df, timeframes=('1h',))
    assert feats['ret_1h_lag1'].isna().all()

def test_mtf_features_are_causal() -> None:
    n = 24 * 12 * 14
    rng = np.random.default_rng(seed=42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df_a = _frame_5m(close)
    close_b = close.copy()
    close_b[n // 2:] *= 1.5
    df_b = _frame_5m(close_b)
    feats_a = build_mtf_features(df_a, timeframes=('1h', '4h'))
    feats_b = build_mtf_features(df_b, timeframes=('1h', '4h'))
    cutoff = n // 2 - 12
    pd.testing.assert_frame_equal(feats_a.iloc[:cutoff], feats_b.iloc[:cutoff], check_dtype=False)

def test_mtf_features_no_future_leakage_at_period_boundary() -> None:
    n = 24 * 12 * 2
    close = np.full(n, 100.0)
    spike_start_idx = 24 * 12 + 6
    spike_start_idx = 150
    spike_end_idx = 156
    close[spike_start_idx:spike_end_idx] = 200.0
    df = _frame_5m(close)
    feats = build_mtf_features(df, timeframes=('1h',))
    assert abs(feats['ret_1h_lag1'].iloc[155]) < 1e-09 or pd.isna(feats['ret_1h_lag1'].iloc[155])

def test_mtf_rejects_tz_naive() -> None:
    df = pd.DataFrame({'close': [1.0, 2.0], 'volume': [10.0, 10.0]}, index=pd.DatetimeIndex(['2026-01-01', '2026-01-02']))
    with pytest.raises(ValueError, match='tz-aware'):
        build_mtf_features(df)

def test_mtf_rejects_missing_close_column() -> None:
    idx = pd.date_range('2026-01-01', periods=5, freq='5min', tz='UTC')
    df = pd.DataFrame({'volume': [1.0] * 5}, index=idx)
    with pytest.raises(KeyError, match='close'):
        build_mtf_features(df)
