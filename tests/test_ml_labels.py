from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tankx.ml.labels import TripleBarrierConfig, label_distribution, triple_barrier_labels


def _frame(close: np.ndarray, *, with_hl: bool=True) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC', name='timestamp')
    cols: dict[str, np.ndarray] = {'close': close}
    if with_hl:
        cols['high'] = close + 0.01
        cols['low'] = close - 0.01
    cols['open'] = close
    cols['volume'] = np.full(n, 100.0)
    return pd.DataFrame(cols, index=idx)

def test_takes_profit_when_price_rises_quickly() -> None:
    n = 200
    close = np.full(n, 100.0)
    close[50:] = 105.0
    rng = np.random.default_rng(seed=0)
    close[:50] = 100.0 + rng.normal(0, 0.1, 50)
    df = _frame(close)
    cfg = TripleBarrierConfig(horizon_bars=12, take_profit_sigma=1.0, stop_loss_sigma=1.0, sigma_lookback=20)
    labels = triple_barrier_labels(df, cfg)
    pre_jump = labels.iloc[40:49]
    assert (pre_jump == 1.0).any()

def test_stops_loss_when_price_falls_quickly() -> None:
    n = 200
    close = np.full(n, 100.0)
    rng = np.random.default_rng(seed=1)
    close[:50] = 100.0 + rng.normal(0, 0.1, 50)
    close[50:] = 95.0
    df = _frame(close)
    cfg = TripleBarrierConfig(horizon_bars=12, take_profit_sigma=1.0, stop_loss_sigma=1.0, sigma_lookback=20)
    labels = triple_barrier_labels(df, cfg)
    pre_drop = labels.iloc[40:49]
    assert (pre_drop == -1.0).any()

def test_flat_price_yields_all_nan_labels() -> None:
    n = 200
    close = np.full(n, 100.0)
    df = _frame(close)
    cfg = TripleBarrierConfig(horizon_bars=12, take_profit_sigma=1.0, stop_loss_sigma=1.0, sigma_lookback=20)
    labels = triple_barrier_labels(df, cfg)
    assert labels.isna().all()

def test_warmup_rows_are_nan() -> None:
    n = 200
    rng = np.random.default_rng(seed=3)
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    df = _frame(close)
    cfg = TripleBarrierConfig(horizon_bars=12, take_profit_sigma=1.0, stop_loss_sigma=1.0, sigma_lookback=50)
    labels = triple_barrier_labels(df, cfg)
    assert labels.iloc[:49].isna().all()
    assert labels.iloc[-12:].isna().all()

def test_label_values_are_signed_unit() -> None:
    n = 500
    rng = np.random.default_rng(seed=4)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = _frame(close)
    labels = triple_barrier_labels(df).dropna()
    assert set(np.unique(labels.to_numpy())) <= {-1.0, 0.0, 1.0}

def test_rejects_missing_close_column() -> None:
    df = pd.DataFrame({'open': [1, 2, 3]})
    with pytest.raises(KeyError, match='close'):
        triple_barrier_labels(df)

def test_label_distribution_sums_to_one() -> None:
    n = 500
    rng = np.random.default_rng(seed=5)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = _frame(close)
    labels = triple_barrier_labels(df)
    dist = label_distribution(labels)
    total = dist['tp_frac'] + dist['sl_frac'] + dist['timeout_frac']
    assert abs(total - 1.0) < 1e-09

def test_wider_barriers_produce_more_timeouts() -> None:
    n = 500
    rng = np.random.default_rng(seed=6)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = _frame(close)
    narrow = triple_barrier_labels(df, TripleBarrierConfig(horizon_bars=12, take_profit_sigma=0.5, stop_loss_sigma=0.5, sigma_lookback=50))
    wide = triple_barrier_labels(df, TripleBarrierConfig(horizon_bars=12, take_profit_sigma=3.0, stop_loss_sigma=3.0, sigma_lookback=50))
    n_dist = label_distribution(narrow)
    w_dist = label_distribution(wide)
    assert w_dist['timeout_frac'] > n_dist['timeout_frac']
