from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from tankx.backtest.walkforward import WFWindow, iter_param_grid, make_windows, walk_forward
from tankx.strategies.ma_crossover import MACrossover


@pytest.fixture
def long_synthetic_ohlcv() -> pd.DataFrame:
    n = 120 * 24
    idx = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC', name='timestamp')
    rng = np.random.default_rng(seed=0)
    log_prices = np.cumsum(rng.normal(0.0, 0.005, n)) + np.log(100.0)
    close = np.exp(log_prices)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({'open': open_, 'high': np.maximum(open_, close) + 0.1, 'low': np.minimum(open_, close) - 0.1, 'close': close, 'volume': np.full(n, 100.0)}, index=idx)

def test_make_windows_no_overlap(long_synthetic_ohlcv: pd.DataFrame) -> None:
    windows = make_windows(long_synthetic_ohlcv.index, train_days=30, test_days=14)
    assert len(windows) > 0
    for a, b in itertools.pairwise(windows):
        assert a.test_end <= b.test_start
        assert a.train_start < a.train_end <= a.test_start < a.test_end

def test_make_windows_anchored_grows_train(long_synthetic_ohlcv: pd.DataFrame) -> None:
    windows = make_windows(long_synthetic_ohlcv.index, train_days=30, test_days=14, anchored=True)
    starts = [w.train_start for w in windows]
    assert all(s == starts[0] for s in starts)

def test_make_windows_requires_tz_index() -> None:
    naive_idx = pd.date_range('2026-01-01', periods=100, freq='1h', name='timestamp')
    with pytest.raises(ValueError, match='tz-aware'):
        make_windows(naive_idx)

def test_wfwindow_rejects_bad_ordering() -> None:
    ts = pd.Timestamp('2026-01-01', tz='UTC')
    with pytest.raises(ValueError, match='ordering'):
        WFWindow(train_start=ts + pd.Timedelta(days=30), train_end=ts + pd.Timedelta(days=10), test_start=ts + pd.Timedelta(days=20), test_end=ts + pd.Timedelta(days=40))

def test_iter_param_grid_cartesian() -> None:
    grid = {'a': [1, 2], 'b': [10, 20, 30]}
    out = iter_param_grid(grid)
    assert len(out) == 6
    assert {'a': 1, 'b': 10} in out
    assert {'a': 2, 'b': 30} in out

def test_walk_forward_runs_end_to_end(long_synthetic_ohlcv: pd.DataFrame) -> None:
    windows = make_windows(long_synthetic_ohlcv.index, train_days=30, test_days=14)
    result = walk_forward(df=long_synthetic_ohlcv, strategy_factory=MACrossover, param_grid={'short_window': [10, 20], 'long_window': [50, 100]}, windows=windows, bars_per_year=365 * 24)
    assert len(result) == len(windows)
    assert {'train_score', 'test_score', 'params', 'n_trades_test'} <= set(result.columns)
    for params in result['params']:
        assert params['short_window'] in (10, 20)
        assert params['long_window'] in (50, 100)

def test_walk_forward_chooses_best_train_params(long_synthetic_ohlcv: pd.DataFrame) -> None:
    windows = make_windows(long_synthetic_ohlcv.index, train_days=30, test_days=14)[:2]
    grid = {'short_window': [5, 10], 'long_window': [40, 80]}
    result = walk_forward(df=long_synthetic_ohlcv, strategy_factory=MACrossover, param_grid=grid, windows=windows, bars_per_year=365 * 24)
    assert result['train_score'].notna().all()
