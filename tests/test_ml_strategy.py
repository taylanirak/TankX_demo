from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tankx.backtest.engine_vec import BacktestConfig, run_backtest
from tankx.strategies.base import VectorizedStrategy
from tankx.strategies.ml_directional import MLDirectionalStrategy


@pytest.fixture
def ml_synthetic_ohlcv(rng: np.random.Generator) -> pd.DataFrame:
    n = 5 * 24 * 12
    idx = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC', name='timestamp')
    log_prices = np.cumsum(rng.normal(0.0, 0.001, n)) + np.log(100.0)
    close = np.exp(log_prices)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.05, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = np.abs(rng.normal(100.0, 30.0, n))
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': vol}, index=idx)

def test_strategy_satisfies_protocol() -> None:
    s = MLDirectionalStrategy(model_name='Lag1')
    assert isinstance(s, VectorizedStrategy)
    assert s.name == 'ML[Lag1/direction]'

def test_strategy_rejects_invalid_train_fraction() -> None:
    with pytest.raises(ValueError, match='train_fraction'):
        MLDirectionalStrategy(train_fraction=0.0)

def test_strategy_returns_zero_during_training_slice(ml_synthetic_ohlcv: pd.DataFrame) -> None:
    s = MLDirectionalStrategy(model_name='Lag1', train_fraction=0.6)
    pos = s.generate_positions(ml_synthetic_ohlcv)
    n = len(ml_synthetic_ohlcv)
    train_end = int(0.6 * n)
    assert (pos.iloc[:train_end - 10] == 0.0).all()

def test_strategy_positions_in_signed_unit_interval(ml_synthetic_ohlcv: pd.DataFrame) -> None:
    s = MLDirectionalStrategy(model_name='Lag1', train_fraction=0.6)
    pos = s.generate_positions(ml_synthetic_ohlcv)
    assert set(np.unique(pos.to_numpy())) <= {-1.0, 0.0, 1.0}

def test_strategy_ridge_direction_yields_some_nonzero_positions(ml_synthetic_ohlcv: pd.DataFrame) -> None:
    s = MLDirectionalStrategy(model_name='Ridge', target_kind='direction', train_fraction=0.6)
    pos = s.generate_positions(ml_synthetic_ohlcv)
    test_slice = pos.iloc[int(0.6 * len(pos)):]
    assert (test_slice != 0).sum() > 0

def test_strategy_ridge_return_yields_some_nonzero_positions(ml_synthetic_ohlcv: pd.DataFrame) -> None:
    s = MLDirectionalStrategy(model_name='Ridge', target_kind='return', train_fraction=0.6, return_band=1e-09)
    pos = s.generate_positions(ml_synthetic_ohlcv)
    test_slice = pos.iloc[int(0.6 * len(pos)):]
    assert (test_slice != 0).sum() > 0

def test_strategy_runs_through_backtest_engine(ml_synthetic_ohlcv: pd.DataFrame) -> None:
    s = MLDirectionalStrategy(model_name='Lag1', train_fraction=0.6)
    pos = s.generate_positions(ml_synthetic_ohlcv)
    cfg = BacktestConfig(fee_bps=0.0, slippage_bps=0.0, execution='next_close')
    result = run_backtest(ml_synthetic_ohlcv, pos, cfg)
    assert len(result.equity) == len(ml_synthetic_ohlcv)
    assert result.equity.iloc[0] == cfg.initial_capital

def test_strategy_handles_too_short_data() -> None:
    rng = np.random.default_rng(seed=0)
    n = 40
    idx = pd.date_range('2026-05-15', periods=n, freq='5min', tz='UTC', name='timestamp')
    close = 100.0 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': np.full(n, 100.0)}, index=idx)
    s = MLDirectionalStrategy(model_name='Lag1')
    pos = s.generate_positions(df)
    assert (pos == 0.0).all()

def test_strategy_fits_lazily(ml_synthetic_ohlcv: pd.DataFrame) -> None:
    s = MLDirectionalStrategy(model_name='Ridge')
    assert s._fitted is False
    s.generate_positions(ml_synthetic_ohlcv)
    assert s._fitted is True
