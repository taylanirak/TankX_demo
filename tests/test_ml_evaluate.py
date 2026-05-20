from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tankx.backtest.walkforward import make_windows
from tankx.ml.evaluate import BenchmarkSpec, evaluate_all, evaluate_one, predictions_to_positions
from tankx.ml.models.registry import MODEL_REGISTRY, list_models, make_predictor
from tankx.ml.training import fit_predict_walkforward


@pytest.fixture
def long_synthetic_ohlcv(rng: np.random.Generator) -> pd.DataFrame:
    n = 60 * 24 * 12
    idx = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC', name='timestamp')
    log_prices = np.cumsum(rng.normal(0.0, 0.001, n)) + np.log(50000.0)
    close = np.exp(log_prices)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0.0, 5.0, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(rng.normal(100.0, 30.0, n))
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=idx)

def test_registry_lists_expected_models() -> None:
    names = list_models()
    assert {'Lag1', 'Ridge', 'MLP', 'LSTM', 'GRU', 'Transformer'} <= set(names)

def test_make_predictor_unknown_raises() -> None:
    with pytest.raises(KeyError, match='Unknown model'):
        make_predictor('NotAModel')

@pytest.mark.parametrize('name', ['Lag1', 'Ridge'])
def test_make_predictor_returns_predictor(name: str) -> None:
    p = make_predictor(name)
    assert hasattr(p, 'fit')
    assert hasattr(p, 'predict')

def test_fit_predict_walkforward_returns_oos_preds(long_synthetic_ohlcv: pd.DataFrame) -> None:
    windows = make_windows(long_synthetic_ohlcv.index, train_days=14, test_days=3)
    assert len(windows) > 0
    result = fit_predict_walkforward(df=long_synthetic_ohlcv, factory=MODEL_REGISTRY['Lag1'], target_kind='direction', windows=windows, model_name='Lag1', symbol='TEST/USDT')
    assert len(result.predictions) > 0
    assert result.predictions.index.is_monotonic_increasing
    assert result.predictions.index.equals(result.actuals.index)
    assert set(np.unique(result.predictions.to_numpy())) <= {0.1, 0.9}

def test_fit_predict_walkforward_ridge_direction(long_synthetic_ohlcv: pd.DataFrame) -> None:
    windows = make_windows(long_synthetic_ohlcv.index, train_days=14, test_days=3)
    result = fit_predict_walkforward(df=long_synthetic_ohlcv, factory=MODEL_REGISTRY['Ridge'], target_kind='direction', windows=windows[:2], model_name='Ridge', symbol='TEST/USDT')
    assert len(result.predictions) > 0
    assert result.predictions.between(0.0, 1.0).all()

def test_predictions_to_positions_direction() -> None:
    pred = pd.Series([0.1, 0.5, 0.6, 0.9])
    pos = predictions_to_positions(pred, kind='direction', direction_band=0.05)
    np.testing.assert_array_equal(pos.to_numpy(), [-1.0, 0.0, 1.0, 1.0])

def test_predictions_to_positions_return() -> None:
    pred = pd.Series([-0.01, -0.001, 0.0, 0.001, 0.01])
    pos = predictions_to_positions(pred, kind='return', return_band=0.002)
    np.testing.assert_array_equal(pos.to_numpy(), [-1.0, 0.0, 0.0, 0.0, 1.0])

def test_predictions_to_positions_threshold_widens_abstention() -> None:
    pred = pd.Series([0.45, 0.48, 0.5, 0.52, 0.55, 0.6])
    pos_tight = predictions_to_positions(pred, kind='direction', direction_band=0.0)
    pos_wide = predictions_to_positions(pred, kind='direction', direction_band=0.1)
    assert (pos_wide == 0.0).sum() > (pos_tight == 0.0).sum()

def test_benchmark_spec_default_direction_band_is_005() -> None:
    from tankx.ml.evaluate import BenchmarkSpec
    spec = BenchmarkSpec(symbol='X/Y', timeframe='5m', target_kind='direction')
    assert spec.direction_band == pytest.approx(0.05)
    assert spec.assume_maker is False
    assert spec.fee_bps_maker == 0.0

def test_evaluate_one_lag1(long_synthetic_ohlcv: pd.DataFrame) -> None:
    spec = BenchmarkSpec(symbol='TEST/USDT', timeframe='5m', target_kind='direction', train_days=14, test_days=3, fee_bps=0.0, slippage_bps=0.0)
    row, _ = evaluate_one(long_synthetic_ohlcv, spec, 'Lag1')
    assert {'model', 'symbol', 'target', 'accuracy', 'strategy_sharpe'} <= row.keys()
    assert row['model'] == 'Lag1'
    assert 0.4 <= row['accuracy'] <= 0.6

def test_evaluate_all_runs_three_classical_models(long_synthetic_ohlcv: pd.DataFrame) -> None:
    spec = BenchmarkSpec(symbol='TEST/USDT', timeframe='5m', target_kind='direction', train_days=14, test_days=3, fee_bps=0.0, slippage_bps=0.0)
    table, trainings = evaluate_all(long_synthetic_ohlcv, spec, model_names=['Lag1', 'Ridge', 'MLP'])
    assert len(table) == 3
    assert set(table['model']) == {'Lag1', 'Ridge', 'MLP'}
    assert set(trainings.keys()) == {'Lag1', 'Ridge', 'MLP'}
    assert (table['n_test'] > 0).all()

def test_evaluate_returns_strategy_columns(long_synthetic_ohlcv: pd.DataFrame) -> None:
    spec = BenchmarkSpec(symbol='TEST/USDT', timeframe='5m', target_kind='direction', train_days=14, test_days=3, fee_bps=10.0, slippage_bps=2.0)
    row, _ = evaluate_one(long_synthetic_ohlcv, spec, 'Lag1')
    assert 'strategy_sharpe' in row
    assert 'strategy_max_drawdown' in row
