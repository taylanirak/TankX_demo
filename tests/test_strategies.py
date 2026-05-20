from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tankx.strategies.base import VectorizedStrategy
from tankx.strategies.bollinger import BollingerMeanReversion
from tankx.strategies.ma_crossover import MACrossover


def _frame_from_close(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC', name='timestamp')
    return pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': np.full(n, 100.0)}, index=idx)

def test_ma_crossover_long_when_short_above_long() -> None:
    n = 300
    close = np.linspace(100.0, 200.0, n)
    df = _frame_from_close(close)
    strat = MACrossover(short_window=10, long_window=50)
    pos = strat.generate_positions(df)
    after_warmup = pos.iloc[50:]
    assert (after_warmup == 1.0).all()

def test_ma_crossover_flat_when_short_below_long() -> None:
    n = 300
    close = np.linspace(200.0, 100.0, n)
    df = _frame_from_close(close)
    strat = MACrossover(short_window=10, long_window=50)
    pos = strat.generate_positions(df)
    after_warmup = pos.iloc[50:]
    assert (after_warmup == 0.0).all()

def test_ma_crossover_warmup_is_nan() -> None:
    n = 200
    close = np.linspace(100.0, 200.0, n)
    df = _frame_from_close(close)
    strat = MACrossover(short_window=10, long_window=50)
    pos = strat.generate_positions(df)
    assert pos.iloc[:49].isna().all()
    assert pos.iloc[50:].notna().all()

def test_ma_crossover_position_aligned_with_input(synthetic_ohlcv: pd.DataFrame) -> None:
    strat = MACrossover(short_window=10, long_window=50)
    pos = strat.generate_positions(synthetic_ohlcv)
    assert pos.index.equals(synthetic_ohlcv.index)

def test_ma_crossover_rejects_short_geq_long() -> None:
    with pytest.raises(ValueError, match='short_window'):
        MACrossover(short_window=50, long_window=10)
    with pytest.raises(ValueError, match='short_window'):
        MACrossover(short_window=50, long_window=50)

def test_ma_crossover_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match='positive'):
        MACrossover(short_window=0, long_window=10)

def test_ma_crossover_satisfies_protocol() -> None:
    strat = MACrossover()
    assert isinstance(strat, VectorizedStrategy)
    assert strat.name == 'MA Crossover'
    assert 'short_window' in strat.params

def test_bollinger_enters_on_spike_below_lower_band() -> None:
    n = 100
    rng = np.random.default_rng(seed=1)
    close = 100.0 + rng.normal(0, 0.5, n).cumsum()
    close[50] -= 10.0
    df = _frame_from_close(close)
    strat = BollingerMeanReversion(window=20, num_std=2.0)
    pos = strat.generate_positions(df)
    assert pos.iloc[50] == 1.0

def test_bollinger_exits_above_mean() -> None:
    n = 100
    close = np.full(n, 100.0)
    close[30] = 80.0
    close[31:] = np.linspace(80, 110, n - 31)
    df = _frame_from_close(close)
    strat = BollingerMeanReversion(window=20, num_std=2.0)
    pos = strat.generate_positions(df)
    entered_at = pos.idxmax() if (pos == 1.0).any() else None
    assert entered_at is not None
    later = pos.loc[entered_at:]
    assert (later == 0.0).any()

def test_bollinger_warmup_nan() -> None:
    n = 50
    close = np.linspace(100, 110, n)
    df = _frame_from_close(close)
    strat = BollingerMeanReversion(window=20, num_std=2.0)
    pos = strat.generate_positions(df)
    assert pos.iloc[:19].isna().all()

def test_bollinger_position_values_only_zero_one(synthetic_ohlcv: pd.DataFrame) -> None:
    strat = BollingerMeanReversion(window=20, num_std=2.0)
    pos = strat.generate_positions(synthetic_ohlcv).dropna()
    assert set(np.unique(pos.to_numpy())).issubset({0.0, 1.0})

def test_bollinger_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match='window'):
        BollingerMeanReversion(window=1, num_std=2.0)
    with pytest.raises(ValueError, match='num_std'):
        BollingerMeanReversion(window=20, num_std=0)

def test_bollinger_satisfies_protocol() -> None:
    strat = BollingerMeanReversion()
    assert isinstance(strat, VectorizedStrategy)

def test_strategies_are_deterministic(synthetic_ohlcv: pd.DataFrame) -> None:
    for strat in (MACrossover(), BollingerMeanReversion()):
        a = strat.generate_positions(synthetic_ohlcv)
        b = strat.generate_positions(synthetic_ohlcv)
        pd.testing.assert_series_equal(a, b)
