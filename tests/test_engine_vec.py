from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tankx.backtest.engine_vec import BacktestConfig, run_backtest
from tankx.strategies.ma_crossover import MACrossover


def _frame(close: np.ndarray, *, open_: np.ndarray | None=None) -> pd.DataFrame:
    n = len(close)
    if open_ is None:
        open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 1
    low = np.minimum(open_, close) - 1
    idx = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC', name='timestamp')
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': np.full(n, 100.0)}, index=idx)

def test_zero_position_yields_zero_pnl(synthetic_ohlcv: pd.DataFrame) -> None:
    flat = pd.Series(0.0, index=synthetic_ohlcv.index, name='position')
    cfg = BacktestConfig(initial_capital=10000.0, fee_bps=10.0, slippage_bps=2.0)
    result = run_backtest(synthetic_ohlcv, flat, cfg)
    assert (result.net_returns == 0).all()
    assert (result.equity == 10000.0).all()
    assert len(result.trades) == 0

def test_constant_long_no_fees_matches_underlying_return() -> None:
    close = np.linspace(100.0, 200.0, 100)
    df = _frame(close)
    pos = pd.Series(1.0, index=df.index)
    cfg = BacktestConfig(initial_capital=1.0, fee_bps=0.0, slippage_bps=0.0, execution='next_close')
    result = run_backtest(df, pos, cfg)
    expected = close[-1] / close[0]
    assert abs(result.equity.iloc[-1] - expected) < 1e-09

def test_equity_starts_at_initial_capital(synthetic_ohlcv: pd.DataFrame) -> None:
    pos = pd.Series(0.0, index=synthetic_ohlcv.index)
    cfg = BacktestConfig(initial_capital=12345.0)
    result = run_backtest(synthetic_ohlcv, pos, cfg)
    assert result.equity.iloc[0] == pytest.approx(12345.0)

def test_misaligned_index_raises(synthetic_ohlcv: pd.DataFrame) -> None:
    bad = pd.Series(0.0, index=synthetic_ohlcv.index[:-1])
    with pytest.raises(ValueError, match='index does not equal'):
        run_backtest(synthetic_ohlcv, bad)

def test_oversized_position_raises(synthetic_ohlcv: pd.DataFrame) -> None:
    pos = pd.Series(2.0, index=synthetic_ohlcv.index)
    with pytest.raises(ValueError, match='magnitude'):
        run_backtest(synthetic_ohlcv, pos)

def test_drawdown_is_non_positive(synthetic_ohlcv: pd.DataFrame) -> None:
    strat = MACrossover(short_window=10, long_window=50)
    pos = strat.generate_positions(synthetic_ohlcv)
    result = run_backtest(synthetic_ohlcv, pos)
    assert (result.drawdown <= 1e-12).all()

def test_fees_proportional_to_position_changes() -> None:
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    df = _frame(close)
    pos = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=df.index)
    cfg = BacktestConfig(initial_capital=1.0, fee_bps=10.0, slippage_bps=0.0, execution='next_close')
    result = run_backtest(df, pos, cfg)
    assert result.fees.sum() == pytest.approx(0.002, abs=1e-12)

def test_trade_reconstruction_round_trip() -> None:
    n = 20
    close = 100.0 + np.arange(n) * 0.5
    df = _frame(close)
    pos = pd.Series([0.0] * 5 + [1.0] * 10 + [0.0] * 5, index=df.index)
    result = run_backtest(df, pos, BacktestConfig(fee_bps=0.0, slippage_bps=0.0, execution='next_close'))
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade['side'] == 1
    assert trade['bars_held'] >= 1

def test_short_position_realized_pnl() -> None:
    close = np.linspace(200.0, 100.0, 50)
    df = _frame(close)
    pos = pd.Series(-1.0, index=df.index)
    cfg = BacktestConfig(initial_capital=1.0, fee_bps=0.0, slippage_bps=0.0, execution='next_close')
    result = run_backtest(df, pos, cfg)
    assert result.equity.iloc[-1] > 1.0

def _make_close_series(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return 100.0 + np.cumsum(rng.normal(0.0, 0.5, n))

@pytest.mark.parametrize('split_at', [50, 100, 150, 200, 250])
def test_no_lookahead_swap_future(split_at: int) -> None:
    n = 400
    prefix_close = _make_close_series(seed=42, n=split_at)
    suffix_a = _make_close_series(seed=1, n=n - split_at) + prefix_close[-1] - 100.0
    suffix_b = _make_close_series(seed=2, n=n - split_at) + prefix_close[-1] - 100.0
    close_a = np.concatenate([prefix_close, suffix_a])
    close_b = np.concatenate([prefix_close, suffix_b])
    df_a = _frame(close_a)
    df_b = _frame(close_b)
    strat = MACrossover(short_window=10, long_window=30)
    pos_a = strat.generate_positions(df_a)
    pos_b = strat.generate_positions(df_b)
    res_a = run_backtest(df_a, pos_a, BacktestConfig(execution='next_close'))
    res_b = run_backtest(df_b, pos_b, BacktestConfig(execution='next_close'))
    past = df_a.index[:split_at]
    pd.testing.assert_series_equal(res_a.equity.loc[past], res_b.equity.loc[past], check_names=False)
    pd.testing.assert_series_equal(res_a.net_returns.loc[past], res_b.net_returns.loc[past], check_names=False)
    pd.testing.assert_series_equal(res_a.positions.loc[past], res_b.positions.loc[past], check_names=False)


@given(split_at=st.integers(min_value=10, max_value=150), suffix_seed_a=st.integers(0, 2 ** 16), suffix_seed_b=st.integers(0, 2 ** 16), short_window=st.integers(2, 8), long_window=st.integers(10, 30), fee_bps=st.floats(min_value=0.0, max_value=20.0))
@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_no_lookahead_property(split_at: int, suffix_seed_a: int, suffix_seed_b: int, short_window: int, long_window: int, fee_bps: float) -> None:
    if short_window >= long_window:
        return
    n_total = max(200, long_window + 100)
    prefix_close = _make_close_series(seed=999, n=split_at + long_window)
    rng_a = np.random.default_rng(seed=suffix_seed_a)
    rng_b = np.random.default_rng(seed=suffix_seed_b)
    suffix_a = np.cumsum(rng_a.normal(0, 0.5, n_total)) + prefix_close[-1] - 100.0
    suffix_b = np.cumsum(rng_b.normal(0, 0.5, n_total)) + prefix_close[-1] - 100.0
    close_a = np.concatenate([prefix_close, suffix_a + 100.0])
    close_b = np.concatenate([prefix_close, suffix_b + 100.0])
    df_a = _frame(close_a)
    df_b = _frame(close_b)
    strat = MACrossover(short_window=short_window, long_window=long_window)
    cfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=0.0, execution='next_close')
    res_a = run_backtest(df_a, strat.generate_positions(df_a), cfg)
    res_b = run_backtest(df_b, strat.generate_positions(df_b), cfg)
    past_idx = df_a.index[:split_at + long_window]
    pd.testing.assert_series_equal(res_a.equity.loc[past_idx], res_b.equity.loc[past_idx], check_names=False)
