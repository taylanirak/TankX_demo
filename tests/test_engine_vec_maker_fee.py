from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tankx.backtest.engine_vec import BacktestConfig, run_backtest


def _frame_from_close(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC', name='timestamp')
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': np.full(n, 10.0)}, index=idx)

def test_default_config_unchanged() -> None:
    cfg = BacktestConfig(fee_bps=10.0, slippage_bps=2.0)
    assert cfg.assume_maker is False
    assert cfg._effective_fee_bps == 10.0
    assert cfg.cost_per_change_fraction == pytest.approx(0.0012)

def test_maker_mode_uses_maker_fee() -> None:
    cfg = BacktestConfig(fee_bps=10.0, fee_bps_maker=0.0, slippage_bps=0.0, assume_maker=True)
    assert cfg._effective_fee_bps == 0.0
    assert cfg.cost_per_change_fraction == 0.0

def test_maker_mode_with_nonzero_maker_fee() -> None:
    cfg = BacktestConfig(fee_bps=10.0, fee_bps_maker=2.0, slippage_bps=1.0, assume_maker=True)
    assert cfg.cost_per_change_fraction == pytest.approx((2.0 + 1.0) / 10000.0)

def test_taker_run_charges_more_than_maker_run() -> None:
    close = np.array([100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0, 100.0, 101.0])
    df = _frame_from_close(close)
    pos = pd.Series([0, 1, 1, 1, 0, 0, -1, -1, 0, 0], dtype=float, index=df.index)
    taker = run_backtest(df, pos, BacktestConfig(fee_bps=10.0, slippage_bps=0.0, execution='next_close'))
    maker = run_backtest(df, pos, BacktestConfig(fee_bps=10.0, fee_bps_maker=0.0, slippage_bps=0.0, assume_maker=True, execution='next_close'))
    assert maker.equity.iloc[-1] > taker.equity.iloc[-1]
    assert (maker.fees == 0.0).all()
    assert (taker.fees > 0.0).any()

def test_assume_maker_false_uses_taker_even_if_maker_zero() -> None:
    close = np.array([100.0, 101.0, 102.0, 103.0, 100.0])
    df = _frame_from_close(close)
    pos = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0], index=df.index)
    cfg = BacktestConfig(fee_bps=20.0, fee_bps_maker=0.0, slippage_bps=0.0, assume_maker=False, execution='next_close')
    result = run_backtest(df, pos, cfg)
    assert (result.fees > 0).any()
