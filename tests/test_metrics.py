from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tankx.stats import metrics


def test_total_return() -> None:
    eq = pd.Series([100.0, 110.0, 121.0])
    assert metrics.total_return(eq) == pytest.approx(0.21)

def test_total_return_empty() -> None:
    assert metrics.total_return(pd.Series(dtype=float)) == 0.0

def test_sharpe_constant_returns_returns_nan_or_inf() -> None:
    constant = pd.Series([0.001] * 100)
    sr = metrics.annualized_sharpe(constant, bars_per_year=252)
    assert math.isnan(sr)

def test_sharpe_signs_correctly() -> None:
    rng = np.random.default_rng(seed=0)
    profit = pd.Series(rng.normal(0.001, 0.005, 1000))
    loss = pd.Series(rng.normal(-0.001, 0.005, 1000))
    assert metrics.annualized_sharpe(profit, 252) > 0
    assert metrics.annualized_sharpe(loss, 252) < 0

def test_sortino_no_downside_is_positive_inf_when_mean_positive() -> None:
    arr = pd.Series([0.01, 0.02, 0.005, 0.03])
    out = metrics.annualized_sortino(arr, bars_per_year=252)
    assert out == float('inf')

def test_max_drawdown_simple() -> None:
    eq = pd.Series([100.0, 110.0, 90.0, 95.0, 80.0, 100.0])
    mdd, peak, trough = metrics.max_drawdown(eq)
    assert mdd == pytest.approx(-30.0 / 110.0)
    assert peak == eq.index[1]
    assert trough == eq.index[4]

def test_max_drawdown_no_loss() -> None:
    eq = pd.Series([100.0, 110.0, 120.0])
    mdd, _, _ = metrics.max_drawdown(eq)
    assert mdd == 0.0

def test_calmar() -> None:
    bars = list(range(101))
    eq = pd.Series([1.0 + 0.01 * b for b in bars] + [1.5])
    bpy = 252
    cagr = metrics.annualized_return(eq, bpy)
    mdd, *_ = metrics.max_drawdown(eq)
    expected = cagr / abs(mdd)
    assert metrics.calmar(eq, bpy) == pytest.approx(expected)

def test_turnover_per_year() -> None:
    pos = pd.Series([0.0] * 50 + [1.0] * 50)
    bpy = 105120
    expected = 1.0 * bpy / 100
    assert metrics.turnover_per_year(pos, bpy) == pytest.approx(expected)

def test_hit_rate() -> None:
    trades = pd.Series([0.01, -0.005, 0.02, -0.01, 0.003])
    assert metrics.hit_rate(trades) == pytest.approx(3 / 5)

def test_profit_factor() -> None:
    trades = pd.Series([1.0, -0.5, 2.0, -1.5])
    pf = metrics.profit_factor(trades)
    assert pf == pytest.approx(3.0 / 2.0)

def test_summarize_returns_full_dict() -> None:
    eq = pd.Series([100.0, 101.0, 102.0, 99.0, 103.0])
    rets = eq.pct_change().fillna(0.0)
    pos = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0])
    out = metrics.summarize(eq, rets, pos, trade_pnls=pd.Series([0.01, -0.03, 0.04]), bars_per_year=252)
    assert {'total_return', 'sharpe', 'max_drawdown', 'calmar', 'num_trades', 'hit_rate', 'profit_factor'} <= out.keys()
