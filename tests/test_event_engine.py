from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tankx.backtest.engine_event import MMBacktestConfig, MMState, Quotes, run_mm_backtest
from tankx.backtest.execution import PoissonFillModel
from tankx.strategies.avellaneda_stoikov import AvellanedaStoikov


def test_fill_probability_decreases_with_distance() -> None:
    m = PoissonFillModel(A=0.5, k=1.5)
    p1 = m.fill_probability(1.0, dt_seconds=1.0)
    p5 = m.fill_probability(5.0, dt_seconds=1.0)
    p10 = m.fill_probability(10.0, dt_seconds=1.0)
    assert p1 > p5 > p10
    assert 0.0 <= p10 < p5 < p1 < 1.0

def test_fill_probability_negative_delta_is_one() -> None:
    m = PoissonFillModel()
    assert m.fill_probability(-1.0, dt_seconds=1.0) == 1.0

def test_fill_probability_zero_delta_increases_with_dt() -> None:
    m = PoissonFillModel(A=1.0, k=1.0)
    p_short = m.fill_probability(0.0, dt_seconds=1.0)
    p_long = m.fill_probability(0.0, dt_seconds=10.0)
    assert p_long > p_short

def test_draw_fills_returns_two_booleans() -> None:
    rng = np.random.default_rng(seed=0)
    m = PoissonFillModel()
    a, b = m.draw_fills(1.0, 1.0, 1.0, rng)
    assert isinstance(a, (bool, np.bool_))
    assert isinstance(b, (bool, np.bool_))

def _state(mid: float, q: float, horizon: float) -> MMState:
    return MMState(ts=pd.Timestamp('2026-05-15', tz='UTC'), mid=mid, inventory=q, cash=0.0, elapsed_seconds=1.0, horizon_left_seconds=horizon)

def test_as_reservation_equals_mid_at_zero_inventory() -> None:
    strat = AvellanedaStoikov(gamma=0.01, sigma=5.0, k=1.5)
    q = strat.quotes(_state(mid=100.0, q=0.0, horizon=60.0))
    assert (q.bid + q.ask) / 2 == pytest.approx(100.0, abs=1e-09)

def test_as_long_inventory_shifts_quotes_down() -> None:
    strat = AvellanedaStoikov()
    q_flat = strat.quotes(_state(100.0, 0.0, 60.0))
    q_long = strat.quotes(_state(100.0, 1.0, 60.0))
    assert q_long.bid < q_flat.bid
    assert q_long.ask < q_flat.ask

def test_as_short_inventory_shifts_quotes_up() -> None:
    strat = AvellanedaStoikov()
    q_flat = strat.quotes(_state(100.0, 0.0, 60.0))
    q_short = strat.quotes(_state(100.0, -1.0, 60.0))
    assert q_short.bid > q_flat.bid
    assert q_short.ask > q_flat.ask

def test_as_spread_widens_with_remaining_horizon() -> None:
    strat = AvellanedaStoikov()
    q_short = strat.quotes(_state(100.0, 0.0, 1.0))
    q_long = strat.quotes(_state(100.0, 0.0, 1000.0))
    spread_short = q_short.ask - q_short.bid
    spread_long = q_long.ask - q_long.bid
    assert spread_long > spread_short

def test_as_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match='gamma'):
        AvellanedaStoikov(gamma=0.0)
    with pytest.raises(ValueError, match='sigma'):
        AvellanedaStoikov(sigma=-1.0)
    with pytest.raises(ValueError, match='k'):
        AvellanedaStoikov(k=0.0)

def _mid_series(n: int=600) -> pd.Series:
    idx = pd.date_range('2026-05-15', periods=n, freq='1s', tz='UTC', name='ts')
    rng = np.random.default_rng(seed=0)
    return pd.Series(50000 + np.cumsum(rng.normal(0, 0.5, n)), index=idx)

def test_event_engine_runs_end_to_end() -> None:
    mid = _mid_series()
    strat = AvellanedaStoikov(gamma=0.01, sigma=5.0, k=1.5)
    fm = PoissonFillModel(A=2.0, k=1.5)
    result = run_mm_backtest(mid, strat, fm, MMBacktestConfig(order_size=0.001))
    assert len(result.timestamps) == len(mid)
    assert result.cash.iloc[0] == result.config.initial_cash
    expected_eq = result.cash + result.inventory * result.mid
    pd.testing.assert_series_equal(result.equity, expected_eq.rename('equity'))

def test_event_engine_requires_tz_aware_index() -> None:
    mid = pd.Series([100.0, 101.0], index=pd.DatetimeIndex(['2026-01-01', '2026-01-02']))
    strat = AvellanedaStoikov()
    with pytest.raises(ValueError, match='tz-aware'):
        run_mm_backtest(mid, strat)

def test_event_engine_inventory_cap_respected() -> None:
    n = 1000
    idx = pd.date_range('2026-05-15', periods=n, freq='1s', tz='UTC', name='ts')
    mid = pd.Series(50000.0, index=idx)
    strat = AvellanedaStoikov(gamma=0.001, sigma=1.0, k=1.5)
    fm = PoissonFillModel(A=10.0, k=1.5)
    cfg = MMBacktestConfig(order_size=0.01, inventory_cap=0.05)
    result = run_mm_backtest(mid, strat, fm, cfg)
    assert result.inventory.abs().max() <= cfg.inventory_cap + 1e-12

def test_event_engine_trades_log_consistent_with_inventory() -> None:
    n = 800
    idx = pd.date_range('2026-05-15', periods=n, freq='1s', tz='UTC', name='ts')
    mid = pd.Series(50000.0 + np.linspace(0, 50, n), index=idx)
    strat = AvellanedaStoikov(gamma=0.01, sigma=2.0, k=1.5)
    fm = PoissonFillModel(A=3.0, k=1.5)
    result = run_mm_backtest(mid, strat, fm, MMBacktestConfig(order_size=0.01, inventory_cap=1.0))
    if len(result.trades) > 0:
        buys = result.trades.loc[result.trades['side'] == 'buy', 'qty'].sum()
        sells = result.trades.loc[result.trades['side'] == 'sell', 'qty'].sum()
        assert math.isclose(buys - sells, result.inventory.iloc[-1], abs_tol=1e-09)

def test_engine_zero_quote_size_zero_pnl() -> None:

    class ZeroFillStrategy:
        name = 'Wide'

        def quotes(self, state: MMState) -> Quotes:
            return Quotes(bid=state.mid - 1000000.0, ask=state.mid + 1000000.0)
    mid = _mid_series(300)
    result = run_mm_backtest(mid, ZeroFillStrategy(), config=MMBacktestConfig())
    assert (result.equity == result.config.initial_cash).all()
    assert len(result.trades) == 0
