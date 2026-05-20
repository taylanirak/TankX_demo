from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from tankx.backtest.execution import PoissonFillModel


@dataclass(frozen=True)
class MMState:
    ts: pd.Timestamp
    mid: float
    inventory: float
    cash: float
    elapsed_seconds: float
    horizon_left_seconds: float

@dataclass(frozen=True)
class Quotes:
    bid: float
    ask: float

class QuotingStrategy(Protocol):
    name: str

    def quotes(self, state: MMState) -> Quotes:
        ...

@dataclass(frozen=True)
class MMBacktestConfig:
    order_size: float = 0.01
    inventory_cap: float = 0.5
    fee_bps_maker: float = 0.0
    initial_cash: float = 10000.0

@dataclass(frozen=True)
class MMBacktestResult:
    timestamps: pd.DatetimeIndex
    mid: pd.Series
    inventory: pd.Series
    cash: pd.Series
    equity: pd.Series
    bids: pd.Series
    asks: pd.Series
    trades: pd.DataFrame
    config: MMBacktestConfig

def run_mm_backtest(mid_series: pd.Series, strategy: QuotingStrategy, fill_model: PoissonFillModel | None=None, config: MMBacktestConfig=MMBacktestConfig(), *, seed: int=0) -> MMBacktestResult:
    if fill_model is None:
        fill_model = PoissonFillModel()
    ts = pd.DatetimeIndex(mid_series.index)
    if ts.tz is None:
        raise ValueError('mid_series must be tz-aware')
    if not ts.is_monotonic_increasing:
        raise ValueError('mid_series index must be sorted')
    mids = mid_series.to_numpy(dtype=np.float64)
    n = len(ts)
    if n < 2:
        raise ValueError('Need >= 2 mid observations to run a simulation')
    inventory = np.zeros(n, dtype=np.float64)
    cash = np.empty(n, dtype=np.float64)
    bids = np.empty(n, dtype=np.float64)
    asks = np.empty(n, dtype=np.float64)
    cash[0] = config.initial_cash
    bids[0] = np.nan
    asks[0] = np.nan
    rng = np.random.default_rng(seed=seed)
    trade_rows: list[dict[str, object]] = []
    fee = config.fee_bps_maker / 10000.0
    for i in range(1, n):
        dt_s = max(1e-09, (ts[i] - ts[i - 1]).total_seconds())
        horizon_left = max(0.0, (ts[-1] - ts[i - 1]).total_seconds())
        state = MMState(ts=ts[i - 1], mid=float(mids[i - 1]), inventory=float(inventory[i - 1]), cash=float(cash[i - 1]), elapsed_seconds=dt_s, horizon_left_seconds=horizon_left)
        try:
            q = strategy.quotes(state)
        except Exception:
            raise
        bids[i] = q.bid
        asks[i] = q.ask
        bid_delta = state.mid - q.bid
        ask_delta = q.ask - state.mid
        bid_filled, ask_filled = fill_model.draw_fills(bid_delta, ask_delta, dt_s, rng)
        cur_inv = float(inventory[i - 1])
        cur_cash = float(cash[i - 1])
        if bid_filled and cur_inv + config.order_size <= config.inventory_cap:
            cost = q.bid * config.order_size
            cur_cash -= cost * (1.0 + fee)
            cur_inv += config.order_size
            trade_rows.append({'ts': ts[i], 'side': 'buy', 'price': q.bid, 'qty': config.order_size, 'inventory_after': cur_inv})
        if ask_filled and cur_inv - config.order_size >= -config.inventory_cap:
            proceeds = q.ask * config.order_size
            cur_cash += proceeds * (1.0 - fee)
            cur_inv -= config.order_size
            trade_rows.append({'ts': ts[i], 'side': 'sell', 'price': q.ask, 'qty': config.order_size, 'inventory_after': cur_inv})
        inventory[i] = cur_inv
        cash[i] = cur_cash
    equity = cash + inventory * mids
    trades = pd.DataFrame(trade_rows, columns=['ts', 'side', 'price', 'qty', 'inventory_after'])
    return MMBacktestResult(timestamps=ts, mid=pd.Series(mids, index=ts, name='mid'), inventory=pd.Series(inventory, index=ts, name='inventory'), cash=pd.Series(cash, index=ts, name='cash'), equity=pd.Series(equity, index=ts, name='equity'), bids=pd.Series(bids, index=ts, name='bid_quote'), asks=pd.Series(asks, index=ts, name='ask_quote'), trades=trades, config=config)
__all__ = ['MMBacktestConfig', 'MMBacktestResult', 'MMState', 'Quotes', 'QuotingStrategy', 'run_mm_backtest']
