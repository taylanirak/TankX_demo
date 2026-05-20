from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from tankx.config import DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS

ExecutionPolicy = Literal['next_open', 'next_close']

@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10000.0
    fee_bps: float = DEFAULT_FEE_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    execution: ExecutionPolicy = 'next_open'
    fee_bps_maker: float = 0.0
    assume_maker: bool = False

    @property
    def _effective_fee_bps(self) -> float:
        return self.fee_bps_maker if self.assume_maker else self.fee_bps

    @property
    def cost_per_change_fraction(self) -> float:
        return (self._effective_fee_bps + self.slippage_bps) / 10000.0

@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    net_returns: pd.Series
    gross_returns: pd.Series
    positions: pd.Series
    raw_positions: pd.Series
    position_changes: pd.Series
    fees: pd.Series
    drawdown: pd.Series
    trades: pd.DataFrame
    config: BacktestConfig

def run_backtest(df: pd.DataFrame, positions: pd.Series, config: BacktestConfig=BacktestConfig()) -> BacktestResult:
    if not positions.index.equals(df.index):
        raise ValueError('positions.index does not equal df.index — backtester refuses to run to avoid silent misalignment.')
    if (df['close'] <= 0).any() or (df['open'] <= 0).any():
        raise ValueError('Non-positive prices in OHLCV; reject before backtest.')
    raw = positions.astype(float).fillna(0.0)
    if raw.abs().max() > 1.0:
        raise ValueError('Position magnitude > 1 detected — this engine assumes unit-sized positions.')
    lagged = raw.shift(1).fillna(0.0)
    lagged.name = 'position'
    if config.execution == 'next_open':
        next_open = df['open'].shift(-1)
        gross_returns = (next_open / df['open'] - 1.0).fillna(0.0)
    elif config.execution == 'next_close':
        gross_returns = df['close'].pct_change().fillna(0.0)
    else:
        raise ValueError(f'Unknown execution policy: {config.execution!r}')
    strategy_returns = lagged * gross_returns
    position_changes = lagged.diff().abs().fillna(lagged.abs())
    fees = position_changes * config.cost_per_change_fraction
    net_returns = strategy_returns - fees
    equity = config.initial_capital * (1.0 + net_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    trades = _reconstruct_trades(df, lagged, equity)
    return BacktestResult(equity=equity.rename('equity'), net_returns=net_returns.rename('net_returns'), gross_returns=gross_returns.rename('gross_returns'), positions=lagged, raw_positions=raw.rename('raw_position'), position_changes=position_changes.rename('position_change'), fees=fees.rename('fees'), drawdown=drawdown.rename('drawdown'), trades=trades, config=config)

def _reconstruct_trades(df: pd.DataFrame, lagged_positions: pd.Series, equity: pd.Series) -> pd.DataFrame:
    pos = lagged_positions.to_numpy()
    n = len(pos)
    if n == 0:
        return _empty_trades_frame()
    entries: list[dict[str, object]] = []
    in_trade = False
    entry_idx = 0
    entry_eq = float(equity.iloc[0])
    side = 0.0
    timestamps = df.index
    for i in range(n):
        cur = pos[i]
        if cur != 0.0 and (not in_trade):
            in_trade = True
            entry_idx = i
            entry_eq = float(equity.iloc[i - 1]) if i > 0 else float(equity.iloc[0])
            side = float(cur)
        elif in_trade and (cur == 0.0 or np.sign(cur) != np.sign(side)):
            exit_idx = i
            exit_eq = float(equity.iloc[exit_idx])
            entries.append({'entry_at': timestamps[entry_idx], 'exit_at': timestamps[exit_idx], 'side': int(side), 'bars_held': exit_idx - entry_idx, 'entry_equity': entry_eq, 'exit_equity': exit_eq, 'pnl_pct': exit_eq / entry_eq - 1.0 if entry_eq else 0.0})
            if cur != 0.0:
                in_trade = True
                entry_idx = i
                entry_eq = exit_eq
                side = float(cur)
            else:
                in_trade = False
    if in_trade:
        exit_idx = n - 1
        exit_eq = float(equity.iloc[exit_idx])
        entries.append({'entry_at': timestamps[entry_idx], 'exit_at': timestamps[exit_idx], 'side': int(side), 'bars_held': exit_idx - entry_idx, 'entry_equity': entry_eq, 'exit_equity': exit_eq, 'pnl_pct': exit_eq / entry_eq - 1.0 if entry_eq else 0.0})
    if not entries:
        return _empty_trades_frame()
    return pd.DataFrame(entries)

def _empty_trades_frame() -> pd.DataFrame:
    return pd.DataFrame({'entry_at': pd.Series(dtype='datetime64[ns, UTC]'), 'exit_at': pd.Series(dtype='datetime64[ns, UTC]'), 'side': pd.Series(dtype='int64'), 'bars_held': pd.Series(dtype='int64'), 'entry_equity': pd.Series(dtype='float64'), 'exit_equity': pd.Series(dtype='float64'), 'pnl_pct': pd.Series(dtype='float64')})
__all__ = ['BacktestConfig', 'BacktestResult', 'ExecutionPolicy', 'run_backtest']
