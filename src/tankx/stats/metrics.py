from __future__ import annotations

import math

import numpy as np
import pandas as pd


def annualized_sharpe(returns: pd.Series | np.ndarray, bars_per_year: int, rf_per_bar: float=0.0) -> float:
    arr = np.asarray(returns, dtype=np.float64) - rf_per_bar
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return float('nan')
    sigma = float(arr.std(ddof=1))
    if sigma < 1e-15 * (abs(float(arr.mean())) + 1e-300):
        return float('nan')
    return float(arr.mean() / sigma * math.sqrt(bars_per_year))

def annualized_sortino(returns: pd.Series | np.ndarray, bars_per_year: int, mar_per_bar: float=0.0) -> float:
    arr = np.asarray(returns, dtype=np.float64) - mar_per_bar
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return float('nan')
    downside = arr[arr < 0]
    if downside.size == 0:
        return float('inf') if arr.mean() > 0 else float('nan')
    denom = math.sqrt((downside ** 2).mean())
    if denom == 0:
        return float('nan')
    return float(arr.mean() / denom * math.sqrt(bars_per_year))

def annualized_return(equity: pd.Series, bars_per_year: int) -> float:
    if len(equity) < 2:
        return float('nan')
    bars = len(equity) - 1
    total = float(equity.iloc[-1] / equity.iloc[0])
    if total <= 0:
        return float('nan')
    return float(total ** (bars_per_year / bars) - 1.0)

def total_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)

def max_drawdown(equity: pd.Series) -> tuple[float, object | None, object | None]:
    if len(equity) == 0:
        return (0.0, None, None)
    cummax = equity.cummax()
    drawdown = equity / cummax - 1.0
    trough_idx = drawdown.idxmin()
    mdd = float(drawdown.loc[trough_idx])
    peak_candidates = equity.loc[:trough_idx]
    peak_idx = peak_candidates[peak_candidates == cummax.loc[trough_idx]].index.max()
    return (mdd, peak_idx, trough_idx)

def calmar(equity: pd.Series, bars_per_year: int) -> float:
    cagr = annualized_return(equity, bars_per_year)
    mdd, _, _ = max_drawdown(equity)
    if mdd == 0 or math.isnan(cagr):
        return float('nan')
    return float(cagr / abs(mdd))

def turnover_per_year(positions: pd.Series, bars_per_year: int) -> float:
    n = len(positions)
    if n < 2:
        return 0.0
    changes = positions.diff().abs().fillna(0.0).sum()
    return float(changes * bars_per_year / n)

def hit_rate(trade_pnls: pd.Series | np.ndarray) -> float:
    arr = np.asarray(trade_pnls, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float('nan')
    return float((arr > 0).mean())

def profit_factor(trade_pnls: pd.Series | np.ndarray) -> float:
    arr = np.asarray(trade_pnls, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float('nan')
    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses == 0:
        return float('inf') if gains > 0 else float('nan')
    return float(gains / losses)

def summarize(equity: pd.Series, net_returns: pd.Series, positions: pd.Series, trade_pnls: pd.Series | None, bars_per_year: int) -> dict[str, float]:
    mdd, _, _ = max_drawdown(equity)
    out: dict[str, float] = {'total_return': total_return(equity), 'annualized_return': annualized_return(equity, bars_per_year), 'sharpe': annualized_sharpe(net_returns, bars_per_year), 'sortino': annualized_sortino(net_returns, bars_per_year), 'max_drawdown': mdd, 'calmar': calmar(equity, bars_per_year), 'turnover_per_year': turnover_per_year(positions, bars_per_year)}
    if trade_pnls is not None and len(trade_pnls):
        out['num_trades'] = float(len(trade_pnls))
        out['hit_rate'] = hit_rate(trade_pnls)
        out['profit_factor'] = profit_factor(trade_pnls)
    else:
        out['num_trades'] = 0.0
        out['hit_rate'] = float('nan')
        out['profit_factor'] = float('nan')
    return out
__all__ = ['annualized_return', 'annualized_sharpe', 'annualized_sortino', 'calmar', 'hit_rate', 'max_drawdown', 'profit_factor', 'summarize', 'total_return', 'turnover_per_year']
