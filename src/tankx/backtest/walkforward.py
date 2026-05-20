from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tankx.backtest.engine_vec import BacktestConfig, BacktestResult, run_backtest
from tankx.stats.metrics import annualized_sharpe


@dataclass(frozen=True)
class WFWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.train_start < self.train_end <= self.test_start < self.test_end:
            raise ValueError(f'Invalid window ordering: {self.train_start} < {self.train_end} <= {self.test_start} < {self.test_end}')

def make_windows(index: pd.DatetimeIndex, *, train_days: int=60, test_days: int=14, step_days: int | None=None, anchored: bool=False) -> list[WFWindow]:
    if step_days is None:
        step_days = test_days
    if index.tz is None:
        raise ValueError('walk-forward windows require a tz-aware DatetimeIndex')
    if step_days <= 0:
        raise ValueError('step_days must be positive')
    first = index[0]
    last = index[-1]
    train_td = pd.Timedelta(days=train_days)
    test_td = pd.Timedelta(days=test_days)
    step_td = pd.Timedelta(days=step_days)
    windows: list[WFWindow] = []
    train_end = first + train_td
    while train_end + test_td <= last:
        train_start = first if anchored else max(first, train_end - train_td)
        windows.append(WFWindow(train_start=train_start, train_end=train_end, test_start=train_end, test_end=train_end + test_td))
        train_end = train_end + step_td
    return windows
StrategyFactory = Callable[..., Any]
ScoringFn = Callable[[BacktestResult], float]

def default_sharpe_score(bars_per_year: int) -> ScoringFn:

    def score(result: BacktestResult) -> float:
        return annualized_sharpe(result.net_returns, bars_per_year)
    return score

def iter_param_grid(grid: Mapping[str, Iterable[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

def _safe_score(score: float) -> float:
    return -math.inf if math.isnan(score) else score

def walk_forward(df: pd.DataFrame, strategy_factory: StrategyFactory, param_grid: Mapping[str, Iterable[Any]], windows: Iterable[WFWindow], *, config: BacktestConfig=BacktestConfig(), score: ScoringFn | None=None, bars_per_year: int | None=None) -> pd.DataFrame:
    if score is None:
        if bars_per_year is None:
            raise ValueError('Provide either `score` or `bars_per_year`')
        score = default_sharpe_score(bars_per_year)
    combos = iter_param_grid(param_grid)
    rows: list[dict[str, Any]] = []
    for window in windows:
        train = df.loc[window.train_start:window.train_end]
        test = df.loc[window.test_start:window.test_end]
        if len(train) == 0 or len(test) == 0:
            continue
        best_score = -math.inf
        best_params: dict[str, Any] | None = None
        for params in combos:
            try:
                strat = strategy_factory(**params)
            except (ValueError, TypeError):
                continue
            positions = strat.generate_positions(train)
            train_result = run_backtest(train, positions, config)
            s = _safe_score(score(train_result))
            if s > best_score:
                best_score = s
                best_params = params
        if best_params is None:
            continue
        strat_best = strategy_factory(**best_params)
        positions_test = strat_best.generate_positions(test)
        test_result = run_backtest(test, positions_test, config)
        test_score = score(test_result)
        rows.append({'train_start': window.train_start, 'train_end': window.train_end, 'test_start': window.test_start, 'test_end': window.test_end, 'train_score': float(best_score), 'test_score': float(test_score), 'params': best_params, 'n_trades_test': len(test_result.trades)})
    return pd.DataFrame(rows)
__all__ = ['WFWindow', 'default_sharpe_score', 'iter_param_grid', 'make_windows', 'walk_forward']
