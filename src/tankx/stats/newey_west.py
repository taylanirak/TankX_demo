from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HACSharpe:
    sharpe: float
    standard_error: float
    t_stat: float
    bars_per_year: int
    lag: int

def auto_nw_lag(n: int) -> int:
    if n < 20:
        return max(1, n // 5)
    return math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))

def newey_west_sharpe(returns: np.ndarray, bars_per_year: int, *, lag: int | None=None) -> HACSharpe:
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    T = arr.size
    if T < 3:
        return HACSharpe(float('nan'), float('nan'), float('nan'), bars_per_year, 0)
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    if sigma == 0:
        return HACSharpe(float('nan'), float('nan'), float('nan'), bars_per_year, 0)
    sr_per_bar = mu / sigma
    L = lag if lag is not None else auto_nw_lag(T)
    L = max(1, min(L, T - 2))
    centered = arr - mu
    gamma0 = float((centered ** 2).mean())
    lrv = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        gamma_k = float((centered[:-k] * centered[k:]).mean())
        lrv += 2.0 * w * gamma_k
    adj = lrv / gamma0 if gamma0 > 0 else 1.0
    se_per_bar = math.sqrt(max(1e-12, (1.0 + 0.5 * sr_per_bar ** 2) * adj / T))
    annualization = math.sqrt(bars_per_year)
    sr_annual = sr_per_bar * annualization
    se_annual = se_per_bar * annualization
    t_stat = sr_per_bar / se_per_bar if se_per_bar > 0 else float('nan')
    return HACSharpe(sharpe=sr_annual, standard_error=se_annual, t_stat=t_stat, bars_per_year=bars_per_year, lag=L)
__all__ = ['HACSharpe', 'auto_nw_lag', 'newey_west_sharpe']
