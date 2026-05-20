from __future__ import annotations

import math

import numpy as np
from scipy import stats


def probabilistic_sharpe_ratio(returns: np.ndarray, *, sr_benchmark_per_bar: float=0.0) -> float:
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    T = arr.size
    if T < 4:
        return float('nan')
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    if sigma == 0 or sigma < 1e-15 * (abs(mu) + 1e-300):
        return float('nan')
    sr_hat = float(mu / sigma)
    skew = float(stats.skew(arr, bias=False))
    kurt = float(stats.kurtosis(arr, fisher=False, bias=False))
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2
    if denom <= 0:
        return float('nan')
    z = (sr_hat - sr_benchmark_per_bar) * math.sqrt(T - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))
__all__ = ['probabilistic_sharpe_ratio']
