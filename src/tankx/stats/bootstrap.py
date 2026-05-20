from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float
    confidence: float
    n_iter: int
    block_size: float

def auto_block_size(returns: np.ndarray) -> float:
    n = len(returns)
    if n < 10:
        return 2.0
    return max(2.0, min(n / 10.0, 1.75 * n ** (1 / 3)))

def stationary_bootstrap_indices(n: int, mean_block_size: float, rng: np.random.Generator) -> np.ndarray:
    if mean_block_size < 1.0:
        raise ValueError('mean_block_size must be >= 1.0')
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    p = 1.0 / mean_block_size
    indices = np.empty(n, dtype=np.int64)
    indices[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            indices[t] = rng.integers(0, n)
        else:
            indices[t] = (indices[t - 1] + 1) % n
    return indices

def stationary_bootstrap_statistic(returns: np.ndarray, statistic: object, *, n_iter: int=2000, confidence: float=0.95, mean_block_size: float | None=None, seed: int=0) -> BootstrapCI:
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size < 10:
        raise ValueError('Bootstrap requires at least 10 non-NaN observations')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be in (0, 1)')
    L = float(mean_block_size if mean_block_size is not None else auto_block_size(arr))
    rng = np.random.default_rng(seed=seed)
    point = float(statistic(arr))  # type: ignore[operator]
    samples = np.empty(n_iter, dtype=np.float64)
    n = arr.size
    for i in range(n_iter):
        idx = stationary_bootstrap_indices(n, L, rng)
        samples[i] = float(statistic(arr[idx]))  # type: ignore[operator]
    samples = samples[~np.isnan(samples)]
    alpha = 1.0 - confidence
    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return BootstrapCI(point=point, lower=lower, upper=upper, confidence=confidence, n_iter=n_iter, block_size=L)

def bootstrap_sharpe_ci(returns: np.ndarray, bars_per_year: int, *, n_iter: int=2000, confidence: float=0.95, mean_block_size: float | None=None, seed: int=0) -> BootstrapCI:

    def sharpe(arr: np.ndarray) -> float:
        if arr.size < 2 or arr.std(ddof=1) == 0.0:
            return float('nan')
        return float(arr.mean() / arr.std(ddof=1) * math.sqrt(bars_per_year))
    return stationary_bootstrap_statistic(returns, sharpe, n_iter=n_iter, confidence=confidence, mean_block_size=mean_block_size, seed=seed)
__all__ = ['BootstrapCI', 'auto_block_size', 'bootstrap_sharpe_ci', 'stationary_bootstrap_indices', 'stationary_bootstrap_statistic']
