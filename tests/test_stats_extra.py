from __future__ import annotations

import math

import numpy as np
import pytest

from tankx.stats.newey_west import auto_nw_lag, newey_west_sharpe
from tankx.stats.psr import probabilistic_sharpe_ratio


def test_auto_nw_lag_small_sample() -> None:
    assert auto_nw_lag(5) >= 1
    assert auto_nw_lag(50) >= 1

def test_auto_nw_lag_large_sample() -> None:
    assert 8 <= auto_nw_lag(10000) <= 14

def test_newey_west_sharpe_matches_naive_when_no_autocorr() -> None:
    rng = np.random.default_rng(seed=2)
    returns = rng.normal(0.001, 0.01, 2000)
    out = newey_west_sharpe(returns, bars_per_year=252)
    naive = returns.mean() / returns.std(ddof=1) * math.sqrt(252)
    assert math.isclose(out.sharpe, naive, abs_tol=1e-09)
    assert out.standard_error > 0
    assert out.lag >= 1

def test_newey_west_handles_zero_std() -> None:
    constant = np.ones(50) * 0.001
    out = newey_west_sharpe(constant, bars_per_year=252)
    assert math.isnan(out.sharpe)

def test_newey_west_too_few_observations() -> None:
    out = newey_west_sharpe(np.array([0.01, 0.02]), bars_per_year=252)
    assert math.isnan(out.sharpe)

def test_psr_high_for_strong_signal() -> None:
    rng = np.random.default_rng(seed=42)
    returns = rng.normal(0.001, 0.001, 2000)
    psr = probabilistic_sharpe_ratio(returns)
    assert 0.99 <= psr <= 1.0

def test_psr_around_50_for_zero_mean() -> None:
    psrs: list[float] = []
    for seed in range(40):
        rng = np.random.default_rng(seed=seed)
        returns = rng.normal(0.0, 0.01, 2000)
        psrs.append(probabilistic_sharpe_ratio(returns))
    mean_psr = float(np.mean(psrs))
    assert 0.3 <= mean_psr <= 0.7

def test_psr_low_for_negative_signal() -> None:
    rng = np.random.default_rng(seed=42)
    returns = rng.normal(-0.001, 0.001, 2000)
    psr = probabilistic_sharpe_ratio(returns)
    assert 0.0 <= psr <= 0.05

def test_psr_too_few_observations_returns_nan() -> None:
    assert math.isnan(probabilistic_sharpe_ratio(np.array([0.01, 0.02])))

def test_psr_handles_zero_std() -> None:
    assert math.isnan(probabilistic_sharpe_ratio(np.ones(100) * 0.001))

@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5])
def test_psr_in_unit_interval(seed: int) -> None:
    rng = np.random.default_rng(seed=seed)
    returns = rng.normal(0.0001 * (seed - 3), 0.01, 1000)
    psr = probabilistic_sharpe_ratio(returns)
    assert 0.0 <= psr <= 1.0
