from __future__ import annotations

import math

import numpy as np
import pytest

from tankx.stats.bootstrap import auto_block_size, bootstrap_sharpe_ci, stationary_bootstrap_indices


def test_auto_block_size_grows_with_n() -> None:
    assert auto_block_size(np.zeros(20)) <= auto_block_size(np.zeros(2000))

def test_indices_have_correct_length() -> None:
    rng = np.random.default_rng(seed=42)
    idx = stationary_bootstrap_indices(n=100, mean_block_size=10.0, rng=rng)
    assert len(idx) == 100
    assert idx.min() >= 0
    assert idx.max() < 100

def test_bootstrap_ci_brackets_point_estimate_iid() -> None:
    rng = np.random.default_rng(seed=0)
    returns = rng.normal(0.001, 0.01, 2000)
    out = bootstrap_sharpe_ci(returns, bars_per_year=252, n_iter=400, seed=1)
    assert out.lower < out.point < out.upper or math.isclose(out.lower, out.point) or math.isclose(out.point, out.upper)

def test_bootstrap_ci_widens_with_higher_confidence() -> None:
    rng = np.random.default_rng(seed=0)
    returns = rng.normal(0.001, 0.01, 1000)
    ci90 = bootstrap_sharpe_ci(returns, bars_per_year=252, n_iter=400, confidence=0.9, seed=1)
    ci99 = bootstrap_sharpe_ci(returns, bars_per_year=252, n_iter=400, confidence=0.99, seed=1)
    assert ci99.upper - ci99.lower >= ci90.upper - ci90.lower

def test_bootstrap_coverage_on_ar1_series() -> None:
    rng_master = np.random.default_rng(seed=7)
    bars_per_year = 252
    true_mu = 0.001
    sigma = 0.01
    rho = 0.3
    n_trials = 50
    n_obs = 600
    inside = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(seed=int(rng_master.integers(0, 2 ** 31)))
        eps = rng.normal(0, sigma, n_obs)
        x = np.empty(n_obs)
        x[0] = eps[0] + true_mu
        for t in range(1, n_obs):
            x[t] = rho * (x[t - 1] - true_mu) + true_mu + eps[t]
        out = bootstrap_sharpe_ci(x, bars_per_year=bars_per_year, n_iter=200, confidence=0.9, seed=trial)
        true_sd = sigma / math.sqrt(1 - rho ** 2)
        true_sr = true_mu / true_sd * math.sqrt(bars_per_year)
        if out.lower <= true_sr <= out.upper:
            inside += 1
    coverage = inside / n_trials
    assert 0.6 <= coverage <= 1.0

def test_bootstrap_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError, match='at least 10'):
        bootstrap_sharpe_ci(np.array([0.01, 0.02]), bars_per_year=252)

def test_bootstrap_is_reproducible() -> None:
    rng_master = np.random.default_rng(seed=99)
    returns = rng_master.normal(0, 0.01, 500)
    a = bootstrap_sharpe_ci(returns, bars_per_year=252, n_iter=300, seed=7)
    b = bootstrap_sharpe_ci(returns, bars_per_year=252, n_iter=300, seed=7)
    assert a.lower == b.lower
    assert a.upper == b.upper
    assert a.point == b.point
