from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tankx.backtest.walkforward import make_windows
from tankx.ml.features import build_features
from tankx.ml.models.registry import MODEL_REGISTRY
from tankx.ml.targets import build_direction_target, build_return_target
from tankx.ml.training import fit_predict_walkforward


def _frame_from_close(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC', name='timestamp')
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    vol = np.full(n, 100.0)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': vol}, index=idx)

@given(split_at=st.integers(min_value=100, max_value=400), suffix_seed_a=st.integers(0, 2 ** 16), suffix_seed_b=st.integers(0, 2 ** 16))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_features_are_causal_property(split_at: int, suffix_seed_a: int, suffix_seed_b: int) -> None:
    n_total = 600
    base = np.random.default_rng(seed=999).normal(0.0, 0.5, split_at)
    base_close = 100.0 + np.cumsum(base)
    rng_a = np.random.default_rng(seed=suffix_seed_a)
    rng_b = np.random.default_rng(seed=suffix_seed_b)
    suffix_a = np.cumsum(rng_a.normal(0, 0.5, n_total - split_at))
    suffix_b = np.cumsum(rng_b.normal(0, 0.5, n_total - split_at))
    close_a = np.concatenate([base_close, base_close[-1] + suffix_a])
    close_b = np.concatenate([base_close, base_close[-1] + suffix_b])
    feats_a = build_features(_frame_from_close(close_a))
    feats_b = build_features(_frame_from_close(close_b))
    pd.testing.assert_frame_equal(feats_a.iloc[:split_at - 1], feats_b.iloc[:split_at - 1], check_dtype=False)

@given(split_at=st.integers(min_value=100, max_value=400))
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_targets_match_log_return_property(split_at: int) -> None:
    rng = np.random.default_rng(seed=split_at)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, 500))
    df = _frame_from_close(close)
    y_return = build_return_target(df)
    y_direction = build_direction_target(df)
    expected = np.log(close[1:] / close[:-1])
    np.testing.assert_allclose(y_return.iloc[:-1].to_numpy(), expected, rtol=1e-12)
    nonnan = y_direction.dropna()
    np.testing.assert_array_equal(nonnan.to_numpy().astype(bool), y_return.loc[nonnan.index].to_numpy() > 0)

def test_walkforward_prefix_invariance_lag1() -> None:
    n_total = 8640
    split_at = n_total // 2
    rng_base = np.random.default_rng(seed=7)
    prefix_close = 100.0 + np.cumsum(rng_base.normal(0, 0.5, split_at))

    def run(suffix_seed: int) -> tuple[pd.Series, pd.Timestamp]:
        rng = np.random.default_rng(seed=suffix_seed)
        suffix = np.cumsum(rng.normal(0, 0.5, n_total - split_at))
        close = np.concatenate([prefix_close, prefix_close[-1] + suffix])
        df = _frame_from_close(close)
        windows = make_windows(pd.DatetimeIndex(df.index), train_days=5, test_days=2)
        result = fit_predict_walkforward(df=df, factory=MODEL_REGISTRY['Lag1'], target_kind='direction', windows=windows, model_name='Lag1', symbol='TEST/USDT')
        return (result.predictions, df.index[split_at - 1])
    preds_a, split_ts = run(1)
    preds_b, _ = run(2)
    prefix_mask_a = preds_a.index < split_ts
    prefix_mask_b = preds_b.index < split_ts
    assert prefix_mask_a.sum() > 0, 'No prefix predictions to compare — adjust window sizes.'
    pd.testing.assert_series_equal(preds_a.loc[prefix_mask_a], preds_b.loc[prefix_mask_b])
