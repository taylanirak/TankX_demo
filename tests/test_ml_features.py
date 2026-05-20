from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tankx.ml.features import FeatureConfig, build_features, feature_column_names
from tankx.ml.scaling import fit_transform_split
from tankx.ml.targets import align_features_and_target, build_direction_target, build_return_target


def test_build_features_columns_match_helper(synthetic_ohlcv: pd.DataFrame) -> None:
    cfg = FeatureConfig()
    feats = build_features(synthetic_ohlcv, cfg)
    assert list(feats.columns) == feature_column_names(cfg)

def test_build_features_index_aligns(synthetic_ohlcv: pd.DataFrame) -> None:
    feats = build_features(synthetic_ohlcv)
    assert feats.index.equals(synthetic_ohlcv.index)

def test_build_features_warmup_is_nan(synthetic_ohlcv: pd.DataFrame) -> None:
    cfg = FeatureConfig(n_lag_returns=5, rolling_windows=(10, 20, 60), volume_z_window=60)
    feats = build_features(synthetic_ohlcv, cfg)
    assert feats.iloc[:50].isna().any(axis=1).all()
    tail = feats.iloc[80:].copy()
    assert tail.notna().all().all()
    assert np.isfinite(tail.to_numpy()).all()

def test_build_features_is_causal(synthetic_ohlcv: pd.DataFrame) -> None:
    half = len(synthetic_ohlcv) // 2
    df_a = synthetic_ohlcv.copy()
    df_b = synthetic_ohlcv.copy()
    df_b.iloc[half:] = df_b.iloc[half:] * 1.5
    feats_a = build_features(df_a)
    feats_b = build_features(df_b)
    pd.testing.assert_frame_equal(feats_a.iloc[:half - 1], feats_b.iloc[:half - 1], check_dtype=False)

def test_build_features_rejects_missing_columns() -> None:
    bad = pd.DataFrame({'open': [1.0, 2.0]})
    with pytest.raises(KeyError):
        build_features(bad)

def test_feature_count_matches_config() -> None:
    cfg = FeatureConfig(n_lag_returns=3, rolling_windows=(5,), include_volume_z=True, include_tod_cyclic=True)
    expected = 3 + 2 * 1 + 1 + 2
    assert len(feature_column_names(cfg)) == expected

def test_return_target_is_next_bar_log_return(synthetic_ohlcv: pd.DataFrame) -> None:
    y = build_return_target(synthetic_ohlcv)
    close = synthetic_ohlcv['close'].to_numpy()
    expected = np.log(close[1:] / close[:-1])
    np.testing.assert_allclose(y.iloc[:-1].to_numpy(), expected, rtol=1e-12)
    assert np.isnan(y.iloc[-1])

def test_direction_target_is_binary(synthetic_ohlcv: pd.DataFrame) -> None:
    y = build_direction_target(synthetic_ohlcv)
    nonnan = y.dropna()
    assert set(np.unique(nonnan.to_numpy())) <= {0.0, 1.0}

def test_direction_target_with_neutral_band_masks_small_moves(synthetic_ohlcv: pd.DataFrame) -> None:
    y_strict = build_direction_target(synthetic_ohlcv)
    y_banded = build_direction_target(synthetic_ohlcv, neutral_band=0.001)
    assert y_banded.isna().sum() >= y_strict.isna().sum()

def test_align_drops_warmup_and_last_row(synthetic_ohlcv: pd.DataFrame) -> None:
    feats = build_features(synthetic_ohlcv)
    y = build_return_target(synthetic_ohlcv)
    X, y_clean = align_features_and_target(feats, y)
    assert len(X) == len(y_clean)
    assert X.notna().all().all()
    assert y_clean.notna().all()
    assert synthetic_ohlcv.index[-1] not in y_clean.index

def test_align_rejects_misaligned_indexes(synthetic_ohlcv: pd.DataFrame) -> None:
    feats = build_features(synthetic_ohlcv)
    y = build_return_target(synthetic_ohlcv).iloc[:-1]
    with pytest.raises(ValueError, match='same index'):
        align_features_and_target(feats, y)

def test_fit_transform_split_zero_mean_unit_var_on_train() -> None:
    rng = np.random.default_rng(seed=0)
    X_train = rng.normal(10.0, 3.0, size=(500, 5))
    X_test = rng.normal(10.0, 3.0, size=(100, 5))
    Xt, Xv, _ = fit_transform_split(X_train, X_test)
    assert Xt.shape == X_train.shape
    assert Xv.shape == X_test.shape
    np.testing.assert_allclose(Xt.mean(axis=0), 0, atol=1e-09)
    np.testing.assert_allclose(Xt.std(axis=0, ddof=0), 1, atol=1e-09)
    assert not np.allclose(Xv.mean(axis=0), 0, atol=0.01)

def test_fit_transform_split_dimension_mismatch_raises() -> None:
    X_train = np.zeros((10, 4))
    X_test = np.zeros((10, 3))
    with pytest.raises(ValueError, match='dim mismatch'):
        fit_transform_split(X_train, X_test)

def test_fit_transform_split_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match='2-D'):
        fit_transform_split(np.zeros(10), np.zeros(10))

def test_fit_transform_split_returns_carry_feature_names() -> None:
    X = np.zeros((20, 2))
    _, _, scaler = fit_transform_split(X, X, feature_names=['a', 'b'])
    assert scaler.feature_names == ['a', 'b']
