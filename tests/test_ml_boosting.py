from __future__ import annotations

import numpy as np
import pytest

from tankx.ml.models.base import Predictor
from tankx.ml.models.boosting import LightGBMPredictor, XGBoostPredictor


@pytest.fixture
def synthetic_xy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 1000
    n_feats = 8
    X = rng.normal(0.0, 1.0, size=(n, n_feats))
    y_return = 0.2 * X[:, 0] + rng.normal(0.0, 1.0, size=n)
    y_direction = (y_return > 0).astype(np.float64)
    return (X, y_direction, y_return)

def test_xgboost_satisfies_protocol() -> None:
    assert isinstance(XGBoostPredictor(), Predictor)

def test_xgboost_direction_runs(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y_dir, _ = synthetic_xy
    m = XGBoostPredictor(n_estimators=20, max_depth=3, target_kind='direction')
    m.fit(X, y_dir)
    pred = m.predict(X)
    assert pred.shape == y_dir.shape
    assert ((pred >= 0) & (pred <= 1)).all()

def test_xgboost_return_runs(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, _, y_ret = synthetic_xy
    m = XGBoostPredictor(n_estimators=20, max_depth=3, target_kind='return')
    m.fit(X, y_ret)
    pred = m.predict(X)
    assert pred.shape == y_ret.shape
    corr = np.corrcoef(pred, y_ret)[0, 1]
    assert corr > 0.1

def test_xgboost_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match='before fit'):
        XGBoostPredictor().predict(np.zeros((5, 3)))

def test_xgboost_deterministic_with_seed(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y_dir, _ = synthetic_xy
    a = XGBoostPredictor(n_estimators=20, seed=42)
    b = XGBoostPredictor(n_estimators=20, seed=42)
    a.fit(X, y_dir)
    b.fit(X, y_dir)
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=1e-09)

def test_lightgbm_satisfies_protocol() -> None:
    assert isinstance(LightGBMPredictor(), Predictor)

def test_lightgbm_direction_runs(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y_dir, _ = synthetic_xy
    m = LightGBMPredictor(n_estimators=20, num_leaves=15, target_kind='direction')
    m.fit(X, y_dir)
    pred = m.predict(X)
    assert pred.shape == y_dir.shape
    assert ((pred >= 0) & (pred <= 1)).all()

def test_lightgbm_return_runs(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, _, y_ret = synthetic_xy
    m = LightGBMPredictor(n_estimators=20, num_leaves=15, target_kind='return')
    m.fit(X, y_ret)
    pred = m.predict(X)
    assert pred.shape == y_ret.shape

def test_lightgbm_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match='before fit'):
        LightGBMPredictor().predict(np.zeros((5, 3)))

def test_lightgbm_deterministic_with_seed(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y_dir, _ = synthetic_xy
    a = LightGBMPredictor(n_estimators=20, seed=42)
    b = LightGBMPredictor(n_estimators=20, seed=42)
    a.fit(X, y_dir)
    b.fit(X, y_dir)
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=1e-09)

def test_registry_has_boosting_entries() -> None:
    from tankx.ml.models.registry import MODEL_REGISTRY, make_predictor
    assert 'XGBoost' in MODEL_REGISTRY
    assert 'LightGBM' in MODEL_REGISTRY
    xgb_p = make_predictor('XGBoost', target_kind='direction')
    assert isinstance(xgb_p, Predictor)
    lgb_p = make_predictor('LightGBM', target_kind='return')
    assert isinstance(lgb_p, Predictor)
