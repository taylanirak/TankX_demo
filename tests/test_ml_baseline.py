from __future__ import annotations

import numpy as np
import pytest

from tankx.ml.models.base import Predictor
from tankx.ml.models.baseline import Lag1Predictor, MLPPredictor, RidgePredictor


@pytest.fixture
def synthetic_xy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 1000
    n_feats = 8
    X = rng.normal(0.0, 1.0, size=(n, n_feats))
    y_return = 0.2 * X[:, 0] + rng.normal(0.0, 1.0, size=n)
    y_direction = (y_return > 0).astype(np.float64)
    return (X, y_direction, y_return)

def test_lag1_satisfies_protocol() -> None:
    assert isinstance(Lag1Predictor(), Predictor)

def test_lag1_direction_predicts_sign_of_first_column(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, _, _ = synthetic_xy
    pred = Lag1Predictor(target_kind='direction').predict(X)
    assert ((pred > 0.5) == (X[:, 0] > 0)).all()

def test_lag1_return_returns_first_column(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, _, _ = synthetic_xy
    pred = Lag1Predictor(target_kind='return').predict(X)
    np.testing.assert_array_equal(pred, X[:, 0])

def test_lag1_fit_is_noop(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y_dir, _ = synthetic_xy
    m = Lag1Predictor()
    m.fit(X, y_dir)
    np.testing.assert_array_equal(m.predict(X), Lag1Predictor().predict(X))

def test_ridge_classification_runs(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y_dir, _ = synthetic_xy
    m = RidgePredictor(target_kind='direction')
    m.fit(X, y_dir)
    pred = m.predict(X)
    assert pred.shape == y_dir.shape
    assert ((pred >= 0) & (pred <= 1)).all()

def test_ridge_regression_runs(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, _, y_ret = synthetic_xy
    m = RidgePredictor(target_kind='return', alpha=0.1)
    m.fit(X, y_ret)
    pred = m.predict(X)
    assert pred.shape == y_ret.shape
    corr = np.corrcoef(pred, y_ret)[0, 1]
    assert corr > 0.1

def test_ridge_predict_before_fit_raises() -> None:
    m = RidgePredictor()
    with pytest.raises(RuntimeError, match='before fit'):
        m.predict(np.zeros((5, 3)))

def test_ridge_satisfies_protocol() -> None:
    assert isinstance(RidgePredictor(), Predictor)

def test_mlp_classification_runs(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y_dir, _ = synthetic_xy
    m = MLPPredictor(hidden_units=(16,), max_iter=50, target_kind='direction')
    m.fit(X, y_dir)
    pred = m.predict(X)
    assert pred.shape == y_dir.shape
    assert ((pred >= 0) & (pred <= 1)).all()

def test_mlp_regression_runs(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, _, y_ret = synthetic_xy
    m = MLPPredictor(hidden_units=(16,), max_iter=50, target_kind='return')
    m.fit(X, y_ret)
    pred = m.predict(X)
    assert pred.shape == y_ret.shape

def test_mlp_predict_before_fit_raises() -> None:
    m = MLPPredictor()
    with pytest.raises(RuntimeError, match='before fit'):
        m.predict(np.zeros((5, 3)))

def test_mlp_satisfies_protocol() -> None:
    assert isinstance(MLPPredictor(), Predictor)

def test_mlp_deterministic_given_seed(synthetic_xy: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y_dir, _ = synthetic_xy
    a = MLPPredictor(hidden_units=(8,), max_iter=30, seed=42)
    b = MLPPredictor(hidden_units=(8,), max_iter=30, seed=42)
    a.fit(X, y_dir)
    b.fit(X, y_dir)
    np.testing.assert_allclose(a.predict(X), b.predict(X))
