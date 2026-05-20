from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

from tankx.ml.models.base import TargetKind


@dataclass
class Lag1Predictor:
    target_kind: TargetKind = 'direction'
    name: str = 'Lag1'
    _params: dict[str, object] = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        ret_lag1 = X[:, 0]
        if self.target_kind == 'direction':
            return np.where(ret_lag1 > 0, 0.9, 0.1)
        return ret_lag1.copy()

@dataclass
class RidgePredictor:
    alpha: float = 1.0
    target_kind: TargetKind = 'direction'
    name: str = 'Ridge'
    _model: object | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.target_kind == 'direction':
            model: object = LogisticRegression(C=1.0 / max(self.alpha, 1e-09), max_iter=500, solver='lbfgs')
        else:
            model = Ridge(alpha=self.alpha)
        model.fit(X, y)  # type: ignore[attr-defined]
        self._model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('RidgePredictor.predict called before fit')
        if self.target_kind == 'direction':
            proba = self._model.predict_proba(X)[:, 1]  # type: ignore[attr-defined]
            return np.asarray(proba, dtype=np.float64)
        return np.asarray(self._model.predict(X), dtype=np.float64)  # type: ignore[attr-defined]

@dataclass
class MLPPredictor:
    hidden_units: tuple[int, ...] = (32,)
    target_kind: TargetKind = 'direction'
    max_iter: int = 200
    name: str = 'MLP'
    seed: int = 0
    _model: object | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.target_kind == 'direction':
            model: object = MLPClassifier(hidden_layer_sizes=self.hidden_units, max_iter=self.max_iter, random_state=self.seed, solver='adam', learning_rate_init=0.001, early_stopping=True, n_iter_no_change=5, validation_fraction=0.1)
        else:
            model = MLPRegressor(hidden_layer_sizes=self.hidden_units, max_iter=self.max_iter, random_state=self.seed, solver='adam', learning_rate_init=0.001, early_stopping=True, n_iter_no_change=5, validation_fraction=0.1)
        model.fit(X, y)  # type: ignore[attr-defined]
        self._model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('MLPPredictor.predict called before fit')
        if self.target_kind == 'direction':
            proba = self._model.predict_proba(X)[:, 1]  # type: ignore[attr-defined]
            return np.asarray(proba, dtype=np.float64)
        return np.asarray(self._model.predict(X), dtype=np.float64)  # type: ignore[attr-defined]
__all__ = ['Lag1Predictor', 'MLPPredictor', 'RidgePredictor']
