from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tankx.ml.models.base import TargetKind


@dataclass
class XGBoostPredictor:
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    target_kind: TargetKind = 'direction'
    seed: int = 0
    name: str = 'XGBoost'
    _model: object | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import xgboost as xgb
        common = {'n_estimators': self.n_estimators, 'max_depth': self.max_depth, 'learning_rate': self.learning_rate, 'subsample': self.subsample, 'colsample_bytree': self.colsample_bytree, 'random_state': self.seed, 'verbosity': 0, 'n_jobs': 1, 'tree_method': 'hist'}
        if self.target_kind == 'direction':
            model: object = xgb.XGBClassifier(eval_metric='logloss', **common)
        else:
            model = xgb.XGBRegressor(**common)
        model.fit(X, y)  # type: ignore[attr-defined]
        self._model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('XGBoostPredictor.predict called before fit')
        if self.target_kind == 'direction':
            proba = self._model.predict_proba(X)[:, 1]  # type: ignore[attr-defined]
            return np.asarray(proba, dtype=np.float64)
        return np.asarray(self._model.predict(X), dtype=np.float64)  # type: ignore[attr-defined]

@dataclass
class LightGBMPredictor:
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    num_leaves: int = 15
    target_kind: TargetKind = 'direction'
    seed: int = 0
    name: str = 'LightGBM'
    _model: object | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import lightgbm as lgb
        common: dict[str, Any] = {'n_estimators': self.n_estimators, 'max_depth': self.max_depth, 'learning_rate': self.learning_rate, 'subsample': self.subsample, 'colsample_bytree': self.colsample_bytree, 'num_leaves': self.num_leaves, 'random_state': self.seed, 'verbose': -1, 'n_jobs': 1, 'min_child_samples': 20}
        if self.target_kind == 'direction':
            model: object = lgb.LGBMClassifier(**common)
        else:
            model = lgb.LGBMRegressor(**common)
        model.fit(X, y)  # type: ignore[attr-defined]
        self._model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('LightGBMPredictor.predict called before fit')
        if self.target_kind == 'direction':
            proba = self._model.predict_proba(X)[:, 1]  # type: ignore[attr-defined]
            return np.asarray(proba, dtype=np.float64)
        return np.asarray(self._model.predict(X), dtype=np.float64)  # type: ignore[attr-defined]
__all__ = ['LightGBMPredictor', 'XGBoostPredictor']
