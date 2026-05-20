from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedScaler:
    scaler: StandardScaler
    feature_names: list[str]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.scaler.transform(X), dtype=np.float64)

def fit_transform_split(X_train: np.ndarray, X_test: np.ndarray, *, feature_names: list[str] | None=None) -> tuple[np.ndarray, np.ndarray, FittedScaler]:
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError('X_train and X_test must be 2-D')
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f'Feature dim mismatch: train has {X_train.shape[1]}, test has {X_test.shape[1]}')
    scaler = StandardScaler()
    scaler.fit(X_train)
    Xt = np.asarray(scaler.transform(X_train), dtype=np.float64)
    Xv = np.asarray(scaler.transform(X_test), dtype=np.float64)
    return (Xt, Xv, FittedScaler(scaler=scaler, feature_names=list(feature_names) if feature_names else []))
__all__ = ['FittedScaler', 'fit_transform_split']
