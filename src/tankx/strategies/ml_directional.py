from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tankx.ml.evaluate import predictions_to_positions
from tankx.ml.features import FeatureConfig, build_features
from tankx.ml.models.base import Predictor, TargetKind
from tankx.ml.models.registry import make_predictor
from tankx.ml.scaling import fit_transform_split
from tankx.ml.targets import align_features_and_target, build_direction_target, build_return_target


@dataclass
class MLDirectionalStrategy:
    model_name: str = 'Ridge'
    target_kind: TargetKind = 'direction'
    train_fraction: float = 0.6
    direction_band: float = 0.0
    return_band: float = 0.0
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = ''
    params: dict[str, Any] = field(init=False, default_factory=dict)
    _predictor: Predictor | None = None
    _fitted: bool = False
    _feature_columns: list[str] = field(default_factory=list)
    _scaler_mean: np.ndarray | None = None
    _scaler_std: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not 0.1 <= self.train_fraction < 1.0:
            raise ValueError('train_fraction must be in [0.1, 1.0)')
        if not self.name:
            object.__setattr__(self, 'name', f'ML[{self.model_name}/{self.target_kind}]')
        object.__setattr__(self, 'params', {'model_name': self.model_name, 'target_kind': self.target_kind, 'train_fraction': self.train_fraction, 'direction_band': self.direction_band, 'return_band': self.return_band, **self.model_kwargs})

    def _build_predictor(self) -> Predictor:
        return make_predictor(self.model_name, target_kind=self.target_kind, **self.model_kwargs)

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        feats = build_features(df, self.feature_config)
        if self.target_kind == 'direction':
            target = build_direction_target(df)
        else:
            target = build_return_target(df)
        X, y = align_features_and_target(feats, target)
        if len(X) < 50:
            return pd.Series(0.0, index=df.index, name='position')
        split = max(10, int(self.train_fraction * len(X)))
        X_train = X.iloc[:split].to_numpy(dtype=np.float64)
        y_train = y.iloc[:split].to_numpy(dtype=np.float64)
        X_test = X.iloc[split:].to_numpy(dtype=np.float64)
        if len(X_test) == 0:
            return pd.Series(0.0, index=df.index, name='position')
        X_train_scaled, X_test_scaled, _ = fit_transform_split(X_train, X_test)
        predictor = self._build_predictor()
        predictor.fit(X_train_scaled, y_train)
        preds = predictor.predict(X_test_scaled)
        pred_idx = X.iloc[split:].index
        pred_series = pd.Series(preds, index=pred_idx, name='pred')
        positions = predictions_to_positions(pred_series, kind=self.target_kind, direction_band=self.direction_band, return_band=self.return_band)
        out = pd.Series(0.0, index=df.index, name='position')
        out.loc[positions.index] = positions.to_numpy()
        self._predictor = predictor
        self._fitted = True
        return out
__all__ = ['MLDirectionalStrategy']
