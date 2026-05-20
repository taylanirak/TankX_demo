from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tankx.backtest.walkforward import WFWindow
from tankx.ml.features import FeatureConfig, build_features
from tankx.ml.labels import TripleBarrierConfig, triple_barrier_labels
from tankx.ml.models.base import Predictor, TargetKind
from tankx.ml.scaling import fit_transform_split
from tankx.ml.targets import align_features_and_target, build_direction_target, build_return_target

PredictorFactory = Callable[..., Predictor]

@dataclass(frozen=True)
class WindowPredictions:
    window: WFWindow
    pred: pd.Series
    actual: pd.Series
    history: list[dict[str, float]]

@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    target_kind: TargetKind
    symbol: str
    predictions: pd.Series
    actuals: pd.Series
    per_window: list[WindowPredictions]

def _build_target(df: pd.DataFrame, kind: TargetKind) -> pd.Series:
    if kind == 'direction':
        return build_direction_target(df)
    if kind == 'triple_barrier':
        return triple_barrier_labels(df, TripleBarrierConfig())
    return build_return_target(df)

def _model_target_kind(spec_kind: TargetKind) -> TargetKind:
    return 'return' if spec_kind == 'triple_barrier' else spec_kind

def fit_predict_walkforward(df: pd.DataFrame, factory: PredictorFactory, target_kind: TargetKind, windows: Iterable[WFWindow], *, feature_config: FeatureConfig=FeatureConfig(), model_kwargs: dict[str, object] | None=None, model_name: str='model', symbol: str='unknown') -> TrainingResult:
    if model_kwargs is None:
        model_kwargs = {}
    feats = build_features(df, feature_config)
    target = _build_target(df, target_kind)
    X_all, y_all = align_features_and_target(feats, target)
    pred_chunks: list[pd.Series] = []
    actual_chunks: list[pd.Series] = []
    per_window: list[WindowPredictions] = []
    for window in windows:
        train_mask = (X_all.index >= window.train_start) & (X_all.index < window.train_end)
        test_mask = (X_all.index >= window.test_start) & (X_all.index < window.test_end)
        X_train = X_all.loc[train_mask].to_numpy(dtype=np.float64)
        y_train = y_all.loc[train_mask].to_numpy(dtype=np.float64)
        X_test = X_all.loc[test_mask].to_numpy(dtype=np.float64)
        y_test = y_all.loc[test_mask].to_numpy(dtype=np.float64)
        test_idx = X_all.loc[test_mask].index
        if len(X_train) < 50 or len(X_test) < 1:
            continue
        Xt, Xv, _ = fit_transform_split(X_train, X_test)
        predictor = factory(target_kind=_model_target_kind(target_kind), **model_kwargs)
        predictor.fit(Xt, y_train)
        preds = predictor.predict(Xv)
        pred_s = pd.Series(preds, index=test_idx, name='pred')
        actual_s = pd.Series(y_test, index=test_idx, name='actual')
        pred_chunks.append(pred_s)
        actual_chunks.append(actual_s)
        history = list(getattr(predictor, 'history', []) or [])
        per_window.append(WindowPredictions(window=window, pred=pred_s, actual=actual_s, history=history))
    if pred_chunks:
        pred_all = pd.concat(pred_chunks).sort_index()
        act_all = pd.concat(actual_chunks).sort_index()
    else:
        pred_all = pd.Series(dtype=np.float64, name='pred')
        act_all = pd.Series(dtype=np.float64, name='actual')
    return TrainingResult(model_name=model_name, target_kind=target_kind, symbol=symbol, predictions=pred_all, actuals=act_all, per_window=per_window)
__all__ = ['PredictorFactory', 'TrainingResult', 'WindowPredictions', 'fit_predict_walkforward']
