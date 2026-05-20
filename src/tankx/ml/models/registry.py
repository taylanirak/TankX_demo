from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tankx.ml.models.base import Predictor
from tankx.ml.models.baseline import Lag1Predictor, MLPPredictor, RidgePredictor
from tankx.ml.models.boosting import LightGBMPredictor, XGBoostPredictor
from tankx.ml.models.deep import GRUPredictor, LSTMPredictor, TransformerPredictor


def _factory(cls: type, **defaults: Any) -> Callable[..., Predictor]:

    def make(target_kind: str='direction', **kwargs: Any) -> Predictor:
        instance: Predictor = cls(target_kind=target_kind, **{**defaults, **kwargs})
        return instance
    return make
MODEL_REGISTRY: dict[str, Callable[..., Predictor]] = {'Lag1': _factory(Lag1Predictor), 'Ridge': _factory(RidgePredictor, alpha=1.0), 'MLP': _factory(MLPPredictor, hidden_units=(32,), max_iter=120), 'XGBoost': _factory(XGBoostPredictor, n_estimators=200, max_depth=4, learning_rate=0.05), 'LightGBM': _factory(LightGBMPredictor, n_estimators=200, max_depth=4, learning_rate=0.05, num_leaves=15), 'LSTM': _factory(LSTMPredictor, seq_len=16, hidden_size=16, epochs=4, batch_size=256), 'GRU': _factory(GRUPredictor, seq_len=16, hidden_size=16, epochs=4, batch_size=256), 'Transformer': _factory(TransformerPredictor, seq_len=16, d_model=16, nhead=2, num_layers=1, epochs=4, batch_size=256)}

def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())

def make_predictor(name: str, target_kind: str='direction', **kwargs: Any) -> Predictor:
    if name not in MODEL_REGISTRY:
        raise KeyError(f'Unknown model {name!r}; known: {list(MODEL_REGISTRY)}')
    return MODEL_REGISTRY[name](target_kind=target_kind, **kwargs)
__all__ = ['MODEL_REGISTRY', 'list_models', 'make_predictor']
