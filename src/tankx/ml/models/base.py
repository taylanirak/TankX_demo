from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

TargetKind = Literal['direction', 'return', 'triple_barrier']

@runtime_checkable
class Predictor(Protocol):
    name: str
    target_kind: TargetKind

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
__all__ = ['Predictor', 'TargetKind']
