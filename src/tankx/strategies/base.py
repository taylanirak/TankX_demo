from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

@runtime_checkable
class VectorizedStrategy(Protocol):
    name: str
    params: dict[str, Any]

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        ...
__all__ = ['VectorizedStrategy']
