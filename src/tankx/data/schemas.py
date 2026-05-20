from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

OHLCV_COLUMNS: Final[tuple[str, ...]] = ('open', 'high', 'low', 'close', 'volume')
OHLCV_INDEX_NAME: Final[str] = 'timestamp'

class SchemaError(ValueError):
    pass

def validate_ohlcv(df: pd.DataFrame, *, allow_zero_volume: bool=True) -> pd.DataFrame:
    _validate_ohlcv_index(df.index)
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f'Missing OHLCV columns: {missing}')
    sub = df[list(OHLCV_COLUMNS)]
    for col in OHLCV_COLUMNS:
        if not pd.api.types.is_float_dtype(sub[col]):
            raise SchemaError(f'Column {col!r} must be float, got {sub[col].dtype}')
    if sub.isna().any().any():
        bad_cols = sub.columns[sub.isna().any()].tolist()
        raise SchemaError(f'NaN values found in OHLCV columns: {bad_cols}')
    if np.isinf(sub.to_numpy()).any():
        raise SchemaError('Infinite values found in OHLCV columns')
    price_cols = ['open', 'high', 'low', 'close']
    if (sub[price_cols] <= 0).any().any():
        raise SchemaError('Non-positive prices found in OHLCV data')
    if (sub['volume'] < 0).any():
        raise SchemaError('Negative volume found in OHLCV data')
    if not allow_zero_volume and (sub['volume'] == 0).any():
        raise SchemaError('Zero volume bars present but allow_zero_volume=False')
    return df

def _validate_ohlcv_index(index: pd.Index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise SchemaError(f'Index must be DatetimeIndex, got {type(index).__name__}')
    if index.tz is None or str(index.tz) != 'UTC':
        raise SchemaError(f'Index must be tz-aware UTC, got tz={index.tz!r}')
    if index.name != OHLCV_INDEX_NAME:
        raise SchemaError(f'Index name must be {OHLCV_INDEX_NAME!r}, got {index.name!r}')
    if not index.is_monotonic_increasing:
        raise SchemaError('Index must be strictly monotonically increasing')
    if index.has_duplicates:
        raise SchemaError('Index has duplicate timestamps')
AGG_TRADES_COLUMNS: Final[tuple[str, ...]] = ('agg_id', 'price', 'qty', 'first_id', 'last_id', 'ts', 'is_buyer_maker')

def validate_agg_trades(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in AGG_TRADES_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f'Missing aggTrades columns: {missing}')
    if not pd.api.types.is_datetime64_any_dtype(df['ts']):
        raise SchemaError(f"Column 'ts' must be datetime64, got {df['ts'].dtype}")
    if df['ts'].dt.tz is None or str(df['ts'].dt.tz) != 'UTC':
        raise SchemaError("Column 'ts' must be tz-aware UTC")
    if not df['ts'].is_monotonic_increasing:
        raise SchemaError("Column 'ts' must be non-decreasing")
    if (df['price'] <= 0).any():
        raise SchemaError('Non-positive prices in aggTrades')
    if (df['qty'] <= 0).any():
        raise SchemaError('Non-positive quantities in aggTrades')
    if df['is_buyer_maker'].dtype != bool:
        raise SchemaError(f"Column 'is_buyer_maker' must be bool, got {df['is_buyer_maker'].dtype}")
    return df
__all__ = ['AGG_TRADES_COLUMNS', 'OHLCV_COLUMNS', 'OHLCV_INDEX_NAME', 'SchemaError', 'validate_agg_trades', 'validate_ohlcv']
