from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path
from tankx.config import (
    DEFAULT_DAYS,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    OHLCV_ROOT,
    timeframe_to_timedelta_seconds,
)
from tankx.data.schemas import OHLCV_COLUMNS, OHLCV_INDEX_NAME, validate_ohlcv

if TYPE_CHECKING:
    import ccxt
CCXT_PAGE_LIMIT = 1000

@dataclass(frozen=True)
class CacheLocation:
    parquet: Path
    manifest: Path

def cache_location(symbol: str, timeframe: str, root: Path=OHLCV_ROOT) -> CacheLocation:
    slug = symbol.replace('/', '').upper()
    return CacheLocation(parquet=root / f'{slug}_{timeframe}.parquet', manifest=root / f'{slug}_{timeframe}.manifest.json')

def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_convert('UTC') if pd.Timestamp.utcnow().tzinfo else pd.Timestamp.utcnow().tz_localize('UTC')

def _read_manifest(loc: CacheLocation) -> dict[str, Any]:
    if not loc.manifest.exists():
        return {}
    return json.loads(loc.manifest.read_text())

def _write_manifest(loc: CacheLocation, manifest: dict[str, Any]) -> None:
    loc.manifest.parent.mkdir(parents=True, exist_ok=True)
    loc.manifest.write_text(json.dumps(manifest, indent=2, default=str))

def _frame_from_ccxt_pages(pages: list[list[list[float]]]) -> pd.DataFrame:
    if not pages:
        return _empty_ohlcv_frame()
    flat = [row for page in pages for row in page]
    df = pd.DataFrame(flat, columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True)
    df = df.drop(columns='ts_ms').set_index('timestamp')
    df.index.name = OHLCV_INDEX_NAME
    df = df[~df.index.duplicated(keep='first')].sort_index()
    for col in OHLCV_COLUMNS:
        df[col] = df[col].astype(np.float64)
    return df

def _empty_ohlcv_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz='UTC', name=OHLCV_INDEX_NAME)
    return pd.DataFrame({c: pd.Series(dtype=np.float64) for c in OHLCV_COLUMNS}, index=idx)

def _fetch_paginated(exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int, until_ms: int, sleep_seconds: float=0.2) -> pd.DataFrame:
    pages: list[list[list[float]]] = []
    cursor = since_ms
    bar_ms = timeframe_to_timedelta_seconds(timeframe) * 1000
    while cursor < until_ms:
        page: list[list[float]] = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=cursor, limit=CCXT_PAGE_LIMIT)
        if not page:
            break
        pages.append(page)
        last_ts_ms = int(page[-1][0])
        next_cursor = last_ts_ms + bar_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(sleep_seconds)
    return _frame_from_ccxt_pages(pages)

def _make_exchange() -> ccxt.Exchange:
    import ccxt
    exchange = ccxt.binance({'enableRateLimit': True})
    exchange.load_markets()
    return exchange

def load_ohlcv(symbol: str=DEFAULT_SYMBOL, timeframe: str=DEFAULT_TIMEFRAME, days: int=DEFAULT_DAYS, *, end: pd.Timestamp | None=None, force_refresh: bool=False, exchange: ccxt.Exchange | None=None, root: Path=OHLCV_ROOT) -> pd.DataFrame:
    loc = cache_location(symbol, timeframe, root=root)
    if end is None:
        end = pd.Timestamp.utcnow()
    if end.tzinfo is None:
        end = end.tz_localize('UTC')
    start = end - pd.Timedelta(days=days)
    start = start.floor(_freq_for(timeframe))
    cached = _empty_ohlcv_frame()
    if not force_refresh and loc.parquet.exists():
        cached = pd.read_parquet(loc.parquet)
        cached.index = pd.DatetimeIndex(cached.index, name=OHLCV_INDEX_NAME)
        if cached.index.tz is None:
            cached.index = cached.index.tz_localize('UTC')
    fetch_since = start
    if len(cached) > 0:
        last_cached_ts = cached.index.max()
        fetch_since = max(start, last_cached_ts + pd.Timedelta(seconds=timeframe_to_timedelta_seconds(timeframe)))
    if fetch_since < end:
        ex = exchange if exchange is not None else _make_exchange()
        fresh = _fetch_paginated(ex, symbol=symbol, timeframe=timeframe, since_ms=int(fetch_since.timestamp() * 1000), until_ms=int(end.timestamp() * 1000))
    else:
        fresh = _empty_ohlcv_frame()
    combined = pd.concat([cached, fresh])
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    combined = combined[combined.index >= start]
    if len(combined):
        validate_ohlcv(combined)
    loc.parquet.parent.mkdir(parents=True, exist_ok=True)
    if len(combined):
        combined.to_parquet(loc.parquet)
        _write_manifest(loc, {'symbol': symbol, 'timeframe': timeframe, 'first_ts': str(combined.index.min()), 'last_ts': str(combined.index.max()), 'rows': len(combined), 'updated_at': str(pd.Timestamp.utcnow())})
    return combined

def _freq_for(timeframe: str) -> str:
    if timeframe.endswith('m'):
        return f'{int(timeframe[:-1])}min'
    if timeframe.endswith('h'):
        return f'{int(timeframe[:-1])}h'
    if timeframe.endswith('d'):
        return f'{int(timeframe[:-1])}D'
    raise ValueError(f'Unknown timeframe: {timeframe!r}')
__all__ = ['CCXT_PAGE_LIMIT', 'CacheLocation', 'cache_location', 'load_ohlcv']
