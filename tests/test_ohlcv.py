from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tankx.data import ohlcv as ohlcv_mod
from tankx.data.ohlcv import cache_location, load_ohlcv
from tankx.data.schemas import (
    OHLCV_COLUMNS,
    OHLCV_INDEX_NAME,
    SchemaError,
    validate_agg_trades,
    validate_ohlcv,
)


class FakeExchange:

    def __init__(self, base_ms: int, bar_ms: int=5 * 60 * 1000, n_bars: int=2400) -> None:
        self.base_ms = base_ms
        self.bar_ms = bar_ms
        self.n_bars = n_bars
        self.calls: list[dict[str, Any]] = []

    def load_markets(self) -> None:
        pass

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
        self.calls.append({'since': since, 'limit': limit})
        end_ms = self.base_ms + self.n_bars * self.bar_ms
        if since >= end_ms:
            return []
        rows: list[list[float]] = []
        ts = max(since, self.base_ms)
        while ts < end_ms and len(rows) < limit:
            i = (ts - self.base_ms) // self.bar_ms
            o = 50000.0 + i * 0.5
            c = o + 1.0
            h = max(o, c) + 5.0
            low = min(o, c) - 5.0
            v = 100.0 + i % 7
            rows.append([float(ts), o, h, low, c, v])
            ts += self.bar_ms
        return rows

def _base_ms() -> int:
    return int(pd.Timestamp('2026-01-01', tz='UTC').timestamp() * 1000)

def _fake_end() -> pd.Timestamp:
    return pd.Timestamp('2026-01-01', tz='UTC') + pd.Timedelta(days=8)

def test_validate_ohlcv_happy_path(synthetic_ohlcv: pd.DataFrame) -> None:
    out = validate_ohlcv(synthetic_ohlcv)
    assert out is synthetic_ohlcv

def test_validate_ohlcv_rejects_tz_naive(synthetic_ohlcv: pd.DataFrame) -> None:
    df = synthetic_ohlcv.copy()
    df.index = df.index.tz_localize(None)
    with pytest.raises(SchemaError, match='tz-aware UTC'):
        validate_ohlcv(df)

def test_validate_ohlcv_rejects_negative_volume(synthetic_ohlcv: pd.DataFrame) -> None:
    df = synthetic_ohlcv.copy()
    df.loc[df.index[0], 'volume'] = -1.0
    with pytest.raises(SchemaError, match='Negative volume'):
        validate_ohlcv(df)

def test_validate_ohlcv_rejects_nan(synthetic_ohlcv: pd.DataFrame) -> None:
    df = synthetic_ohlcv.copy()
    df.loc[df.index[5], 'close'] = np.nan
    with pytest.raises(SchemaError, match='NaN'):
        validate_ohlcv(df)

def test_validate_ohlcv_rejects_duplicate_index(synthetic_ohlcv: pd.DataFrame) -> None:
    df = pd.concat([synthetic_ohlcv, synthetic_ohlcv.iloc[[0]]])
    with pytest.raises(SchemaError, match=r'duplicate|monotonic'):
        validate_ohlcv(df)

def test_validate_ohlcv_rejects_zero_price(synthetic_ohlcv: pd.DataFrame) -> None:
    df = synthetic_ohlcv.copy()
    df.loc[df.index[10], 'open'] = 0.0
    with pytest.raises(SchemaError, match='Non-positive prices'):
        validate_ohlcv(df)

def test_load_ohlcv_pagination(tmp_path: Path) -> None:
    base = _base_ms()
    fake = FakeExchange(base_ms=base, n_bars=2400)
    df = load_ohlcv(symbol='BTC/USDT', timeframe='5m', days=8, end=_fake_end(), exchange=fake, root=tmp_path)
    assert len(df) > 0
    assert len(fake.calls) >= 2
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    validate_ohlcv(df)

def test_load_ohlcv_writes_parquet_and_manifest(tmp_path: Path) -> None:
    fake = FakeExchange(base_ms=_base_ms())
    df = load_ohlcv(symbol='BTC/USDT', timeframe='5m', days=2, end=_fake_end(), exchange=fake, root=tmp_path)
    loc = cache_location('BTC/USDT', '5m', root=tmp_path)
    assert loc.parquet.exists()
    assert loc.manifest.exists()
    reloaded = pd.read_parquet(loc.parquet)
    assert len(reloaded) == len(df)

def test_load_ohlcv_is_idempotent(tmp_path: Path) -> None:
    fake = FakeExchange(base_ms=_base_ms())
    end = _fake_end()
    load_ohlcv(symbol='BTC/USDT', timeframe='5m', days=2, end=end, exchange=fake, root=tmp_path)
    calls_before = len(fake.calls)
    load_ohlcv(symbol='BTC/USDT', timeframe='5m', days=2, end=end, exchange=fake, root=tmp_path)
    calls_after = len(fake.calls)
    assert calls_after - calls_before <= 1

def test_load_ohlcv_force_refresh_redownloads(tmp_path: Path) -> None:
    fake = FakeExchange(base_ms=_base_ms())
    load_ohlcv(symbol='BTC/USDT', timeframe='5m', days=2, end=_fake_end(), exchange=fake, root=tmp_path)
    calls_first = len(fake.calls)
    load_ohlcv(symbol='BTC/USDT', timeframe='5m', days=2, end=_fake_end(), exchange=fake, root=tmp_path, force_refresh=True)
    assert len(fake.calls) > calls_first

def test_cache_location_naming() -> None:
    loc = cache_location('BTC/USDT', '5m', root=Path('/tmp'))
    assert loc.parquet.name == 'BTCUSDT_5m.parquet'
    assert loc.manifest.name == 'BTCUSDT_5m.manifest.json'

def test_validate_agg_trades_happy_path() -> None:
    df = pd.DataFrame({'agg_id': [1, 2, 3], 'price': [100.0, 100.5, 101.0], 'qty': [0.1, 0.2, 0.3], 'first_id': [10, 11, 12], 'last_id': [10, 11, 12], 'ts': pd.to_datetime([1700000000000, 1700000001000, 1700000002000], unit='ms', utc=True), 'is_buyer_maker': [False, True, False]})
    out = validate_agg_trades(df)
    assert out is df

def test_validate_agg_trades_rejects_non_monotonic_ts() -> None:
    df = pd.DataFrame({'agg_id': [1, 2], 'price': [100.0, 101.0], 'qty': [0.1, 0.2], 'first_id': [10, 11], 'last_id': [10, 11], 'ts': pd.to_datetime([1700000001000, 1700000000000], unit='ms', utc=True), 'is_buyer_maker': [False, True]})
    with pytest.raises(SchemaError, match=r'non-decreasing|monotonic'):
        validate_agg_trades(df)

@pytest.mark.parametrize('col', ['open', 'high', 'low', 'close', 'volume'])
def test_ohlcv_columns_constant(col: str) -> None:
    assert col in OHLCV_COLUMNS
    assert OHLCV_INDEX_NAME == 'timestamp'
    assert ohlcv_mod.CCXT_PAGE_LIMIT == 1000
