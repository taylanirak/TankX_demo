from __future__ import annotations

import datetime as dt
import io
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import requests

from tankx.config import BINANCE_VISION_BASE, TRADES_ROOT
from tankx.data.schemas import AGG_TRADES_COLUMNS, validate_agg_trades

if TYPE_CHECKING:
    from collections.abc import Iterable
AGG_TRADES_RAW_COLUMNS = ('agg_id', 'price', 'qty', 'first_id', 'last_id', 'ts', 'is_buyer_maker', 'is_best_match')

def _daily_url(symbol: str, date: dt.date) -> str:
    slug = symbol.replace('/', '').upper()
    return f'{BINANCE_VISION_BASE}/data/spot/daily/aggTrades/{slug}/{slug}-aggTrades-{date.isoformat()}.zip'

def _daily_parquet_path(symbol: str, date: dt.date, root: Path) -> Path:
    slug = symbol.replace('/', '').upper()
    return root / slug / f'{slug}-aggTrades-{date.isoformat()}.parquet'

def _parse_agg_trades_csv(blob: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(blob), header=None, names=AGG_TRADES_RAW_COLUMNS, dtype={'is_buyer_maker': str, 'is_best_match': str})
    if not pd.api.types.is_numeric_dtype(df['agg_id']):
        df = pd.read_csv(io.BytesIO(blob), header=0, names=AGG_TRADES_RAW_COLUMNS, dtype={'is_buyer_maker': str, 'is_best_match': str})
    df['price'] = df['price'].astype('float64')
    df['qty'] = df['qty'].astype('float64')
    df['agg_id'] = df['agg_id'].astype('int64')
    df['first_id'] = df['first_id'].astype('int64')
    df['last_id'] = df['last_id'].astype('int64')
    df['ts'] = _coerce_unix_ts(df['ts'].astype('int64'))
    df['is_buyer_maker'] = df['is_buyer_maker'].str.lower().map({'true': True, 'false': False}).astype(bool)
    return df[list(AGG_TRADES_COLUMNS)]

def _coerce_unix_ts(values: pd.Series) -> pd.Series:
    sample = int(values.iloc[len(values) // 2])
    if sample < 1000000000000:
        return pd.to_datetime(values, unit='s', utc=True)
    if sample < 1000000000000000:
        return pd.to_datetime(values, unit='ms', utc=True)
    return pd.to_datetime(values, unit='us', utc=True)

def download_agg_trades_daily(symbol: str, date: dt.date, *, root: Path=TRADES_ROOT, force: bool=False, session: requests.Session | None=None, retries: int=3, backoff: float=1.5) -> Path:
    parquet_path = _daily_parquet_path(symbol, date, root)
    if parquet_path.exists() and (not force):
        return parquet_path
    sess = session or requests.Session()
    url = _daily_url(symbol, date)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = sess.get(url, timeout=60)
            if resp.status_code == 404:
                raise FileNotFoundError(f'No aggTrades archive at {url}')
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                names = zf.namelist()
                if not names:
                    raise RuntimeError(f'Empty zip at {url}')
                blob = zf.read(names[0])
            df = _parse_agg_trades_csv(blob)
            validate_agg_trades(df)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path)
            return parquet_path
        except (requests.RequestException, zipfile.BadZipFile) as exc:
            last_exc = exc
            time.sleep(backoff ** attempt)
    assert last_exc is not None
    raise last_exc

def _daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)

def load_agg_trades(symbol: str, start: dt.date | str | pd.Timestamp, end: dt.date | str | pd.Timestamp, *, root: Path=TRADES_ROOT, session: requests.Session | None=None) -> pd.DataFrame:
    s = _coerce_date(start)
    e = _coerce_date(end)
    if e < s:
        raise ValueError(f'end ({e}) must be >= start ({s})')
    frames: list[pd.DataFrame] = []
    for d in _daterange(s, e):
        path = download_agg_trades_daily(symbol, d, root=root, session=session)
        frames.append(pd.read_parquet(path))
    out = pd.concat(frames, ignore_index=True).sort_values('ts', kind='mergesort')
    validate_agg_trades(out)
    return out.reset_index(drop=True)

def _coerce_date(x: dt.date | str | pd.Timestamp) -> dt.date:
    if isinstance(x, dt.date) and (not isinstance(x, dt.datetime)):
        return x
    return pd.Timestamp(x).date()
__all__ = ['AGG_TRADES_RAW_COLUMNS', 'download_agg_trades_daily', 'load_agg_trades']
