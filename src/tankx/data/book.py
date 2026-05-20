from __future__ import annotations

import datetime as dt
import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from tankx.config import BINANCE_VISION_BASE, BOOK_ROOT

BOOK_DEPTH_COLUMNS = ('ts', 'percentage', 'depth', 'notional')
PERCENTAGE_LEVELS = (-5.0, -4.0, -3.0, -2.0, -1.0, -0.2, 0.2, 1.0, 2.0, 3.0, 4.0, 5.0)

def _daily_url(symbol: str, date: dt.date) -> str:
    slug = symbol.replace('/', '').upper().replace('USDT-PERP', 'USDT')
    return f'{BINANCE_VISION_BASE}/data/futures/um/daily/bookDepth/{slug}/{slug}-bookDepth-{date.isoformat()}.zip'

def _daily_parquet_path(symbol: str, date: dt.date, root: Path) -> Path:
    slug = symbol.replace('/', '').upper().replace('USDT-PERP', 'USDT')
    return root / slug / f'{slug}-bookDepth-{date.isoformat()}.parquet'

def _parse_book_depth_csv(blob: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(blob), header=None, names=list(BOOK_DEPTH_COLUMNS))
    if not pd.api.types.is_numeric_dtype(df['percentage']):
        df = pd.read_csv(io.BytesIO(blob), header=0, names=list(BOOK_DEPTH_COLUMNS))
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df['percentage'] = df['percentage'].astype('float64')
    df['depth'] = df['depth'].astype('float64')
    df['notional'] = df['notional'].astype('float64')
    return df

def download_book_depth_daily(symbol: str, date: dt.date, *, root: Path=BOOK_ROOT, force: bool=False, session: requests.Session | None=None, retries: int=3, backoff: float=1.5) -> Path:
    path = _daily_parquet_path(symbol, date, root)
    if path.exists() and (not force):
        return path
    sess = session or requests.Session()
    url = _daily_url(symbol, date)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = sess.get(url, timeout=60)
            if resp.status_code == 404:
                raise FileNotFoundError(f'No bookDepth archive at {url}')
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                names = zf.namelist()
                if not names:
                    raise RuntimeError(f'Empty zip at {url}')
                blob = zf.read(names[0])
            df = _parse_book_depth_csv(blob)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            return path
        except (requests.RequestException, zipfile.BadZipFile) as exc:
            last_exc = exc
            time.sleep(backoff ** attempt)
    assert last_exc is not None
    raise last_exc

def load_book_depth(symbol: str, start: dt.date | str | pd.Timestamp, end: dt.date | str | pd.Timestamp, *, root: Path=BOOK_ROOT, session: requests.Session | None=None) -> pd.DataFrame:
    s = _coerce_date(start)
    e = _coerce_date(end)
    if e < s:
        raise ValueError(f'end ({e}) must be >= start ({s})')
    frames: list[pd.DataFrame] = []
    d = s
    while d <= e:
        path = download_book_depth_daily(symbol, d, root=root, session=session)
        frames.append(pd.read_parquet(path))
        d += dt.timedelta(days=1)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values('ts', kind='mergesort').reset_index(drop=True)

def _coerce_date(x: dt.date | str | pd.Timestamp) -> dt.date:
    if isinstance(x, dt.date) and (not isinstance(x, dt.datetime)):
        return x
    return pd.Timestamp(x).date()
__all__ = ['BOOK_DEPTH_COLUMNS', 'PERCENTAGE_LEVELS', 'download_book_depth_daily', 'load_book_depth']
