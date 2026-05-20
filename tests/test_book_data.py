from __future__ import annotations

import datetime as dt
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from tankx.data.book import (
    BOOK_DEPTH_COLUMNS,
    _daily_url,
    _parse_book_depth_csv,
    download_book_depth_daily,
    load_book_depth,
)


def _make_zip(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('inner.csv', csv_text)
    return buf.getvalue()
SAMPLE_CSV = '2026-05-15 00:00:09,-5.00,9710.5,770710741.8\n2026-05-15 00:00:09,-1.00,2395.5,193344273.8\n2026-05-15 00:00:09,-0.20,583.4,47245348.2\n2026-05-15 00:00:09,0.20,771.8,62605474.7\n2026-05-15 00:00:09,1.00,2016.1,164026341.2\n2026-05-15 00:00:09,5.00,11285.9,933805577.3\n2026-05-15 00:00:30,-5.00,9750.5,772710741.0\n2026-05-15 00:00:30,-0.20,600.0,48000000.0\n2026-05-15 00:00:30,0.20,750.0,60000000.0\n2026-05-15 00:00:30,5.00,11200.0,930000000.0\n'

class FakeResponse:

    def __init__(self, status: int, content: bytes) -> None:
        self.status_code = status
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError('HTTP error')

class FakeSession:

    def __init__(self, content: bytes, status: int=200) -> None:
        self.content = content
        self.status = status
        self.calls = 0

    def get(self, url: str, timeout: int) -> FakeResponse:
        self.calls += 1
        return FakeResponse(self.status, self.content)

def test_daily_url_normalizes_symbol() -> None:
    url = _daily_url('BTC/USDT', dt.date(2026, 5, 15))
    assert 'BTCUSDT-bookDepth-2026-05-15.zip' in url
    url2 = _daily_url('BTCUSDT-PERP', dt.date(2026, 5, 15))
    assert 'BTCUSDT-bookDepth-2026-05-15.zip' in url2

def test_parse_book_depth_csv() -> None:
    df = _parse_book_depth_csv(SAMPLE_CSV.encode())
    assert list(df.columns) == list(BOOK_DEPTH_COLUMNS)
    assert len(df) == 10
    assert df['ts'].iloc[0] == pd.Timestamp('2026-05-15 00:00:09', tz='UTC')
    assert df['percentage'].iloc[2] == -0.2
    assert df['depth'].iloc[3] == pytest.approx(771.8)

def test_download_book_depth_writes_parquet(tmp_path: Path) -> None:
    fake = FakeSession(content=_make_zip(SAMPLE_CSV))
    path = download_book_depth_daily('BTC/USDT', dt.date(2026, 5, 15), root=tmp_path, session=fake)
    assert path.exists()
    df = pd.read_parquet(path)
    assert len(df) == 10

def test_download_book_depth_uses_cache(tmp_path: Path) -> None:
    fake = FakeSession(content=_make_zip(SAMPLE_CSV))
    download_book_depth_daily('BTC/USDT', dt.date(2026, 5, 15), root=tmp_path, session=fake)
    download_book_depth_daily('BTC/USDT', dt.date(2026, 5, 15), root=tmp_path, session=fake)
    assert fake.calls == 1

def test_load_book_depth_concats(tmp_path: Path) -> None:
    fake = FakeSession(content=_make_zip(SAMPLE_CSV))
    out = load_book_depth('BTC/USDT', start=dt.date(2026, 5, 15), end=dt.date(2026, 5, 16), root=tmp_path, session=fake)
    assert len(out) == 20
    assert fake.calls == 2

def test_load_rejects_inverted_range(tmp_path: Path) -> None:
    fake = FakeSession(content=_make_zip(SAMPLE_CSV))
    with pytest.raises(ValueError, match='>= start'):
        load_book_depth('BTC/USDT', start=dt.date(2026, 5, 16), end=dt.date(2026, 5, 15), root=tmp_path, session=fake)

def test_download_404_raises(tmp_path: Path) -> None:
    fake = FakeSession(content=b'', status=404)
    with pytest.raises(FileNotFoundError):
        download_book_depth_daily('BTC/USDT', dt.date(2026, 5, 15), root=tmp_path, session=fake)
