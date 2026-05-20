from __future__ import annotations

import datetime as dt
import io
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tankx.data import trades as trades_mod
from tankx.data.trades import (
    _daily_url,
    _parse_agg_trades_csv,
    download_agg_trades_daily,
    load_agg_trades,
)


def _make_zip(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('inner.csv', csv_text)
    return buf.getvalue()

class FakeResponse:

    def __init__(self, status: int, content: bytes) -> None:
        self.status_code = status
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f'HTTP {self.status_code}')

class FakeSession:

    def __init__(self, content: bytes, status: int=200) -> None:
        self.content = content
        self.status = status
        self.calls = 0

    def get(self, url: str, timeout: int) -> FakeResponse:
        self.calls += 1
        return FakeResponse(self.status, self.content)
SAMPLE_CSV = '1,50000.10,0.001,100,100,1717200000000,true,true\n2,50001.20,0.002,101,101,1717200000100,false,true\n3,50000.50,0.003,102,103,1717200000200,true,true\n'

def test_daily_url_format() -> None:
    url = _daily_url('BTC/USDT', dt.date(2026, 5, 1))
    assert url.endswith('BTCUSDT/BTCUSDT-aggTrades-2026-05-01.zip')

def test_parse_agg_trades_basic() -> None:
    df = _parse_agg_trades_csv(SAMPLE_CSV.encode())
    assert len(df) == 3
    assert df['price'].iloc[1] == pytest.approx(50001.2)
    assert df['qty'].iloc[2] == pytest.approx(0.003)
    assert df['is_buyer_maker'].iloc[0]
    assert not df['is_buyer_maker'].iloc[1]
    assert str(df['ts'].dt.tz) == 'UTC'
    assert df['ts'].is_monotonic_increasing

def test_parse_agg_trades_with_header_row() -> None:
    csv_with_header = 'agg_id,price,qty,first_id,last_id,ts,is_buyer_maker,is_best_match\n' + SAMPLE_CSV
    df = _parse_agg_trades_csv(csv_with_header.encode())
    assert len(df) == 3
    assert df['price'].iloc[0] == pytest.approx(50000.1)

def test_download_agg_trades_writes_parquet(tmp_path: Path) -> None:
    fake = FakeSession(content=_make_zip(SAMPLE_CSV))
    path = download_agg_trades_daily('BTC/USDT', dt.date(2026, 5, 1), root=tmp_path, session=fake)
    assert path.exists()
    df = pd.read_parquet(path)
    assert len(df) == 3
    assert fake.calls == 1

def test_download_agg_trades_uses_cache(tmp_path: Path) -> None:
    fake = FakeSession(content=_make_zip(SAMPLE_CSV))
    download_agg_trades_daily('BTC/USDT', dt.date(2026, 5, 1), root=tmp_path, session=fake)
    download_agg_trades_daily('BTC/USDT', dt.date(2026, 5, 1), root=tmp_path, session=fake)
    assert fake.calls == 1

def test_download_agg_trades_retries_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    class FlakySession:

        def __init__(self) -> None:
            self.calls = 0

        def get(self, url: str, timeout: int) -> Any:
            self.calls += 1
            if self.calls < 2:
                import requests
                raise requests.ConnectionError('simulated')
            return FakeResponse(200, _make_zip(SAMPLE_CSV))
    monkeypatch.setattr(trades_mod.time, 'sleep', lambda _s: None)
    flaky = FlakySession()
    download_agg_trades_daily('BTC/USDT', dt.date(2026, 5, 1), root=tmp_path, session=flaky, retries=3)
    assert flaky.calls == 2

def test_load_agg_trades_concatenates_dates(tmp_path: Path) -> None:
    fake = FakeSession(content=_make_zip(SAMPLE_CSV))
    out = load_agg_trades('BTC/USDT', start=dt.date(2026, 5, 1), end=dt.date(2026, 5, 3), root=tmp_path, session=fake)
    assert len(out) == 9
    assert fake.calls == 3
    assert out['ts'].is_monotonic_increasing

def test_load_agg_trades_rejects_inverted_range(tmp_path: Path) -> None:
    fake = FakeSession(content=_make_zip(SAMPLE_CSV))
    with pytest.raises(ValueError, match='>= start'):
        load_agg_trades('BTC/USDT', start=dt.date(2026, 5, 3), end=dt.date(2026, 5, 1), root=tmp_path, session=fake)

def test_download_404_raises(tmp_path: Path) -> None:
    fake = FakeSession(content=b'', status=404)
    with pytest.raises(FileNotFoundError):
        download_agg_trades_daily('BTC/USDT', dt.date(2026, 5, 1), root=tmp_path, session=fake)
