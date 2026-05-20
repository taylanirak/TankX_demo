from __future__ import annotations

import tankx
from tankx import config


def test_package_imports() -> None:
    assert tankx.__version__ == '0.1.0'

def test_bars_per_year_5m() -> None:
    assert config.bars_per_year('5m') == 365 * 24 * 12

def test_bars_per_year_unknown_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match='Unknown timeframe'):
        config.bars_per_year('7m')

def test_timeframe_to_seconds() -> None:
    assert config.timeframe_to_timedelta_seconds('5m') == 300
    assert config.timeframe_to_timedelta_seconds('1h') == 3600
    assert config.timeframe_to_timedelta_seconds('1d') == 86400

def test_default_symbol_in_supported() -> None:
    assert config.DEFAULT_SYMBOL in config.SUPPORTED_SYMBOLS
