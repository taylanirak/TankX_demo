from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_ROOT: Final[Path] = PROJECT_ROOT / 'data'
OHLCV_ROOT: Final[Path] = DATA_ROOT / 'ohlcv'
TRADES_ROOT: Final[Path] = DATA_ROOT / 'trades'
BOOK_ROOT: Final[Path] = DATA_ROOT / 'book'
DEFAULT_SYMBOL: Final[str] = 'BTC/USDT'
DEFAULT_TIMEFRAME: Final[str] = '5m'
DEFAULT_DAYS: Final[int] = 365
SUPPORTED_SYMBOLS: Final[tuple[str, ...]] = ('BTC/USDT', 'ETH/USDT', 'SOL/USDT')
SUPPORTED_TIMEFRAMES: Final[tuple[str, ...]] = ('1m', '5m', '15m', '1h', '4h', '1d')
BARS_PER_YEAR: Final[dict[str, int]] = {'1m': 365 * 24 * 60, '5m': 365 * 24 * 12, '15m': 365 * 24 * 4, '1h': 365 * 24, '4h': 365 * 6, '1d': 365}

def bars_per_year(timeframe: str) -> int:
    try:
        return BARS_PER_YEAR[timeframe]
    except KeyError as exc:
        raise ValueError(f'Unknown timeframe {timeframe!r}; expected one of {sorted(BARS_PER_YEAR)}') from exc

def timeframe_to_timedelta_seconds(timeframe: str) -> int:
    units = {'m': 60, 'h': 3600, 'd': 86400}
    suffix = timeframe[-1]
    if suffix not in units:
        raise ValueError(f'Unknown timeframe {timeframe!r}')
    return int(timeframe[:-1]) * units[suffix]
BINANCE_SPOT_TAKER_BPS: Final[float] = 10.0
BINANCE_SPOT_MAKER_BPS: Final[float] = 10.0
DEFAULT_FEE_BPS: Final[float] = BINANCE_SPOT_TAKER_BPS
DEFAULT_SLIPPAGE_BPS: Final[float] = 2.0
BINANCE_VISION_BASE: Final[str] = 'https://data.binance.vision'
'Public archive of Binance spot/futures data (no auth).'
__all__ = ['BARS_PER_YEAR', 'BINANCE_SPOT_MAKER_BPS', 'BINANCE_SPOT_TAKER_BPS', 'BINANCE_VISION_BASE', 'BOOK_ROOT', 'DATA_ROOT', 'DEFAULT_DAYS', 'DEFAULT_FEE_BPS', 'DEFAULT_SLIPPAGE_BPS', 'DEFAULT_SYMBOL', 'DEFAULT_TIMEFRAME', 'OHLCV_ROOT', 'PROJECT_ROOT', 'SUPPORTED_SYMBOLS', 'SUPPORTED_TIMEFRAMES', 'TRADES_ROOT', 'bars_per_year', 'timeframe_to_timedelta_seconds']
