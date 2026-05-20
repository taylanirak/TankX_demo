from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from tankx.config import DEFAULT_DAYS, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, SUPPORTED_TIMEFRAMES

if TYPE_CHECKING:
    from collections.abc import Sequence
from tankx.data.ohlcv import load_ohlcv


def _parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--symbol', default=DEFAULT_SYMBOL, help="Binance spot pair, e.g. 'BTC/USDT'")
    parser.add_argument('--timeframe', default=DEFAULT_TIMEFRAME, choices=list(SUPPORTED_TIMEFRAMES), help='Candle timeframe')
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS, help='Lookback window in days')
    parser.add_argument('--force-refresh', action='store_true', help='Ignore the local parquet cache and redownload')
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None=None) -> int:
    args = _parse_args(argv)
    print(f'Fetching {args.symbol} {args.timeframe} for the last {args.days} days...')
    df = load_ohlcv(symbol=args.symbol, timeframe=args.timeframe, days=args.days, force_refresh=args.force_refresh)
    if df.empty:
        print('No rows returned.')
        return 1
    print(f"Got {len(df):,} rows from {df.index.min()} to {df.index.max()}\n  high={df['high'].max():.2f}  low={df['low'].min():.2f}  close[-1]={df['close'].iloc[-1]:.2f}")
    return 0
if __name__ == '__main__':
    sys.exit(main())
