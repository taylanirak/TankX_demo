from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from tankx.config import DEFAULT_DAYS, DEFAULT_TIMEFRAME, SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES
from tankx.data.ohlcv import load_ohlcv

if TYPE_CHECKING:
    from collections.abc import Sequence

def _parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--symbols', nargs='+', default=list(SUPPORTED_SYMBOLS), help='Binance spot pairs, e.g. BTC/USDT ETH/USDT SOL/USDT')
    parser.add_argument('--timeframe', default=DEFAULT_TIMEFRAME, choices=list(SUPPORTED_TIMEFRAMES))
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS)
    parser.add_argument('--force-refresh', action='store_true')
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None=None) -> int:
    args = _parse_args(argv)
    for symbol in args.symbols:
        print(f'Fetching {symbol} {args.timeframe} for {args.days} days...')
        df = load_ohlcv(symbol=symbol, timeframe=args.timeframe, days=args.days, force_refresh=args.force_refresh)
        if df.empty:
            print(f'  WARNING: no rows returned for {symbol}')
            continue
        print(f"  {len(df):,} rows, {df.index.min()} -> {df.index.max()}, close[-1]={df['close'].iloc[-1]:.2f}")
    return 0
if __name__ == '__main__':
    sys.exit(main())
