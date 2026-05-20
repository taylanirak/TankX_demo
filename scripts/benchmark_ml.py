from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tankx.config import (
    DEFAULT_FEE_BPS,
    DEFAULT_SLIPPAGE_BPS,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
)
from tankx.data.ohlcv import cache_location
from tankx.ml.evaluate import BenchmarkSpec, evaluate_all
from tankx.ml.features import FeatureConfig
from tankx.ml.models.registry import list_models

if TYPE_CHECKING:
    from collections.abc import Sequence
DEFAULT_WINDOWS_BY_TF = {'5m': (90, 60), '1h': (120, 30), '1d': (365, 60)}

def _load_cached_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    loc = cache_location(symbol, timeframe)
    if not loc.parquet.exists():
        raise FileNotFoundError(f'No cached parquet for {symbol} {timeframe}. Run: python scripts/fetch_multi_symbol.py --symbols {symbol} --timeframe {timeframe}')
    df = pd.read_parquet(loc.parquet)
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index.name = 'timestamp'
    return df

def _parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--symbols', nargs='+', default=list(SUPPORTED_SYMBOLS))
    parser.add_argument('--timeframes', nargs='+', default=['5m', '1h', '1d'], choices=list(SUPPORTED_TIMEFRAMES))
    parser.add_argument('--targets', nargs='+', default=['direction', 'triple_barrier'], choices=['direction', 'return', 'triple_barrier'])
    parser.add_argument('--fee-modes', nargs='+', default=['taker', 'maker'], choices=['taker', 'maker'])
    parser.add_argument('--models', nargs='+', default=list_models())
    parser.add_argument('--fee-bps', type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument('--slippage-bps', type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument('--maker-bps', type=float, default=0.0)
    parser.add_argument('--direction-band', type=float, default=0.05, help='Abstain band on prediction probability (direction target).')
    parser.add_argument('--return-band', type=float, default=0.0, help='Abstain band for return / triple-barrier predictions.')
    parser.add_argument('--include-mtf', action='store_true', help='Add multi-timeframe features (5m benchmark only).')
    parser.add_argument('--out', default='data/ml_benchmark_v2.md')
    parser.add_argument('--out-parquet', default='data/ml_benchmark_v2.parquet')
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None=None) -> int:
    args = _parse_args(argv)
    rows: list[pd.DataFrame] = []
    grand_t0 = time.perf_counter()
    feature_config = FeatureConfig(include_multi_timeframe=args.include_mtf, mtf_timeframes=('1h', '4h'))
    for symbol in args.symbols:
        for timeframe in args.timeframes:
            try:
                df = _load_cached_ohlcv(symbol, timeframe)
            except FileNotFoundError as exc:
                print(exc)
                continue
            train_days, test_days = DEFAULT_WINDOWS_BY_TF.get(timeframe, (60, 14))
            for target in args.targets:
                for fee_mode in args.fee_modes:
                    spec = BenchmarkSpec(symbol=symbol, timeframe=timeframe, target_kind=target, feature_config=feature_config, train_days=train_days, test_days=test_days, fee_bps=args.fee_bps, fee_bps_maker=args.maker_bps, slippage_bps=args.slippage_bps, direction_band=args.direction_band, return_band=args.return_band, assume_maker=fee_mode == 'maker')
                    print(f'Benchmarking {symbol} {timeframe} / {target} / {fee_mode} ...')
                    t0 = time.perf_counter()
                    table, _ = evaluate_all(df, spec, model_names=args.models)
                    elapsed = time.perf_counter() - t0
                    print(f'  {len(table)} rows in {elapsed:.1f}s')
                    rows.append(table)
    if not rows:
        print('No benchmark rows produced.')
        return 1
    benchmark = pd.concat(rows, ignore_index=True)
    out_parquet = Path(args.out_parquet)
    out_md = Path(args.out)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_parquet(out_parquet)
    _write_markdown(benchmark, out_md, total_elapsed=time.perf_counter() - grand_t0)
    print(f'Wrote {out_parquet} and {out_md} (total: {time.perf_counter() - grand_t0:.0f}s)')
    return 0

def _write_markdown(benchmark: pd.DataFrame, out: Path, *, total_elapsed: float) -> None:
    sections: list[str] = ['# ML Benchmark v2 Report\n', f'_Generated: {pd.Timestamp.utcnow().isoformat()} UTC_  \n_Total rows: {len(benchmark)}, runtime: {total_elapsed:.0f}s_  \n_Targets: direction (binary) + triple_barrier (López de Prado)._  \n_Fee modes: taker (10 bps) and maker (0 bps) — same predictions, different fee assumption._\n']
    sections.append('\n## Direction accuracy by timeframe\n')
    dir_pivot = benchmark[(benchmark['target'] == 'direction') & (benchmark['fee_mode'] == 'taker')].pivot_table(index='model', columns=['timeframe', 'symbol'], values='accuracy', aggfunc='mean')
    sections.append(dir_pivot.to_markdown(floatfmt='.3f'))
    sections.append('\n')
    sections.append('\n## Triple-barrier sign accuracy by timeframe (timeouts excluded)\n')
    tb_pivot = benchmark[(benchmark['target'] == 'triple_barrier') & (benchmark['fee_mode'] == 'taker')].pivot_table(index='model', columns=['timeframe', 'symbol'], values='accuracy', aggfunc='mean')
    if not tb_pivot.empty:
        sections.append(tb_pivot.to_markdown(floatfmt='.3f'))
        sections.append('\n')
    sections.append('\n## Strategy Sharpe — taker vs maker fees\n')
    for target in sorted(benchmark['target'].unique()):
        sub = benchmark[benchmark['target'] == target]
        if sub.empty:
            continue
        sections.append(f'\n### Target: `{target}`\n')
        sharpe_pivot = sub.pivot_table(index=['timeframe', 'symbol', 'model'], columns='fee_mode', values='strategy_sharpe', aggfunc='mean')
        sections.append(sharpe_pivot.to_markdown(floatfmt='.2f'))
        sections.append('\n')
    sections.append('\n---\n\n## Reading the table\n* `direction` target = binary up/down classification; chance = 0.50.\n* `triple_barrier` target = López de Prado labeling. Accuracy excludes\n  the timeout class (where the model abstains). Chance = 0.50.\n* `fee_mode = taker` charges the configured retail-taker fee on each\n  position change. `maker = 0 bps` is the counterfactual: what would\n  happen if the same predictions were executed maker-only.\n* The wider the timeframe, the better the signal-to-noise; expect 5m\n  to be near 50% accuracy and 1d to be the most informative.\n')
    out.write_text(''.join(sections), encoding='utf-8')
if __name__ == '__main__':
    sys.exit(main())
