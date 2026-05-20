from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from tankx.backtest.engine_event import MMBacktestConfig, run_mm_backtest
from tankx.backtest.engine_vec import BacktestConfig, BacktestResult, run_backtest
from tankx.backtest.execution import PoissonFillModel
from tankx.backtest.walkforward import default_sharpe_score, make_windows, walk_forward
from tankx.config import (
    DEFAULT_FEE_BPS,
    DEFAULT_SLIPPAGE_BPS,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
    bars_per_year,
)
from tankx.data.ohlcv import cache_location, load_ohlcv
from tankx.data.trades import _daily_parquet_path
from tankx.microstructure.features import (
    effective_spread,
    kyle_lambda,
    order_flow_imbalance,
    trade_size_distribution,
)
from tankx.orderbook.replay import collect_top_of_book_series, replay_book_depth
from tankx.quality.anomaly import run_all_checks
from tankx.stats.bootstrap import bootstrap_sharpe_ci
from tankx.stats.metrics import (
    annualized_sharpe,
    calmar,
    max_drawdown,
    summarize,
    turnover_per_year,
)
from tankx.stats.psr import probabilistic_sharpe_ratio
from tankx.strategies.avellaneda_stoikov import AvellanedaStoikov
from tankx.strategies.bollinger import BollingerMeanReversion
from tankx.strategies.ma_crossover import MACrossover
from tankx.viz.charts import (
    bbo_chart,
    book_ladder_chart,
    drawdown_chart,
    effective_spread_chart,
    equity_curve,
    imbalance_chart,
    kyle_lambda_chart,
    mm_inventory_chart,
    mm_pnl_decomposition_chart,
    mm_quotes_vs_mid_chart,
    ofi_chart,
    price_with_signals,
    spread_chart,
    trade_arrival_heatmap,
    trade_pnl_histogram,
    trade_size_chart,
    volume_distribution,
)

st.set_page_config(page_title='tankx-research', layout='wide')

@st.cache_data(ttl=3600, show_spinner='Loading OHLCV from parquet cache...')
def _cached_load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    loc = cache_location(symbol, timeframe)
    if not loc.parquet.exists():
        return load_ohlcv(symbol=symbol, timeframe=timeframe, days=365)
    df = pd.read_parquet(loc.parquet)
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index.name = 'timestamp'
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_quality_report(_df: pd.DataFrame, timeframe: str) -> object:
    return run_all_checks(_df, timeframe)

@st.cache_data(ttl=3600, show_spinner='Loading aggTrades parquet cache...')
def _cached_load_trades(symbol: str) -> pd.DataFrame:
    import glob
    slug = symbol.replace('/', '').upper()
    pattern = str(_daily_parquet_path(symbol, __import__('datetime').date(2000, 1, 1), root=__import__('pathlib').Path('data/trades')).parent / f'{slug}-aggTrades-*.parquet')
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    out = pd.concat(frames, ignore_index=True).sort_values('ts', kind='mergesort')
    return out.reset_index(drop=True)

def _render_sidebar() -> dict[str, object]:
    st.sidebar.title('tankx-research')
    st.sidebar.caption('HFT-flavored crypto backtester / microstructure lab')
    with st.sidebar.form('controls'):
        symbol = st.selectbox('Symbol', SUPPORTED_SYMBOLS, index=0)
        timeframe = st.selectbox('Timeframe', [t for t in SUPPORTED_TIMEFRAMES if t in {'1m', '5m', '15m', '1h'}], index=1)
        strategy_name = st.selectbox('Strategy', ['MA Crossover', 'Bollinger Mean Reversion'], index=0)
        st.markdown('**Strategy parameters**')
        if strategy_name == 'MA Crossover':
            short_window = st.slider('Short SMA window', 5, 100, 20, step=5)
            long_window = st.slider('Long SMA window', 20, 500, 100, step=10)
            strat_params = {'short_window': short_window, 'long_window': long_window}
        else:
            window = st.slider('Window', 5, 100, 20, step=5)
            num_std = st.slider('Num std', 1.0, 3.0, 2.0, step=0.1)
            strat_params = {'window': window, 'num_std': float(num_std)}
        st.markdown('**Cost model**')
        fee_bps = st.slider('Fee (bps)', 0.0, 50.0, float(DEFAULT_FEE_BPS), step=0.5)
        slippage_bps = st.slider('Slippage (bps)', 0.0, 50.0, float(DEFAULT_SLIPPAGE_BPS), step=0.5)
        initial_capital = st.number_input('Initial capital (USD)', min_value=100.0, max_value=10000000.0, value=10000.0, step=1000.0)
        st.markdown('**Validation**')
        wfa = st.checkbox('Walk-forward (recommended)', value=True, help='When on, the reported equity/metrics are aggregated from out-of-sample test windows. When off, you see the in-sample fit, which is optimistic.')
        if wfa:
            train_days = st.slider('WFA train days', 7, 120, 30, step=1)
            test_days = st.slider('WFA test days', 1, 60, 7, step=1)
        else:
            train_days = 0
            test_days = 0
        submitted = st.form_submit_button('Run backtest', use_container_width=True)
    return {'submitted': submitted, 'symbol': symbol, 'timeframe': timeframe, 'strategy_name': strategy_name, 'strat_params': strat_params, 'fee_bps': fee_bps, 'slippage_bps': slippage_bps, 'initial_capital': initial_capital, 'wfa': wfa, 'train_days': train_days, 'test_days': test_days}

def _strategy_factory(name: str):
    return MACrossover if name == 'MA Crossover' else BollingerMeanReversion

def _tab_backtest(cfg: dict[str, object], df: pd.DataFrame) -> None:
    factory = _strategy_factory(str(cfg['strategy_name']))
    bcfg = BacktestConfig(initial_capital=float(cfg['initial_capital']), fee_bps=float(cfg['fee_bps']), slippage_bps=float(cfg['slippage_bps']), execution='next_open')
    bpy = bars_per_year(str(cfg['timeframe']))
    wf_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if bool(cfg['wfa']):
        windows = make_windows(df.index, train_days=int(cfg['train_days']), test_days=int(cfg['test_days']))
        if not windows:
            st.warning('Not enough data for the requested walk-forward windows. Falling back to a single in-sample fit.')
            strat = factory(**cfg['strat_params'])
            positions = strat.generate_positions(df)
            result = run_backtest(df, positions, bcfg)
        else:
            grid = {k: [v] for k, v in cfg['strat_params'].items()}
            wf_report = walk_forward(df=df, strategy_factory=factory, param_grid=grid, windows=windows, config=bcfg, score=default_sharpe_score(bpy))
            test_slices = []
            for window in windows:
                test_slice = df.loc[window.test_start:window.test_end]
                if test_slice.empty:
                    continue
                strat = factory(**cfg['strat_params'])
                pos = strat.generate_positions(test_slice)
                test_slices.append(run_backtest(test_slice, pos, bcfg))
                wf_windows.append((window.test_start, window.test_end))
            if not test_slices:
                st.warning('Walk-forward produced no usable test slices.')
                return
            equity_parts = []
            running = float(cfg['initial_capital'])
            for r in test_slices:
                eq = r.equity * (running / r.equity.iloc[0])
                equity_parts.append(eq)
                running = float(eq.iloc[-1])
            stitched_eq = pd.concat(equity_parts).sort_index()
            net_returns = pd.concat([r.net_returns for r in test_slices]).sort_index()
            positions = pd.concat([r.positions for r in test_slices]).sort_index()
            trades_all = pd.concat([r.trades for r in test_slices], ignore_index=True)
            result = BacktestResult(equity=stitched_eq.rename('equity'), net_returns=net_returns, gross_returns=pd.concat([r.gross_returns for r in test_slices]).sort_index(), positions=positions, raw_positions=positions, position_changes=positions.diff().abs().fillna(0.0), fees=pd.concat([r.fees for r in test_slices]).sort_index(), drawdown=stitched_eq / stitched_eq.cummax() - 1.0, trades=trades_all, config=bcfg)
            st.success(f"Out-of-sample (walk-forward): {len(windows)} non-overlapping test windows, mean train Sharpe {wf_report['train_score'].mean():.2f}, mean test Sharpe {wf_report['test_score'].mean():.2f}.")
    else:
        st.warning('In-sample mode active — these numbers are optimistic. Turn on the walk-forward checkbox in the sidebar for credible OOS metrics.')
        strat = factory(**cfg['strat_params'])
        positions = strat.generate_positions(df)
        result = run_backtest(df, positions, bcfg)
    mdd, _, _ = max_drawdown(result.equity)
    sr = annualized_sharpe(result.net_returns, bpy)
    cal = calmar(result.equity, bpy)
    turn = turnover_per_year(result.positions, bpy)
    psr = probabilistic_sharpe_ratio(result.net_returns.to_numpy())
    cols = st.columns(5)
    cols[0].metric('Sharpe', f'{sr:.2f}', help='Annualized, naive (no autocorrelation adj)')
    badge = 'OK' if psr > 0.95 else 'weak'
    cols[1].metric('PSR(>0)', f'{psr:.2f}', delta=badge, delta_color='normal' if psr > 0.95 else 'off', help='Probabilistic Sharpe Ratio (Bailey & López de Prado 2012)')
    cols[2].metric('Max Drawdown', f'{mdd:.1%}')
    cols[3].metric('Calmar', 'n/a' if math.isnan(cal) else f'{cal:.2f}')
    cols[4].metric('Turnover/yr', f'{turn:.0f}')
    rets = result.net_returns.dropna().to_numpy()
    if len(rets) >= 50:
        ci = bootstrap_sharpe_ci(rets, bars_per_year=bpy, n_iter=400, confidence=0.95)
        st.caption(f'Stationary-bootstrap 95% CI for annualized Sharpe: **[{ci.lower:.2f}, {ci.upper:.2f}]** (point = {ci.point:.2f}, block ≈ {ci.block_size:.0f} bars, {ci.n_iter} resamples). If the interval crosses zero, the Sharpe is not statistically distinguishable from random.')
    st.plotly_chart(equity_curve(result.equity, wf_test_windows=wf_windows), use_container_width=True)
    st.plotly_chart(drawdown_chart(result.drawdown), use_container_width=True)
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.plotly_chart(price_with_signals(df, result.positions, title='Price with strategy signals'), use_container_width=True)
    with col_right:
        if len(result.trades):
            trades = result.trades.copy()
            trades['pnl_pct'] = trades['pnl_pct'].map(lambda x: f'{x:+.2%}')
            st.markdown('**Trade log**')
            st.dataframe(trades, use_container_width=True, height=380)
        else:
            st.info('No trades on this window.')
    if len(result.trades):
        trade_pnls = result.trades['exit_equity'] / result.trades['entry_equity'] - 1.0
        st.plotly_chart(trade_pnl_histogram(trade_pnls), use_container_width=True)
    with st.expander('Full metrics dict'):
        m = summarize(result.equity, result.net_returns, result.positions, trade_pnls=None, bars_per_year=bpy)
        st.json(m)

def _tab_microstructure(symbol: str) -> None:
    trades = _cached_load_trades(symbol)
    if trades.empty:
        st.info(f"""No locally cached aggTrades for **{symbol}**.\n\nHydrate the cache with:\n```bash\npython -c "import datetime as dt; from tankx.data.trades import download_agg_trades_daily; [download_agg_trades_daily('{symbol}', dt.date(2026, 5, d)) for d in range(13, 19)]"\n```""")
        return
    n = len(trades)
    span_start = trades['ts'].min()
    span_end = trades['ts'].max()
    buyer_frac = float((~trades['is_buyer_maker']).mean())
    cols = st.columns(4)
    cols[0].metric('Trades', f'{n:,}')
    cols[1].metric('Span (UTC)', f'{span_start.date()} → {span_end.date()}')
    cols[2].metric('Taker-buy fraction', f'{buyer_frac:.2%}')
    cols[3].metric('Price range', f"{trades['price'].min():,.0f} - {trades['price'].max():,.0f}")
    st.markdown('---')
    st.markdown("Microstructure features below are computed from Binance Vision aggTrades. Formulas: OFI = rolling signed-vol ratio (Cartea-Jaimungal-Penalva 2015); effective spread = 2|P-mid|/mid (Roll 1984); Kyle's lambda = OLS of |delta_p| on |signed volume| (Kyle 1985).")
    st.plotly_chart(trade_arrival_heatmap(trades['ts']), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        tsd = trade_size_distribution(trades)
        st.plotly_chart(trade_size_chart(trades['qty'], p99=tsd.p99), use_container_width=True)
        st.caption(f'p50={tsd.p50:.4f}, p99={tsd.p99:.4f}, p99.9={tsd.p999:.4f}, Hill tail-index alpha={tsd.hill_alpha:.2f} (alpha<2 implies infinite variance — a fat-tailed regime).')
    with col_b:
        ofi = order_flow_imbalance(trades, window='30s')
        ofi_min = ofi.resample('1min').last().dropna()
        st.plotly_chart(ofi_chart(ofi_min, z_threshold=0.5), use_container_width=True)
    mid = trades.set_index(pd.DatetimeIndex(trades['ts']))['price'].resample('1s').last().ffill()
    eff = effective_spread(trades, mid)
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(effective_spread_chart(eff), use_container_width=True)
    with col_d:
        lam, frame = kyle_lambda(trades, bar='1min')
        st.plotly_chart(kyle_lambda_chart(frame, lam), use_container_width=True)

def _list_book_dates() -> list[str]:
    import glob
    from pathlib import Path
    pattern = 'data/book/BTCUSDT/BTCUSDT-bookDepth-*.parquet'
    files = sorted(Path(p).stem.split('-')[-3:] for p in glob.glob(pattern))
    return ['-'.join(parts) for parts in files]

@st.cache_data(ttl=3600, show_spinner='Reconstructing order book...')
def _cached_top_of_book(date_str: str) -> pd.DataFrame:
    import datetime as dt2

    from tankx.data.book import load_book_depth
    from tankx.data.trades import load_agg_trades
    y, m, d = (int(x) for x in date_str.split('-'))
    date = dt2.date(y, m, d)
    book = load_book_depth('BTC/USDT', start=date, end=date)
    trades = load_agg_trades('BTC/USDT', start=date, end=date)
    ref = trades.set_index(pd.DatetimeIndex(trades['ts']))['price'].groupby(level=0).last().resample('1s').last().ffill()
    snapshots = list(replay_book_depth(book, ref))
    return collect_top_of_book_series(snapshots)

def _tab_order_book() -> None:
    st.markdown("Order-book replay uses Binance **futures (USD-margined) bookDepth** archives (BTCUSDT-PERP), because Binance's public *spot* archive does not include order-book data. The mid price for percentage-offset decoding is taken from same-day spot aggTrades — small basis differences are ignored.")
    dates = _list_book_dates()
    if not dates:
        st.info('No locally cached bookDepth parquets. Run:\n```bash\npython -c "import datetime as dt; from tankx.data.book import download_book_depth_daily; [download_book_depth_daily(\'BTC/USDT\', dt.date(2026, 5, d)) for d in range(13, 19)]"\n```')
        return
    chosen = st.selectbox('Date', dates, index=len(dates) - 1)
    tob = _cached_top_of_book(chosen)
    if tob.empty:
        st.warning('Replay produced no snapshots for that date.')
        return
    cols = st.columns(4)
    cols[0].metric('Snapshots', f'{len(tob):,}')
    cols[1].metric('Mean spread (bps)', f"{(tob['spread'] / tob['mid'] * 10000).mean():.2f}")
    cols[2].metric('Mean imbalance', f"{tob['imbalance'].mean():+.3f}")
    cols[3].metric('Price range', f"{tob['mid'].min():,.0f} - {tob['mid'].max():,.0f}")
    st.plotly_chart(bbo_chart(tob), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(spread_chart(tob), use_container_width=True)
    with col_b:
        st.plotly_chart(imbalance_chart(tob), use_container_width=True)
    st.markdown('---')
    st.markdown('**Order-book ladder snapshot**')
    if len(tob) > 1:
        slider_idx = st.slider('Snapshot index', 0, len(tob) - 1, value=len(tob) // 2)
        bbo_row = tob.iloc[slider_idx]
        ts_str = str(tob.index[slider_idx])
        st.caption(f"At {ts_str}: bid={bbo_row['bid_price']:.2f} ({bbo_row['bid_size']:.2f}), ask={bbo_row['ask_price']:.2f} ({bbo_row['ask_size']:.2f}), spread={bbo_row['spread']:.4f}, microprice={bbo_row['microprice']:.4f}, imbalance={bbo_row['imbalance']:+.3f}")
        bids = [(float(bbo_row['bid_price']), float(bbo_row['bid_size']))]
        asks = [(float(bbo_row['ask_price']), float(bbo_row['ask_size']))]
        st.plotly_chart(book_ladder_chart(bids, asks), use_container_width=True)

def _tab_market_making() -> None:
    st.markdown('Closed-form Avellaneda-Stoikov (2008) quoting against a real reconstructed mid from Binance Vision bookDepth. Fills are simulated by a Poisson model calibrated to the same `k` as the strategy expects (`P(fill within dt) = 1 - exp(-A * exp(-k * delta) * dt)`).')
    dates = _list_book_dates()
    if not dates:
        st.info('Hydrate the bookDepth cache first (see Order Book tab).')
        return
    with st.form('mm_controls'):
        col_a, col_b = st.columns(2)
        with col_a:
            chosen = st.selectbox('Date', dates, index=len(dates) - 1)
            gamma = st.select_slider('Risk aversion gamma', options=[1e-06, 1e-05, 0.0001, 0.001, 0.01], value=1e-05, format_func=lambda x: f'{x:.0e}')
            sigma = st.slider('Per-second mid vol sigma (USD)', 0.5, 50.0, 5.0, step=0.5)
        with col_b:
            k = st.select_slider('Fill-decay k (1/USD)', options=[0.001, 0.01, 0.1, 1.0], value=0.01, format_func=lambda x: f'{x:.3f}')
            A = st.slider('Base arrival rate A (events/s)', 0.1, 20.0, 3.0, step=0.1)
            order_size = st.number_input('Order size (BTC)', min_value=0.0001, max_value=1.0, value=0.001, step=0.0001, format='%.4f')
            inv_cap = st.number_input('Inventory cap (BTC)', min_value=0.001, max_value=10.0, value=0.05, step=0.005, format='%.3f')
        submitted = st.form_submit_button('Run market-making sim', use_container_width=True)
    if not submitted:
        st.info('Set parameters and click Run to simulate.')
        return
    tob = _cached_top_of_book(chosen)
    if tob.empty:
        st.warning('No reconstructed book snapshots for that date.')
        return
    mid = tob['mid']
    strat = AvellanedaStoikov(gamma=float(gamma), sigma=float(sigma), k=float(k))
    fill = PoissonFillModel(A=float(A), k=float(k))
    cfg = MMBacktestConfig(order_size=float(order_size), inventory_cap=float(inv_cap), fee_bps_maker=0.0, initial_cash=10000.0)
    result = run_mm_backtest(mid, strat, fill, cfg, seed=20260520)
    n_trades = len(result.trades)
    final_pnl = float(result.equity.iloc[-1] - result.config.initial_cash)
    sharpe = result.equity.diff().mean() / result.equity.diff().std() * len(result.equity) ** 0.5
    cols = st.columns(4)
    cols[0].metric('Fills', f'{n_trades:,}')
    cols[1].metric('Final PnL (USD)', f'{final_pnl:+,.2f}')
    cols[2].metric('Inventory at end (BTC)', f'{result.inventory.iloc[-1]:+.4f}')
    cols[3].metric('Per-step Sharpe (intra-day)', f'{sharpe:+.2f}')
    st.plotly_chart(mm_quotes_vs_mid_chart(result.mid, result.bids, result.asks), use_container_width=True)
    st.plotly_chart(mm_inventory_chart(result.inventory, cap=cfg.inventory_cap), use_container_width=True)
    st.plotly_chart(mm_pnl_decomposition_chart(result.equity, result.cash, result.inventory, result.mid), use_container_width=True)
    if n_trades > 0:
        with st.expander(f'Trade log ({n_trades} fills)'):
            st.dataframe(result.trades, use_container_width=True, height=300)
    st.caption('**Interpretation note.** PnL decomposition: cash drift = realized spread capture from completed round-trips; inventory mark-to-market is unrealized and goes to zero when inventory returns to zero. A well-behaved AS quoter flattens inventory by the end of the day — large terminal inventory means gamma is too low (or the price moved adversely).')

def _tab_ml_models() -> None:
    from pathlib import Path

    import plotly.graph_objects as go

    from tankx.ml.evaluate import BenchmarkSpec, evaluate_one
    from tankx.ml.models.registry import list_models as _list_models
    from tankx.strategies.ml_directional import MLDirectionalStrategy
    st.markdown('**ML benchmark (v2).** Eight predictors (Lag1, Ridge, MLP, XGBoost, LightGBM, LSTM, GRU, Transformer) on causal lagged-return + rolling + volume + ToD features, walk-forward validated. Targets: direction (binary) and triple-barrier (Lopez de Prado: TP/SL/timeout). Fee modes: taker (10 bps) and maker (0 bps counterfactual).')
    bench_path_v2 = Path('data/ml_benchmark_v2.parquet')
    bench_path_v1 = Path('data/ml_benchmark.parquet')
    bench = None
    if bench_path_v2.exists():
        bench = pd.read_parquet(bench_path_v2)
        st.success(f'Loaded v2 benchmark from {bench_path_v2} ({len(bench)} rows). Refresh with `python scripts/benchmark_ml.py`.')
    elif bench_path_v1.exists():
        bench = pd.read_parquet(bench_path_v1)
        st.info(f'Loaded v1 benchmark from {bench_path_v1}. Upgrade with `python scripts/benchmark_ml.py` (writes v2).')
    else:
        st.info('No cached benchmark yet. The on-the-fly mode below trains a single (symbol, timeframe, model, target) combination. For the full v2 table run:\n```\npython scripts/benchmark_ml.py\n```')
    with st.form('ml_controls'):
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            symbol = st.selectbox('Symbol', SUPPORTED_SYMBOLS, index=0)
        with col_b:
            timeframe = st.selectbox('Timeframe', ['5m', '1h', '1d'], index=1)
        with col_c:
            target = st.selectbox('Target', ['direction', 'triple_barrier', 'return'], index=0)
        with col_d:
            model_name = st.selectbox('Model', _list_models(), index=1)
        col_e, col_f, col_g = st.columns(3)
        with col_e:
            fee_mode = st.radio('Fee mode', ['taker', 'maker'], horizontal=True)
        with col_f:
            direction_band = st.slider('Direction band (abstain)', 0.0, 0.2, 0.05, step=0.01)
        with col_g:
            include_mtf = st.checkbox('Add multi-timeframe features', value=False)
        submitted = st.form_submit_button('Run on-the-fly', use_container_width=True)
    if bench is not None:
        st.markdown('### Cached benchmark (filterable)')
        sub = bench.copy()
        if 'timeframe' in sub.columns:
            sub = sub[sub['timeframe'] == timeframe]
        if 'target' in sub.columns:
            sub = sub[sub['target'] == target]
        sub = sub[sub['symbol'] == symbol]
        if 'fee_mode' in sub.columns:
            sub = sub[sub['fee_mode'] == fee_mode]
        st.dataframe(sub.drop(columns=['symbol'], errors='ignore'), use_container_width=True, hide_index=True)
    if not submitted:
        st.info('Pick a (symbol, model, target) and click Run on-the-fly to train.')
        return
    df = _cached_load_ohlcv(symbol, timeframe)
    if df.empty:
        st.error(f'No cached OHLCV for {symbol} {timeframe}. Run scripts/fetch_multi_symbol.py first.')
        return
    train_days, test_days = {'5m': (30, 7), '1h': (90, 14), '1d': (365, 30)}.get(timeframe, (60, 14))
    from tankx.ml.features import FeatureConfig
    spec = BenchmarkSpec(symbol=symbol, timeframe=timeframe, target_kind=target, feature_config=FeatureConfig(include_multi_timeframe=include_mtf), train_days=train_days, test_days=test_days, fee_bps=10.0, slippage_bps=2.0, direction_band=direction_band, fee_bps_maker=0.0, assume_maker=fee_mode == 'maker')
    with st.spinner(f'Training {model_name} on {symbol} {timeframe} / {target} / {fee_mode} ...'):
        row, training_result = evaluate_one(df, spec, model_name)
    st.markdown('### On-the-fly result')
    metrics_cols = st.columns(4)
    if target == 'direction':
        metrics_cols[0].metric('Accuracy', f"{row.get('accuracy', float('nan')):.3f}")
        metrics_cols[1].metric('F1', f"{row.get('f1', float('nan')):.3f}")
        metrics_cols[2].metric('AUC', f"{row.get('auc', float('nan')):.3f}")
    elif target == 'triple_barrier':
        metrics_cols[0].metric('Sign Acc.', f"{row.get('accuracy', float('nan')):.3f}")
        metrics_cols[1].metric('F1', f"{row.get('f1', float('nan')):.3f}")
        metrics_cols[2].metric('Timeout %', f"{row.get('timeout_frac', float('nan')):.1%}")
    else:
        metrics_cols[0].metric('RMSE', f"{row.get('rmse', float('nan')):.6f}")
        metrics_cols[1].metric('MAE', f"{row.get('mae', float('nan')):.6f}")
        metrics_cols[2].metric('R^2', f"{row.get('r2', float('nan')):.4f}")
    metrics_cols[3].metric('Strategy Sharpe', f"{row.get('strategy_sharpe', float('nan')):.2f}")
    st.caption(f"Turnover (sum |delta pos|): {row.get('turnover', float('nan')):.1f} | Fee mode: {fee_mode}")
    st.markdown('### Strategy equity (single-fit wrapper, 60% train slice)')
    strat = MLDirectionalStrategy(model_name=model_name, target_kind='return' if target == 'triple_barrier' else target, train_fraction=0.6, direction_band=direction_band)
    positions = strat.generate_positions(df)
    bcfg = BacktestConfig(fee_bps=10.0, slippage_bps=2.0, execution='next_close', fee_bps_maker=0.0, assume_maker=fee_mode == 'maker')
    result = run_backtest(df, positions, bcfg)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity.values, mode='lines', name=f'{model_name} ({target})', line={'width': 1.5}))
    fig.update_layout(title=f'Single-fit equity — {model_name} / {target}', xaxis_title='Time', yaxis_title='Equity (USD)', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=380)
    st.plotly_chart(fig, use_container_width=True)
    if training_result.per_window and training_result.per_window[0].history:
        st.markdown('### Training loss curves (first walk-forward window)')
        history_df = pd.DataFrame(training_result.per_window[0].history)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=history_df['epoch'], y=history_df['train_loss'], mode='lines+markers', name='train'))
        fig2.add_trace(go.Scatter(x=history_df['epoch'], y=history_df['val_loss'], mode='lines+markers', name='val'))
        fig2.update_layout(xaxis_title='Epoch', yaxis_title='Loss', margin={'l': 40, 'r': 20, 't': 30, 'b': 40}, height=300)
        st.plotly_chart(fig2, use_container_width=True)
    if target == 'return' and len(training_result.predictions) > 0:
        st.markdown('### Prediction vs actual (OOS rows)')
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=training_result.actuals.values, y=training_result.predictions.values, mode='markers', marker={'size': 3, 'opacity': 0.5}, name='OOS'))
        lo = float(min(training_result.actuals.min(), training_result.predictions.min()))
        hi = float(max(training_result.actuals.max(), training_result.predictions.max()))
        fig3.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='y=x', line={'color': 'red', 'dash': 'dot'}))
        fig3.update_layout(xaxis_title='Actual log return', yaxis_title='Predicted log return', margin={'l': 40, 'r': 20, 't': 30, 'b': 40}, height=380)
        st.plotly_chart(fig3, use_container_width=True)
    st.caption('**Honesty note.** Direction accuracy near 50% means the model is essentially flipping a coin. On retail-fee BTC 5m, *any* model whose strategy Sharpe is positive after fees has either found a real edge — or is suffering from selection bias. The walk-forward + no-lookahead property tests (`tests/test_ml_no_lookahead.py`) rule out the second cause; the first is rare. Treat all positive numbers below as conditional on the next 30 days looking like the last 30.')

def _tab_data_quality(df: pd.DataFrame, timeframe: str) -> None:
    report = _cached_quality_report(df, timeframe)
    total = report.total_rows
    severity = report.severity
    color = {'clean': 'green', 'minor': 'orange', 'major': 'red'}[severity]
    st.markdown(f'### Severity: :{color}[{severity.upper()}] over {total:,} bars')
    summary = report.summary_frame()
    st.dataframe(summary, use_container_width=True, hide_index=True)
    findings = report.findings
    for check_name, frame in findings.items():
        if len(frame) == 0:
            continue
        with st.expander(f'{check_name} — {len(frame)} flagged rows'):
            st.dataframe(frame.head(20), use_container_width=True)
    if 'volume_outliers' in findings:
        st.plotly_chart(volume_distribution(df, findings['volume_outliers']), use_container_width=True)
    all_findings_rows: list[pd.DataFrame] = []
    for check_name, frame in findings.items():
        if len(frame) == 0:
            continue
        f = frame.copy()
        f['check'] = check_name
        all_findings_rows.append(f.reset_index())
    if all_findings_rows:
        merged = pd.concat(all_findings_rows, ignore_index=True)
        st.download_button('Download all findings (CSV)', data=merged.to_csv(index=False).encode('utf-8'), file_name=f'anomalies_{timeframe}.csv', mime='text/csv')

def main() -> None:
    cfg = _render_sidebar()
    st.title('tankx-research: backtest, microstructure, and market making')
    st.caption('Walk-forward by default. PSR badge marks Sharpe ratios that are statistically distinguishable from random.')
    df = _cached_load_ohlcv(str(cfg['symbol']), str(cfg['timeframe']))
    if df.empty:
        st.error(f"No cached data available. Run `python scripts/fetch_data.py --symbol {cfg['symbol']} --timeframe {cfg['timeframe']} --days 365` first.")
        st.stop()
    tabs = st.tabs(['Strategy & Backtest', 'Microstructure', 'Order Book Replay', 'Market Making', 'ML Models', 'Data Quality'])
    with tabs[0]:
        _tab_backtest(cfg, df)
    with tabs[1]:
        _tab_microstructure(str(cfg['symbol']))
    with tabs[2]:
        _tab_order_book()
    with tabs[3]:
        _tab_market_making()
    with tabs[4]:
        _tab_ml_models()
    with tabs[5]:
        _tab_data_quality(df, str(cfg['timeframe']))
if __name__ == '__main__':
    main()
