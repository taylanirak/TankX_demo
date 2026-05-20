from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def lttb_downsample(x: np.ndarray, y: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    n_in = len(x)
    if n_out >= n_in or n_out < 3:
        return (x, y)
    bucket_size = (n_in - 2) / (n_out - 2)
    out_x = np.empty(n_out, dtype=x.dtype)
    out_y = np.empty(n_out, dtype=y.dtype)
    out_x[0] = x[0]
    out_y[0] = y[0]
    a = 0
    for i in range(n_out - 2):
        start = int(np.floor((i + 0) * bucket_size)) + 1
        end = int(np.floor((i + 1) * bucket_size)) + 1
        end = min(end, n_in - 1)
        next_start = end
        next_end = int(np.floor((i + 2) * bucket_size)) + 1
        next_end = min(next_end, n_in - 1)
        avg_x = float(x[next_start:next_end].mean())
        avg_y = float(y[next_start:next_end].mean())
        xs = x[start:end].astype(np.float64)
        ys = y[start:end].astype(np.float64)
        area = np.abs((float(x[a]) - avg_x) * (ys - float(y[a])) - (xs - float(x[a])) * (float(y[a]) - avg_y))
        chosen = start if len(area) == 0 else start + int(np.argmax(area))
        out_x[i + 1] = x[chosen]
        out_y[i + 1] = y[chosen]
        a = chosen
    out_x[-1] = x[-1]
    out_y[-1] = y[-1]
    return (out_x, out_y)

def _maybe_downsample(series: pd.Series, max_points: int=5000) -> pd.Series:
    if len(series) <= max_points or len(series) < 4:
        return series
    x = series.index.astype('int64').to_numpy()
    y = series.to_numpy(dtype=np.float64)
    new_x, new_y = lttb_downsample(x, y, max_points)
    return pd.Series(new_y, index=pd.to_datetime(new_x, utc=True))

def equity_curve(equity: pd.Series, *, title: str='Equity Curve', wf_test_windows: list[tuple[pd.Timestamp, pd.Timestamp]] | None=None) -> go.Figure:
    ds = _maybe_downsample(equity)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds.index, y=ds.values, mode='lines', name='Equity', line={'color': '#1f77b4', 'width': 1.5}))
    if wf_test_windows:
        for start, end in wf_test_windows:
            fig.add_vrect(x0=start, x1=end, fillcolor='rgba(31,119,180,0.08)', line_width=0, annotation_text='', layer='below')
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Equity', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=420)
    return fig

def drawdown_chart(drawdown: pd.Series) -> go.Figure:
    ds = _maybe_downsample(drawdown)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds.index, y=ds.values, mode='lines', fill='tozeroy', name='Drawdown', line={'color': '#d62728', 'width': 0.8}, fillcolor='rgba(214,39,40,0.25)'))
    fig.update_layout(title='Drawdown', xaxis_title='Time', yaxis_title='Drawdown', yaxis={'tickformat': '.0%'}, margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=260)
    return fig

def price_with_signals(df: pd.DataFrame, positions: pd.Series, *, title: str='Price with Signals') -> go.Figure:
    ds = _maybe_downsample(df['close'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds.index, y=ds.values, mode='lines', name='Close', line={'color': '#7f7f7f', 'width': 1.0}))
    changes = positions.diff().fillna(positions.iloc[0])
    entries = positions.index[changes > 0]
    exits = positions.index[changes < 0]
    if len(entries):
        fig.add_trace(go.Scatter(x=entries, y=df.loc[entries, 'close'], mode='markers', name='Entry', marker={'color': '#2ca02c', 'size': 7, 'symbol': 'triangle-up'}))
    if len(exits):
        fig.add_trace(go.Scatter(x=exits, y=df.loc[exits, 'close'], mode='markers', name='Exit', marker={'color': '#d62728', 'size': 7, 'symbol': 'triangle-down'}))
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Close', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=380)
    return fig

def trade_pnl_histogram(trade_pnls: pd.Series) -> go.Figure:
    fig = go.Figure()
    if len(trade_pnls):
        fig.add_trace(go.Histogram(x=trade_pnls, marker={'color': '#1f77b4'}, name='Per-trade PnL'))
    fig.update_layout(title='Per-trade PnL distribution', xaxis_title='PnL', xaxis={'tickformat': '.1%'}, yaxis_title='Trades', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=300)
    return fig

def ofi_chart(ofi: pd.Series, *, z_threshold: float=2.0) -> go.Figure:
    ds = _maybe_downsample(ofi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds.index, y=ds.values, mode='lines', name='OFI', line={'color': '#1f77b4', 'width': 1.0}))
    fig.add_hline(y=z_threshold, line_dash='dot', line_color='rgba(214,39,40,0.5)')
    fig.add_hline(y=-z_threshold, line_dash='dot', line_color='rgba(44,160,44,0.5)')
    fig.update_layout(title='Order Flow Imbalance (rolling)', xaxis_title='Time', yaxis_title='OFI', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=320)
    return fig

def trade_size_chart(qty: pd.Series, *, p99: float | None=None) -> go.Figure:
    pos = qty[qty > 0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pos, name='Trade size', xbins={'size': 0.001}, marker={'color': '#1f77b4'}))
    if p99 is not None:
        fig.add_vline(x=float(p99), line_color='rgba(214,39,40,0.8)', line_width=2, annotation_text='p99 (whale)')
    fig.update_layout(title='Trade size distribution', xaxis_title='Quantity per trade', xaxis_type='log', yaxis_title='Trades', yaxis_type='log', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=320)
    return fig

def trade_arrival_heatmap(ts: pd.Series) -> go.Figure:
    df = pd.DataFrame({'ts': pd.DatetimeIndex(ts)})
    df['dow'] = df['ts'].dt.dayofweek
    df['hour'] = df['ts'].dt.hour
    counts = df.groupby(['dow', 'hour']).size().unstack(fill_value=0)
    counts = counts.reindex(index=range(7), columns=range(24), fill_value=0)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig = go.Figure(go.Heatmap(z=counts.values, x=list(range(24)), y=days, colorscale='Blues', colorbar={'title': 'Trades'}))
    fig.update_layout(title='Trade arrival heatmap (UTC)', xaxis_title='Hour', yaxis_title='Day of week', margin={'l': 60, 'r': 20, 't': 40, 'b': 40}, height=320)
    return fig

def effective_spread_chart(eff: pd.Series) -> go.Figure:
    bps = eff.dropna() * 10000.0
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=bps, marker={'color': '#1f77b4'}, name='Effective spread', xbins={'start': 0, 'end': float(bps.quantile(0.999)), 'size': 0.05}))
    fig.update_layout(title='Effective spread (bps) — Roll 1984 proxy', xaxis_title='bps', yaxis_title='Trades', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=300)
    return fig

def kyle_lambda_chart(frame: pd.DataFrame, lam: float) -> go.Figure:
    fig = go.Figure()
    if len(frame):
        fig.add_trace(go.Scatter(x=frame['abs_signed_vol'], y=frame['delta_p_abs'], mode='markers', name='bars', marker={'color': '#1f77b4', 'size': 5, 'opacity': 0.5}))
        x_max = float(frame['abs_signed_vol'].max())
        fig.add_trace(go.Scatter(x=[0, x_max], y=[0, lam * x_max], mode='lines', name=f'lambda={lam:.4f}', line={'color': '#d62728', 'width': 2}))
    fig.update_layout(title=f"Kyle's lambda regression: lambda = {lam:.4f}", xaxis_title='|signed volume| per bar', yaxis_title='|delta price| per bar', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=380)
    return fig

def bbo_chart(top_of_book: pd.DataFrame) -> go.Figure:
    df = top_of_book
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['bid_price'], name='Bid', line={'color': '#2ca02c', 'width': 1}))
    fig.add_trace(go.Scatter(x=df.index, y=df['ask_price'], name='Ask', line={'color': '#d62728', 'width': 1}))
    fig.add_trace(go.Scatter(x=df.index, y=df['mid'], name='Mid', line={'color': '#1f77b4', 'width': 1.5}))
    fig.update_layout(title='Top-of-book over time', xaxis_title='Time', yaxis_title='Price (USD)', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=360)
    return fig

def spread_chart(top_of_book: pd.DataFrame) -> go.Figure:
    bps = top_of_book['spread'] / top_of_book['mid'] * 10000.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bps.index, y=bps.values, name='Spread (bps)', line={'color': '#9467bd', 'width': 1}))
    fig.update_layout(title='Spread over time (bps of mid)', xaxis_title='Time', yaxis_title='bps', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=260)
    return fig

def imbalance_chart(top_of_book: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=top_of_book.index, y=top_of_book['imbalance'], name='Imbalance', line={'color': '#ff7f0e', 'width': 1}))
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(0,0,0,0.4)')
    fig.update_layout(title='Top-of-book imbalance (bid_size - ask_size) / total', xaxis_title='Time', yaxis_title='Imbalance', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=260)
    return fig

def mm_inventory_chart(inventory: pd.Series, *, cap: float | None=None) -> go.Figure:
    ds = _maybe_downsample(inventory)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds.index, y=ds.values, mode='lines', name='Inventory', line={'color': '#9467bd', 'width': 1.5}))
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(0,0,0,0.4)')
    if cap is not None:
        fig.add_hline(y=cap, line_dash='dash', line_color='rgba(214,39,40,0.6)')
        fig.add_hline(y=-cap, line_dash='dash', line_color='rgba(214,39,40,0.6)')
    fig.update_layout(title='MM inventory over time', xaxis_title='Time', yaxis_title='Inventory (base asset)', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=300)
    return fig

def mm_quotes_vs_mid_chart(mid: pd.Series, bids: pd.Series, asks: pd.Series) -> go.Figure:
    n = len(mid)
    if n > 6000:
        step = n // 5000
        mid = mid.iloc[::step]
        bids = bids.iloc[::step]
        asks = asks.iloc[::step]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mid.index, y=mid.values, name='Mid', line={'color': '#1f77b4', 'width': 1}))
    fig.add_trace(go.Scatter(x=bids.index, y=bids.values, name='Our bid', line={'color': '#2ca02c', 'width': 0.8, 'dash': 'dot'}))
    fig.add_trace(go.Scatter(x=asks.index, y=asks.values, name='Our ask', line={'color': '#d62728', 'width': 0.8, 'dash': 'dot'}))
    fig.update_layout(title='MM quotes vs mid', xaxis_title='Time', yaxis_title='Price (USD)', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=360)
    return fig

def mm_pnl_decomposition_chart(equity: pd.Series, cash: pd.Series, inventory: pd.Series, mid: pd.Series) -> go.Figure:
    spread_capture = cash - cash.iloc[0]
    inv_pnl = inventory * mid
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=(equity - equity.iloc[0]).values, name='Total PnL', line={'color': '#1f77b4', 'width': 1.5}))
    fig.add_trace(go.Scatter(x=spread_capture.index, y=spread_capture.values, name='Spread capture (cash drift)', line={'color': '#2ca02c', 'width': 1, 'dash': 'dash'}))
    fig.add_trace(go.Scatter(x=inv_pnl.index, y=inv_pnl.values, name='Inventory mark-to-market', line={'color': '#ff7f0e', 'width': 1, 'dash': 'dot'}))
    fig.update_layout(title='MM PnL decomposition', xaxis_title='Time', yaxis_title='PnL (USD)', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=320)
    return fig

def book_ladder_chart(bids: list[tuple[float, float]], asks: list[tuple[float, float]], *, title: str='Order book ladder') -> go.Figure:
    fig = go.Figure()
    if bids:
        bid_prices, bid_sizes = zip(*bids, strict=False)
        fig.add_trace(go.Bar(x=[-s for s in bid_sizes], y=[f'{p:.2f}' for p in bid_prices], orientation='h', name='Bids', marker={'color': '#2ca02c'}))
    if asks:
        ask_prices, ask_sizes = zip(*asks, strict=False)
        fig.add_trace(go.Bar(x=list(ask_sizes), y=[f'{p:.2f}' for p in ask_prices], orientation='h', name='Asks', marker={'color': '#d62728'}))
    fig.update_layout(title=title, xaxis_title='Size (negative = bids)', yaxis_title='Price (USD)', barmode='overlay', margin={'l': 80, 'r': 20, 't': 40, 'b': 40}, height=400)
    return fig

def volume_distribution(df: pd.DataFrame, outliers: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['volume'], name='All bars', marker={'color': '#1f77b4'}, opacity=0.7))
    if len(outliers):
        fig.add_trace(go.Histogram(x=outliers['volume'], name=f'Outliers ({len(outliers)})', marker={'color': '#d62728'}, opacity=0.9))
    fig.update_layout(title='Volume distribution', xaxis_title='Volume per bar', yaxis_title='Bars', barmode='overlay', margin={'l': 40, 'r': 20, 't': 40, 'b': 40}, height=300)
    return fig
__all__ = ['bbo_chart', 'book_ladder_chart', 'drawdown_chart', 'effective_spread_chart', 'equity_curve', 'imbalance_chart', 'kyle_lambda_chart', 'lttb_downsample', 'mm_inventory_chart', 'mm_pnl_decomposition_chart', 'mm_quotes_vs_mid_chart', 'ofi_chart', 'price_with_signals', 'spread_chart', 'trade_arrival_heatmap', 'trade_pnl_histogram', 'trade_size_chart', 'volume_distribution']
