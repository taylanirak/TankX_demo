# Crypto Backtesting Mini-Framework — Build Spec

> **Note to Claude Code:** This is a complete, self-contained spec. Build the project end-to-end in the order described under "Implementation Plan". After each step, run the code to verify it works before moving on. Ask me only if a requirement is ambiguous; otherwise, make a reasonable choice and document it in the README.

---

## 1. Context & Motivation

I'm preparing for a job interview at **TankX**, a high-frequency crypto market-making firm based in Istanbul. The role is a quantitative/data analyst position. The job description emphasizes:

- Building **research pipelines and backtesting frameworks**
- Analyzing **large amounts of market and trade data**
- Running **statistical experiments** to evaluate strategy configurations
- Developing **dashboards (Streamlit/Grafana)** to track P&L, system health, alpha
- **Data validation** mindset — attention to anomalies, integrity, consistency
- Acting as the **bridge between raw exchange data and production-ready signals**

My background is ML research (cyber security, telecom signal processing, biomedical), not finance. I want to ship a small but credible end-to-end project in **one evening (~3-4 focused hours)** that demonstrates I can handle this exact kind of work.

The project should be:
- Completable in one sitting
- Deployable to a public GitHub repo
- Runnable with `streamlit run app.py` after a single `pip install -r requirements.txt`
- Honest about its limitations (this is not real production HFT)

---

## 2. High-Level Goal (one sentence)

**Build a minimal but real crypto backtesting framework that fetches Binance market data, runs two classical strategies on it, computes performance metrics, validates data integrity, and presents everything in an interactive Streamlit dashboard.**

---

## 3. Tech Stack

- Python 3.10+
- `ccxt` — public Binance market data (no API key required for public endpoints)
- `pandas`, `numpy` — data manipulation
- `plotly` — interactive charts (preferred over matplotlib for Streamlit)
- `streamlit` — dashboard
- `scipy` — statistical tests

All dependencies must be pinned in `requirements.txt`.

---

## 4. Project Structure

```
crypto-backtest/
├── data/
│   └── btc_5m.csv               # Cached historical data (created by data_loader)
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Fetch & cache Binance OHLCV
│   ├── strategies.py            # Strategy classes/functions
│   ├── backtester.py            # Backtest engine
│   ├── metrics.py               # Performance metrics
│   ├── statistical_tests.py     # Bootstrap CI for Sharpe
│   └── anomaly_detector.py      # Data validation checks
├── app.py                       # Streamlit dashboard (entry point)
├── requirements.txt
├── README.md
└── .gitignore                   # ignore data/*.csv, __pycache__, .venv
```

---

## 5. Implementation Plan (build in this order, verify each step)

### Step 1 — `requirements.txt` and venv setup

Create `requirements.txt`:
```
ccxt>=4.0
pandas>=2.0
numpy>=1.24
plotly>=5.0
streamlit>=1.30
scipy>=1.10
```

### Step 2 — `src/data_loader.py`

**Purpose:** Fetch OHLCV (Open/High/Low/Close/Volume) data from Binance via ccxt's public API and cache it as CSV.

**Function signature:**
```python
def load_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    days: int = 180,
    cache_path: str = "data/btc_5m.csv",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        timestamp (datetime, UTC), open, high, low, close, volume
    indexed by timestamp.

    Fetches in chunks of 1000 candles (ccxt limit) and stitches them.
    Caches to CSV; on subsequent calls, loads from cache unless force_refresh=True.
    """
```

**Implementation notes:**
- Use `ccxt.binance()` (no API key needed for public OHLCV).
- Binance returns max 1000 candles per call. To get 180 days of 5m data (~52,000 candles), loop with `since` parameter, sleep ~0.2s between calls.
- Convert ccxt's millisecond timestamps to pandas UTC datetimes.
- Deduplicate on timestamp (ccxt sometimes returns overlapping batches).
- Save to CSV; on load, parse timestamp column as datetime.

**Verify:** Run a small script that calls `load_ohlcv(days=7)` and prints `df.head()`, `df.tail()`, `len(df)`, `df.dtypes`. Expect ~2000 rows for 7 days of 5m data.

### Step 3 — `src/strategies.py`

Implement two strategies. Each strategy is a function that takes the OHLCV DataFrame and returns a `pd.Series` of **target positions** indexed by timestamp, where:
- `+1` = long (hold 1 unit)
- `0` = flat (no position)
- (Optionally `-1` = short, but we'll skip short for simplicity in v1.)

**Strategy A — Moving Average Crossover (momentum):**
```python
def ma_crossover(
    df: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.Series:
    """
    Position = 1 when SMA(short) > SMA(long), else 0.
    Use df['close'] for SMA calculation.
    Returns a Series aligned with df.index.
    """
```

**Strategy B — Bollinger Band Mean Reversion:**
```python
def bollinger_mean_reversion(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """
    Compute rolling mean and std of close over `window`.
    Upper band = mean + num_std * std, Lower band = mean - num_std * std.
    Logic (long-only mean reversion):
        - Enter long (position = 1) when close crosses below Lower band.
        - Exit (position = 0) when close crosses above the rolling mean.
    """
```

**Notes:**
- Both functions must produce a position series that is **shifted by 1 bar** when used in the backtester (you act on the next bar's open, not the same bar's close). The shift happens inside the backtester, not in the strategy itself.
- Handle NaN at the start (warm-up period) by returning 0.

### Step 4 — `src/backtester.py`

**Purpose:** Given an OHLCV DataFrame and a position series, simulate the strategy and return an equity curve.

**Function signature:**
```python
def run_backtest(
    df: pd.DataFrame,
    positions: pd.Series,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,  # 0.1% per side, taker fee
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        close, position, position_change, returns, strategy_returns,
        fees, net_returns, equity, drawdown

    Assumptions:
        - Positions are shifted by 1 bar (act on next bar's close).
        - Returns are simple close-to-close log returns or pct change (state which).
        - Fee is applied on every position change (entry and exit).
        - Position size is fixed at 1 unit of notional capital (no leverage, no compounding within trade).
    """
```

**Logic:**
1. `df['returns'] = df['close'].pct_change()`
2. `pos = positions.shift(1).fillna(0)` — act on next bar
3. `df['strategy_returns'] = pos * df['returns']`
4. `df['position_change'] = pos.diff().abs().fillna(0)`
5. `df['fees'] = df['position_change'] * fee_rate`
6. `df['net_returns'] = df['strategy_returns'] - df['fees']`
7. `df['equity'] = initial_capital * (1 + df['net_returns']).cumprod()`
8. `df['drawdown'] = df['equity'] / df['equity'].cummax() - 1`

**Verify:** Run with `ma_crossover` on 90 days of BTC/USDT 5m data. Print final equity, max drawdown, number of position changes. Sanity check: no NaN in equity after the warm-up.

### Step 5 — `src/metrics.py`

**Function:**
```python
def compute_metrics(
    backtest_df: pd.DataFrame,
    bars_per_year: int = 365 * 24 * 12,  # 5m bars
) -> dict:
    """
    Returns a dict with:
        total_return_pct      — final equity / initial - 1, as %
        annualized_return_pct — geometric annualization
        sharpe                — annualized, assuming risk-free = 0
        max_drawdown_pct      — most negative value of drawdown column
        num_trades            — total position changes / 2 (entries+exits / 2)
        win_rate              — fraction of trades with positive PnL
        avg_trade_return_pct  — mean per-trade return
    """
```

**Notes:**
- Sharpe: `mean(net_returns) / std(net_returns) * sqrt(bars_per_year)`.
- For win_rate and avg_trade_return: group consecutive bars with the same position into "trades"; compute the return of each trade. Use the position_change column to detect trade boundaries.
- Handle edge case: zero trades.

### Step 6 — `src/statistical_tests.py`

**Function:**
```python
def bootstrap_sharpe_ci(
    trade_returns: pd.Series,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    bars_per_year: int = 365 * 24 * 12,
) -> tuple[float, float, float]:
    """
    Bootstrap a 95% CI for the Sharpe ratio.

    Procedure:
        1. Resample trade_returns with replacement, len(trade_returns) samples.
        2. Compute Sharpe on the resample.
        3. Repeat n_iterations times.
        4. Return (point_estimate, lower_bound, upper_bound).
    """
```

**Use case:** A Sharpe of 1.5 with CI [0.2, 2.8] is barely significant; a Sharpe of 1.5 with CI [1.2, 1.8] is solid evidence.

### Step 7 — `src/anomaly_detector.py`

**Purpose:** Validate the integrity of the raw OHLCV data. This is the single most important module for the TankX interview narrative — the JD literally says "data validation mindset, attention to data integrity, anomalies, consistency."

**Function:**
```python
def detect_anomalies(df: pd.DataFrame, timeframe: str = "5m") -> dict:
    """
    Returns a dict with:
        gaps             — DataFrame of timestamp gaps (missing candles)
        duplicates       — DataFrame of duplicated timestamps
        volume_outliers  — rows where volume z-score > 3 (rolling 100-bar window)
        return_outliers  — rows where abs(returns) z-score > 3 (rolling 100-bar window)
        zero_volume      — rows with volume == 0
        summary          — dict of counts per category
    """
```

**Implementation:**
- For gaps: compute `df.index.to_series().diff()` and compare to expected timedelta (5 min). Anything > 1.5x is a gap.
- For volume_outliers: rolling z-score, threshold at 3.
- For return_outliers: same but on `close.pct_change()`.
- Summary: `{"total_rows": ..., "gap_count": ..., "duplicate_count": ..., ...}`.

### Step 8 — `app.py` (Streamlit)

The dashboard. Use `st.set_page_config(layout="wide")`.

**Layout:**

**Sidebar:**
- `st.selectbox` — Strategy: ["MA Crossover", "Bollinger Mean Reversion"]
- Conditional parameter sliders based on strategy choice
  - MA: `short_window` (5–100), `long_window` (50–500)
  - Bollinger: `window` (10–50), `num_std` (1.0–3.0)
- `st.slider` for `fee_rate` (0–0.005, default 0.001)
- `st.slider` for `initial_capital` (1k–100k)
- `st.button("Run Backtest")` — or run live on parameter change (use `@st.cache_data` to keep data load cheap)

**Main area — two tabs:** `st.tabs(["Backtest", "Data Quality"])`

**Tab 1 — Backtest:**
1. **Metrics row** — `st.columns(4)` with `st.metric` boxes: Total Return, Sharpe, Max Drawdown, # Trades.
2. **Equity curve** — plotly line chart of `equity` over time, with markers for trade entries/exits.
3. **Drawdown chart** — plotly area chart of `drawdown` (filled red).
4. **Price chart with signals** — plotly candlestick (or line) with buy/sell markers overlaid.
5. **Statistical significance** — show `st.metric` with Sharpe point estimate and 95% CI from bootstrap. Add a sentence: "If the CI crosses zero, the result is not statistically distinguishable from random."

**Tab 2 — Data Quality:**
1. Summary table from `detect_anomalies`.
2. Section per anomaly type, each with `st.dataframe` of flagged rows (limit 20 rows display).
3. A small bar chart showing volume distribution with outliers highlighted.

**Caching:**
- `@st.cache_data` on `load_ohlcv` so the data only loads once per session.
- `@st.cache_data` on `detect_anomalies` keyed by data hash.

### Step 9 — `README.md`

The README is graded as much as the code by anyone serious. Use this exact structure:

```markdown
# Crypto Backtesting Mini-Framework

A minimal end-to-end backtesting framework for crypto trading strategies, with an interactive Streamlit dashboard and built-in data quality checks.

## Motivation
Brief paragraph: built as a personal exploration of quantitative trading workflows — fetching real exchange data, implementing classical strategies, validating data integrity, and stress-testing results with bootstrap statistical significance. Built in one evening to test how far one can go with a clean minimal stack.

## Features
- Fetches Binance OHLCV data via ccxt (public API, no key)
- Two strategies: MA Crossover (momentum) and Bollinger Bands (mean reversion)
- Backtest engine with transaction costs
- Interactive Streamlit dashboard with parameter tuning
- Data quality / anomaly detection module
- Bootstrap-based statistical significance test for Sharpe ratio

## Architecture
ASCII or mermaid diagram of: data_loader → strategies → backtester → metrics → dashboard, with anomaly_detector as parallel branch.

## Quickstart
```bash
git clone <repo>
cd crypto-backtest
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
(directory tree)

## Strategies Implemented
Brief paragraph on each, with formula.

## Metrics
List with definitions: Total Return, Annualized Return, Sharpe, Max Drawdown, Win Rate, Trade Count.

## Data Validation Approach
Brief description of each check: gaps, duplicates, volume/return outliers, zero volume.

## Limitations & Honest Caveats
This is **not** a production trading system. Specifically:
- No slippage modeling beyond a fixed fee rate.
- No order book / market impact modeling.
- No latency modeling.
- In-sample only (no train/test split or walk-forward validation).
- Limited to spot, long-only positions.
- No regime-change adaptation.
A real HFT/market-making system would model all of these.

## Future Work
- Walk-forward validation with rolling windows.
- Slippage model based on order book depth.
- Multi-symbol portfolio backtest.
- Live paper trading via ccxt's order endpoints.

## Tech Stack
List.
```

---

## 6. Strategy & Metric Formulas (precise)

### Simple Moving Average
```
SMA(t, n) = mean(close[t-n+1 : t+1])
```

### MA Crossover signal
```
position(t) = 1 if SMA(t, short) > SMA(t, long) else 0
```

### Bollinger Bands
```
mean(t) = SMA(t, window)
std(t)  = rolling_std(close, window)(t)
upper(t) = mean(t) + num_std * std(t)
lower(t) = mean(t) - num_std * std(t)
```

### Bollinger Mean Reversion signal
```
if not in position and close(t) < lower(t):  enter (position = 1)
if in position and close(t) > mean(t):       exit  (position = 0)
```

### Sharpe Ratio (annualized)
```
sharpe = mean(net_returns) / std(net_returns) * sqrt(bars_per_year)
```
For 5-minute bars: `bars_per_year = 365 * 24 * 12 = 105,120`.

### Max Drawdown
```
running_max(t) = max(equity[0 : t+1])
drawdown(t)    = equity(t) / running_max(t) - 1
max_drawdown   = min(drawdown)
```

---

## 7. Acceptance Criteria

The project is "done" when all of these hold:

- [ ] `pip install -r requirements.txt && streamlit run app.py` works from a fresh clone.
- [ ] App loads BTC/USDT 5m data for the last 90+ days within ~30 seconds on first run, instant on subsequent runs (caching).
- [ ] Both strategies produce a valid equity curve with no NaN/Inf anomalies.
- [ ] Changing a slider triggers re-computation and updates all charts in under 2 seconds.
- [ ] Data Quality tab shows at least a few flagged anomalies on real Binance data (volume spikes are common during news events).
- [ ] Bootstrap CI is computed and displayed.
- [ ] README has all sections listed in Step 9.
- [ ] Code has docstrings on every public function; type hints throughout.
- [ ] `.gitignore` excludes `data/*.csv` and `__pycache__`.

---

## 8. Common Pitfalls to Avoid

1. **Look-ahead bias.** Do NOT use the same bar's close to decide AND execute. Always shift positions by 1 bar in the backtester.
2. **Survivorship bias.** Not relevant here (we use BTC only), but mention it in the README's Limitations.
3. **Overfit Sharpe.** Don't tune parameters until Sharpe is huge — that's overfitting. The point is to show the **process**, not to find a winning strategy.
4. **NaN propagation.** Indicators have a warm-up period; the first N bars will be NaN. Fill those positions with 0.
5. **Stale data cache.** If you change the timeframe or symbol, the cache should invalidate. Include symbol and timeframe in the cache filename.
6. **Streamlit re-running everything on every interaction.** Use `@st.cache_data` liberally on data fetch and anomaly detection.
7. **Plotly performance with large data.** 50k+ candles can be slow. Use line charts (not candlestick) for the equity curve; downsample if needed.

---

## 9. Stretch Goals (only if time permits, in priority order)

1. **Walk-forward validation:** split data into train/test windows, optimize parameters on train, evaluate on test. Show the out-of-sample Sharpe.
2. **A second symbol** (ETH/USDT) and a dropdown to switch.
3. **Trade log table:** list every trade with entry time, exit time, return, duration.
4. **Position sizing using volatility targeting:** scale position inversely with rolling volatility.
5. **Compare strategies side by side:** run both, plot equity curves on the same chart.

Do **not** start on stretch goals until the acceptance criteria are met.

---

## 10. Tone / Style Guidelines

- Code: clean, typed, docstring'd. Prefer pure functions over classes where possible (this is a small project, not a framework).
- Comments: minimal; let the code speak. One-line docstrings for short functions, multi-line for the public API.
- Variable names: explicit (`short_window`, not `sw`). Time-related variables in `_at` suffix if needed (`entry_at`).
- No emojis in code or comments. Plain professional.

---

## 11. Final Note

If anything in this spec is ambiguous, pick the simpler interpretation and document the choice in the README under "Implementation Decisions". I'd rather have a working v1 that's slightly different from spec than a perfect implementation that took twice as long.

End of spec.
