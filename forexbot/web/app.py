"""
ForexBot Pro — TradingView-style Backtest + Live FX Dashboard (IST)
Walk-forward backtesting with per-bar indicator recomputation for genuine results.
"""

import math
import os
import time as _time
from dataclasses import dataclass
from datetime import timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Secrets / Config
# ---------------------------------------------------------------------------

def _get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


TWELVEDATA_API_KEY = _get_secret("TWELVEDATA_API_KEY", "")

IST = ZoneInfo("Asia/Kolkata")
UTC = timezone.utc

TF_MAP = {
    "5m":  {"td": "5min",  "yf": "5m",  "period_days": 60,    "bar_minutes": 5,    "htf": "30m"},
    "15m": {"td": "15min", "yf": "15m", "period_days": 120,   "bar_minutes": 15,   "htf": "4h"},
    "30m": {"td": "30min", "yf": "30m", "period_days": 180,   "bar_minutes": 30,   "htf": "4h"},
    "4h":  {"td": "4h",    "yf": "60m", "period_days": 730,   "bar_minutes": 240,  "htf": "1d"},
    "1d":  {"td": "1day",  "yf": "1d",  "period_days": 11000, "bar_minutes": 1440, "htf": "1d"},
}

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF"]

# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    stop: float
    target: float
    pnl_pct: float
    pnl_usd: float
    r_multiple: float
    bars: int
    reason: str


# ---------------------------------------------------------------------------
# Strategy Profiles
# ---------------------------------------------------------------------------

STRATEGY_PROFILES = {
    "strat1": {
        "name": "Strategy 1 — Trend Breakout Pro",
        "description": (
            "Trades breakout continuation in trend direction. "
            "Uses EMA(20/50) alignment, RSI 52-72 for longs / 28-48 for shorts, "
            "20-bar channel breakout, and NATR volatility filter. "
            "Best on trending pairs like GBPUSD, EURUSD on 4h/1d."
        ),
    },
    "strat2": {
        "name": "Strategy 2 — Mean Reversion Bounce",
        "description": (
            "Fades stretched moves back toward trend equilibrium. "
            "Buys RSI <= 35 in uptrend (EMA fast >= slow), sells RSI >= 65 in downtrend. "
            "Wider NATR tolerance (up to 3%). Best on range-bound or consolidating markets."
        ),
    },
    "strat3": {
        "name": "Strategy 3 — Momentum Pullback Sniper",
        "description": (
            "Enters pullbacks inside strong trend after breakout confirmation. "
            "RSI 45-58 for longs / 42-55 for shorts (not at extremes). "
            "Requires 20-bar breakout + trend alignment. Tighter volatility. "
            "Best selective strategy — fewer trades, higher conviction."
        ),
    },
}


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

def _inject_autorefresh(enabled: bool, seconds: int) -> None:
    if not enabled:
        return
    ms = max(5, int(seconds)) * 1000
    st.markdown(
        f'<script>setTimeout(function(){{window.location.reload();}},{ms});</script>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data loaders  (cached, long TTL for large daily datasets)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Fetching from TwelveData...")
def _fetch_twelvedata(pair: str, timeframe: str, outputsize: int = 5000) -> pd.DataFrame:
    if not TWELVEDATA_API_KEY:
        return pd.DataFrame()
    symbol = f"{pair[:3]}/{pair[3:]}"
    interval = TF_MAP[timeframe]["td"]
    final_outputsize = 40000 if interval == "1day" else outputsize
    params = {
        "symbol": symbol, "interval": interval, "outputsize": final_outputsize,
        "apikey": TWELVEDATA_API_KEY, "format": "JSON", "timezone": "UTC",
    }
    try:
        resp = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=60)
        resp.raise_for_status()
        values = resp.json().get("values", [])
        if not values:
            return pd.DataFrame()
        rows = [
            {"datetime": v["datetime"],
             "Open": float(v["open"]), "High": float(v["high"]),
             "Low": float(v["low"]), "Close": float(v["close"]),
             "Volume": float(v.get("volume", 0) or 0)}
            for v in values
        ]
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df.set_index("datetime").sort_index()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Fetching from yfinance...")
def _fetch_yfinance(pair: str, timeframe: str) -> pd.DataFrame:
    ticker = f"{pair}=X"
    interval = TF_MAP[timeframe]["yf"]
    period_days = TF_MAP[timeframe]["period_days"]
    try:
        if timeframe in ("1d", "4h"):
            hist = yf.download(ticker, period="max", interval="1d" if timeframe == "1d" else interval,
                               auto_adjust=False, progress=False)
        else:
            hist = yf.download(ticker, period=f"{period_days}d", interval=interval,
                               auto_adjust=False, progress=False)
        if hist.empty:
            return pd.DataFrame()
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [c[0] for c in hist.columns]
        df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        else:
            df.index = df.index.tz_convert(UTC)
        if timeframe == "4h":
            df = df.resample("4h").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            ).dropna()
        return df
    except Exception:
        return pd.DataFrame()


def load_ohlcv(pair: str, timeframe: str) -> tuple[pd.DataFrame, str]:
    td = _fetch_twelvedata(pair, timeframe)
    if not td.empty:
        return td, "TwelveData"
    yf_data = _fetch_yfinance(pair, timeframe)
    if not yf_data.empty:
        return yf_data, "yfinance"
    return pd.DataFrame(), "none"


def to_ist(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.tz_convert(IST)
    return out


# ---------------------------------------------------------------------------
# Walk-forward indicator computation (per-bar rolling window)
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_indicators_walkforward(df: pd.DataFrame, bar_idx: int, lookback: int = 200) -> dict | None:
    """
    Compute indicators using ONLY data available up to bar_idx (walk-forward).
    Uses a rolling lookback window so each bar sees only past data — no lookahead.
    Returns None if not enough history.
    """
    start = max(0, bar_idx - lookback)
    window = df.iloc[start:bar_idx + 1]
    if len(window) < 60:
        return None

    close = window["Close"]
    high = window["High"]
    low = window["Low"]

    ema_fast = _ema(close, 20)
    ema_slow = _ema(close, 50)
    rsi_14 = _rsi(close, 14)
    atr_14 = _atr(high, low, close, 14)
    natr = atr_14 / close

    rolling_high = high.rolling(20).max().shift(1)
    rolling_low = low.rolling(20).min().shift(1)
    breakout_up = close.iloc[-1] > rolling_high.iloc[-1] if pd.notna(rolling_high.iloc[-1]) else False
    breakout_dn = close.iloc[-1] < rolling_low.iloc[-1] if pd.notna(rolling_low.iloc[-1]) else False

    return {
        "ema_fast": float(ema_fast.iloc[-1]),
        "ema_slow": float(ema_slow.iloc[-1]),
        "rsi_14": float(rsi_14.iloc[-1]) if pd.notna(rsi_14.iloc[-1]) else 50.0,
        "atr_14": float(atr_14.iloc[-1]) if pd.notna(atr_14.iloc[-1]) else 0.0,
        "natr": float(natr.iloc[-1]) if pd.notna(natr.iloc[-1]) else 0.0,
        "breakout_up": bool(breakout_up),
        "breakout_dn": bool(breakout_dn),
    }


# ---------------------------------------------------------------------------
# Strategy signal generators
# ---------------------------------------------------------------------------

def _signal_strat1(ind: dict) -> tuple[bool, bool]:
    """Trend Breakout Pro"""
    ef, es, rsi = ind["ema_fast"], ind["ema_slow"], ind["rsi_14"]
    natr, bu, bd = ind["natr"], ind["breakout_up"], ind["breakout_dn"]
    long = (ef > es) and (52 <= rsi <= 72) and bu and (0.0004 <= natr <= 0.02)
    short = (ef < es) and (28 <= rsi <= 48) and bd and (0.0004 <= natr <= 0.02)
    return long, short


def _signal_strat2(ind: dict) -> tuple[bool, bool]:
    """Mean Reversion Bounce"""
    ef, es, rsi, natr = ind["ema_fast"], ind["ema_slow"], ind["rsi_14"], ind["natr"]
    long = (ef >= es) and (rsi <= 35) and (0.0004 <= natr <= 0.03)
    short = (ef <= es) and (rsi >= 65) and (0.0004 <= natr <= 0.03)
    return long, short


def _signal_strat3(ind: dict) -> tuple[bool, bool]:
    """Momentum Pullback Sniper"""
    ef, es, rsi = ind["ema_fast"], ind["ema_slow"], ind["rsi_14"]
    natr, bu, bd = ind["natr"], ind["breakout_up"], ind["breakout_dn"]
    long = (ef > es) and (45 <= rsi <= 58) and bu and (0.0003 <= natr <= 0.02)
    short = (ef < es) and (42 <= rsi <= 55) and bd and (0.0003 <= natr <= 0.02)
    return long, short


SIGNAL_FUNCS = {
    "strat1": _signal_strat1,
    "strat2": _signal_strat2,
    "strat3": _signal_strat3,
}


# ---------------------------------------------------------------------------
# Max hold bars per timeframe
# ---------------------------------------------------------------------------

def _max_hold_bars(timeframe: str) -> int:
    return {"5m": 96, "15m": 64, "30m": 40, "4h": 14, "1d": 12}.get(timeframe, 20)


# ---------------------------------------------------------------------------
# HTF trend builder
# ---------------------------------------------------------------------------

def _build_htf_trend(pair: str, timeframe: str) -> pd.Series:
    htf = TF_MAP[timeframe]["htf"]
    htf_df, _ = load_ohlcv(pair, htf)
    if htf_df.empty:
        return pd.Series(dtype=float)
    close = htf_df["Close"]
    ema_f = _ema(close, 20)
    ema_s = _ema(close, 50)
    trend = np.where(ema_f > ema_s, 1, -1)
    return pd.Series(trend, index=htf_df.index)


# ---------------------------------------------------------------------------
# REAL Walk-Forward Backtest Engine
# ---------------------------------------------------------------------------

def run_backtest(
    pair: str,
    timeframe: str,
    df: pd.DataFrame,
    sl_atr: float,
    tp_atr: float,
    risk_pct: float,
    spread_pips: float,
    use_mtf_filter: bool,
    strategy_key: str,
    progress_cb=None,
) -> tuple[pd.DataFrame, list[Trade], dict]:
    """
    Walk-forward backtest: indicators are recomputed per bar using only
    historically available data (no lookahead bias). Processing time is
    proportional to the number of candles — this is a REAL computation.
    """
    n = len(df)
    if n < 400:
        return pd.DataFrame(), [], {"error": f"Need >= 400 candles for robust test (got {n})."}

    pip_size = 0.01 if pair.endswith("JPY") else 0.0001
    spread = spread_pips * pip_size
    signal_fn = SIGNAL_FUNCS.get(strategy_key, _signal_strat1)

    htf_trend = _build_htf_trend(pair, timeframe) if use_mtf_filter else pd.Series(dtype=float)

    start_equity = 10_000.0
    equity = start_equity
    peak = equity
    max_dd = 0.0
    gross_win = 0.0
    gross_loss = 0.0

    position = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_ts = None
    bars_held = 0
    stop = 0.0
    target = 0.0
    risk_amount = 0.0

    trades: list[Trade] = []
    equity_curve: list[tuple[pd.Timestamp, float]] = []

    # Walk-forward: start from bar 200 so we have enough lookback
    start_bar = 200
    total_bars = n - start_bar
    last_progress_time = _time.monotonic()

    for loop_idx, bar_idx in enumerate(range(start_bar, n)):
        ts = df.index[bar_idx]
        row = df.iloc[bar_idx]

        # ---- Progress callback (throttled to ~20 updates/sec) ----
        now_mono = _time.monotonic()
        if progress_cb and (loop_idx == 0 or loop_idx == total_bars - 1 or (now_mono - last_progress_time) >= 0.05):
            progress_cb(loop_idx + 1, total_bars, ts)
            last_progress_time = now_mono

        # ---- Walk-forward indicator computation (the real work per bar) ----
        ind = compute_indicators_walkforward(df, bar_idx, lookback=200)
        if ind is None:
            equity_curve.append((ts, equity))
            continue

        close_price = float(row["Close"])

        # ---- Signal generation ----
        long_signal, short_signal = signal_fn(ind)

        # ---- HTF filter ----
        if use_mtf_filter and not htf_trend.empty:
            trend_slice = htf_trend[htf_trend.index <= ts]
            if not trend_slice.empty:
                trend_val = int(trend_slice.iloc[-1])
                if trend_val != 1:
                    long_signal = False
                if trend_val != -1:
                    short_signal = False

        # ---- Position management ----
        if position == 0:
            if long_signal or short_signal:
                position = 1 if long_signal else -1
                entry_price = close_price + spread / 2 if position == 1 else close_price - spread / 2
                entry_atr = ind["atr_14"]
                entry_ts = ts
                bars_held = 0
                stop = entry_price - sl_atr * entry_atr if position == 1 else entry_price + sl_atr * entry_atr
                target = entry_price + tp_atr * entry_atr if position == 1 else entry_price - tp_atr * entry_atr
                risk_amount = max(10.0, equity * (risk_pct / 100.0))
        else:
            bars_held += 1
            exit_reason = None
            exit_price = close_price
            bar_low = float(row["Low"])
            bar_high = float(row["High"])

            if position == 1 and bar_low <= stop:
                exit_price = stop
                exit_reason = "SL"
            elif position == -1 and bar_high >= stop:
                exit_price = stop
                exit_reason = "SL"
            elif position == 1 and bar_high >= target:
                exit_price = target
                exit_reason = "TP"
            elif position == -1 and bar_low <= target:
                exit_price = target
                exit_reason = "TP"
            elif bars_held >= _max_hold_bars(timeframe):
                exit_reason = "TIME"
            elif (position == 1 and short_signal) or (position == -1 and long_signal):
                exit_reason = "FLIP"

            if exit_reason:
                exit_exec = exit_price - spread / 2 if position == 1 else exit_price + spread / 2
                move = (exit_exec - entry_price) * position
                sl_move = abs(entry_price - stop)
                r_multiple = (move / sl_move) if sl_move > 0 else 0.0
                pnl_usd = risk_amount * r_multiple
                pnl_pct = (pnl_usd / equity) * 100 if equity > 0 else 0.0
                equity += pnl_usd

                if pnl_usd >= 0:
                    gross_win += pnl_usd
                else:
                    gross_loss += abs(pnl_usd)

                trades.append(Trade(
                    side="BUY" if position == 1 else "SELL",
                    entry_time=entry_ts, exit_time=ts,
                    entry=entry_price, exit=exit_exec,
                    stop=stop, target=target,
                    pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                    r_multiple=r_multiple, bars=bars_held,
                    reason=exit_reason,
                ))

                position = 0
                entry_price = 0.0
                entry_atr = 0.0
                entry_ts = None
                bars_held = 0
                stop = 0.0
                target = 0.0
                risk_amount = 0.0

        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
        equity_curve.append((ts, equity))

    # ---- Compute stats ----
    eq = pd.DataFrame(equity_curve, columns=["time", "equity"]).set_index("time")
    returns = eq["equity"].pct_change().dropna()
    bars_per_year = (365 * 24 * 60) / TF_MAP[timeframe]["bar_minutes"]
    sharpe = (returns.mean() / returns.std() * math.sqrt(bars_per_year)) if returns.std() > 0 else 0.0

    win_rate = (sum(1 for t in trades if t.pnl_usd > 0) / len(trades) * 100) if trades else 0.0
    avg_r = float(np.mean([t.r_multiple for t in trades])) if trades else 0.0
    expectancy = float(np.mean([t.pnl_usd for t in trades])) if trades else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)

    stats = {
        "total_return_pct": ((equity - start_equity) / start_equity) * 100,
        "ending_equity": equity,
        "trades": len(trades),
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd * 100,
        "sharpe": sharpe,
        "avg_r": avg_r,
        "expectancy_usd": expectancy,
        "longs": sum(1 for t in trades if t.side == "BUY"),
        "shorts": sum(1 for t in trades if t.side == "SELL"),
    }
    return eq, trades, stats


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------

def build_trade_overlay_chart(df_ist: pd.DataFrame, trades: list[Trade], pair: str, timeframe: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_ist.index, open=df_ist["Open"], high=df_ist["High"],
        low=df_ist["Low"], close=df_ist["Close"], name=f"{pair} {timeframe}",
    ))

    entry_x, entry_y, exit_x, exit_y = [], [], [], []

    for trade in trades:
        start = trade.entry_time.tz_convert(IST)
        end = trade.exit_time.tz_convert(IST)
        win = trade.pnl_usd >= 0
        profit_color = "rgba(16,185,129,0.17)" if win else "rgba(239,68,68,0.17)"

        fig.add_shape(type="rect", x0=start, x1=end,
                      y0=min(trade.entry, trade.stop), y1=max(trade.entry, trade.stop),
                      fillcolor="rgba(239,68,68,0.14)", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=start, x1=end,
                      y0=min(trade.entry, trade.target), y1=max(trade.entry, trade.target),
                      fillcolor="rgba(16,185,129,0.14)", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=start, x1=end,
                      y0=min(trade.entry, trade.exit), y1=max(trade.entry, trade.exit),
                      fillcolor=profit_color, line=dict(width=0), layer="below")

        fig.add_annotation(
            x=end, y=trade.exit,
            text=f"{trade.pnl_pct:+.2f}% | {trade.r_multiple:+.2f}R",
            showarrow=True, arrowhead=2,
            font=dict(size=10, color="#111827"),
            bgcolor="rgba(255,255,255,0.85)",
        )
        entry_x.append(start); entry_y.append(trade.entry)
        exit_x.append(end); exit_y.append(trade.exit)

    if entry_x:
        fig.add_trace(go.Scatter(x=entry_x, y=entry_y, mode="markers",
                                 marker=dict(symbol="triangle-up", size=10, color="#2563eb"), name="Entries"))
    if exit_x:
        fig.add_trace(go.Scatter(x=exit_x, y=exit_y, mode="markers",
                                 marker=dict(symbol="x", size=9, color="#111827"), name="Exits"))

    fig.update_layout(height=700, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=25, b=10),
                      legend=dict(orientation="h"))
    return fig


# ---------------------------------------------------------------------------
# Live signal helper
# ---------------------------------------------------------------------------

def _last_signal(df: pd.DataFrame) -> str:
    if len(df) < 60:
        return "NO_DATA"
    ind = compute_indicators_walkforward(df, len(df) - 1, lookback=200)
    if ind is None:
        return "NO_DATA"
    long, short = _signal_strat1(ind)
    if long:
        return "BUY"
    if short:
        return "SELL"
    return "HOLD"


# ---------------------------------------------------------------------------
# Main Streamlit App
# ---------------------------------------------------------------------------

def app() -> None:
    st.set_page_config(page_title="ForexBot Pro", layout="wide")
    st.title("ForexBot Pro — TradingView-style Backtest + Live FX Dashboard (IST)")
    st.caption("Live data + walk-forward backtesting (5m, 15m, 30m, 4h, 1d) with entry/exit markers and RR zones.")

    # ---- Sidebar ----
    with st.sidebar:
        st.subheader("Execution")
        pair = st.selectbox("Pair", PAIRS, index=1)
        timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "4h", "1d"], index=3)
        auto_refresh = st.checkbox("Realtime refresh", value=True)
        refresh_sec = st.slider("Refresh seconds", 10, 180, 30)

        st.subheader("Backtest params")
        default_start = pd.Timestamp("2015-01-01").date()
        default_end = pd.Timestamp("2026-02-01").date()
        bt_start_date = st.date_input("Backtest start date", value=default_start)
        bt_end_date = st.date_input("Backtest end date", value=default_end)
        sl_atr = st.number_input("SL ATR", 0.5, 5.0, 1.5, 0.1)
        tp_atr = st.number_input("TP ATR", 0.8, 10.0, 2.5, 0.1)
        risk_pct = st.number_input("Risk per trade (%)", 0.1, 3.0, 1.0, 0.1)
        spread_pips = st.number_input("Spread (pips)", 0.0, 5.0, 0.8, 0.1)
        use_mtf_filter = st.checkbox("Use higher-timeframe trend filter", value=True)

        strategy_key = st.selectbox(
            "Backtest algorithm",
            options=list(STRATEGY_PROFILES.keys()),
            index=0,
            format_func=lambda k: STRATEGY_PROFILES[k]["name"],
        )
        st.caption(STRATEGY_PROFILES[strategy_key]["description"])

        run_backtest_btn = st.button("Run robust backtest", use_container_width=True)
        auto_rank_btn = st.button("Auto-Rank & Run Best", use_container_width=True)
        reset_backtest_btn = st.button("Reset backtest data", use_container_width=True)

    _inject_autorefresh(auto_refresh, refresh_sec)

    # ---- Load data ----
    df, source = load_ohlcv(pair, timeframe)
    if df.empty:
        st.error("No live data from TwelveData/yfinance right now.")
        st.stop()

    last_utc = df.index.max()
    last_ist = last_utc.astimezone(IST)
    now_ist = pd.Timestamp.now(tz=IST)
    st.caption(
        f"Source: {source} | Last candle UTC: {last_utc:%Y-%m-%d %H:%M:%S} "
        f"| Last candle IST: {last_ist:%Y-%m-%d %H:%M:%S} | Now IST: {now_ist:%Y-%m-%d %H:%M:%S}"
    )

    tabs = st.tabs(["Backtest Chart", "Realtime Dashboard"])

    # ================================================================
    # TAB 1 — Backtest
    # ================================================================
    with tabs[0]:
        if bt_start_date > bt_end_date:
            st.error("Backtest start date must be before end date.")
            st.stop()

        bt_start_utc = pd.Timestamp(bt_start_date).tz_localize(UTC)
        bt_end_utc = (pd.Timestamp(bt_end_date) + pd.Timedelta(days=1)).tz_localize(UTC)
        dated_df = df[(df.index >= bt_start_utc) & (df.index < bt_end_utc)].copy()

        if dated_df.empty:
            st.warning("No candles in selected range for this timeframe/source.")
            st.stop()

        total_available = len(dated_df)
        st.caption(f"Total candles available in range: **{total_available:,}** ({bt_start_date} to {bt_end_date})")

        # NO artificial cap — slider goes up to ALL available candles
        replay_bars = st.slider(
            "Replay candles (last N bars from selected date range)",
            min_value=min(200, total_available),
            max_value=total_available,
            value=total_available,  # default: use ALL candles
            step=max(1, total_available // 200),
        )
        replay_df = dated_df.tail(replay_bars)
        replay_df_ist = to_ist(replay_df)

        est_time_per_bar = 0.002  # ~2ms per bar for walk-forward computation
        est_seconds = int(len(replay_df) * est_time_per_bar)
        est_min, est_sec = divmod(est_seconds, 60)
        st.caption(
            f"Using last **{len(replay_df):,}** candles for replay | "
            f"Estimated backtest time: ~{est_min}m {est_sec}s (walk-forward per-bar computation)"
        )

        # ---- Reset ----
        if reset_backtest_btn:
            st.session_state.pop("last_bt", None)
            st.session_state.pop("last_bt_meta", None)
            st.cache_data.clear()
            st.success("Backtest state reset. Click a backtest button for a fresh run.")

        current_meta = {
            "pair": pair, "timeframe": timeframe, "replay_bars": replay_bars,
            "bt_start": str(bt_start_date), "bt_end": str(bt_end_date),
            "sl_atr": float(sl_atr), "tp_atr": float(tp_atr),
            "risk_pct": float(risk_pct), "spread_pips": float(spread_pips),
            "use_mtf_filter": bool(use_mtf_filter), "strategy_key": strategy_key,
        }

        # ---- Run backtest ----
        if run_backtest_btn or auto_rank_btn:
            progress_bar = st.progress(0, text="Initialising walk-forward backtest engine...")
            status_text = st.empty()
            started_at = _time.monotonic()

            def _on_progress(done: int, total: int, ts_point: pd.Timestamp) -> None:
                ratio = done / total if total > 0 else 0.0
                pct = int(ratio * 100)
                elapsed = _time.monotonic() - started_at
                eta_sec = int((elapsed / ratio) - elapsed) if ratio > 0.001 else 0
                mins, secs = divmod(eta_sec, 60)
                elapsed_mins, elapsed_secs = divmod(int(elapsed), 60)
                current_ist = ts_point.tz_convert(IST).strftime("%Y-%m-%d %H:%M")
                progress_bar.progress(
                    min(100, max(0, pct)),
                    text=(
                        f"Walk-forward: {pct}% | Bar {done:,}/{total:,} "
                        f"| Elapsed: {elapsed_mins}m {elapsed_secs}s "
                        f"| ETA: {mins}m {secs}s | At: {current_ist} IST"
                    ),
                )

            final_strategy = strategy_key

            # ---- Auto-Rank mode ----
            if auto_rank_btn:
                status_text.info(f"Auto-ranking all 3 strategies on {pair} {timeframe} ({len(replay_df):,} bars)...")
                rank_results = []
                for i, key in enumerate(STRATEGY_PROFILES.keys()):
                    rank_started = _time.monotonic()
                    progress_bar.progress(
                        int((i / len(STRATEGY_PROFILES)) * 30),
                        text=f"Ranking: running '{STRATEGY_PROFILES[key]['name']}'...",
                    )
                    _, _, s = run_backtest(
                        pair=pair, timeframe=timeframe, df=replay_df,
                        sl_atr=sl_atr, tp_atr=tp_atr, risk_pct=risk_pct,
                        spread_pips=spread_pips, use_mtf_filter=use_mtf_filter,
                        strategy_key=key,
                    )
                    rank_elapsed = _time.monotonic() - rank_started
                    score = (s.get("sharpe", -99) * max(0, s.get("profit_factor", 0))) if "error" not in s else -999
                    rank_results.append({"key": key, "score": score, "stats": s, "time": rank_elapsed})

                # Show ranking table
                rank_df = pd.DataFrame([
                    {
                        "Strategy": STRATEGY_PROFILES[r["key"]]["name"],
                        "Return %": f"{r['stats'].get('total_return_pct', 0):+.2f}",
                        "Win Rate": f"{r['stats'].get('win_rate_pct', 0):.1f}%",
                        "Sharpe": f"{r['stats'].get('sharpe', 0):.2f}",
                        "PF": f"{r['stats'].get('profit_factor', 0):.2f}",
                        "Score": f"{r['score']:.3f}",
                        "Time": f"{r['time']:.1f}s",
                    }
                    for r in sorted(rank_results, key=lambda x: x["score"], reverse=True)
                ])
                st.subheader("Strategy Ranking")
                st.dataframe(rank_df, use_container_width=True)

                best = max(rank_results, key=lambda x: x["score"])
                final_strategy = best["key"]
                status_text.success(
                    f"Winner: **{STRATEGY_PROFILES[final_strategy]['name']}** "
                    f"(Score: {best['score']:.3f}). Running detailed backtest with progress..."
                )
                started_at = _time.monotonic()

            # ---- Final detailed backtest run ----
            eq, trades, stats = run_backtest(
                pair=pair, timeframe=timeframe, df=replay_df,
                sl_atr=sl_atr, tp_atr=tp_atr, risk_pct=risk_pct,
                spread_pips=spread_pips, use_mtf_filter=use_mtf_filter,
                strategy_key=final_strategy,
                progress_cb=_on_progress,
            )

            elapsed_total = _time.monotonic() - started_at
            progress_bar.progress(100, text=f"Backtest complete: 100% ({elapsed_total:.1f}s for {len(replay_df):,} bars)")

            st.session_state["last_bt"] = (eq, trades, stats)
            current_meta["strategy_key"] = final_strategy
            st.session_state["last_bt_meta"] = current_meta

        # ---- Display results ----
        if "last_bt" in st.session_state:
            eq, trades, stats = st.session_state["last_bt"]
            last_meta = st.session_state.get("last_bt_meta", {})

            if last_meta and last_meta != current_meta:
                st.warning("Parameters changed since last run. Click a backtest button to refresh.")

            if "error" in stats:
                st.warning(stats["error"])
            else:
                m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
                m1.metric("Return", f"{stats['total_return_pct']:+.2f}%")
                m2.metric("Trades", f"{stats['trades']}")
                m3.metric("Win rate", f"{stats['win_rate_pct']:.1f}%")
                pf_str = "inf" if math.isinf(stats["profit_factor"]) else f"{stats['profit_factor']:.2f}"
                m4.metric("Profit factor", pf_str)
                m5.metric("Max DD", f"{stats['max_drawdown_pct']:.2f}%")
                m6.metric("Sharpe", f"{stats['sharpe']:.2f}")
                m7.metric("Avg R", f"{stats['avg_r']:+.2f}")

                st.plotly_chart(
                    build_trade_overlay_chart(replay_df_ist, trades, pair, timeframe),
                    use_container_width=True,
                )

                if not eq.empty:
                    eq_ist = eq.copy()
                    eq_ist.index = eq_ist.index.tz_convert(IST)
                    st.subheader("Equity Curve")
                    st.line_chart(eq_ist["equity"], use_container_width=True)

                if trades:
                    st.subheader(f"Trade Log ({len(trades)} trades)")
                    trades_df = pd.DataFrame([
                        {
                            "Side": t.side,
                            "Entry IST": t.entry_time.tz_convert(IST).strftime("%Y-%m-%d %H:%M"),
                            "Exit IST": t.exit_time.tz_convert(IST).strftime("%Y-%m-%d %H:%M"),
                            "Entry": round(t.entry, 5),
                            "Stop": round(t.stop, 5),
                            "Target": round(t.target, 5),
                            "Exit": round(t.exit, 5),
                            "R": round(t.r_multiple, 2),
                            "PnL $": round(t.pnl_usd, 2),
                            "PnL %": round(t.pnl_pct, 2),
                            "Bars": t.bars,
                            "Reason": t.reason,
                        }
                        for t in trades
                    ])
                    st.dataframe(trades_df, use_container_width=True, height=400)
        else:
            st.info(
                "No backtest results yet. Configure parameters in the sidebar "
                "and click **Run robust backtest** or **Auto-Rank & Run Best**."
            )
            st.plotly_chart(
                build_trade_overlay_chart(replay_df_ist, [], pair, timeframe),
                use_container_width=True,
            )

    # ================================================================
    # TAB 2 — Realtime Dashboard
    # ================================================================
    with tabs[1]:
        st.subheader("Realtime Pair Signals (IST)")
        rows = []
        for p in PAIRS:
            d, src = load_ohlcv(p, timeframe)
            if d.empty:
                rows.append({"Pair": p, "Signal": "NO_DATA", "Close": None, "Last IST": None, "Source": src})
                continue
            sig = _last_signal(d)
            last_close = float(d["Close"].iloc[-1])
            last_t = d.index.max().tz_convert(IST)
            rows.append({
                "Pair": p,
                "Signal": sig,
                "Close": round(last_close, 5),
                "Last IST": last_t.strftime("%Y-%m-%d %H:%M:%S"),
                "Source": src,
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        live_df_ist = to_ist(df.tail(500))
        st.plotly_chart(
            build_trade_overlay_chart(live_df_ist, [], pair, timeframe),
            use_container_width=True,
        )


if __name__ == "__main__":
    app()
