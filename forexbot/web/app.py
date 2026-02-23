import math
import os
from dataclasses import dataclass
from datetime import timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from zoneinfo import ZoneInfo


def _get_secret(key: str, default: str = "") -> str:
    """Read secret from Streamlit Cloud secrets, fallback to env vars."""
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
    "5m": {"td": "5min", "yf": "5m", "period_days": 30, "bar_minutes": 5, "htf": "30m"},
    "15m": {"td": "15min", "yf": "15m", "period_days": 60, "bar_minutes": 15, "htf": "4h"},
    "30m": {"td": "30min", "yf": "30m", "period_days": 90, "bar_minutes": 30, "htf": "4h"},
    "4h": {"td": "4h", "yf": "60m", "period_days": 365, "bar_minutes": 240, "htf": "1d"},
    "1d": {"td": "1day", "yf": "1d", "period_days": 3650, "bar_minutes": 1440, "htf": "1d"},
}

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF"]


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


STRATEGY_PROFILES = {
    "strat1": {
        "name": "Strategy 1 — Trend Breakout Pro",
        "description": "Trades breakout continuation in trend direction using EMA alignment, RSI strength, and volatility guard.",
    },
    "strat2": {
        "name": "Strategy 2 — Mean Reversion Bounce",
        "description": "Fades stretched moves back toward trend equilibrium using RSI extremes and EMA context.",
    },
    "strat3": {
        "name": "Strategy 3 — Momentum Pullback Sniper",
        "description": "Enters pullbacks inside strong trend after breakout confirmation and momentum re-acceleration.",
    },
}


def _inject_autorefresh(enabled: bool, seconds: int) -> None:
    if not enabled:
        return
    ms = max(5, int(seconds)) * 1000
    st.markdown(
        f"""
        <script>
            setTimeout(function() {{ window.location.reload(); }}, {ms});
        </script>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_twelvedata(pair: str, timeframe: str, outputsize: int = 5000) -> pd.DataFrame:
    if not TWELVEDATA_API_KEY:
        return pd.DataFrame()
    symbol = f"{pair[:3]}/{pair[3:]}"
    interval = TF_MAP[timeframe]["td"]
    # For daily, we can get much more data. Let's maximize it.
    # 5000 is the max for intraday, but for daily we can get more by omitting it.
    final_outputsize = 40000 if interval == "1day" else outputsize
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": final_outputsize,
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON",
        "timezone": "UTC",
    }
    try:
        response = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=30)
        response.raise_for_status()
        values = response.json().get("values", [])
        if not values:
            return pd.DataFrame()
        rows = [
            {
                "datetime": v["datetime"],
                "Open": float(v["open"]),
                "High": float(v["high"]),
                "Low": float(v["low"]),
                "Close": float(v["close"]),
                "Volume": float(v.get("volume", 0) or 0),
            }
            for v in values
        ]
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df.set_index("datetime").sort_index()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_yfinance(pair: str, timeframe: str) -> pd.DataFrame:
    ticker = f"{pair}=X"
    interval = TF_MAP[timeframe]["yf"]
    period_days = TF_MAP[timeframe]["period_days"]
    try:
        # Use "max" period for daily to get all available history
        if timeframe == "1d":
            hist = yf.download(ticker, period="max", interval=interval, auto_adjust=False, progress=False)
        else:
            # For intraday, fetch a much larger window to ensure we have enough data for backtests
            hist = yf.download(ticker, period=f"{period_days}d", interval=interval, auto_adjust=False, progress=False)
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
            df = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema_fast"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["ema_slow"] = data["Close"].ewm(span=50, adjust=False).mean()
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["rsi_14"] = 100 - (100 / (1 + rs))
    tr = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - data["Close"].shift(1)).abs(),
            (data["Low"] - data["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr_14"] = tr.ewm(span=14, adjust=False).mean()
    data["breakout_up"] = data["Close"] > data["High"].rolling(20).max().shift(1)
    data["breakout_dn"] = data["Close"] < data["Low"].rolling(20).min().shift(1)
    data["natr"] = data["atr_14"] / data["Close"]
    return data.dropna()


def _max_hold_bars(timeframe: str) -> int:
    return {"5m": 96, "15m": 64, "30m": 40, "4h": 14, "1d": 12}[timeframe]


def _build_htf_trend(pair: str, timeframe: str) -> pd.Series:
    htf = TF_MAP[timeframe]["htf"]
    htf_df, _ = load_ohlcv(pair, htf)
    if htf_df.empty:
        return pd.Series(dtype=float)
    htf_ind = add_indicators(htf_df)
    if htf_ind.empty:
        return pd.Series(dtype=float)
    trend = np.where(htf_ind["ema_fast"] > htf_ind["ema_slow"], 1, -1)
    return pd.Series(trend, index=htf_ind.index)


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
    data = add_indicators(df)
    if len(data) < 400:
        return pd.DataFrame(), [], {"error": "Need at least 400 candles for robust test."}

    pip_size = 0.01 if pair.endswith("JPY") else 0.0001
    spread = spread_pips * pip_size
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

    total_bars = len(data)
    for idx, (ts, row) in enumerate(data.iterrows(), start=1):
        if progress_cb and (idx == 1 or idx % 10 == 0 or idx == total_bars):
            progress_cb(idx, total_bars, ts)

        close_price = float(row["Close"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        rsi_14 = float(row["rsi_14"])
        breakout_up = bool(row["breakout_up"])
        breakout_dn = bool(row["breakout_dn"])
        natr = float(row["natr"])

        if strategy_key == "strat2":
            long_signal = (ema_fast >= ema_slow) and (rsi_14 <= 35) and (0.0004 <= natr <= 0.03)
            short_signal = (ema_fast <= ema_slow) and (rsi_14 >= 65) and (0.0004 <= natr <= 0.03)
        elif strategy_key == "strat3":
            long_signal = (ema_fast > ema_slow) and (45 <= rsi_14 <= 58) and breakout_up and (0.0003 <= natr <= 0.02)
            short_signal = (ema_fast < ema_slow) and (42 <= rsi_14 <= 55) and breakout_dn and (0.0003 <= natr <= 0.02)
        else:
            long_signal = (ema_fast > ema_slow) and (52 <= rsi_14 <= 72) and breakout_up and (0.0004 <= natr <= 0.02)
            short_signal = (ema_fast < ema_slow) and (28 <= rsi_14 <= 48) and breakout_dn and (0.0004 <= natr <= 0.02)

        if use_mtf_filter and not htf_trend.empty:
            trend_slice = htf_trend[htf_trend.index <= ts]
            if not trend_slice.empty:
                trend = int(trend_slice.iloc[-1])
                if trend != 1:
                    long_signal = False
                if trend != -1:
                    short_signal = False

        if position == 0:
            if long_signal or short_signal:
                position = 1 if long_signal else -1
                entry_price = close_price + spread / 2 if position == 1 else close_price - spread / 2
                entry_atr = float(row["atr_14"])
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

                trades.append(
                    Trade(
                        side="BUY" if position == 1 else "SELL",
                        entry_time=entry_ts,
                        exit_time=ts,
                        entry=entry_price,
                        exit=exit_exec,
                        stop=stop,
                        target=target,
                        pnl_pct=pnl_pct,
                        pnl_usd=pnl_usd,
                        r_multiple=r_multiple,
                        bars=bars_held,
                        reason=exit_reason,
                    )
                )

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

    eq = pd.DataFrame(equity_curve, columns=["time", "equity"]).set_index("time")
    returns = eq["equity"].pct_change().dropna()
    bars_per_year = (365 * 24 * 60) / TF_MAP[timeframe]["bar_minutes"]
    sharpe = (returns.mean() / returns.std() * math.sqrt(bars_per_year)) if returns.std() > 0 else 0.0

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
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


def build_trade_overlay_chart(df_ist: pd.DataFrame, trades: list[Trade], pair: str, timeframe: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df_ist.index,
            open=df_ist["Open"],
            high=df_ist["High"],
            low=df_ist["Low"],
            close=df_ist["Close"],
            name=f"{pair} {timeframe}",
        )
    )

    entry_x = []
    entry_y = []
    exit_x = []
    exit_y = []

    for trade in trades:
        start = trade.entry_time.tz_convert(IST)
        end = trade.exit_time.tz_convert(IST)
        profit_color = "rgba(16, 185, 129, 0.17)" if trade.pnl_usd >= 0 else "rgba(239, 68, 68, 0.17)"
        risk_color = "rgba(239, 68, 68, 0.14)"
        reward_color = "rgba(16, 185, 129, 0.14)"

        # Risk zone
        y0_risk = min(trade.entry, trade.stop)
        y1_risk = max(trade.entry, trade.stop)
        fig.add_shape(type="rect", x0=start, x1=end, y0=y0_risk, y1=y1_risk, fillcolor=risk_color, line=dict(width=0), layer="below")
        # Reward zone
        y0_reward = min(trade.entry, trade.target)
        y1_reward = max(trade.entry, trade.target)
        fig.add_shape(type="rect", x0=start, x1=end, y0=y0_reward, y1=y1_reward, fillcolor=reward_color, line=dict(width=0), layer="below")
        # Net result zone
        y0_net = min(trade.entry, trade.exit)
        y1_net = max(trade.entry, trade.exit)
        fig.add_shape(type="rect", x0=start, x1=end, y0=y0_net, y1=y1_net, fillcolor=profit_color, line=dict(width=0), layer="below")

        fig.add_annotation(
            x=end,
            y=trade.exit,
            text=f"{trade.pnl_pct:+.2f}% | {trade.r_multiple:+.2f}R",
            showarrow=True,
            arrowhead=2,
            font=dict(size=10, color="#111827"),
            bgcolor="rgba(255,255,255,0.85)",
        )

        entry_x.append(start)
        entry_y.append(trade.entry)
        exit_x.append(end)
        exit_y.append(trade.exit)

    if entry_x:
        fig.add_trace(go.Scatter(x=entry_x, y=entry_y, mode="markers", marker=dict(symbol="triangle-up", size=10, color="#2563eb"), name="Entries"))
    if exit_x:
        fig.add_trace(go.Scatter(x=exit_x, y=exit_y, mode="markers", marker=dict(symbol="x", size=9, color="#111827"), name="Exits"))

    fig.update_layout(height=680, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h"))
    return fig


def _last_signal(df: pd.DataFrame) -> str:
    d = add_indicators(df)
    if d.empty:
        return "NO_DATA"
    r = d.iloc[-1]
    long_signal = (r["ema_fast"] > r["ema_slow"]) and (52 <= r["rsi_14"] <= 72) and bool(r["breakout_up"]) and (0.0004 <= r["natr"] <= 0.02)
    short_signal = (r["ema_fast"] < r["ema_slow"]) and (28 <= r["rsi_14"] <= 48) and bool(r["breakout_dn"]) and (0.0004 <= r["natr"] <= 0.02)
    if long_signal:
        return "BUY"
    if short_signal:
        return "SELL"
    return "HOLD"


def app() -> None:
    st.set_page_config(page_title="ForexBot Pro", layout="wide")
    st.title("ForexBot Pro — TradingView-style Backtest + Live FX Dashboard (IST)")
    st.caption("Live market data + robust multi-timeframe backtest (5m, 15m, 30m, 4h, 1d), with entry/exit markers and RR zones.")

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
            format_func=lambda key: STRATEGY_PROFILES[key]["name"],
        )
        st.caption(STRATEGY_PROFILES[strategy_key]["description"])
        run_backtest_btn = st.button("Run robust backtest", use_container_width=True)
        auto_rank_btn = st.button("Auto-Rank & Run Best", use_container_width=True)
        reset_backtest_btn = st.button("Reset backtest data", use_container_width=True)

    _inject_autorefresh(auto_refresh, refresh_sec)

    df, source = load_ohlcv(pair, timeframe)
    if df.empty:
        st.error("No live data from TwelveData/yfinance right now.")
        st.stop()

    last_utc = df.index.max()
    last_ist = last_utc.astimezone(IST)
    now_ist = pd.Timestamp.now(tz=IST)
    st.caption(f"Source: {source} | Last candle UTC: {last_utc:%Y-%m-%d %H:%M:%S} | Last candle IST: {last_ist:%Y-%m-%d %H:%M:%S} | Now IST: {now_ist:%Y-%m-%d %H:%M:%S}")

    tabs = st.tabs(["Backtest Chart", "Realtime Dashboard"])

    with tabs[0]:
        if bt_start_date > bt_end_date:
            st.error("Backtest start date must be before end date.")
            st.stop()

        bt_start_utc = pd.Timestamp(bt_start_date).tz_localize(UTC)
        bt_end_utc_exclusive = (pd.Timestamp(bt_end_date) + pd.Timedelta(days=1)).tz_localize(UTC)
        dated_df = df[(df.index >= bt_start_utc) & (df.index < bt_end_utc_exclusive)].copy()

        if dated_df.empty:
            st.warning("No candles available in selected range for this timeframe/source.")
            st.stop()

        replay_cap = min(2500, len(dated_df))
        replay_default = min(1200, len(dated_df))
        replay_bars = st.slider("Replay candles (last N bars from selected date range)", min_value=min(200, replay_cap), max_value=replay_cap, value=max(min(200, replay_default), replay_default), step=50 if replay_cap >= 250 else 10)
        replay_df = dated_df.tail(replay_bars)
        replay_df_ist = to_ist(replay_df)
        st.caption(f"Backtest window (IST display): {bt_start_date} → {bt_end_date} | Candles in window: {len(dated_df)} | Using last {len(replay_df)} candles for replay")

        if reset_backtest_btn:
            st.session_state.pop("last_bt", None)
            st.session_state.pop("last_bt_meta", None)
            st.cache_data.clear()
            st.success("Backtest state reset. Fresh data will be used for next run, and live trading view is now clean.")

        current_meta = {
            "pair": pair,
            "timeframe": timeframe,
            "replay_bars": replay_bars,
            "bt_start": str(bt_start_date),
            "bt_end": str(bt_end_date),
            "sl_atr": float(sl_atr),
            "tp_atr": float(tp_atr),
            "risk_pct": float(risk_pct),
            "spread_pips": float(spread_pips),
            "use_mtf_filter": bool(use_mtf_filter),
            "strategy_key": strategy_key,
        }

        if run_backtest_btn or auto_rank_btn:
            progress_placeholder = st.empty()
            progress_bar = st.progress(0, text="Starting backtest...")
            started_at = pd.Timestamp.now(tz=UTC)

            def _on_progress(done: int, total: int, ts_point: pd.Timestamp) -> None:
                ratio = done / total if total > 0 else 0.0
                pct = int(ratio * 100)
                elapsed = (pd.Timestamp.now(tz=UTC) - started_at).total_seconds()
                eta_sec = int((elapsed / ratio) - elapsed) if ratio > 0 and elapsed > 0 else 0
                current_ist = ts_point.tz_convert(IST).strftime("%Y-%m-%d %H:%M")
                progress_bar.progress(
                    min(100, max(0, pct)),
                    text=f"Backtest progress: {pct}% | {done}/{total} bars | ETA: ~{eta_sec}s | At candle: {current_ist} IST",
                )

            final_strategy = strategy_key
            if auto_rank_btn:
                st.info(f"Auto-ranking strategies for {pair} on {timeframe}...")
                rank_results = []
                for i, key in enumerate(STRATEGY_PROFILES.keys()):
                    st.progress((i + 1) / len(STRATEGY_PROFILES), text=f"Testing '{STRATEGY_PROFILES[key]['name']}'...")
                    _, _, stats = run_backtest(
                        pair=pair,
                        timeframe=timeframe,
                        df=replay_df,
                        sl_atr=sl_atr,
                        tp_atr=tp_atr,
                        risk_pct=risk_pct,
                        spread_pips=spread_pips,
                        use_mtf_filter=use_mtf_filter,
                        strategy_key=key,
                    )
                    score = (stats.get("sharpe", -99) * stats.get("profit_factor", 0)) if "error" not in stats else -999
                    rank_results.append({"key": key, "score": score, "stats": stats})

                best_strat = max(rank_results, key=lambda x: x["score"])
                final_strategy = best_strat["key"]
                st.success(f"Best strategy found: **{STRATEGY_PROFILES[final_strategy]['name']}** (Score: {best_strat['score']:.2f})")
                st.caption("Now running full backtest with the winning strategy...")

            eq, trades, stats = run_backtest(
                pair=pair,
                timeframe=timeframe,
                df=replay_df,
                sl_atr=sl_atr,
                tp_atr=tp_atr,
                risk_pct=risk_pct,
                spread_pips=spread_pips,
                use_mtf_filter=use_mtf_filter,
                strategy_key=final_strategy,
                progress_cb=_on_progress,
            )
            progress_bar.progress(100, text="Backtest complete: 100%")
            progress_placeholder.info("Run complete. Results below are from this explicit run only.")
            st.session_state["last_bt"] = (eq, trades, stats)
            st.session_state["last_bt_meta"] = current_meta
            st.session_state["last_bt_meta"]["strategy_key"] = final_strategy  # Update meta with the one that ran

        if "last_bt" not in st.session_state:
            st.info("No backtest results yet. Configure params and click 'Run robust backtest'.")
            st.stop()

        eq, trades, stats = st.session_state["last_bt"]
        last_meta = st.session_state.get("last_bt_meta", {})
        if last_meta and last_meta != current_meta:
            st.warning("Parameters changed after last run. Click 'Run robust backtest' to refresh results.")

        if "error" in stats:
            st.warning(stats["error"])
        else:
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            m1.metric("Return", f"{stats['total_return_pct']:+.2f}%")
            m2.metric("Trades", f"{stats['trades']}")
            m3.metric("Win rate", f"{stats['win_rate_pct']:.1f}%")
            m4.metric("Profit factor", "inf" if math.isinf(stats["profit_factor"]) else f"{stats['profit_factor']:.2f}")
            m5.metric("Max DD", f"{stats['max_drawdown_pct']:.2f}%")
            m6.metric("Sharpe", f"{stats['sharpe']:.2f}")
            m7.metric("Avg R", f"{stats['avg_r']:+.2f}")

            st.plotly_chart(build_trade_overlay_chart(replay_df_ist, trades, pair, timeframe), use_container_width=True)

            if not eq.empty:
                eq_ist = eq.copy()
                eq_ist.index = eq_ist.index.tz_convert(IST)
                st.line_chart(eq_ist["equity"], use_container_width=True)

            if trades:
                trades_df = pd.DataFrame([
                    {
                        "side": t.side,
                        "entry_IST": t.entry_time.tz_convert(IST).strftime("%Y-%m-%d %H:%M"),
                        "exit_IST": t.exit_time.tz_convert(IST).strftime("%Y-%m-%d %H:%M"),
                        "entry": round(t.entry, 5),
                        "stop": round(t.stop, 5),
                        "target": round(t.target, 5),
                        "exit": round(t.exit, 5),
                        "R": round(t.r_multiple, 2),
                        "pnl_$": round(t.pnl_usd, 2),
                        "pnl_%": round(t.pnl_pct, 2),
                        "bars": t.bars,
                        "reason": t.reason,
                    }
                    for t in trades
                ])
                st.dataframe(trades_df.tail(300), use_container_width=True)

    with tabs[1]:
        st.subheader("Realtime Pair Signals (IST)")
        rows = []
        for p in PAIRS:
            d, src = load_ohlcv(p, timeframe)
            if d.empty:
                rows.append({"pair": p, "signal": "NO_DATA", "close": None, "last_ist": None, "source": src})
                continue
            sig = _last_signal(d)
            last_close = float(d["Close"].iloc[-1])
            last_t = d.index.max().tz_convert(IST)
            rows.append(
                {
                    "pair": p,
                    "signal": sig,
                    "close": round(last_close, 5),
                    "last_ist": last_t.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": src,
                }
            )

        table = pd.DataFrame(rows)
        st.dataframe(table, use_container_width=True)

        live_df_ist = to_ist(df.tail(350))
        st.plotly_chart(build_trade_overlay_chart(live_df_ist, [], pair, timeframe), use_container_width=True)


if __name__ == "__main__":
    app()
