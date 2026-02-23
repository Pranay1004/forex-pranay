"""
data/feed.py — Robust multi-source OHLCV data provider.

Data sources (tried in priority order):
  1. TwelveData API  — up to 30+ years of D1/W1; good H1 depth
  2. FMP API         — D1 history back to 1990s for major pairs
  3. yfinance        — fallback (H1 limited to ~730 days, D1 ~20 years)

All data is cached locally as Parquet. The pipeline validates, deduplicates,
and merges data across sources so the bot always gets the deepest available
history in a single clean DataFrame.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from forexbot.config import (
    CACHE_DIR,
    DATA_YEARS,
    DATA_YEARS_D1,
    DATA_YEARS_H1,
    DATA_YEARS_W1,
    D1_INTERVAL,
    FMP_API_KEY,
    H1_INTERVAL,
    PAIRS,
    TIMEFRAMES,
    TWELVEDATA_API_KEY,
    W1_INTERVAL,
    YFINANCE_SUFFIX,
)

logger = logging.getLogger(__name__)

# ─── Cache TTLs ──────────────────────────────────────────────────────────────
_CACHE_TTL = {
    H1_INTERVAL: 3600,           # 1 hour
    D1_INTERVAL: 86400,          # 1 day
    W1_INTERVAL: 86400 * 7,      # 1 week
}

_YEARS_BY_INTERVAL = {
    H1_INTERVAL: DATA_YEARS_H1,
    D1_INTERVAL: DATA_YEARS_D1,
    W1_INTERVAL: DATA_YEARS_W1,
}

# ─── TwelveData config ───────────────────────────────────────────────────────
_TD_INTERVAL_MAP = {
    H1_INTERVAL: "1h",
    D1_INTERVAL: "1day",
    W1_INTERVAL: "1week",
}
_TD_MAX_ROWS = 5000
_TD_CREDITS_PER_MINUTE = 8          # free tier limit
_TD_LAST_REQUEST_TS: list[float] = []  # rolling window of request timestamps

# ─── FMP config ──────────────────────────────────────────────────────────────
_FMP_BASE = "https://financialmodelingprep.com/stable"


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_path(pair: str, interval: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{pair}_{interval}.parquet"


def _is_stale(path: Path, ttl: int) -> bool:
    if not path.exists():
        return True
    return (time.time() - path.stat().st_mtime) > ttl


def _validate_ohlcv(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Ensure OHLCV integrity: correct columns, no dups, sorted, no null Close."""
    if df.empty:
        return df
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("%s: Missing columns %s", label, missing)
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["Close"], inplace=True)
    df = df[df["Close"] > 0]
    return df


def _merge_dataframes(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    """Merge two OHLCV DataFrames, preferring primary where overlapping."""
    if primary.empty:
        return secondary
    if secondary.empty:
        return primary
    combined = pd.concat([secondary, primary])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    return combined


def _td_rate_limit():
    """Wait if necessary to respect TwelveData free-tier rate limit (8 req/min)."""
    now = time.time()
    # Purge timestamps older than 60s
    while _TD_LAST_REQUEST_TS and (now - _TD_LAST_REQUEST_TS[0]) > 60:
        _TD_LAST_REQUEST_TS.pop(0)
    if len(_TD_LAST_REQUEST_TS) >= _TD_CREDITS_PER_MINUTE:
        wait_until = _TD_LAST_REQUEST_TS[0] + 61
        sleep_sec = max(0, wait_until - now)
        if sleep_sec > 0:
            logger.info("TwelveData: rate limit — waiting %.0fs", sleep_sec)
            time.sleep(sleep_sec)
    _TD_LAST_REQUEST_TS.append(time.time())


# ═══════════════════════════════════════════════════════════════════════════════
#  SOURCE 1: TwelveData API
# ═══════════════════════════════════════════════════════════════════════════════

def _twelvedata_download(pair: str, interval: str, years: int) -> pd.DataFrame:
    """Download OHLCV from TwelveData REST API."""
    if not TWELVEDATA_API_KEY:
        return pd.DataFrame()

    td_interval = _TD_INTERVAL_MAP.get(interval)
    if td_interval is None:
        return pd.DataFrame()

    symbol = f"{pair[:3]}/{pair[3:]}"
    end_date = datetime.now(tz=timezone.utc)
    start_date = end_date - timedelta(days=365 * years)

    all_chunks: list[pd.DataFrame] = []
    cursor_end = end_date

    for _attempt in range(40):  # safety limit for pagination + retries
        _td_rate_limit()  # respect free-tier rate limit

        params = {
            "symbol": symbol,
            "interval": td_interval,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": cursor_end.strftime("%Y-%m-%d %H:%M:%S"),
            "outputsize": _TD_MAX_ROWS,
            "apikey": TWELVEDATA_API_KEY,
            "format": "JSON",
            "timezone": "UTC",
        }

        try:
            resp = requests.get(
                "https://api.twelvedata.com/time_series",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if "values" not in data:
                msg = data.get("message", "no message")
                if "exceeded" in msg.lower() or "limit" in msg.lower() or "run out" in msg.lower():
                    # Rate limited — wait 65s and retry this iteration
                    logger.info("TwelveData rate limited for %s — waiting 65s to retry…", pair)
                    time.sleep(65)
                    _TD_LAST_REQUEST_TS.clear()
                    continue
                else:
                    logger.debug("TwelveData no values for %s @ %s: %s",
                                 pair, interval, msg)
                break

            values = data["values"]
            if not values:
                break

            rows = []
            for v in values:
                rows.append({
                    "datetime": v["datetime"],
                    "Open": float(v["open"]),
                    "High": float(v["high"]),
                    "Low": float(v["low"]),
                    "Close": float(v["close"]),
                    "Volume": int(float(v.get("volume", 0))),
                })

            chunk = pd.DataFrame(rows)
            chunk["datetime"] = pd.to_datetime(chunk["datetime"], utc=True)
            chunk.set_index("datetime", inplace=True)
            chunk.sort_index(inplace=True)
            all_chunks.append(chunk)

            if len(values) < _TD_MAX_ROWS:
                break

            earliest = chunk.index.min()
            if earliest <= start_date:
                break
            cursor_end = earliest - timedelta(seconds=1)

        except requests.exceptions.RequestException as exc:
            logger.warning("TwelveData request error [%s %s]: %s", pair, interval, exc)
            break
        except Exception as exc:
            logger.warning("TwelveData parse error [%s %s]: %s", pair, interval, exc)
            break

    if not all_chunks:
        return pd.DataFrame()

    result = pd.concat(all_chunks)
    result = _validate_ohlcv(result, f"TwelveData/{pair}/{interval}")
    if not result.empty:
        logger.info("TwelveData: %s @ %s → %d bars (%s to %s)",
                    pair, interval, len(result),
                    result.index.min().strftime("%Y-%m-%d"),
                    result.index.max().strftime("%Y-%m-%d"))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  SOURCE 2: Financial Modeling Prep (FMP) — D1 / W1 only
# ═══════════════════════════════════════════════════════════════════════════════

def _fmp_download(pair: str, interval: str, years: int) -> pd.DataFrame:
    """Download D1 OHLCV from FMP. Only supports daily data for forex."""
    if not FMP_API_KEY:
        return pd.DataFrame()

    if interval not in (D1_INTERVAL, W1_INTERVAL):
        return pd.DataFrame()

    end_date = datetime.now(tz=timezone.utc)
    start_date = end_date - timedelta(days=365 * years)

    try:
        url = f"{_FMP_BASE}/historical-price-eod/full"
        params = {
            "symbol": pair,
            "apikey": FMP_API_KEY,
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and "historical" in data:
            prices = data["historical"]
        elif isinstance(data, list):
            prices = data
        else:
            logger.debug("FMP unexpected response for %s: %s", pair, type(data))
            return pd.DataFrame()

        if not prices:
            return pd.DataFrame()

        rows = []
        for p in prices:
            try:
                dt = pd.to_datetime(p.get("date", p.get("Date", "")))
                rows.append({
                    "datetime": dt,
                    "Open": float(p.get("open", p.get("Open", 0))),
                    "High": float(p.get("high", p.get("High", 0))),
                    "Low": float(p.get("low", p.get("Low", 0))),
                    "Close": float(p.get("close", p.get("Close", 0))),
                    "Volume": int(float(p.get("volume", p.get("Volume", 0)))),
                })
            except (ValueError, TypeError):
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        df = _validate_ohlcv(df, f"FMP/{pair}/{interval}")

        # If weekly is needed, resample from daily
        if interval == W1_INTERVAL and not df.empty:
            df = df.resample("W-FRI").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna(subset=["Close"])

        if not df.empty:
            logger.info("FMP: %s @ %s → %d bars (%s to %s)",
                        pair, interval, len(df),
                        df.index.min().strftime("%Y-%m-%d"),
                        df.index.max().strftime("%Y-%m-%d"))
        return df

    except requests.exceptions.RequestException as exc:
        logger.warning("FMP request error [%s %s]: %s", pair, interval, exc)
        return pd.DataFrame()
    except Exception as exc:
        logger.warning("FMP parse error [%s %s]: %s", pair, interval, exc)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  SOURCE 3: yfinance — Fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _yfinance_download(pair: str, interval: str, years: int) -> pd.DataFrame:
    """Download OHLCV from yfinance (fallback)."""
    ticker = f"{pair}{YFINANCE_SUFFIX}"
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=365 * years)

    if interval == H1_INTERVAL:
        chunks: list[pd.DataFrame] = []
        chunk_days = 700
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=chunk_days), end)
            try:
                df = yf.download(
                    ticker,
                    start=cursor.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
                if not df.empty:
                    chunks.append(df)
            except Exception as exc:
                logger.warning("yfinance chunk error [%s %s]: %s", pair, interval, exc)
            cursor = chunk_end + timedelta(days=1)
        raw = pd.concat(chunks) if chunks else pd.DataFrame()
    else:
        try:
            raw = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            logger.error("yfinance download error [%s %s]: %s", pair, interval, exc)
            raw = pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = _validate_ohlcv(raw, f"yfinance/{pair}/{interval}")
    if not raw.empty:
        logger.info("yfinance: %s @ %s → %d bars (%s to %s)",
                    pair, interval, len(raw),
                    raw.index.min().strftime("%Y-%m-%d"),
                    raw.index.max().strftime("%Y-%m-%d"))
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIFIED LOADER — tries sources in order, merges, caches
# ═══════════════════════════════════════════════════════════════════════════════

def load_ohlcv(
    pair: str,
    interval: str = H1_INTERVAL,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV data for a pair/interval, using multi-source pipeline.

    Priority:
      1. TwelveData (deepest history for D1/W1, also good for H1)
      2. FMP (D1/W1 backup, very deep history)
      3. yfinance (always available but limited depth)

    Results are merged and cached as Parquet.
    """
    years = _YEARS_BY_INTERVAL.get(interval, DATA_YEARS)
    ttl = _CACHE_TTL.get(interval, 86400)
    path = _cache_path(pair, interval)

    # Check cache first
    cached: Optional[pd.DataFrame] = None
    if use_cache and not _is_stale(path, ttl):
        try:
            cached = pd.read_parquet(path)
            if not cached.empty:
                logger.debug("Cache hit: %s @ %s (%d bars)", pair, interval, len(cached))
                return cached
        except Exception as exc:
            logger.warning("Cache read failed for %s @ %s: %s", pair, interval, exc)

    # ── Download from multiple sources and merge ──────────────────────────────
    merged = pd.DataFrame()

    # Source 1: TwelveData
    try:
        td_data = _twelvedata_download(pair, interval, years)
        if not td_data.empty:
            merged = _merge_dataframes(merged, td_data)
    except Exception as exc:
        logger.warning("TwelveData source failed for %s @ %s: %s", pair, interval, exc)

    # Source 2: FMP (D1/W1 only)
    if interval in (D1_INTERVAL, W1_INTERVAL):
        try:
            fmp_data = _fmp_download(pair, interval, years)
            if not fmp_data.empty:
                merged = _merge_dataframes(merged, fmp_data)
        except Exception as exc:
            logger.warning("FMP source failed for %s @ %s: %s", pair, interval, exc)

    # Source 3: yfinance (always try — good for recent data)
    try:
        yf_data = _yfinance_download(pair, interval, years)
        if not yf_data.empty:
            merged = _merge_dataframes(merged, yf_data)
    except Exception as exc:
        logger.warning("yfinance source failed for %s @ %s: %s", pair, interval, exc)

    # Final validation
    if not merged.empty:
        merged = _validate_ohlcv(merged, f"merged/{pair}/{interval}")

    # Fall back to stale cache if all sources failed
    if merged.empty and cached is not None and not cached.empty:
        logger.warning("All sources failed for %s @ %s; using stale cache (%d bars)",
                       pair, interval, len(cached))
        return cached

    # Save to cache
    if not merged.empty and use_cache:
        try:
            merged.to_parquet(path)
        except Exception as exc:
            logger.warning("Cache write failed for %s @ %s: %s", pair, interval, exc)

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API (unchanged signatures for backwards compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def refresh_all_pairs(
    intervals: Optional[list[str]] = None,
    use_cache: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Download/refresh data for all pairs and all timeframes."""
    if intervals is None:
        intervals = TIMEFRAMES
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for pair in PAIRS:
        result[pair] = {}
        for interval in intervals:
            df = load_ohlcv(pair, interval, use_cache=use_cache)
            result[pair][interval] = df
            if not df.empty:
                span_years = (df.index.max() - df.index.min()).days / 365.25
                logger.info(
                    "Loaded %s @ %s: %d bars (%.1f years) — %.4f to %.4f",
                    pair, interval, len(df), span_years,
                    df["Close"].iloc[0], df["Close"].iloc[-1],
                )
            else:
                logger.warning("No data for %s @ %s", pair, interval)
    return result


def get_latest_close(pair_data: dict[str, pd.DataFrame]) -> float:
    df = pair_data.get(H1_INTERVAL, pd.DataFrame())
    if df.empty:
        return 0.0
    return float(df["Close"].iloc[-1])


def seconds_to_next_candle(interval: str = H1_INTERVAL) -> int:
    now = datetime.now(tz=timezone.utc)
    if interval == H1_INTERVAL:
        next_candle = now.replace(minute=0, second=5, microsecond=0) + timedelta(hours=1)
    else:
        next_candle = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=5, microsecond=0
        )
    return max(int((next_candle - now).total_seconds()), 0)


def resample_to_h4(df_h1: pd.DataFrame) -> pd.DataFrame:
    """Resample H1 data to H4."""
    if df_h1.empty:
        return pd.DataFrame()
    h4 = df_h1.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna(subset=["Close"])
    return h4


def get_mtf_bias(
    ohlcv_data: dict[str, pd.DataFrame],
    pair: str,
) -> dict[str, int]:
    """Compute directional bias across all available timeframes."""
    biases: dict[str, int] = {}

    def _compute_bias(df: pd.DataFrame, sma_fast: int = 20, sma_slow: int = 50) -> int:
        if df is None or len(df) < sma_slow + 10:
            return 0
        close = df["Close"]
        sma_f = close.rolling(sma_fast).mean()
        sma_s = close.rolling(sma_slow).mean()
        roc = close.pct_change(10)

        last_c = float(close.iloc[-1])
        last_sf = float(sma_f.iloc[-1])
        last_ss = float(sma_s.iloc[-1])
        last_roc = float(roc.iloc[-1]) if not pd.isna(roc.iloc[-1]) else 0.0

        if last_c > last_sf > last_ss and last_roc > 0:
            return 1
        elif last_c < last_sf < last_ss and last_roc < 0:
            return -1
        return 0

    df_h1 = ohlcv_data.get(H1_INTERVAL)
    biases["H1"] = _compute_bias(df_h1, 20, 50)

    if df_h1 is not None and len(df_h1) > 100:
        df_h4 = resample_to_h4(df_h1)
        biases["H4"] = _compute_bias(df_h4, 20, 50)
    else:
        biases["H4"] = 0

    df_d1 = ohlcv_data.get(D1_INTERVAL)
    biases["D1"] = _compute_bias(df_d1, 20, 50)

    df_w1 = ohlcv_data.get(W1_INTERVAL)
    biases["W1"] = _compute_bias(df_w1, 10, 30)

    return biases


def mtf_confirms_signal(
    biases: dict[str, int],
    signal: int,
    min_agree: int = 3,
) -> bool:
    """Check if enough timeframes agree with the proposed signal direction."""
    if signal == 0:
        return True
    agree_count = sum(1 for b in biases.values() if b == signal)
    return agree_count >= min_agree
