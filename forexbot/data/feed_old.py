"""
data/feed.py — OHLCV data provider using yfinance with local Parquet cache.
Downloads H1 (2yr), D1 (30yr), W1 (30yr) data for the top 5 FOREX pairs.
Refreshes stale data automatically. Supports H4 resampling from H1.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from forexbot.config import (
    CACHE_DIR, DATA_YEARS, DATA_YEARS_H1, DATA_YEARS_D1, DATA_YEARS_W1,
    D1_INTERVAL, H1_INTERVAL, W1_INTERVAL, TIMEFRAMES, PAIRS, YFINANCE_SUFFIX
)

logger = logging.getLogger(__name__)

# Parquet cache TTL: re-download if older than this (seconds)
_CACHE_TTL_H1 = 3600       # 1 hour
_CACHE_TTL_D1 = 86400      # 1 day
_CACHE_TTL_W1 = 86400 * 7  # 1 week

# Data depth per interval
_YEARS_BY_INTERVAL: dict[str, int] = {
    H1_INTERVAL: DATA_YEARS_H1,
    D1_INTERVAL: DATA_YEARS_D1,
    W1_INTERVAL: DATA_YEARS_W1,
}


def _cache_path(pair: str, interval: str) -> Path:
    """Return the Parquet cache file path for a given pair and interval."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{pair}_{interval}.parquet"


def _is_stale(path: Path, ttl: int) -> bool:
    """Return True if the cache file does not exist or is older than ttl seconds."""
    if not path.exists():
        return True
    age = time.time() - path.stat().st_mtime
    return age > ttl


def _ticker_symbol(pair: str) -> str:
    """Convert pair name to yfinance ticker, e.g. EURUSD → EURUSD=X."""
    return f"{pair}{YFINANCE_SUFFIX}"


def _download_pair(pair: str, interval: str, years: int = None) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance for a single pair.

    Args:
        pair: Currency pair string e.g. "EURUSD".
        interval: yfinance interval string e.g. "1h", "1d", or "1wk".
        years: Number of years of history to fetch. Auto-detected per interval if None.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] and DatetimeIndex (UTC).
    """
    if years is None:
        years = _YEARS_BY_INTERVAL.get(interval, DATA_YEARS)

    ticker = _ticker_symbol(pair)
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=365 * years)

    # yfinance H1 is limited to ~730 days; chunk for longer periods
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
        logger.warning("No data returned for %s @ %s", pair, interval)
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    df.dropna(subset=["Close"], inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def load_ohlcv(
    pair: str,
    interval: str = H1_INTERVAL,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV data for a pair, using the Parquet cache when available and fresh.

    Args:
        pair: Currency pair e.g. "EURUSD".
        interval: Data interval ("1h" or "1d").
        use_cache: Whether to use the local Parquet cache.

    Returns:
        DataFrame with OHLCV data indexed by UTC datetime.
    """
    ttl = _CACHE_TTL_H1 if interval == H1_INTERVAL else (
        _CACHE_TTL_W1 if interval == W1_INTERVAL else _CACHE_TTL_D1
    )
    path = _cache_path(pair, interval)

    cached: Optional[pd.DataFrame] = None
    if use_cache and not _is_stale(path, ttl):
        try:
            cached = pd.read_parquet(path)
            logger.debug("Cache hit: %s @ %s (%d rows)", pair, interval, len(cached))
            return cached
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", pair, exc)

    logger.info("Downloading %s @ %s from yfinance …", pair, interval)
    fresh = _download_pair(pair, interval)

    if fresh.empty and cached is not None:
        logger.warning("Download failed; returning stale cache for %s", pair)
        return cached

    if not fresh.empty and use_cache:
        try:
            fresh.to_parquet(path)
            logger.debug("Cached %s @ %s → %s", pair, interval, path)
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", pair, exc)

    return fresh if not fresh.empty else (cached if cached is not None else pd.DataFrame())


def refresh_all_pairs(
    intervals: Optional[list[str]] = None,
    use_cache: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Download / refresh OHLCV data for all pairs and all configured timeframes.
    H1 (2yr), D1 (30yr), W1 (30yr).

    Args:
        intervals: List of interval strings. Defaults to TIMEFRAMES [H1, D1, W1].
        use_cache: Forward to load_ohlcv.

    Returns:
        Nested dict: {pair: {interval: DataFrame}}.
    """
    if intervals is None:
        intervals = TIMEFRAMES

    result: dict[str, dict[str, pd.DataFrame]] = {}
    for pair in PAIRS:
        result[pair] = {}
        for interval in intervals:
            df = load_ohlcv(pair, interval, use_cache=use_cache)
            result[pair][interval] = df
            logger.info(
                "Loaded %s @ %s: %d bars (%.4f–%.4f)",
                pair, interval, len(df),
                df["Close"].iloc[0] if not df.empty else 0,
                df["Close"].iloc[-1] if not df.empty else 0,
            )
    return result


def get_latest_close(pair_data: dict[str, pd.DataFrame]) -> float:
    """Return the most recent close price from the H1 feed."""
    df = pair_data.get(H1_INTERVAL, pd.DataFrame())
    if df.empty:
        return 0.0
    return float(df["Close"].iloc[-1])


def seconds_to_next_candle(interval: str = H1_INTERVAL) -> int:
    """
    Compute seconds remaining until the next full candle close for the given interval.

    Args:
        interval: "1h" or "1d".

    Returns:
        Seconds as an integer.
    """
    now = datetime.now(tz=timezone.utc)
    if interval == H1_INTERVAL:
        next_candle = now.replace(minute=0, second=5, microsecond=0) + timedelta(hours=1)
    else:  # D1
        next_candle = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=5, microsecond=0
        )
    delta = int((next_candle - now).total_seconds())
    return max(delta, 0)


def resample_to_h4(df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Resample H1 OHLCV data to H4 (4-hour) bars for multi-timeframe analysis.

    Args:
        df_h1: DataFrame with H1 OHLCV data.

    Returns:
        DataFrame with H4 OHLCV data.
    """
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
    """
    Compute directional bias from each available timeframe using
    price vs SMA crossovers + momentum.

    Returns dict {timeframe_label: direction} where:
      1 = bullish, -1 = bearish, 0 = neutral

    Timeframes checked: H1, H4 (resampled), D1, W1
    """
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

        # Bullish: price > fast SMA > slow SMA AND positive momentum
        if last_c > last_sf > last_ss and last_roc > 0:
            return 1
        # Bearish: price < fast SMA < slow SMA AND negative momentum
        elif last_c < last_sf < last_ss and last_roc < 0:
            return -1
        return 0

    # H1
    df_h1 = ohlcv_data.get(H1_INTERVAL)
    biases["H1"] = _compute_bias(df_h1, 20, 50)

    # H4 (resampled from H1)
    if df_h1 is not None and len(df_h1) > 100:
        df_h4 = resample_to_h4(df_h1)
        biases["H4"] = _compute_bias(df_h4, 20, 50)
    else:
        biases["H4"] = 0

    # D1
    df_d1 = ohlcv_data.get(D1_INTERVAL)
    biases["D1"] = _compute_bias(df_d1, 20, 50)

    # W1
    df_w1 = ohlcv_data.get(W1_INTERVAL)
    biases["W1"] = _compute_bias(df_w1, 10, 30)  # shorter SMAs for weekly

    return biases


def mtf_confirms_signal(
    biases: dict[str, int],
    signal: int,
    min_agree: int = 3,
) -> bool:
    """
    Check whether enough timeframes agree with the proposed trade direction.

    For BUY (signal=1): count TFs with bias == 1
    For SELL (signal=-1): count TFs with bias == -1
    HOLD always passes.

    Args:
        biases: {TF_label: direction} from get_mtf_bias().
        signal: Proposed trade direction (1, -1, or 0).
        min_agree: Minimum number of TFs that must agree.

    Returns:
        True if signal is confirmed by enough timeframes.
    """
    if signal == 0:
        return True  # HOLD always passes
    agree_count = sum(1 for b in biases.values() if b == signal)
    return agree_count >= min_agree
