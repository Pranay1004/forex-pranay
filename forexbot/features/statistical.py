"""
features/statistical.py — Statistical and quantitative features.
Includes Hurst Exponent, autocorrelation, entropy, fractal dimension,
ADF test, DFA, GARCH volatility, Kalman filter trend, and cross-pair correlations.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


# ─── Hurst Exponent (R/S method) ─────────────────────────────────────────────

def _hurst_rs(series: np.ndarray) -> float:
    """Compute Hurst Exponent via R/S (rescaled range) analysis."""
    n = len(series)
    if n < 20:
        return 0.5
    lags = range(10, n // 2, max(1, n // 20))
    rs_list, lag_list = [], []
    for lag in lags:
        chunks = [series[i:i + lag] for i in range(0, n - lag + 1, lag)]
        rs_vals = []
        for chunk in chunks:
            if len(chunk) < 4:
                continue
            mean_adj = chunk - chunk.mean()
            cumdev = np.cumsum(mean_adj)
            r = cumdev.max() - cumdev.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            rs_list.append(np.mean(rs_vals))
            lag_list.append(lag)
    if len(lag_list) < 2:
        return 0.5
    try:
        slope, _, _, _, _ = stats.linregress(np.log(lag_list), np.log(rs_list))
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5


def rolling_hurst(returns: pd.Series, window: int = 100) -> pd.Series:
    """Rolling Hurst Exponent over a given window."""
    result = pd.Series(np.nan, index=returns.index, dtype=float)
    arr = returns.values
    for i in range(window, len(arr) + 1):
        result.iloc[i - 1] = _hurst_rs(arr[i - window:i])
    return result


# ─── Higuchi Fractal Dimension ────────────────────────────────────────────────

def _higuchi_fd(series: np.ndarray, kmax: int = 5) -> float:
    """Compute Higuchi fractal dimension of a 1-D array."""
    n = len(series)
    if n < 10:
        return 1.5
    lk, ks = [], []
    for k in range(1, kmax + 1):
        lengths = []
        for m in range(1, k + 1):
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue
            sub = series[indices]
            km = (n - 1) / (np.floor((n - m) / k) * k)
            lm = (np.sum(np.abs(np.diff(sub))) * km) / k
            lengths.append(lm)
        if lengths:
            lk.append(np.mean(lengths))
            ks.append(k)
    if len(ks) < 2:
        return 1.5
    try:
        slope, _, _, _, _ = stats.linregress(np.log(ks), np.log(lk))
        return float(-slope)
    except Exception:
        return 1.5


def rolling_fractal_dim(returns: pd.Series, window: int = 50) -> pd.Series:
    """Rolling Higuchi fractal dimension."""
    result = pd.Series(np.nan, index=returns.index, dtype=float)
    arr = returns.values
    for i in range(window, len(arr) + 1):
        result.iloc[i - 1] = _higuchi_fd(arr[i - window:i])
    return result


# ─── Shannon Entropy ──────────────────────────────────────────────────────────

def _shannon_entropy(series: np.ndarray, bins: int = 20) -> float:
    """Compute Shannon entropy of a return series by binning into a distribution."""
    series = series[~np.isnan(series)]
    if len(series) < 5:
        return 0.0
    counts, _ = np.histogram(series, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def rolling_entropy(returns: pd.Series, window: int = 30) -> pd.Series:
    """Rolling Shannon entropy."""
    return returns.rolling(window).apply(lambda x: _shannon_entropy(x.values), raw=False)


# ─── Detrended Fluctuation Analysis ──────────────────────────────────────────

def _dfa_exponent(series: np.ndarray) -> float:
    """Compute DFA scaling exponent (quadratic detrend)."""
    n = len(series)
    if n < 30:
        return 0.5
    y = np.cumsum(series - series.mean())
    scales, fluctuations = [], []
    for scale in np.unique(np.geomspace(4, n // 4, num=10, dtype=int)):
        if scale < 4:
            continue
        n_segments = n // scale
        if n_segments < 2:
            continue
        F2 = []
        for seg in range(n_segments):
            seg_y = y[seg * scale:(seg + 1) * scale]
            x_seg = np.arange(len(seg_y))
            try:
                coeffs = np.polyfit(x_seg, seg_y, 2)
                trend = np.polyval(coeffs, x_seg)
                F2.append(np.mean((seg_y - trend) ** 2))
            except Exception:
                continue
        if F2:
            fluctuations.append(np.sqrt(np.mean(F2)))
            scales.append(scale)
    if len(scales) < 2:
        return 0.5
    try:
        slope, _, _, _, _ = stats.linregress(np.log(scales), np.log(fluctuations))
        return float(np.clip(slope, 0.0, 2.0))
    except Exception:
        return 0.5


# ─── ADF p-value ─────────────────────────────────────────────────────────────

def rolling_adf_pvalue(series: pd.Series, window: int = 100) -> pd.Series:
    """Rolling Augmented Dickey-Fuller p-value."""
    result = pd.Series(np.nan, index=series.index, dtype=float)
    arr = series.values
    for i in range(window, len(arr) + 1):
        try:
            p = adfuller(arr[i - window:i], autolag="AIC")[1]
            result.iloc[i - 1] = float(p)
        except Exception:
            result.iloc[i - 1] = 1.0
    return result


# ─── GARCH Conditional Volatility ─────────────────────────────────────────────

def garch_volatility(returns: pd.Series, window: int = 252) -> pd.Series:
    """
    Fit GARCH(1,1) on a rolling window and return conditional volatility.
    Falls back to rolling std if arch is unavailable.
    """
    result = returns.rolling(20).std().rename("GARCH_VOL")
    try:
        import warnings
        from arch import arch_model  # type: ignore

        # Suppress DataScaleWarning from arch; we're already scaling by 100
        warnings.filterwarnings("ignore", category=Warning, module="arch")

        arr = returns.fillna(0.0).values * 100
        result_vals = np.full(len(arr), np.nan)
        step = max(1, window // 4)
        for i in range(window, len(arr), step):
            try:
                am = arch_model(arr[i - window:i], vol="Garch", p=1, q=1, dist="Normal")
                res = am.fit(disp="off", show_warning=False)
                fc = res.forecast(horizon=1, reindex=False)
                vol = float(np.sqrt(fc.variance.values[-1, 0])) / 100
                # Fill forward until next estimation
                end = min(i + step, len(arr))
                result_vals[i:end] = vol
            except Exception:
                pass
        result = pd.Series(result_vals, index=returns.index, name="GARCH_VOL")
    except ImportError:
        logger.warning("arch library not found; using rolling std for GARCH proxy")
    return result


# ─── Kalman Filter Trend ──────────────────────────────────────────────────────

def kalman_trend(close: pd.Series) -> pd.Series:
    """
    Estimate trend using a simple Kalman filter (random walk + observation noise).
    Falls back to EMA if pykalman is unavailable.
    """
    try:
        from pykalman import KalmanFilter  # type: ignore

        observations = close.ffill().values.reshape(-1, 1)
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=observations[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01,
        )
        state_means, _ = kf.filter(observations)
        return pd.Series(state_means.flatten(), index=close.index, name="KF_TREND")
    except Exception:
        logger.debug("Kalman fallback to EMA-50")
        return close.ewm(span=50).mean().rename("KF_TREND")


# ─── Main Builder ─────────────────────────────────────────────────────────────

def build_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all statistical and quantitative features.

    Args:
        df: OHLCV DataFrame (already has technical features from technical.py).

    Returns:
        DataFrame with additional statistical feature columns.
    """
    out = df.copy()
    c = out["Close"]
    log_ret = np.log(c / c.shift(1))

    # Autocorrelations (lag 1, 5, 10)
    for lag in (1, 5, 10):
        out[f"AUTOCORR_{lag}"] = log_ret.rolling(50).apply(
            lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else np.nan,
            raw=False,
        )

    # Skewness and Kurtosis (rolling 50-bar)
    out["SKEW_50"] = log_ret.rolling(50).skew()
    out["KURT_50"] = log_ret.rolling(50).apply(
        lambda x: float(kurtosis(x, nan_policy="omit")), raw=True
    )

    # Shannon Entropy
    out["ENTROPY_30"] = rolling_entropy(log_ret, window=30)

    # Hurst (rolling 100-bar) — expensive; compute every 10th row, interpolate
    logger.debug("Computing Hurst exponent (this may be slow)…")
    hurst_vals = pd.Series(np.nan, index=out.index, dtype=float)
    arr = log_ret.fillna(0.0).values
    step = 10
    for i in range(100, len(arr) + 1, step):
        h = _hurst_rs(arr[max(0, i - 100):i])
        hurst_vals.iloc[i - 1] = h
    out["HURST_100"] = hurst_vals.interpolate(method="linear")

    # Fractal Dimension (rolling 50)
    out["FRACTAL_DIM_50"] = rolling_fractal_dim(log_ret.fillna(0.0), window=50)

    # ADF p-value (rolling 100) — compute every 20th row for performance
    adf_vals = pd.Series(np.nan, index=out.index, dtype=float)
    for i in range(100, len(out) + 1, 20):
        try:
            p = adfuller(c.values[max(0, i - 100):i], autolag="AIC")[1]
            adf_vals.iloc[i - 1] = float(p)
        except Exception:
            pass
    out["ADF_PVAL_100"] = adf_vals.interpolate(method="linear")

    # DFA exponent (rolling 200)
    dfa_vals = pd.Series(np.nan, index=out.index, dtype=float)
    for i in range(200, len(out) + 1, 50):
        sub = log_ret.fillna(0.0).values[max(0, i - 200):i]
        dfa_vals.iloc[i - 1] = _dfa_exponent(sub)
    out["DFA_EXP_200"] = dfa_vals.interpolate(method="linear")

    # GARCH conditional volatility
    out["GARCH_VOL"] = garch_volatility(log_ret.fillna(0.0))

    # Kalman Filter trend
    out["KF_TREND"] = kalman_trend(c)
    out["KF_TREND_DIFF"] = out["KF_TREND"].diff()

    # Order flow imbalance (if not already present from technical.py)
    if "ORDER_FLOW_IMB" not in out.columns:
        bar_range = (out["High"] - out["Low"]).replace(0, np.nan)
        out["ORDER_FLOW_IMB"] = (c - out["Open"]) / bar_range

    # Relative volume
    if "REL_VOLUME" not in out.columns:
        out["REL_VOLUME"] = out["Volume"] / out["Volume"].rolling(20).mean().replace(0, np.nan)

    logger.debug("Statistical features built: %d columns total", out.shape[1])
    return out


def build_cross_pair_features(
    all_data: dict[str, pd.DataFrame],
    target_pair: str,
) -> pd.DataFrame:
    """
    Add cross-pair correlation features and DXY beta to a pair's DataFrame.

    Args:
        all_data: Dict of {pair: DataFrame} with Close prices already aligned.
        target_pair: The pair whose DataFrame we're enriching.

    Returns:
        Enriched DataFrame with cross-correlation and beta columns.
    """
    from forexbot.config import PAIRS

    df = all_data[target_pair].copy()
    target_ret = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)

    # Build a DXY proxy: equal-weight USD-positive pairs vs USD-negative pairs
    # USD positive (USD is quote): EURUSD(-), GBPUSD(-), AUDUSD(-), NZDUSD(-), GBPJPY(n/a)
    # Simple proxy: average of USDJPY, USDCAD, USDCHF log returns (USD base)
    usd_base_pairs = [p for p in PAIRS if p.startswith("USD") and p in all_data]
    if usd_base_pairs:
        dxy_rets = pd.concat(
            [np.log(all_data[p]["Close"] / all_data[p]["Close"].shift(1)) for p in usd_base_pairs],
            axis=1,
        ).mean(axis=1).fillna(0.0)
        # Align index
        dxy_rets = dxy_rets.reindex(df.index).fillna(0.0)
        rolling_cov = target_ret.rolling(20).cov(dxy_rets)
        rolling_var = dxy_rets.rolling(20).var().replace(0, np.nan)
        df["BETA_DXY_20"] = rolling_cov / rolling_var
        df["CORR_DXY_20"] = target_ret.rolling(20).corr(dxy_rets)
    else:
        df["BETA_DXY_20"] = np.nan
        df["CORR_DXY_20"] = np.nan

    # Rolling 20-bar correlation with each other pair
    for other in PAIRS:
        if other == target_pair or other not in all_data:
            continue
        other_ret = np.log(
            all_data[other]["Close"] / all_data[other]["Close"].shift(1)
        ).reindex(df.index).fillna(0.0)
        df[f"CORR_{other}_20"] = target_ret.rolling(20).corr(other_ret)

    return df
