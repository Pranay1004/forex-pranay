"""
features/technical.py — 60+ technical indicators via the `ta` library.
Returns a single flat DataFrame of features appended to the OHLCV data.
No TA-Lib dependency.  No pandas-ta dependency.
"""

import logging

import numpy as np
import pandas as pd
import ta  # pip install ta

logger = logging.getLogger(__name__)


# ─── Manual helpers (indicators not covered by the `ta` library) ──────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def _dema(series: pd.Series, period: int) -> pd.Series:
    e1 = _ema(series, period)
    return 2 * e1 - _ema(e1, period)


def _tema(series: pd.Series, period: int) -> pd.Series:
    e1 = _ema(series, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def _kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    result = series.copy().astype(float)
    for i in range(period, len(series)):
        direction = abs(series.iloc[i] - series.iloc[i - period])
        vol = series.iloc[i - period: i + 1].diff().abs().sum()
        er = direction / vol if vol != 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        result.iloc[i] = result.iloc[i - 1] + sc * (series.iloc[i] - result.iloc[i - 1])
    result.iloc[:period] = np.nan
    return result


def _hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average."""
    half = max(1, period // 2)
    return _wma(2 * _wma(series, half) - _wma(series, period), max(1, int(period ** 0.5)))


def _atr_manual(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 10, multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Returns (supertrend_line, direction): direction 1=bullish, -1=bearish."""
    atr = _atr_manual(high, low, close, period)
    hl2 = (high + low) / 2
    upper = (hl2 + multiplier * atr).copy()
    lower = (hl2 - multiplier * atr).copy()
    st = pd.Series(np.nan, index=close.index)
    direction = pd.Series(0, index=close.index)

    for i in range(1, len(close)):
        pu, pl = upper.iloc[i - 1], lower.iloc[i - 1]
        cu, cl = upper.iloc[i], lower.iloc[i]
        upper.iloc[i] = cu if cu < pu or close.iloc[i - 1] > pu else pu
        lower.iloc[i] = cl if cl > pl or close.iloc[i - 1] < pl else pl
        prev = st.iloc[i - 1]
        if pd.isna(prev) or prev == pu:
            if close.iloc[i] <= upper.iloc[i]:
                st.iloc[i], direction.iloc[i] = upper.iloc[i], -1
            else:
                st.iloc[i], direction.iloc[i] = lower.iloc[i], 1
        else:
            if close.iloc[i] >= lower.iloc[i]:
                st.iloc[i], direction.iloc[i] = lower.iloc[i], 1
            else:
                st.iloc[i], direction.iloc[i] = upper.iloc[i], -1
    return st, direction


def _psar_manual(
    high: pd.Series, low: pd.Series, close: pd.Series,
    iaf: float = 0.02, maxaf: float = 0.2,
) -> pd.Series:
    """Parabolic SAR."""
    n = len(close)
    psar = close.copy().astype(float)
    bull = True
    af = iaf
    hp = high.iloc[0]
    lp = low.iloc[0]
    for i in range(2, n):
        psar.iloc[i] = psar.iloc[i - 1] + af * ((hp if bull else lp) - psar.iloc[i - 1])
        reverse = False
        if bull:
            if low.iloc[i] < psar.iloc[i]:
                bull, reverse, psar.iloc[i], lp, af = False, True, hp, low.iloc[i], iaf
        else:
            if high.iloc[i] > psar.iloc[i]:
                bull, reverse, psar.iloc[i], hp, af = True, True, lp, high.iloc[i], iaf
        if not reverse:
            if bull:
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + iaf, maxaf)
                psar.iloc[i] = min(psar.iloc[i], low.iloc[i - 1], low.iloc[i - 2])
            else:
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + iaf, maxaf)
                psar.iloc[i] = max(psar.iloc[i], high.iloc[i - 1], high.iloc[i - 2])
    return psar


def _candlestick_patterns(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> dict[str, pd.Series]:
    body = c - o
    body_size = body.abs()
    bar_range = h - l
    upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - l
    avg_body = body_size.rolling(10).mean()
    rnz = bar_range.replace(0, np.nan)

    patterns: dict[str, pd.Series] = {}
    patterns["CDL_DOJI"] = (body_size / rnz < 0.1).astype(float)
    patterns["CDL_HAMMER"] = (
        (lower_shadow > 2 * body_size) & (upper_shadow < body_size) & (body_size > 0)
    ).astype(float)
    patterns["CDL_SHOOTINGSTAR"] = (
        (upper_shadow > 2 * body_size) & (lower_shadow < body_size) & (body_size > 0)
    ).astype(float)
    patterns["CDL_HARAMI"] = (
        (h < h.shift(1)) & (l > l.shift(1)) & (body_size < body_size.shift(1) * 0.5)
    ).astype(float)
    bull_eng = (body > 0) & (o < c.shift(1)) & (c > o.shift(1))
    bear_eng = (body < 0) & (o > c.shift(1)) & (c < o.shift(1))
    patterns["CDL_ENGULFING"] = (bull_eng.astype(int) - bear_eng.astype(int)).astype(float)
    patterns["CDL_SPINNINGTOP"] = (
        (body_size < avg_body * 0.5) & (upper_shadow > body_size) & (lower_shadow > body_size)
    ).astype(float)
    patterns["CDL_MARUBOZU"] = (
        (body_size > avg_body * 1.5)
        & (upper_shadow < body_size * 0.05)
        & (lower_shadow < body_size * 0.05)
    ).astype(float)
    patterns["CDL_MORNINGSTAR"] = (
        (body.shift(2) < -avg_body.shift(2))
        & (body_size.shift(1) < avg_body.shift(1) * 0.5)
        & (body > avg_body)
    ).astype(float)
    patterns["CDL_EVENINGSTAR"] = (
        (body.shift(2) > avg_body.shift(2))
        & (body_size.shift(1) < avg_body.shift(1) * 0.5)
        & (body < -avg_body)
    ).astype(float)
    return patterns


def _roc_manual(series: pd.Series, period: int) -> pd.Series:
    return ((series - series.shift(period)) / series.shift(period).replace(0, np.nan)) * 100


def _natr_manual(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return (_atr_manual(high, low, close, period) / close.replace(0, np.nan)) * 100


def _swing_highs_lows(
    high: pd.Series, low: pd.Series, window: int = 5
) -> tuple[pd.Series, pd.Series]:
    sh = (high == high.rolling(window, center=True).max()).astype(float)
    sl = (low == low.rolling(window, center=True).min()).astype(float)
    return sh, sl


# ─── Main builder ─────────────────────────────────────────────────────────────

def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 60+ technical indicators and append them to the OHLCV DataFrame.

    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume] and DatetimeIndex.

    Returns:
        Extended DataFrame (NaNs at start of each series are expected).
    """
    out = df.copy()
    o, h, l, c, v = out["Open"], out["High"], out["Low"], out["Close"], out["Volume"]

    # ── TREND ─────────────────────────────────────────────────────────────────
    for p in (10, 20, 50, 200):
        out[f"SMA_{p}"] = ta.trend.SMAIndicator(close=c, window=p).sma_indicator()
    for p in (9, 21, 55, 200):
        out[f"EMA_{p}"] = ta.trend.EMAIndicator(close=c, window=p).ema_indicator()

    out["DEMA_20"] = _dema(c, 20)
    out["TEMA_20"] = _tema(c, 20)
    out["WMA_20"] = _wma(c, 20)
    out["KAMA_10"] = _kama(c, period=10)
    out["HMA_14"] = _hma(c, 14)

    st_line, st_dir = _supertrend(h, l, c, period=10, multiplier=3.0)
    out["SUPERT_10_3.0"] = st_line
    out["SUPERT_d_10_3.0"] = st_dir

    out["PSAR"] = _psar_manual(h, l, c)

    ich = ta.trend.IchimokuIndicator(high=h, low=l, window1=9, window2=26, window3=52)
    out["ICH_conv"] = ich.ichimoku_conversion_line()
    out["ICH_base"] = ich.ichimoku_base_line()
    out["ICH_a"] = ich.ichimoku_a()
    out["ICH_b"] = ich.ichimoku_b()

    adx_ind = ta.trend.ADXIndicator(high=h, low=l, close=c, window=14)
    out["ADX_14"] = adx_ind.adx()
    out["ADX_pos_14"] = adx_ind.adx_pos()
    out["ADX_neg_14"] = adx_ind.adx_neg()

    aroon = ta.trend.AroonIndicator(high=h, low=l, window=25)
    out["AROONU_25"] = aroon.aroon_up()
    out["AROOND_25"] = aroon.aroon_down()
    out["AROON_IND_25"] = aroon.aroon_indicator()

    vortex = ta.trend.VortexIndicator(high=h, low=l, close=c, window=14)
    out["VTXP_14"] = vortex.vortex_indicator_pos()
    out["VTXN_14"] = vortex.vortex_indicator_neg()

    # ── MOMENTUM ──────────────────────────────────────────────────────────────
    for p in (7, 14, 21):
        out[f"RSI_{p}"] = ta.momentum.RSIIndicator(close=c, window=p).rsi()

    srsi = ta.momentum.StochRSIIndicator(close=c, window=14, smooth1=3, smooth2=3)
    out["STOCHRSId_14_14_3_3"] = srsi.stochrsi_d()
    out["STOCHRSIk_14_14_3_3"] = srsi.stochrsi_k()

    macd_ind = ta.trend.MACD(close=c, window_slow=26, window_fast=12, window_sign=9)
    out["MACD_12_26_9"] = macd_ind.macd()
    out["MACDs_12_26_9"] = macd_ind.macd_signal()
    out["MACDh_12_26_9"] = macd_ind.macd_diff()
    out["MACD_hist_slope"] = out["MACDh_12_26_9"].diff()

    for p in (14, 20):
        out[f"CCI_{p}"] = ta.trend.CCIIndicator(high=h, low=l, close=c, window=p).cci()

    out["MOM_10"] = c - c.shift(10)

    for p in (5, 10, 20):
        out[f"ROC_{p}"] = ta.momentum.ROCIndicator(close=c, window=p).roc()

    out["WILLR_14"] = ta.momentum.WilliamsRIndicator(high=h, low=l, close=c, lbp=14).williams_r()
    out["UO"] = ta.momentum.UltimateOscillator(
        high=h, low=l, close=c, window1=7, window2=14, window3=28
    ).ultimate_oscillator()
    out["AO"] = ta.momentum.AwesomeOscillatorIndicator(
        high=h, low=l, window1=5, window2=34
    ).awesome_oscillator()

    tsi = ta.momentum.TSIIndicator(close=c, window_slow=25, window_fast=13)
    out["TSI_25_13"] = tsi.tsi()

    # Coppock Curve: WMA(10) of ROC(14) + ROC(11)
    out["COPPOCK"] = _wma(_roc_manual(c, 14) + _roc_manual(c, 11), 10)

    # ── VOLATILITY ────────────────────────────────────────────────────────────
    for p in (7, 14, 21):
        out[f"ATR_{p}"] = ta.volatility.AverageTrueRange(
            high=h, low=l, close=c, window=p
        ).average_true_range()

    bb = ta.volatility.BollingerBands(close=c, window=20, window_dev=2)
    out["BBU_20_2.0"] = bb.bollinger_hband()
    out["BBM_20_2.0"] = bb.bollinger_mavg()
    out["BBL_20_2.0"] = bb.bollinger_lband()
    out["BBW_20_2.0"] = bb.bollinger_wband()
    out["BBP_20_2.0"] = bb.bollinger_pband()

    kc = ta.volatility.KeltnerChannel(high=h, low=l, close=c, window=20)
    out["KCUe_20_2"] = kc.keltner_channel_hband()
    out["KCMe_20_2"] = kc.keltner_channel_mband()
    out["KCLe_20_2"] = kc.keltner_channel_lband()

    dc = ta.volatility.DonchianChannel(high=h, low=l, close=c, window=20)
    out["DCU_20"] = dc.donchian_channel_hband()
    out["DCM_20"] = dc.donchian_channel_mband()
    out["DCL_20"] = dc.donchian_channel_lband()

    log_ret = np.log(c / c.shift(1))
    for p in (10, 20, 60):
        out[f"HV_{p}"] = log_ret.rolling(p).std() * np.sqrt(252 * 24)

    out["NATR_14"] = _natr_manual(h, l, c, 14)

    # ── CANDLESTICK PATTERNS ──────────────────────────────────────────────────
    for name, series in _candlestick_patterns(o, h, l, c).items():
        out[name] = series

    # ── PRICE ACTION ──────────────────────────────────────────────────────────
    out["SWING_HIGH"], out["SWING_LOW"] = _swing_highs_lows(h, l, window=5)

    high50, low50 = h.rolling(50).max(), l.rolling(50).min()
    rng50 = (high50 - low50).replace(0, np.nan)
    out["PRICE_RANGE_POS"] = ((c - low50) / rng50).clip(0, 1)

    out["GAP"] = ((o / c.shift(1) - 1).abs() > 0.001).astype(float)

    for p in (1, 3, 5, 10, 20):
        out[f"LOG_RET_{p}"] = np.log(c / c.shift(p))

    roll20 = c.rolling(20)
    out["ZSCORE_20"] = (c - roll20.mean()) / roll20.std().replace(0, np.nan)

    high52 = c.rolling(365 * 24, min_periods=100).max()
    low52 = c.rolling(365 * 24, min_periods=100).min()
    out["DIST_52W_HIGH_PCT"] = (c / high52 - 1) * 100
    out["DIST_52W_LOW_PCT"] = (c / low52 - 1) * 100

    # ── ORDER FLOW PROXY ──────────────────────────────────────────────────────
    out["ORDER_FLOW_IMB"] = (c - o) / (h - l).replace(0, np.nan)
    
    # Handle zero volume (common in Forex/CFD data)
    avg_vol = v.rolling(20).mean()
    out["REL_VOLUME"] = v / avg_vol.replace(0, np.nan)
    # If volume is all zero, fill with 1.0 (neutral) to avoid dropping all data
    if out["REL_VOLUME"].isna().all():
         out["REL_VOLUME"] = 1.0
    else:
         pass  # Sparse NaNs handled by ffill later


    logger.debug("Technical features built: %d columns", out.shape[1])
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns (excludes raw OHLCV)."""
    base = {"Open", "High", "Low", "Close", "Volume"}
    return [col for col in df.columns if col not in base]
