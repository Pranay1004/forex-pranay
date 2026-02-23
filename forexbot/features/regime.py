"""
features/regime.py — Market regime detection via Hidden Markov Model.
States: 0=RANGING, 1=TRENDING_UP, 2=TRENDING_DOWN.
Uses hmmlearn GaussianHMM on [log_return, ATR_norm, ADX, RSI_14_norm].
Recalibrates weekly on a rolling 6-month window.
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Suppress HMM/sklearn convergence warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="hmmlearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from forexbot.config import (
    HMM_RECALIBRATE_DAYS,
    HMM_STATES,
    HMM_WINDOW_MONTHS,
    REGIME_LABELS,
)

logger = logging.getLogger(__name__)

_RECALIBRATE_SECONDS = HMM_RECALIBRATE_DAYS * 86400
_WINDOW_BARS = HMM_WINDOW_MONTHS * 30 * 24  # approx 6 months of H1 bars


@dataclass
class RegimeModel:
    """Holds a fitted HMM and its last calibration metadata."""

    pair: str
    model: Optional[object] = None          # hmmlearn GaussianHMM
    last_calibrated: float = 0.0
    state_map: dict[int, int] = field(default_factory=dict)   # raw→semantic state
    is_fitted: bool = False


def _build_hmm_features(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Extract the 4-feature observation matrix for HMM training/inference.

    Features: log_return, ATR_norm, ADX_14, RSI_14_norm

    Args:
        df: DataFrame containing necessary columns.

    Returns:
        2-D float32 array of shape (T, 4), or None if columns are missing.
    """
    required_checks = ["Close", "High", "Low"]
    for col in required_checks:
        if col not in df.columns:
            logger.warning("Missing column %s for regime features", col)
            return None

    log_ret = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)

    # ATR norm: ATR / Close
    atr_col = next((c for c in df.columns if c.startswith("ATR_14")), None)
    if atr_col:
        atr_norm = (df[atr_col] / df["Close"]).fillna(0.0)
    else:
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        raw_atr = tr.ewm(span=14, adjust=False).mean()
        atr_norm = (raw_atr / df["Close"]).fillna(0.0)

    # ADX
    adx_col = next((c for c in df.columns if c.upper().startswith("ADX_14")), None)
    if adx_col:
        adx = df[adx_col].fillna(25.0) / 100.0
    else:
        adx = pd.Series(0.25, index=df.index)

    # RSI_14 normalised to [0, 1]
    rsi_col = next((c for c in df.columns if "RSI_14" in c.upper() and "STOCH" not in c.upper()), None)
    if rsi_col:
        rsi_norm = (df[rsi_col].fillna(50.0) / 100.0)
    else:
        rsi_norm = pd.Series(0.5, index=df.index)

    X = np.column_stack([
        log_ret.values,
        atr_norm.values,
        adx.values,
        rsi_norm.values,
    ]).astype(np.float32)

    # Replace any inf / nan with 0
    X = np.where(np.isfinite(X), X, 0.0)
    return X


def _semantic_state_map(model, X: np.ndarray) -> dict[int, int]:
    """
    Map raw HMM states 0..N to semantic labels by ranking mean log_return.
    State with lowest mean return → TRENDING_DOWN (2)
    State with highest mean return → TRENDING_UP (1)
    Remaining → RANGING (0)

    Args:
        model: Fitted GaussianHMM.
        X: Observation array used to compute state means.

    Returns:
        Dict {raw_state: semantic_state}.
    """
    try:
        states = model.predict(X)
        means_by_state = {}
        for s in range(HMM_STATES):
            mask = states == s
            if mask.sum() > 0:
                means_by_state[s] = X[mask, 0].mean()   # mean log_return
            else:
                means_by_state[s] = 0.0
        sorted_states = sorted(means_by_state, key=means_by_state.get)
        # Lowest → TRENDING_DOWN=2, Highest → TRENDING_UP=1, Middle → RANGING=0
        mapping = {
            sorted_states[0]: 2,   # most negative returns
            sorted_states[-1]: 1,  # most positive returns
            sorted_states[1]: 0,   # middle = ranging
        }
        return mapping
    except Exception as exc:
        logger.warning("State mapping error: %s", exc)
        return {i: i for i in range(HMM_STATES)}


def calibrate_regime_model(df: pd.DataFrame, pair: str) -> RegimeModel:
    """
    Fit a fresh GaussianHMM on the most recent 6 months of H1 data.

    Args:
        df: Full-featured DataFrame for the pair.
        pair: Currency pair name.

    Returns:
        Fitted RegimeModel.
    """
    rm = RegimeModel(pair=pair)
    try:
        from hmmlearn import hmm  # type: ignore

        X = _build_hmm_features(df)
        if X is None or len(X) < 100:
            logger.warning("%s: insufficient data for HMM calibration", pair)
            return rm

        # Rolling 6-month window
        X_train = X[-_WINDOW_BARS:] if len(X) > _WINDOW_BARS else X

        model = hmm.GaussianHMM(
            n_components=HMM_STATES,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        model.fit(X_train)

        rm.model = model
        rm.state_map = _semantic_state_map(model, X_train)
        rm.last_calibrated = time.time()
        rm.is_fitted = True
        logger.info("%s: HMM calibrated (states: %s)", pair, rm.state_map)

    except ImportError:
        logger.error("hmmlearn not installed. Install with: pip install hmmlearn")
    except Exception as exc:
        logger.error("%s: HMM calibration failed: %s", pair, exc)

    return rm


def needs_recalibration(rm: RegimeModel) -> bool:
    """Return True if the regime model should be recalibrated."""
    if not rm.is_fitted:
        return True
    return (time.time() - rm.last_calibrated) > _RECALIBRATE_SECONDS


def detect_regime(df: pd.DataFrame, rm: RegimeModel) -> tuple[int, np.ndarray]:
    """
    Predict the current market regime for the latest bar.

    Args:
        df: DataFrame with features (at least the last ~200 rows).
        rm: Fitted RegimeModel.

    Returns:
        (current_regime: int 0/1/2, probabilities: ndarray of shape (3,))

    Falls back to RANGING=0 if model is not fitted.
    """
    default_probs = np.array([1.0, 0.0, 0.0], dtype=float)

    if not rm.is_fitted or rm.model is None:
        return 0, default_probs

    try:
        X = _build_hmm_features(df)
        if X is None or len(X) == 0:
            return 0, default_probs

        # Use last 200 bars for inference to avoid cold-start issues
        X_inf = X[-200:] if len(X) > 200 else X

        # predict_proba → (T, n_states)
        log_proba = rm.model.predict_proba(X_inf)  # shape (T, 3)
        probs_raw = log_proba[-1]                  # last bar

        # Remap probabilities from raw states to semantic states
        semantic_probs = np.zeros(HMM_STATES, dtype=float)
        for raw_s, sem_s in rm.state_map.items():
            if raw_s < len(probs_raw):
                semantic_probs[sem_s] += probs_raw[raw_s]

        current_regime = int(np.argmax(semantic_probs))
        return current_regime, semantic_probs

    except Exception as exc:
        logger.warning("%s: regime detection failed: %s", rm.pair, exc)
        return 0, default_probs


def regime_label(regime: int) -> str:
    """Return human-readable label for a regime integer."""
    return REGIME_LABELS.get(regime, "UNKNOWN")
