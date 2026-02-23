"""
models/ensemble.py — Weighted soft-vote ensemble: TabPFN + XGBoost + LightGBM + Sentiment.
Outputs BUY / HOLD / SELL probabilities and the final signal with confidence.
Weights are adjusted by market regime.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from forexbot.config import CONFIDENCE_THRESHOLD, ENSEMBLE_WEIGHTS
from forexbot.models.classical import ClassicalModels
from forexbot.models.hf_sentiment import SentimentEngine
from forexbot.models.hf_tabular import TabPFNWrapper

logger = logging.getLogger(__name__)

# Signal integer constants
BUY = 1
HOLD = 0
SELL = -1

# Class probability index: index 0 = BUY, index 1 = HOLD, index 2 = SELL
_BUY_IDX = 0
_HOLD_IDX = 1
_SELL_IDX = 2


def _regime_adjusted_weights(regime: int) -> dict[str, float]:
    """
    Return ensemble weights adjusted for the current market regime.

    Regime 0 = RANGING  → favour sentiment + TabPFN (mean-reversion)
    Regime 1/2 = TRENDING → favour XGBoost + LightGBM (momentum)

    Args:
        regime: Integer regime code (0, 1, or 2).

    Returns:
        Dict of {model_name: weight} summing to 1.0.
    """
    w = ENSEMBLE_WEIGHTS.copy()

    if regime == 0:  # RANGING
        w["sentiment"] = min(1.0, w["sentiment"] + 0.10)
        w["tabpfn"] = min(1.0, w["tabpfn"] + 0.05)
        # Reduce momentum models proportionally
        reduction = 0.15 / 2
        w["xgboost"] = max(0.05, w["xgboost"] - reduction)
        w["lightgbm"] = max(0.05, w["lightgbm"] - reduction)
    elif regime in (1, 2):  # TRENDING
        w["xgboost"] = min(1.0, w["xgboost"] + 0.10)
        w["lightgbm"] = min(1.0, w["lightgbm"] + 0.10)
        # Reduce sentiment contribution
        w["sentiment"] = max(0.05, w["sentiment"] - 0.10)
        w["tabpfn"] = max(0.05, w["tabpfn"] - 0.10)

    # Normalise to sum = 1.0
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


@dataclass
class EnsembleResult:
    """
    Output from the ensemble for a single pair at a single bar.

    Attributes:
        signal: Final signal integer (1=BUY, 0=HOLD, -1=SELL).
        confidence: Max class probability after ensemble voting.
        buy_prob: Ensemble BUY probability.
        hold_prob: Ensemble HOLD probability.
        sell_prob: Ensemble SELL probability.
        model_probas: Per-model probability arrays {model_name: [p_buy, p_hold, p_sell]}.
        weights_used: Regime-adjusted weights actually used.
    """
    signal: int
    confidence: float
    buy_prob: float
    hold_prob: float
    sell_prob: float
    model_probas: dict[str, np.ndarray]
    weights_used: dict[str, float]


def run_ensemble(
    pair: str,
    X_latest: np.ndarray,
    classical: ClassicalModels,
    tabpfn: TabPFNWrapper,
    sentiment_engine: SentimentEngine,
    headlines: list[str],
    regime: int,
) -> EnsembleResult:
    """
    Combine all model outputs into a weighted soft-vote signal.

    Args:
        pair: Currency pair.
        X_latest: Feature vector for the latest bar, shape (1, n_features).
        classical: Fitted ClassicalModels (XGBoost + LightGBM).
        tabpfn: Fitted TabPFNWrapper.
        sentiment_engine: Loaded SentimentEngine.
        headlines: Latest headlines for sentiment scoring.
        regime: Current market regime (0=RANGING, 1=TRENDING_UP, 2=TRENDING_DOWN).

    Returns:
        EnsembleResult with final signal and all intermediate probabilities.
    """
    uniform = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    model_probas: dict[str, np.ndarray] = {}

    # ── Model predictions ──────────────────────────────────────────────────────
    # TabPFN
    try:
        model_probas["tabpfn"] = tabpfn.predict_proba(X_latest)
    except Exception as exc:
        logger.warning("%s: TabPFN ensemble error: %s", pair, exc)
        model_probas["tabpfn"] = uniform.copy()

    # Classical
    try:
        xgb_p, lgb_p = classical.predict_proba(X_latest)
        model_probas["xgboost"] = xgb_p
        model_probas["lightgbm"] = lgb_p
    except Exception as exc:
        logger.warning("%s: Classical ensemble error: %s", pair, exc)
        model_probas["xgboost"] = uniform.copy()
        model_probas["lightgbm"] = uniform.copy()

    # Sentiment
    try:
        sent_score = sentiment_engine.score(pair, headlines)
        model_probas["sentiment"] = sentiment_engine.sentiment_to_class_probs(sent_score)
    except Exception as exc:
        logger.warning("%s: Sentiment ensemble error: %s", pair, exc)
        model_probas["sentiment"] = uniform.copy()

    # ── Regime-adjusted weighted vote ─────────────────────────────────────────
    weights = _regime_adjusted_weights(regime)

    combined = np.zeros(3, dtype=float)
    for model_name, proba in model_probas.items():
        w = weights.get(model_name, 0.0)
        p = np.array(proba, dtype=float)
        # Guard against bad shapes
        if p.shape == (3,):
            combined += w * p
        else:
            combined += w * uniform

    # Normalise
    total = combined.sum()
    if total > 0:
        combined /= total

    buy_prob = float(combined[_BUY_IDX])
    hold_prob = float(combined[_HOLD_IDX])
    sell_prob = float(combined[_SELL_IDX])

    # Final signal: highest probability class (must exceed confidence threshold)
    max_idx = int(np.argmax(combined))
    confidence = float(combined[max_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        signal = HOLD
    elif max_idx == _BUY_IDX:
        signal = BUY
    elif max_idx == _SELL_IDX:
        signal = SELL
    else:
        signal = HOLD

    logger.debug(
        "%s ensemble: BUY=%.2f HOLD=%.2f SELL=%.2f → signal=%d conf=%.2f (regime=%d)",
        pair, buy_prob, hold_prob, sell_prob, signal, confidence, regime,
    )

    return EnsembleResult(
        signal=signal,
        confidence=confidence,
        buy_prob=buy_prob,
        hold_prob=hold_prob,
        sell_prob=sell_prob,
        model_probas=model_probas,
        weights_used=weights,
    )
