"""
strategy/signal.py — Convert ensemble output to actionable trade signals with position sizing.
Applies confidence threshold and delegates to risk module for sizing.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from forexbot.config import CONFIDENCE_THRESHOLD
from forexbot.models.ensemble import EnsembleResult, BUY, SELL, HOLD
from forexbot.strategy.risk import TradeParameters, calculate_trade_parameters
from forexbot.config import (
    MIN_DIRECTIONAL_PROB,
    MIN_PROB_EDGE,
    PERFORMANCE_GUARD_MIN_RR,
    PERFORMANCE_GUARD_MIN_WIN_RATE,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """
    Full trade signal ready for execution by the paper trading engine.

    Attributes:
        pair: Currency pair.
        direction: 1=BUY, -1=SELL, 0=HOLD (no trade).
        confidence: Model ensemble confidence.
        entry_price: Current market price.
        params: Position sizing and SL/TP levels (None if HOLD).
        regime: Market regime at signal time.
        sentiment_score: Sentiment score at signal time.
        ensemble: Full ensemble result.
    """
    pair: str
    direction: int
    confidence: float
    entry_price: float
    params: Optional[TradeParameters]
    regime: int
    sentiment_score: float
    ensemble: EnsembleResult


def generate_signal(
    pair: str,
    ensemble_result: EnsembleResult,
    current_price: float,
    atr: float,
    account_balance: float,
    regime: int,
    sentiment_score: float,
    win_rate: float = 0.50,
    avg_win: float = 1.0,
    avg_loss: float = 1.0,
) -> TradeSignal:
    """
    Convert ensemble output to a fully parametrised trade signal.

    Args:
        pair: Currency pair.
        ensemble_result: Output from run_ensemble().
        current_price: Latest close price.
        atr: Current ATR(14) value.
        account_balance: Current paper balance.
        regime: Current market regime.
        sentiment_score: Current combined sentiment score.
        win_rate: Rolling win rate from recent trades.
        avg_win: Average winning pip gain (for Kelly).
        avg_loss: Average losing pip loss (for Kelly).

    Returns:
        TradeSignal dataclass with all trade parameters.
    """
    direction = ensemble_result.signal
    confidence = ensemble_result.confidence

    if direction == HOLD or confidence < CONFIDENCE_THRESHOLD:
        return TradeSignal(
            pair=pair,
            direction=HOLD,
            confidence=confidence,
            entry_price=current_price,
            params=None,
            regime=regime,
            sentiment_score=sentiment_score,
            ensemble=ensemble_result,
        )

    buy_prob = float(ensemble_result.buy_prob)
    hold_prob = float(ensemble_result.hold_prob)
    sell_prob = float(ensemble_result.sell_prob)
    probs = np.array([buy_prob, hold_prob, sell_prob], dtype=float)
    sorted_probs = np.sort(probs)
    prob_edge = float(sorted_probs[-1] - sorted_probs[-2])
    directional_prob = buy_prob if direction == BUY else sell_prob

    if directional_prob < MIN_DIRECTIONAL_PROB or prob_edge < MIN_PROB_EDGE:
        logger.info(
            "%s: signal blocked by quality gate | dir_prob=%.3f edge=%.3f",
            pair,
            directional_prob,
            prob_edge,
        )
        return TradeSignal(
            pair=pair,
            direction=HOLD,
            confidence=confidence,
            entry_price=current_price,
            params=None,
            regime=regime,
            sentiment_score=sentiment_score,
            ensemble=ensemble_result,
        )

    if (direction == BUY and regime == 2) or (direction == SELL and regime == 1):
        logger.info("%s: signal blocked by regime alignment | direction=%d regime=%d", pair, direction, regime)
        return TradeSignal(
            pair=pair,
            direction=HOLD,
            confidence=confidence,
            entry_price=current_price,
            params=None,
            regime=regime,
            sentiment_score=sentiment_score,
            ensemble=ensemble_result,
        )

    rr = (avg_win / avg_loss) if avg_loss > 0 else 1.0
    if win_rate < PERFORMANCE_GUARD_MIN_WIN_RATE and rr < PERFORMANCE_GUARD_MIN_RR:
        logger.info(
            "%s: signal blocked by performance guard | win_rate=%.3f rr=%.3f",
            pair,
            win_rate,
            rr,
        )
        return TradeSignal(
            pair=pair,
            direction=HOLD,
            confidence=confidence,
            entry_price=current_price,
            params=None,
            regime=regime,
            sentiment_score=sentiment_score,
            ensemble=ensemble_result,
        )

    # Compute trade parameters
    if atr <= 0 or not np.isfinite(atr):
        logger.warning("%s: Invalid ATR=%.6f — defaulting to HOLD", pair, atr)
        return TradeSignal(
            pair=pair,
            direction=HOLD,
            confidence=confidence,
            entry_price=current_price,
            params=None,
            regime=regime,
            sentiment_score=sentiment_score,
            ensemble=ensemble_result,
        )

    params = calculate_trade_parameters(
        entry_price=current_price,
        direction=direction,
        atr=atr,
        account_balance=account_balance,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )

    logger.info(
        "%s: Signal %s | conf=%.2f | entry=%.5f SL=%.5f TP=%.5f | size=%.2f",
        pair,
        "BUY" if direction == BUY else "SELL",
        confidence,
        current_price,
        params.stop_loss,
        params.take_profit,
        params.position_size,
    )

    return TradeSignal(
        pair=pair,
        direction=direction,
        confidence=confidence,
        entry_price=current_price,
        params=params,
        regime=regime,
        sentiment_score=sentiment_score,
        ensemble=ensemble_result,
    )


def get_current_atr(df: pd.DataFrame) -> float:
    """
    Extract the most recent ATR(14) from a featured DataFrame.

    Args:
        df: DataFrame with ATR columns (produced by technical.py).

    Returns:
        ATR value as float, or 0.0 if unavailable.
    """
    atr_col = next((c for c in df.columns if "ATR_14" in c), None)
    if atr_col is None or df.empty:
        return 0.0
    val = df[atr_col].dropna().iloc[-1] if not df[atr_col].dropna().empty else 0.0
    return float(val)
