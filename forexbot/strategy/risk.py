"""
strategy/risk.py — Position sizing, stop-loss, and take-profit calculation.
Uses fractional Kelly Criterion (half-Kelly) with ATR-based SL/TP.
Includes max drawdown circuit breaker logic.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from forexbot.config import (
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    KELLY_FRACTION,
    KELLY_WIN_RATE_WINDOW,
    MAX_DAILY_DRAWDOWN_PCT,
    MAX_TOTAL_DRAWDOWN_PCT,
    RISK_PER_TRADE_PCT,
    TRAILING_STOP_ATR_TRIGGER,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeParameters:
    """
    Full risk parameters for a single trade.

    Attributes:
        position_size: Notional size of the trade in account currency units.
        stop_loss: Stop-loss price level.
        take_profit: Take-profit price level.
        stop_distance: Distance from entry to stop (in price units).
        tp_distance: Distance from entry to take-profit.
        atr: ATR value used for sizing.
        kelly_f: Raw Kelly fraction (before half-Kelly scaling).
        risk_amount: Dollar amount at risk.
    """
    position_size: float
    stop_loss: float
    take_profit: float
    stop_distance: float
    tp_distance: float
    atr: float
    kelly_f: float
    risk_amount: float


def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Compute the half-Kelly fraction.

    Args:
        win_rate: Empirical win rate ∈ [0, 1].
        avg_win: Mean profit on winning trades (positive value).
        avg_loss: Mean loss on losing trades (positive value).

    Returns:
        Half-Kelly fraction ∈ [0, 0.25] (capped for safety).
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return RISK_PER_TRADE_PCT  # fallback to 1% risk

    b = avg_win / avg_loss  # win/loss ratio
    q = 1.0 - win_rate
    full_kelly = (win_rate * b - q) / b
    half_kelly = KELLY_FRACTION * full_kelly

    # Cap at 25% and floor at 0 to avoid ruin
    return float(np.clip(half_kelly, 0.0, 0.25))


def calculate_trade_parameters(
    entry_price: float,
    direction: int,          # 1=BUY, -1=SELL
    atr: float,
    account_balance: float,
    win_rate: float = 0.50,
    avg_win: float = 1.0,
    avg_loss: float = 1.0,
) -> TradeParameters:
    """
    Calculate all trade parameters for a given entry.

    Args:
        entry_price: Current price at entry.
        direction: 1 for BUY (long), -1 for SELL (short).
        atr: Current ATR(14) value.
        account_balance: Current paper account balance.
        win_rate: Rolling win rate from recent trades.
        avg_win: Average winning trade return (same units as avg_loss).
        avg_loss: Average losing trade return.

    Returns:
        TradeParameters dataclass.
    """
    kelly_f = kelly_criterion(win_rate, avg_win, avg_loss)
    position_size = kelly_f * account_balance

    # Clamp position size to max 1% risk
    max_risk = RISK_PER_TRADE_PCT * account_balance
    stop_distance = ATR_SL_MULTIPLIER * atr
    tp_distance = ATR_TP_MULTIPLIER * atr

    # Risk-based position size
    if stop_distance > 0:
        risk_based_size = max_risk / stop_distance * entry_price
    else:
        risk_based_size = position_size

    # Use smaller of Kelly and risk-based
    position_size = min(position_size, risk_based_size)
    risk_amount = (position_size / entry_price) * stop_distance

    if direction == 1:   # BUY / long
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + tp_distance
    else:                 # SELL / short
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - tp_distance

    return TradeParameters(
        position_size=round(position_size, 2),
        stop_loss=round(stop_loss, 6),
        take_profit=round(take_profit, 6),
        stop_distance=round(stop_distance, 6),
        tp_distance=round(tp_distance, 6),
        atr=round(atr, 6),
        kelly_f=round(kelly_f, 4),
        risk_amount=round(risk_amount, 2),
    )


def should_trail_stop(
    direction: int,
    entry_price: float,
    current_price: float,
    atr: float,
    stop_moved_to_breakeven: bool,
) -> bool:
    """
    Determine if trailing stop should be moved to breakeven.

    Args:
        direction: 1=BUY, -1=SELL.
        entry_price: Trade entry price.
        current_price: Current price.
        atr: Current ATR value.
        stop_moved_to_breakeven: Whether stop has already been moved.

    Returns:
        True if stop should be moved to breakeven.
    """
    if stop_moved_to_breakeven:
        return False
    if direction == 1:
        return (current_price - entry_price) >= (TRAILING_STOP_ATR_TRIGGER * atr)
    else:
        return (entry_price - current_price) >= (TRAILING_STOP_ATR_TRIGGER * atr)


@dataclass
class DrawdownState:
    """
    Tracks drawdown for the circuit breaker logic.

    Attributes:
        daily_peak_balance: Highest balance reached today.
        total_peak_balance: Highest balance ever reached.
        daily_trades_halted: Whether new trades are blocked for today.
        bot_halted: Whether the bot is fully stopped.
        halt_reason: Human-readable reason string if halted.
    """
    daily_peak_balance: float
    total_peak_balance: float
    daily_trades_halted: bool = False
    bot_halted: bool = False
    halt_reason: str = ""

    def update(self, current_balance: float) -> None:
        """
        Check drawdown levels and trigger circuit breakers if needed.

        Args:
            current_balance: Current account balance.
        """
        # Update peaks
        if current_balance > self.total_peak_balance:
            self.total_peak_balance = current_balance
        if current_balance > self.daily_peak_balance:
            self.daily_peak_balance = current_balance

        # Check total drawdown
        total_dd = (self.total_peak_balance - current_balance) / self.total_peak_balance
        if total_dd >= MAX_TOTAL_DRAWDOWN_PCT:
            self.bot_halted = True
            self.halt_reason = (
                f"MAX TOTAL DRAWDOWN {total_dd*100:.1f}% exceeded "
                f"({MAX_TOTAL_DRAWDOWN_PCT*100:.0f}% limit)"
            )
            logger.critical("BOT HALTED: %s", self.halt_reason)
            return

        # Check daily drawdown
        daily_dd = (self.daily_peak_balance - current_balance) / self.daily_peak_balance
        if daily_dd >= MAX_DAILY_DRAWDOWN_PCT:
            self.daily_trades_halted = True
            self.halt_reason = (
                f"DAILY DRAWDOWN {daily_dd*100:.1f}% exceeded "
                f"({MAX_DAILY_DRAWDOWN_PCT*100:.0f}% limit)"
            )
            logger.warning("DAILY TRADES HALTED: %s", self.halt_reason)

    def reset_daily(self, current_balance: float) -> None:
        """Reset daily tracking at the start of a new trading day."""
        self.daily_peak_balance = current_balance
        self.daily_trades_halted = False
        if not self.bot_halted:
            self.halt_reason = ""

    def can_trade(self) -> tuple[bool, str]:
        """Return (True/False, reason) for whether new trades can be opened."""
        if self.bot_halted:
            return False, self.halt_reason
        if self.daily_trades_halted:
            return False, self.halt_reason
        return True, ""
