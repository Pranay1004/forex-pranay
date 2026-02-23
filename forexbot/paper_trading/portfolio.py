"""
paper_trading/portfolio.py — In-memory portfolio state: balances, open positions, P&L.
Tracks per-pair and aggregate statistics needed for Kelly sizing and display.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import numpy as np

from forexbot.config import (
    MAX_CONCURRENT_POSITIONS,
    MAX_POSITIONS_PER_PAIR,
    PIP_SIZE,
    SLIPPAGE_PIPS,
    SPREAD_PIPS,
    STARTING_BALANCE,
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Single open paper trade position.

    Attributes:
        trade_id: Unique identifier.
        pair: Currency pair.
        direction: 1=LONG, -1=SHORT.
        entry_price: Fill price (entry + spread).
        stop_loss: Stop-loss level.
        take_profit: Take-profit level.
        position_size: Notional USD size.
        lot_size: Computed lot equivalent.
        bars_open: Number of H1 bars since entry.
        entry_time: UTC datetime of entry.
        atr_at_entry: ATR value at entry.
        stop_moved_to_breakeven: Whether trailing stop has been applied.
        regime_at_entry: Market regime at entry.
        sentiment_at_entry: Sentiment score at entry.
        model_confidence: Ensemble confidence at entry.
    """
    trade_id: str
    pair: str
    direction: int
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    lot_size: float
    bars_open: int = 0
    entry_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    atr_at_entry: float = 0.0
    stop_moved_to_breakeven: bool = False
    regime_at_entry: int = 0
    sentiment_at_entry: float = 0.0
    model_confidence: float = 0.0

    def current_pnl_pips(self, current_price: float) -> float:
        """Unrealised P&L in pips."""
        pip = PIP_SIZE.get(self.pair, 0.0001)
        if pip <= 0:
            return 0.0
        if self.direction == 1:
            return (current_price - self.entry_price) / pip
        else:
            return (self.entry_price - current_price) / pip

    def current_pnl_usd(self, current_price: float) -> float:
        """Unrealised P&L in USD (approximate: pips × pip_value)."""
        pip = PIP_SIZE.get(self.pair, 0.0001)
        pips = self.current_pnl_pips(current_price)
        pip_value = (pip / current_price) * self.position_size
        return pips * pip_value


@dataclass
class ClosedTrade:
    """
    Record of a completed trade (for statistics and CSV saving).
    """
    trade_id: str
    pair: str
    direction: int
    entry_price: float
    exit_price: float
    position_size: float
    lot_size: float
    stop_loss: float
    take_profit: float
    pnl_pips: float
    pnl_usd: float
    duration_bars: int
    entry_time: datetime
    exit_time: datetime
    exit_reason: str           # "TP", "SL", "TRAILING", "TIME", "REGIME_FLIP", "MANUAL"
    model_confidence: float
    regime_at_entry: int
    sentiment_at_entry: float


@dataclass
class Portfolio:
    """
    Full paper trading portfolio state.

    Attributes:
        balance: Current paper account balance.
        open_positions: Dict {trade_id: Position}.
        closed_trades: All closed trade records (chronologically).
        start_of_day_balance: Balance at beginning of current trading day.
    """
    balance: float = STARTING_BALANCE
    open_positions: dict[str, Position] = field(default_factory=dict)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    start_of_day_balance: float = STARTING_BALANCE

    # ── Open position management ───────────────────────────────────────────────

    def can_open_position(self, pair: str) -> tuple[bool, str]:
        """Check whether a new position can be opened for the pair."""
        total_open = len(self.open_positions)
        if total_open >= MAX_CONCURRENT_POSITIONS:
            return False, f"Max concurrent positions ({MAX_CONCURRENT_POSITIONS}) reached"
        pair_open = sum(1 for p in self.open_positions.values() if p.pair == pair)
        if pair_open >= MAX_POSITIONS_PER_PAIR:
            return False, f"Already have {pair_open} position(s) in {pair}"
        return True, ""

    def open_position(
        self,
        pair: str,
        direction: int,
        raw_entry: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        atr: float,
        regime: int = 0,
        sentiment: float = 0.0,
        confidence: float = 0.0,
    ) -> Optional[Position]:
        """
        Open a new paper position with spread + slippage simulation.

        Args:
            pair: Currency pair.
            direction: 1=BUY, -1=SELL.
            raw_entry: Raw price before spread/slippage.
            stop_loss: Pre-calculated stop loss.
            take_profit: Pre-calculated take profit.
            position_size: Notional trade size (USD).
            atr: ATR at entry.
            regime: Market regime.
            sentiment: Sentiment score.
            confidence: Model confidence.

        Returns:
            Opened Position, or None if checks fail.
        """
        can_open, reason = self.can_open_position(pair)
        if not can_open:
            logger.debug("Cannot open %s: %s", pair, reason)
            return None

        pip = PIP_SIZE.get(pair, 0.0001)
        spread = SPREAD_PIPS.get(pair, 1.5) * pip
        slippage = np.random.uniform(0, SLIPPAGE_PIPS) * pip

        # Apply spread: buyer pays ask (entry + spread), seller receives bid (entry - spread)
        if direction == 1:
            fill_price = raw_entry + spread / 2 + slippage
        else:
            fill_price = raw_entry - spread / 2 - slippage

        lot_size = position_size / raw_entry if raw_entry > 0 else 0.0

        pos = Position(
            trade_id=str(uuid4())[:8],
            pair=pair,
            direction=direction,
            entry_price=fill_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            lot_size=lot_size,
            atr_at_entry=atr,
            regime_at_entry=regime,
            sentiment_at_entry=sentiment,
            model_confidence=confidence,
        )
        self.open_positions[pos.trade_id] = pos
        logger.info(
            "OPENED %s %s @ %.5f (fill=%.5f) SL=%.5f TP=%.5f size=%.2f",
            "LONG" if direction == 1 else "SHORT",
            pair, raw_entry, fill_price, stop_loss, take_profit, position_size,
        )
        return pos

    def close_position(
        self,
        trade_id: str,
        exit_price: float,
        reason: str,
    ) -> Optional[ClosedTrade]:
        """
        Close an open position and record the trade.

        Args:
            trade_id: The position's unique ID.
            exit_price: Price at exit.
            reason: Exit reason string.

        Returns:
            ClosedTrade record, or None if trade_id not found.
        """
        pos = self.open_positions.pop(trade_id, None)
        if pos is None:
            logger.warning("close_position: trade_id %s not found", trade_id)
            return None

        pip = PIP_SIZE.get(pos.pair, 0.0001)
        # Apply spread on exit
        spread = SPREAD_PIPS.get(pos.pair, 1.5) * pip
        slippage = np.random.uniform(0, SLIPPAGE_PIPS) * pip
        if pos.direction == 1:
            fill_exit = exit_price - spread / 2 - slippage
        else:
            fill_exit = exit_price + spread / 2 + slippage

        if pos.direction == 1:
            pnl_pips = (fill_exit - pos.entry_price) / pip
        else:
            pnl_pips = (pos.entry_price - fill_exit) / pip

        pip_value = pip / fill_exit * pos.position_size
        pnl_usd = pnl_pips * pip_value

        self.balance += pnl_usd

        ct = ClosedTrade(
            trade_id=pos.trade_id,
            pair=pos.pair,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=fill_exit,
            position_size=pos.position_size,
            lot_size=pos.lot_size,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            pnl_pips=round(pnl_pips, 1),
            pnl_usd=round(pnl_usd, 2),
            duration_bars=pos.bars_open,
            entry_time=pos.entry_time,
            exit_time=datetime.now(tz=timezone.utc),
            exit_reason=reason,
            model_confidence=pos.model_confidence,
            regime_at_entry=pos.regime_at_entry,
            sentiment_at_entry=pos.sentiment_at_entry,
        )
        self.closed_trades.append(ct)

        logger.info(
            "CLOSED %s %s @ %.5f | pnl_pips=%.1f pnl_usd=%.2f reason=%s",
            "LONG" if pos.direction == 1 else "SHORT",
            pos.pair, fill_exit, pnl_pips, pnl_usd, reason,
        )
        return ct

    # ── Statistics ─────────────────────────────────────────────────────────────

    def recent_win_rate(self, n: int = 50) -> float:
        """Rolling win rate over the last n closed trades."""
        trades = self.closed_trades[-n:]
        if not trades:
            return 0.50
        wins = sum(1 for t in trades if t.pnl_usd > 0)
        return wins / len(trades)

    def recent_avg_win_loss(self, n: int = 50) -> tuple[float, float]:
        """Return (avg_win_pips, avg_loss_pips) from last n trades."""
        trades = self.closed_trades[-n:]
        win_pips = [t.pnl_pips for t in trades if t.pnl_pips > 0]
        loss_pips = [abs(t.pnl_pips) for t in trades if t.pnl_pips < 0]
        avg_win = float(np.mean(win_pips)) if win_pips else 1.0
        avg_loss = float(np.mean(loss_pips)) if loss_pips else 1.0
        return avg_win, avg_loss

    def today_pnl(self) -> float:
        """P&L since start of current day (USD)."""
        return self.balance - self.start_of_day_balance

    def max_drawdown(self) -> float:
        """Maximum drawdown fraction from peak balance across all time."""
        if not self.closed_trades:
            return 0.0
        cumulative = STARTING_BALANCE
        peak = cumulative
        max_dd = 0.0
        for t in self.closed_trades:
            cumulative += t.pnl_usd
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def profit_factor(self) -> float:
        """Gross profit / gross loss ratio."""
        gross_profit = sum(t.pnl_usd for t in self.closed_trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in self.closed_trades if t.pnl_usd < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 1.0
        return gross_profit / gross_loss
