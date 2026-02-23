"""
paper_trading/engine.py — Paper trading execution loop.
Manages position opens/closes, trailing stops, time exits, and regime-flip exits.
Appends every closed trade to CSV.
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from forexbot.config import (
    ATR_SL_MULTIPLIER,
    SPREAD_PIPS,
    TIME_EXIT_BARS,
    TRADES_FILE,
)
from forexbot.paper_trading.portfolio import ClosedTrade, Portfolio, Position
from forexbot.strategy.risk import DrawdownState, should_trail_stop
from forexbot.strategy.signal import TradeSignal

logger = logging.getLogger(__name__)

_TRADE_CSV_FIELDNAMES = [
    "trade_id", "pair", "direction", "entry_price", "exit_price",
    "position_size", "lot_size", "stop_loss", "take_profit",
    "pnl_pips", "pnl_usd", "duration_bars", "entry_time", "exit_time",
    "exit_reason", "model_confidence", "regime_at_entry", "sentiment_at_entry",
]


def _ensure_csv(path: Path) -> None:
    """Create the CSV with header row if it doesn't yet exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_TRADE_CSV_FIELDNAMES)
            writer.writeheader()


def _append_trade_csv(trade: ClosedTrade, path: Path) -> None:
    """Append a single closed trade to the CSV log."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TRADE_CSV_FIELDNAMES)
        writer.writerow({
            "trade_id":         trade.trade_id,
            "pair":             trade.pair,
            "direction":        trade.direction,
            "entry_price":      trade.entry_price,
            "exit_price":       trade.exit_price,
            "position_size":    trade.position_size,
            "lot_size":         trade.lot_size,
            "stop_loss":        trade.stop_loss,
            "take_profit":      trade.take_profit,
            "pnl_pips":         trade.pnl_pips,
            "pnl_usd":          trade.pnl_usd,
            "duration_bars":    trade.duration_bars,
            "entry_time":       trade.entry_time.isoformat(),
            "exit_time":        trade.exit_time.isoformat(),
            "exit_reason":      trade.exit_reason,
            "model_confidence": trade.model_confidence,
            "regime_at_entry":  trade.regime_at_entry,
            "sentiment_at_entry": trade.sentiment_at_entry,
        })


class PaperTradingEngine:
    """
    Orchestrates paper trade execution, position monitoring, and exit logic.

    Attributes:
        portfolio: The shared Portfolio holding open/closed positions.
        drawdown: DrawdownState for circuit breaker monitoring.
    """

    def __init__(self, portfolio: Portfolio, drawdown: DrawdownState) -> None:
        """
        Initialise the engine.

        Args:
            portfolio: Pre-existing Portfolio instance.
            drawdown: Pre-existing DrawdownState instance.
        """
        self.portfolio = portfolio
        self.drawdown = drawdown
        _ensure_csv(TRADES_FILE)

    def try_open(
        self,
        signal: TradeSignal,
        atr: float,
    ) -> bool:
        """
        Attempt to open a new position from a trade signal.

        Args:
            signal: A TradeSignal with direction != HOLD.
            atr: Latest ATR for the pair (for reference).

        Returns:
            True if position was opened.
        """
        # Circuit breaker check
        can_trade, reason = self.drawdown.can_trade()
        if not can_trade:
            logger.warning("Trade blocked by circuit breaker: %s", reason)
            return False

        if signal.direction == 0 or signal.params is None:
            return False

        pos = self.portfolio.open_position(
            pair=signal.pair,
            direction=signal.direction,
            raw_entry=signal.entry_price,
            stop_loss=signal.params.stop_loss,
            take_profit=signal.params.take_profit,
            position_size=signal.params.position_size,
            atr=atr,
            regime=signal.regime,
            sentiment=signal.sentiment_score,
            confidence=signal.confidence,
        )
        return pos is not None

    def update_positions(
        self,
        current_prices: dict[str, float],
        current_atrs: dict[str, float],
        current_regimes: dict[str, int],
    ) -> list[ClosedTrade]:
        """
        Advance all open positions by one bar and handle exits.

        Args:
            current_prices: {pair: latest_close_price}.
            current_atrs: {pair: latest_ATR_14}.
            current_regimes: {pair: regime_int}.

        Returns:
            List of ClosedTrade records generated this bar.
        """
        closed_this_bar: list[ClosedTrade] = []
        trade_ids = list(self.portfolio.open_positions.keys())

        for trade_id in trade_ids:
            pos = self.portfolio.open_positions.get(trade_id)
            if pos is None:
                continue

            price = current_prices.get(pos.pair)
            if price is None or price <= 0:
                pos.bars_open += 1
                continue

            atr = current_atrs.get(pos.pair, pos.atr_at_entry)
            regime = current_regimes.get(pos.pair, pos.regime_at_entry)

            exit_reason: Optional[str] = None

            # 1. Take Profit
            if pos.direction == 1 and price >= pos.take_profit:
                exit_reason = "TP"
            elif pos.direction == -1 and price <= pos.take_profit:
                exit_reason = "TP"

            # 2. Stop Loss
            if exit_reason is None:
                if pos.direction == 1 and price <= pos.stop_loss:
                    exit_reason = "SL"
                elif pos.direction == -1 and price >= pos.stop_loss:
                    exit_reason = "SL"

            # 3. Time exit
            if exit_reason is None and pos.bars_open >= TIME_EXIT_BARS:
                exit_reason = "TIME"

            # 4. Regime flip exit (0 ↔ 2)
            if exit_reason is None:
                entry_regime = pos.regime_at_entry
                if (entry_regime == 0 and regime == 2) or (entry_regime == 2 and regime == 0):
                    exit_reason = "REGIME_FLIP"

            # 5. Trailing stop: move SL to breakeven if price moved 1*ATR in favour
            if exit_reason is None and atr > 0:
                if should_trail_stop(pos.direction, pos.entry_price, price, atr, pos.stop_moved_to_breakeven):
                    pos.stop_loss = pos.entry_price  # move to breakeven
                    pos.stop_moved_to_breakeven = True
                    logger.info("%s %s: Trailing stop moved to breakeven %.5f",
                                pos.pair, trade_id, pos.entry_price)

            if exit_reason:
                closed = self.portfolio.close_position(trade_id, price, exit_reason)
                if closed:
                    closed_this_bar.append(closed)
                    _append_trade_csv(closed, TRADES_FILE)
                    self.drawdown.update(self.portfolio.balance)
            else:
                pos.bars_open += 1

        return closed_this_bar

    def widen_stops_for_news(
        self,
        pair: str,
        atr_multiplier: float = 0.5,
    ) -> None:
        """
        Widen stop losses for open positions in a pair during news events.

        Args:
            pair: Currency pair with upcoming high-impact event.
            atr_multiplier: Additional ATR multiples to add to stop distance.
        """
        for pos in self.portfolio.open_positions.values():
            if pos.pair != pair:
                continue
            extension = atr_multiplier * pos.atr_at_entry
            if pos.direction == 1:
                pos.stop_loss -= extension
            else:
                pos.stop_loss += extension
            logger.info("%s: Widened stop by %.5f for news event", pair, extension)
