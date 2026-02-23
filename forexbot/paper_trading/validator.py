"""
paper_trading/validator.py — Walk-forward backtest for model acceptance validation.
Runs 5 expanding folds on historical H1 data before live paper trading begins.
Accepts the model only if all performance criteria are met.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from forexbot.config import (
    LABEL_FORWARD_BARS,
    PIP_SIZE,
    SPREAD_PIPS,
    STARTING_BALANCE,
    VALIDATION_MAX_DRAWDOWN,
    VALIDATION_MIN_PROFIT_FACTOR,
    VALIDATION_MIN_SHARPE,
    VALIDATION_MIN_WIN_RATE,
    WALK_FORWARD_FOLDS,
)

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Performance metrics for a single walk-forward fold."""
    fold_num: int
    train_bars: int
    test_bars: int
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_duration_bars: float


@dataclass
class ValidationReport:
    """Aggregated walk-forward validation results."""
    folds: list[FoldResult]
    mean_win_rate: float
    mean_profit_factor: float
    mean_sharpe: float
    mean_max_drawdown: float
    passed: bool
    fail_reasons: list[str]


def _compute_fold_metrics(
    trades_pnl_usd: list[float],
    trades_duration: list[int],
    initial_balance: float,
) -> tuple[float, float, float, float]:
    """
    Compute win_rate, profit_factor, annualised_sharpe, max_drawdown for a list of P&L values.

    Returns:
        (win_rate, profit_factor, sharpe, max_drawdown)
    """
    if not trades_pnl_usd:
        return 0.0, 1.0, 0.0, 0.0

    wins = [p for p in trades_pnl_usd if p > 0]
    losses = [p for p in trades_pnl_usd if p < 0]

    win_rate = len(wins) / len(trades_pnl_usd)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 1.0)

    # Sharpe: annualised assuming ~252*24 bars/year for H1
    returns_arr = np.array(trades_pnl_usd) / initial_balance
    if returns_arr.std() > 0:
        # trades per year approximation
        ann_factor = np.sqrt(252 * 24 / max(len(returns_arr), 1))
        sharpe = float(returns_arr.mean() / returns_arr.std() * ann_factor)
    else:
        sharpe = 0.0

    # Max drawdown
    equity = initial_balance
    peak = equity
    max_dd = 0.0
    for pnl in trades_pnl_usd:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

    avg_dur = float(np.mean(trades_duration)) if trades_duration else 0.0
    return win_rate, profit_factor, sharpe, max_dd


def _simulate_trades_on_test(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    pair: str,
) -> tuple[list[float], list[int]]:
    """
    Simulate trades on the test set given predicted signals.

    Args:
        test_df: OHLCV DataFrame for the test period.
        predictions: Array of signals (1=BUY, 0=HOLD, -1=SELL) aligned with test_df index.
        pair: Currency pair (for pip size and spread).

    Returns:
        (list_of_pnl_usd, list_of_duration_bars)
    """
    pip = PIP_SIZE.get(pair, 0.0001)
    spread = SPREAD_PIPS.get(pair, 1.5) * pip
    pnl_list: list[float] = []
    duration_list: list[int] = []

    closes = test_df["Close"].values
    n = len(closes)

    entry_idx: Optional[int] = None
    entry_price: Optional[float] = None
    direction: Optional[int] = None

    for i, signal in enumerate(predictions):
        # Close open position if exists
        if entry_idx is not None and direction is not None and entry_price is not None:
            bars_open = i - entry_idx
            if bars_open >= LABEL_FORWARD_BARS or signal != direction:
                exit_price = closes[min(i, n - 1)] - (spread / 2 if direction == 1 else -spread / 2)
                if direction == 1:
                    pnl_pips = (exit_price - entry_price) / pip
                else:
                    pnl_pips = (entry_price - exit_price) / pip
                pip_value = pip / exit_price * 1000  # $1000 position
                pnl_list.append(pnl_pips * pip_value)
                duration_list.append(bars_open)
                entry_idx = None
                entry_price = None
                direction = None

        # Open new position
        if signal != 0 and entry_idx is None and i < n - 1:
            entry_price = closes[i] + (spread / 2 if signal == 1 else -spread / 2)
            entry_idx = i
            direction = signal

    # Close any remaining open position at end of period
    if entry_idx is not None and direction is not None and entry_price is not None:
        exit_price = closes[-1]
        pip_val = pip / exit_price * 1000
        if direction == 1:
            pnl = (exit_price - entry_price) / pip * pip_val
        else:
            pnl = (entry_price - exit_price) / pip * pip_val
        pnl_list.append(pnl)
        duration_list.append(n - entry_idx)

    return pnl_list, duration_list


def run_walk_forward_validation(
    df: pd.DataFrame,
    pair: str,
    feature_cols: list[str],
    model_trainer,          # callable(df_train, pair, feature_cols) → fitted model
    model_predictor,        # callable(model, X) → np.ndarray of signals
) -> ValidationReport:
    """
    Run 5-fold walk-forward validation and return a ValidationReport.

    Args:
        df: Full featured DataFrame with 'label' column.
        pair: Currency pair.
        feature_cols: Feature column names to use.
        model_trainer: Function to train a model on a training fold.
        model_predictor: Function to generate predictions from a trained model.

    Returns:
        ValidationReport with per-fold metrics, aggregates, and pass/fail verdict.
    """
    from forexbot.models.classical import build_labels

    df = df.copy()
    df["label"] = build_labels(df)
    df = df.dropna(subset=feature_cols + ["label"])

    n = len(df)
    if n < 500:
        logger.warning("%s: Only %d bars for validation — insufficient", pair, n)
        return ValidationReport(
            folds=[],
            mean_win_rate=0.0,
            mean_profit_factor=1.0,
            mean_sharpe=0.0,
            mean_max_drawdown=1.0,
            passed=False,
            fail_reasons=["Insufficient historical data for walk-forward validation"],
        )

    # Compute fold boundaries
    min_train = n // (WALK_FORWARD_FOLDS + 1)
    test_size = min_train

    fold_results: list[FoldResult] = []

    for fold in range(WALK_FORWARD_FOLDS):
        train_end = min_train + fold * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n)

        if test_start >= n:
            break

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        if len(train_df) < 100 or len(test_df) < 20:
            continue

        logger.info(
            "%s fold %d/%d: train=%d test=%d",
            pair, fold + 1, WALK_FORWARD_FOLDS, len(train_df), len(test_df),
        )

        try:
            model = model_trainer(train_df, pair, feature_cols)
            X_test = test_df[feature_cols].values.astype(np.float32)
            X_test = np.nan_to_num(X_test, nan=0.0)
            raw_preds = model_predictor(model, X_test)
            # Convert class indices back to signals
            _idx2sig = {0: 1, 1: 0, 2: -1}
            predictions = np.array([_idx2sig.get(int(p), 0) for p in raw_preds])
        except Exception as exc:
            logger.error("%s fold %d training/prediction failed: %s", pair, fold + 1, exc)
            continue

        pnl_list, dur_list = _simulate_trades_on_test(test_df, predictions, pair)
        win_rate, pf, sharpe, max_dd = _compute_fold_metrics(
            pnl_list, dur_list, STARTING_BALANCE
        )

        fold_results.append(FoldResult(
            fold_num=fold + 1,
            train_bars=len(train_df),
            test_bars=len(test_df),
            total_trades=len(pnl_list),
            win_rate=win_rate,
            profit_factor=pf,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            avg_duration_bars=float(np.mean(dur_list)) if dur_list else 0.0,
        ))

    if not fold_results:
        return ValidationReport(
            folds=[],
            mean_win_rate=0.0,
            mean_profit_factor=1.0,
            mean_sharpe=0.0,
            mean_max_drawdown=1.0,
            passed=False,
            fail_reasons=["No folds completed successfully"],
        )

    mean_wr = float(np.mean([f.win_rate for f in fold_results]))
    mean_pf = float(np.mean([f.profit_factor for f in fold_results]))
    mean_sh = float(np.mean([f.sharpe_ratio for f in fold_results]))
    mean_dd = float(np.mean([f.max_drawdown for f in fold_results]))

    fail_reasons: list[str] = []
    if mean_wr < VALIDATION_MIN_WIN_RATE:
        fail_reasons.append(f"Win rate {mean_wr:.2%} < {VALIDATION_MIN_WIN_RATE:.2%}")
    if mean_pf < VALIDATION_MIN_PROFIT_FACTOR:
        fail_reasons.append(f"Profit factor {mean_pf:.2f} < {VALIDATION_MIN_PROFIT_FACTOR}")
    if mean_dd > VALIDATION_MAX_DRAWDOWN:
        fail_reasons.append(f"Max drawdown {mean_dd:.2%} > {VALIDATION_MAX_DRAWDOWN:.2%}")
    if mean_sh < VALIDATION_MIN_SHARPE:
        fail_reasons.append(f"Sharpe ratio {mean_sh:.2f} < {VALIDATION_MIN_SHARPE}")

    return ValidationReport(
        folds=fold_results,
        mean_win_rate=mean_wr,
        mean_profit_factor=mean_pf,
        mean_sharpe=mean_sh,
        mean_max_drawdown=mean_dd,
        passed=len(fail_reasons) == 0,
        fail_reasons=fail_reasons,
    )
