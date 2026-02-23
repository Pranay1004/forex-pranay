"""
display/terminal.py — Rich-based foreground terminal dashboard.
Renders a live updating UI with balance, signals, open positions, recent trades,
sentiment, and model confidence panels. No background threads.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from forexbot.config import (
    DISPLAY_REFRESH_SECONDS,
    PAIRS,
    RECENT_TRADES_SHOWN,
    REGIME_LABELS,
    VERSION,
)
from forexbot.features.regime import regime_label
from forexbot.models.ensemble import BUY, SELL
from forexbot.paper_trading.portfolio import Portfolio, Position
from forexbot.paper_trading.engine import PaperTradingEngine

logger = logging.getLogger(__name__)


def _signal_badge(direction: int, confidence: float) -> Text:
    """Return a styled Rich Text for the signal cell."""
    if direction == BUY:
        return Text(f"BUY {confidence:.0%}", style="bold green")
    elif direction == SELL:
        return Text(f"SELL {confidence:.0%}", style="bold red")
    else:
        return Text("HOLD", style="dim white")


def _pnl_text(pnl_usd: float, pnl_pips: Optional[float] = None) -> Text:
    """Return coloured P&L text."""
    pip_part = f" ({pnl_pips:+.1f}p)" if pnl_pips is not None else ""
    if pnl_usd >= 0:
        return Text(f"+${pnl_usd:.2f}{pip_part}", style="green")
    else:
        return Text(f"-${abs(pnl_usd):.2f}{pip_part}", style="red")


def build_header_panel(
    portfolio: Portfolio,
    start_time: datetime,
    regimes: dict[str, int],
) -> Panel:
    """Build the top header panel with balance and regime summary."""
    now = datetime.now(tz=timezone.utc)
    elapsed = now - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))
    today_pnl = portfolio.today_pnl()
    pnl_text = f"+${today_pnl:.2f}" if today_pnl >= 0 else f"-${abs(today_pnl):.2f}"
    pnl_style = "green" if today_pnl >= 0 else "red"

    regime_parts = []
    for pair, r in list(regimes.items())[:5]:
        label = {0: "RANGING", 1: "TREND↑", 2: "TREND↓"}.get(r, "?")
        style = {"RANGING": "yellow", "TREND↑": "green", "TREND↓": "red"}.get(label, "white")
        regime_parts.append(f"[{style}]{pair}={label}[/{style}]")

    regime_line = "  ".join(regime_parts)

    header_text = (
        f"[bold cyan]FOREXBOT v{VERSION}[/bold cyan]  |  "
        f"Balance: [bold white]${portfolio.balance:,.2f}[/bold white]  |  "
        f"P&L today: [{pnl_style}]{pnl_text}[/{pnl_style}]\n"
        f"Open: {len(portfolio.open_positions)}  |  "
        f"Closed today: {sum(1 for t in portfolio.closed_trades if t.exit_time.date() == now.date())}\n"
        f"Regime: {regime_line}"
    )
    return Panel(header_text, title="[bold]FOREXBOT LIVE DASHBOARD[/bold]", border_style="cyan")


def build_signals_table(
    signals_display: dict[str, tuple[int, float]],
    open_positions: dict,
) -> Table:
    """Build the signals panel table."""
    table = Table(title="SIGNALS", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("PAIR", style="bold white", width=10)
    table.add_column("SIGNAL", width=14)
    table.add_column("STATUS", width=8)

    open_pairs = {p.pair for p in open_positions.values()}

    for pair in PAIRS:
        direction, confidence = signals_display.get(pair, (0, 0.0))
        status = "[OPEN]" if pair in open_pairs else ""
        table.add_row(pair, _signal_badge(direction, confidence), Text(status, style="cyan"))

    return table


def build_positions_table(open_positions: dict, current_prices: dict[str, float]) -> Table:
    """Build the open positions panel table."""
    table = Table(title="OPEN POSITIONS", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("PAIR", width=8)
    table.add_column("DIR", width=6)
    table.add_column("ENTRY", width=10)
    table.add_column("P&L", width=14)
    table.add_column("BARS", width=5)

    for pos in list(open_positions.values())[:8]:
        price = current_prices.get(pos.pair, pos.entry_price)
        pnl_pips = pos.current_pnl_pips(price)
        pnl_usd = pos.current_pnl_usd(price)
        direction_text = Text("LONG", style="green") if pos.direction == 1 else Text("SHORT", style="red")
        table.add_row(
            pos.pair,
            direction_text,
            f"{pos.entry_price:.5f}",
            _pnl_text(pnl_usd, pnl_pips),
            str(pos.bars_open),
        )

    if not open_positions:
        table.add_row("—", "—", "—", "—", "—")

    return table


def build_recent_trades_table(portfolio: Portfolio) -> Table:
    """Build the recent trades history table."""
    table = Table(title=f"RECENT TRADES (last {RECENT_TRADES_SHOWN})", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("PAIR", width=8)
    table.add_column("DIR", width=6)
    table.add_column("P&L", width=16)
    table.add_column("DUR", width=6)
    table.add_column("CONF", width=6)
    table.add_column("EXIT", width=10)

    recent = portfolio.closed_trades[-RECENT_TRADES_SHOWN:]
    for t in reversed(recent):
        direction_text = Text("L", style="green") if t.direction == 1 else Text("S", style="red")
        table.add_row(
            t.pair,
            direction_text,
            _pnl_text(t.pnl_usd, t.pnl_pips),
            f"{t.duration_bars}h",
            f"{t.model_confidence:.2f}",
            t.exit_reason,
        )

    if not portfolio.closed_trades:
        table.add_row("—", "—", "—", "—", "—", "—")

    return table


def build_sentiment_panel(sentiment_scores: dict[str, float]) -> Panel:
    """Build the sentiment scores panel."""
    parts = []
    for pair, score in sorted(sentiment_scores.items()):
        if score > 0.2:
            label, style = "Bullish", "green"
        elif score < -0.2:
            label, style = "Bearish", "red"
        else:
            label, style = "Neutral", "yellow"
        parts.append(f"[{style}]{pair}: {label} {score:+.2f}[/{style}]")
    line = "   ".join(parts) if parts else "No sentiment data"
    return Panel(line, title="SENTIMENT", border_style="magenta")


def build_model_confidence_panel(
    model_probas_by_pair: dict[str, dict[str, list]],
    focused_pair: Optional[str] = None,
) -> Panel:
    """Build a panel showing model probability breakdowns for a selected pair."""
    if focused_pair is None:
        focused_pair = PAIRS[0]
    probas = model_probas_by_pair.get(focused_pair, {})
    lines = []
    for model_name, proba in probas.items():
        if len(proba) == 3:
            lines.append(
                f"[bold]{model_name:10s}[/bold]: "
                f"BUY=[green]{proba[0]:.2f}[/green] "
                f"HOLD={proba[1]:.2f} "
                f"SELL=[red]{proba[2]:.2f}[/red]"
            )
    content = "\n".join(lines) if lines else "No model data"
    return Panel(content, title=f"MODEL CONFIDENCE ({focused_pair})", border_style="blue")


def build_footer_panel(start_time: datetime, seconds_to_next: int) -> Panel:
    """Build the bottom footer bar."""
    now = datetime.now(tz=timezone.utc)
    elapsed = now - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))
    mins, secs = divmod(seconds_to_next, 60)
    footer = (
        f"[dim]ELAPSED: {elapsed_str}[/dim]  |  "
        f"[bold yellow]NEXT CANDLE: {mins:02d}:{secs:02d}[/bold yellow]  |  "
        f"[dim]Ctrl+C to Stop  |  RefreshEvery {DISPLAY_REFRESH_SECONDS}s[/dim]"
    )
    return Panel(footer, border_style="dim")


def render_full_layout(
    portfolio: Portfolio,
    start_time: datetime,
    regimes: dict[str, int],
    signals_display: dict[str, tuple[int, float]],
    current_prices: dict[str, float],
    sentiment_scores: dict[str, float],
    model_probas_by_pair: dict[str, dict[str, list]],
    seconds_to_next: int,
    focused_pair: Optional[str] = None,
) -> Layout:
    """
    Build the complete Rich Layout for one display refresh.

    Args:
        portfolio: Current portfolio state.
        start_time: Bot startup time.
        regimes: {pair: regime_int}.
        signals_display: {pair: (direction, confidence)}.
        current_prices: {pair: latest_price}.
        sentiment_scores: {pair: sentiment_float}.
        model_probas_by_pair: {pair: {model: [buy,hold,sell]}}.
        seconds_to_next: Seconds until next candle.
        focused_pair: Which pair to show in model confidence panel.

    Returns:
        Fully populated Rich Layout.
    """
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="middle", size=16),
        Layout(name="recent_trades", size=14),
        Layout(name="sentiment_model", size=8),
        Layout(name="footer", size=3),
    )

    layout["header"].update(build_header_panel(portfolio, start_time, regimes))

    layout["middle"].split_row(
        Layout(name="signals", ratio=1),
        Layout(name="positions", ratio=2),
    )
    layout["middle"]["signals"].update(
        Panel(build_signals_table(signals_display, portfolio.open_positions), border_style="white")
    )
    layout["middle"]["positions"].update(
        Panel(build_positions_table(portfolio.open_positions, current_prices), border_style="white")
    )

    layout["recent_trades"].update(
        Panel(build_recent_trades_table(portfolio), border_style="white")
    )

    layout["sentiment_model"].split_row(
        Layout(name="sentiment", ratio=1),
        Layout(name="model_conf", ratio=1),
    )
    layout["sentiment_model"]["sentiment"].update(build_sentiment_panel(sentiment_scores))
    layout["sentiment_model"]["model_conf"].update(
        build_model_confidence_panel(model_probas_by_pair, focused_pair)
    )

    layout["footer"].update(build_footer_panel(start_time, seconds_to_next))

    return layout


def print_banner(console: Console) -> None:
    """Print the startup banner to the console."""
    banner = """
┌──────────────────────────────────────────────────────────────┐
│  FOREXBOT  v{version}  |  Paper Trading Mode                  │
│  Pairs: 10  |  Models: FinBERT + TabPFN + XGB + LGB         │
│  Press Ctrl+C to stop                                        │
└──────────────────────────────────────────────────────────────┘
""".format(version=VERSION)
    console.print(banner, style="bold cyan")


def print_validation_results(console: Console, report, pair: str) -> None:
    """Display walk-forward validation results in a Rich table."""
    from forexbot.paper_trading.validator import ValidationReport, FoldResult

    table = Table(title=f"Walk-Forward Validation: {pair}", box=box.ROUNDED)
    table.add_column("Fold", style="bold")
    table.add_column("Train", justify="right")
    table.add_column("Test", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Prof.Factor", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("MaxDD", justify="right")

    for fold in report.folds:
        wr_style = "green" if fold.win_rate >= 0.48 else "red"
        pf_style = "green" if fold.profit_factor >= 1.15 else "red"
        table.add_row(
            str(fold.fold_num),
            str(fold.train_bars),
            str(fold.test_bars),
            str(fold.total_trades),
            f"[{wr_style}]{fold.win_rate:.2%}[/{wr_style}]",
            f"[{pf_style}]{fold.profit_factor:.2f}[/{pf_style}]",
            f"{fold.sharpe_ratio:.2f}",
            f"{fold.max_drawdown:.2%}",
        )

    console.print(table)

    verdict_style = "bold green" if report.passed else "bold red"
    verdict_text = "PASSED ✓" if report.passed else "FAILED ✗"
    console.print(f"\nValidation result: [{verdict_style}]{verdict_text}[/{verdict_style}]")
    if not report.passed:
        for reason in report.fail_reasons:
            console.print(f"  • {reason}", style="red")
