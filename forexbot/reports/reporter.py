"""
reports/reporter.py — Daily HTML performance report generator.
Produces a self-contained HTML file with equity curve, trade stats, and per-pair breakdown.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from forexbot.config import PAIRS, REPORTS_DIR, STARTING_BALANCE, VERSION
from forexbot.paper_trading.portfolio import ClosedTrade, Portfolio

logger = logging.getLogger(__name__)


def _trades_to_rows(trades: list[ClosedTrade]) -> str:
    """Convert a list of closed trades to HTML table rows."""
    rows = []
    for t in trades:
        direction = "LONG" if t.direction == 1 else "SHORT"
        pnl_style = "color:green" if t.pnl_usd >= 0 else "color:red"
        pnl_str = f"+${t.pnl_usd:.2f} ({t.pnl_pips:+.1f}p)" if t.pnl_usd >= 0 \
            else f"-${abs(t.pnl_usd):.2f} ({t.pnl_pips:.1f}p)"
        rows.append(
            f"<tr>"
            f"<td>{t.trade_id}</td>"
            f"<td>{t.pair}</td>"
            f"<td>{direction}</td>"
            f"<td>{t.entry_price:.5f}</td>"
            f"<td>{t.exit_price:.5f}</td>"
            f"<td style='{pnl_style}'>{pnl_str}</td>"
            f"<td>{t.duration_bars}h</td>"
            f"<td>{t.exit_reason}</td>"
            f"<td>{t.model_confidence:.2f}</td>"
            f"<td>{t.entry_time.strftime('%H:%M')}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _equity_curve_svg(trades: list[ClosedTrade], width: int = 800, height: int = 200) -> str:
    """Generate a simple inline SVG equity curve."""
    if not trades:
        return "<p>No trades yet.</p>"

    equity = [STARTING_BALANCE]
    for t in trades:
        equity.append(equity[-1] + t.pnl_usd)

    min_eq = min(equity)
    max_eq = max(equity)
    eq_range = max_eq - min_eq if max_eq != min_eq else 1.0

    def _x(i: int) -> float:
        return i / max(len(equity) - 1, 1) * (width - 40) + 20

    def _y(val: float) -> float:
        return height - 20 - ((val - min_eq) / eq_range) * (height - 40)

    points = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(equity))
    final_color = "green" if equity[-1] >= STARTING_BALANCE else "red"

    return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="#1a1a2e"/>
  <polyline points="{points}" fill="none" stroke="{final_color}" stroke-width="2"/>
  <text x="20" y="15" fill="#aaa" font-size="11">Equity: ${equity[-1]:,.2f}</text>
  <text x="20" y="{height-5}" fill="#aaa" font-size="10">Start: ${equity[0]:,.2f}</text>
</svg>"""


def _per_pair_stats(trades: list[ClosedTrade]) -> str:
    """Generate HTML table with per-pair performance stats."""
    by_pair: dict[str, list[ClosedTrade]] = {p: [] for p in PAIRS}
    for t in trades:
        if t.pair in by_pair:
            by_pair[t.pair].append(t)

    rows = []
    for pair, pair_trades in by_pair.items():
        if not pair_trades:
            rows.append(f"<tr><td>{pair}</td><td colspan='5'>—</td></tr>")
            continue
        total = len(pair_trades)
        wins = sum(1 for t in pair_trades if t.pnl_usd > 0)
        wr = wins / total if total > 0 else 0
        total_pnl = sum(t.pnl_usd for t in pair_trades)
        avg_conf = np.mean([t.model_confidence for t in pair_trades])
        pnl_style = "color:green" if total_pnl >= 0 else "color:red"
        rows.append(
            f"<tr>"
            f"<td>{pair}</td>"
            f"<td>{total}</td>"
            f"<td>{wr:.1%}</td>"
            f"<td style='{pnl_style}'>{total_pnl:+.2f}</td>"
            f"<td>{avg_conf:.2f}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def generate_daily_report(portfolio: Portfolio, report_date: Optional[datetime] = None) -> Path:
    """
    Generate a self-contained daily HTML performance report.

    Args:
        portfolio: Current Portfolio instance.
        report_date: Date of the report (defaults to today UTC).

    Returns:
        Path to the saved HTML file.
    """
    if report_date is None:
        report_date = datetime.now(tz=timezone.utc)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = REPORTS_DIR / f"report_{report_date.strftime('%Y-%m-%d')}.html"

    today_trades = [
        t for t in portfolio.closed_trades
        if t.exit_time.date() == report_date.date()
    ]

    all_trades = portfolio.closed_trades
    win_rate = portfolio.recent_win_rate(len(all_trades) or 1)
    pf = portfolio.profit_factor()
    max_dd = portfolio.max_drawdown()

    eq_svg = _equity_curve_svg(all_trades)
    trade_rows = _trades_to_rows(today_trades)
    pair_stats = _per_pair_stats(all_trades)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ForexBot Daily Report — {report_date.strftime('%Y-%m-%d')}</title>
<style>
  body {{ background:#1a1a2e; color:#e0e0e0; font-family:monospace; padding:20px; }}
  h1,h2 {{ color:#00d4ff; }}
  table {{ border-collapse:collapse; width:100%; margin-bottom:20px; }}
  th {{ background:#0f3460; color:#e0e0e0; padding:8px; text-align:left; }}
  td {{ padding:6px; border-bottom:1px solid #333; }}
  .stat {{ display:inline-block; margin:10px 20px; }}
  .stat-value {{ font-size:1.5em; font-weight:bold; color:#00d4ff; }}
  .stat-label {{ font-size:0.8em; color:#aaa; }}
</style>
</head>
<body>
<h1>🤖 ForexBot v{VERSION} — Daily Report</h1>
<p>Generated: {report_date.strftime('%Y-%m-%d %H:%M UTC')}</p>

<div>
  <div class="stat">
    <div class="stat-value">${portfolio.balance:,.2f}</div>
    <div class="stat-label">Current Balance</div>
  </div>
  <div class="stat">
    <div class="stat-value" style="color:{'#00ff88' if portfolio.today_pnl()>=0 else '#ff4444'}">
      {portfolio.today_pnl():+.2f}
    </div>
    <div class="stat-label">Today P&L (USD)</div>
  </div>
  <div class="stat">
    <div class="stat-value">{win_rate:.1%}</div>
    <div class="stat-label">Win Rate (all-time)</div>
  </div>
  <div class="stat">
    <div class="stat-value">{pf:.2f}</div>
    <div class="stat-label">Profit Factor</div>
  </div>
  <div class="stat">
    <div class="stat-value" style="color:#ff9944">{max_dd:.1%}</div>
    <div class="stat-label">Max Drawdown</div>
  </div>
  <div class="stat">
    <div class="stat-value">{len(all_trades)}</div>
    <div class="stat-label">Total Trades</div>
  </div>
</div>

<h2>Equity Curve</h2>
{eq_svg}

<h2>Per-Pair Performance</h2>
<table>
  <tr><th>Pair</th><th>Trades</th><th>Win Rate</th><th>P&L (USD)</th><th>Avg Conf</th></tr>
  {pair_stats}
</table>

<h2>Today's Trades ({len(today_trades)})</h2>
<table>
  <tr>
    <th>ID</th><th>Pair</th><th>Dir</th><th>Entry</th><th>Exit</th>
    <th>P&L</th><th>Dur</th><th>Exit</th><th>Conf</th><th>Time</th>
  </tr>
  {trade_rows if trade_rows else '<tr><td colspan="10">No trades today</td></tr>'}
</table>

<p style="color:#555; font-size:0.8em; margin-top:40px;">
  ForexBot v{VERSION} — Paper Trading Only — Not financial advice
</p>
</body>
</html>"""

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Daily report saved: %s", filename)
    except Exception as exc:
        logger.error("Failed to save report: %s", exc)

    return filename
