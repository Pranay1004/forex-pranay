"""
strategy/filters.py — Pre-trade filters: session window, news blackout, spread check.
All filters return (allowed: bool, reason: str).
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from forexbot.config import (
    CONFIDENCE_THRESHOLD,
    SESSION_WINDOWS,
    SPREAD_PIPS,
)
from forexbot.data.news import NewsStore, is_news_blackout

logger = logging.getLogger(__name__)


def session_filter(pair: str, utc_hour: Optional[int] = None) -> tuple[bool, str]:
    """
    Check if the current UTC hour is within the active trading session for the pair.

    Args:
        pair: Currency pair e.g. "EURUSD".
        utc_hour: Override hour for testing; defaults to current UTC hour.

    Returns:
        (allowed, reason_string)
    """
    if utc_hour is None:
        utc_hour = datetime.now(tz=timezone.utc).hour

    window = SESSION_WINDOWS.get(pair)
    if window is None:
        return True, ""   # Unknown pair — allow by default

    start_h, end_h = window
    if start_h < end_h:
        in_session = start_h <= utc_hour < end_h
    else:
        # Wraps midnight (e.g. 21:00–06:00)
        in_session = utc_hour >= start_h or utc_hour < end_h

    if not in_session:
        return False, f"Outside session window {start_h:02d}:00–{end_h:02d}:00 UTC"
    return True, ""


def news_blackout_filter(
    pair: str,
    news_store: NewsStore,
) -> tuple[bool, str]:
    """
    Block trading if a high-impact event is imminent for the pair's currencies.

    Args:
        pair: Currency pair.
        news_store: Current NewsStore with calendar events.

    Returns:
        (allowed, reason_string) — allowed=False means blocked.
    """
    blackout, reason = is_news_blackout(news_store, pair)
    if blackout:
        logger.info("%s: News blackout — %s", pair, reason)
        return False, f"News blackout: {reason}"
    return True, ""


def spread_filter(
    pair: str,
    current_spread_pips: Optional[float] = None,
    max_spread_multiplier: float = 2.0,
) -> tuple[bool, str]:
    """
    Reject trade if spread exceeds acceptable threshold.
    Uses configured typical spread × multiplier as maximum allowed spread.

    Args:
        pair: Currency pair.
        current_spread_pips: Actual current spread in pips. If None, accept.
        max_spread_multiplier: Allow up to this multiple of typical spread.

    Returns:
        (allowed, reason_string)
    """
    typical = SPREAD_PIPS.get(pair, 2.0)
    max_allowed = typical * max_spread_multiplier

    if current_spread_pips is None:
        return True, ""

    if current_spread_pips > max_allowed:
        return (
            False,
            f"Spread {current_spread_pips:.1f} pips exceeds max {max_allowed:.1f} pips",
        )
    return True, ""


def volatility_filter(
    pair: str,
    natr: float,
    min_natr: float = 0.0001,
    max_natr: float = 0.03,
) -> tuple[bool, str]:
    """
    Block trading in extremely low or extremely high volatility.

    Args:
        pair: Currency pair.
        natr: Current Normalised ATR value.
        min_natr: Minimum acceptable NATR (too quiet).
        max_natr: Maximum acceptable NATR (too volatile / news spike).

    Returns:
        (allowed, reason_string)
    """
    if not np.isfinite(natr) or natr <= 0:
        return True, ""   # missing data → allow

    if natr < min_natr:
        return False, f"NATR {natr:.5f} below min {min_natr:.5f} (market too quiet)"
    if natr > max_natr:
        return False, f"NATR {natr:.5f} above max {max_natr:.5f} (extreme volatility)"
    return True, ""


def apply_all_filters(
    pair: str,
    news_store: NewsStore,
    natr: float = 0.001,
    current_spread_pips: Optional[float] = None,
) -> tuple[bool, str]:
    """
    Apply all pre-trade filters in sequence.

    Args:
        pair: Currency pair.
        news_store: NewsStore with events.
        natr: Current NATR value.
        current_spread_pips: Optional real-time spread.

    Returns:
        (allowed: bool, rejection_reason: str)
    """
    checks = [
        session_filter(pair),
        news_blackout_filter(pair, news_store),
        spread_filter(pair, current_spread_pips),
        volatility_filter(pair, natr),
    ]

    for allowed, reason in checks:
        if not allowed:
            logger.info("%s: FILTER BLOCKED — %s", pair, reason)
            return False, reason

    return True, ""
