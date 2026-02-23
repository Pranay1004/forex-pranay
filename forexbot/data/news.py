"""
data/news.py — Economic news fetching and Forex Factory calendar parsing.
Sources: NewsAPI (free tier) + ForexFactory XML calendar.
Provides headline caching, news blackout detection, and raw headline delivery
to the sentiment pipeline.
"""

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

from forexbot.config import (
    CURRENCY_KEYWORDS,
    NEWS_BLACKOUT_MINUTES_AFTER,
    NEWS_BLACKOUT_MINUTES_BEFORE,
    NEWS_CACHE_SECONDS,
    NEWSAPI_KEY,
    PAIRS,
)

logger = logging.getLogger(__name__)

FF_CALENDAR_URL = "https://www.forexfactory.com/ff_calendar_thisweek.xml"
NEWSAPI_BASE = "https://newsapi.org/v2/everything"

# Currency codes referenced in pair names
_ALL_CURRENCIES = ["EUR", "USD", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"]


@dataclass
class NewsHeadline:
    """Single news headline."""
    title: str
    source: str
    published_at: datetime
    url: str


@dataclass
class ForexEvent:
    """Single Forex Factory calendar event."""
    title: str
    currency: str
    impact: str          # "High", "Medium", "Low"
    event_dt: datetime


@dataclass
class NewsStore:
    """In-memory store with TTL caching for headlines and calendar events."""
    headlines: dict[str, list[NewsHeadline]] = field(default_factory=dict)
    events: list[ForexEvent] = field(default_factory=list)
    _last_headline_fetch: float = 0.0
    _last_event_fetch: float = 0.0

    def headlines_stale(self) -> bool:
        """Return True if headlines cache is expired."""
        return (time.time() - self._last_headline_fetch) > NEWS_CACHE_SECONDS

    def events_stale(self) -> bool:
        """Return True if events cache is expired."""
        return (time.time() - self._last_event_fetch) > NEWS_CACHE_SECONDS


def _currencies_for_pair(pair: str) -> list[str]:
    """Extract the two currency codes from a pair string."""
    for c1 in _ALL_CURRENCIES:
        for c2 in _ALL_CURRENCIES:
            if pair == f"{c1}{c2}":
                return [c1, c2]
    return []


def _fetch_newsapi_headlines(currency: str, api_key: str) -> list[NewsHeadline]:
    """
    Fetch up to 20 recent headlines for a currency from NewsAPI.

    Args:
        currency: Three-letter currency code e.g. "EUR".
        api_key: NewsAPI key.

    Returns:
        List of NewsHeadline objects.
    """
    if not api_key:
        return []

    keywords = CURRENCY_KEYWORDS.get(currency, [currency])
    query = " OR ".join(f'"{kw}"' for kw in keywords[:4])
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(NEWSAPI_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        headlines: list[NewsHeadline] = []
        for art in data.get("articles", []):
            try:
                pub = datetime.fromisoformat(
                    art["publishedAt"].replace("Z", "+00:00")
                )
            except Exception:
                pub = datetime.now(tz=timezone.utc)
            headlines.append(
                NewsHeadline(
                    title=art.get("title") or "",
                    source=art.get("source", {}).get("name", ""),
                    published_at=pub,
                    url=art.get("url", ""),
                )
            )
        return headlines
    except Exception as exc:
        logger.warning("NewsAPI fetch error for %s: %s", currency, exc)
        return []


def _fetch_forexfactory_events() -> list[ForexEvent]:
    """
    Scrape ForexFactory's publicly available weekly XML calendar.

    Returns:
        List of ForexEvent objects (only High/Medium impact events).
    """
    events: list[ForexEvent] = []
    try:
        resp = requests.get(FF_CALENDAR_URL, timeout=15, headers={"User-Agent": "ForexBot/1.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        current_date: Optional[datetime] = None
        for item in root.findall("./channel/item"):
            title_el = item.find("title")
            date_el = item.find("date")
            time_el = item.find("time")
            currency_el = item.find("currency")
            impact_el = item.find("impact")

            if title_el is None or currency_el is None or impact_el is None:
                continue

            title = (title_el.text or "").strip()
            currency = (currency_el.text or "").strip().upper()
            impact = (impact_el.text or "").strip()

            if impact not in ("High", "Medium"):
                continue

            # Parse date/time — ForexFactory uses "Month DD, YYYY" / "H:MM[am/pm]"
            date_str = (date_el.text or "").strip() if date_el is not None else ""
            time_str = (time_el.text or "").strip() if time_el is not None else ""

            try:
                if date_str:
                    current_date = datetime.strptime(date_str, "%B %d, %Y").replace(
                        tzinfo=timezone.utc
                    )
                if current_date and time_str and time_str.lower() not in ("", "all day", "tentative"):
                    dt = datetime.strptime(f"{current_date.strftime('%Y-%m-%d')} {time_str}", "%Y-%m-%d %I:%M%p")
                    event_dt = dt.replace(tzinfo=timezone.utc)
                elif current_date:
                    event_dt = current_date
                else:
                    continue
            except Exception:
                event_dt = current_date or datetime.now(tz=timezone.utc)

            events.append(ForexEvent(title=title, currency=currency, impact=impact, event_dt=event_dt))

    except Exception as exc:
        logger.warning("ForexFactory calendar fetch error: %s", exc)

    return events


def refresh_headlines(store: NewsStore) -> None:
    """
    Refresh headline cache for all relevant currencies.

    Args:
        store: The shared NewsStore to update in-place.
    """
    if not store.headlines_stale():
        return

    currencies: set[str] = set()
    for pair in PAIRS:
        for c in _currencies_for_pair(pair):
            currencies.add(c)

    for currency in currencies:
        headlines = _fetch_newsapi_headlines(currency, NEWSAPI_KEY)
        store.headlines[currency] = headlines
        logger.debug("Fetched %d headlines for %s", len(headlines), currency)

    store._last_headline_fetch = time.time()


def refresh_events(store: NewsStore) -> None:
    """
    Refresh ForexFactory calendar events in the store.

    Args:
        store: The shared NewsStore to update in-place.
    """
    if not store.events_stale():
        return
    store.events = _fetch_forexfactory_events()
    store._last_event_fetch = time.time()
    logger.debug("Fetched %d ForexFactory events", len(store.events))


def get_headlines_for_pair(store: NewsStore, pair: str) -> list[NewsHeadline]:
    """
    Return combined headlines relevant to a pair (both currencies).

    Args:
        store: Current NewsStore.
        pair: Currency pair e.g. "EURUSD".

    Returns:
        List of up to 40 recent headlines.
    """
    currencies = _currencies_for_pair(pair)
    seen: set[str] = set()
    combined: list[NewsHeadline] = []
    for currency in currencies:
        for h in store.headlines.get(currency, []):
            if h.title not in seen:
                seen.add(h.title)
                combined.append(h)
    combined.sort(key=lambda h: h.published_at, reverse=True)
    return combined[:40]


def is_news_blackout(store: NewsStore, pair: str) -> tuple[bool, str]:
    """
    Check whether the current time falls within a news blackout window for a pair.

    Args:
        store: Current NewsStore.
        pair: Currency pair.

    Returns:
        (is_blackout: bool, reason: str)
    """
    currencies = _currencies_for_pair(pair)
    now = datetime.now(tz=timezone.utc)

    for event in store.events:
        if event.currency not in currencies:
            continue
        if event.impact != "High":
            continue
        minutes_diff = (event.event_dt - now).total_seconds() / 60
        if -NEWS_BLACKOUT_MINUTES_AFTER <= minutes_diff <= NEWS_BLACKOUT_MINUTES_BEFORE:
            return True, f"{event.title} @ {event.event_dt.strftime('%H:%M UTC')}"

    return False, ""


def get_recent_headlines_text(store: NewsStore, pair: str, n: int = 20) -> list[str]:
    """
    Return last n headline titles for a pair, used by sentiment models.

    Args:
        store: NewsStore.
        pair: Currency pair.
        n: Max number of headlines.

    Returns:
        List of headline strings.
    """
    headlines = get_headlines_for_pair(store, pair)
    return [h.title for h in headlines[:n] if h.title]
