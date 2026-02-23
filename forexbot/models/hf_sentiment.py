"""
models/hf_sentiment.py — FinBERT + VADER sentiment scoring for FOREX pairs.
Loads ProsusAI/finbert once at startup. Results are cached for 1 hour.
Final score: 0.6 * finbert_score + 0.4 * vader_score  ∈ [-1, +1].
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from forexbot.config import (
    FINBERT_MODEL,
    NEWS_CACHE_SECONDS,
    PAIRS,
    SENTIMENT_FINBERT_WEIGHT,
    SENTIMENT_VADER_WEIGHT,
)

logger = logging.getLogger(__name__)


# ─── VADER helper ─────────────────────────────────────────────────────────────

def _vader_score(texts: list[str]) -> float:
    """
    Aggregate VADER compound scores across a list of texts.

    Args:
        texts: List of headline strings.

    Returns:
        Mean compound score ∈ [-1, +1].
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(t)["compound"] for t in texts if t]
        return float(np.mean(scores)) if scores else 0.0
    except Exception as exc:
        logger.warning("VADER error: %s", exc)
        return 0.0


# ─── FinBERT wrapper ──────────────────────────────────────────────────────────

@dataclass
class SentimentEngine:
    """
    Wraps FinBERT and VADER for headline sentiment scoring.

    Attributes:
        _pipeline: HuggingFace text-classification pipeline (loaded lazily).
        _cache: {pair: (score, timestamp)} cached sentiment results.
    """

    _pipeline: Optional[object] = field(default=None, repr=False)
    _cache: dict[str, tuple[float, float]] = field(default_factory=dict)
    _pipeline_loaded: bool = False

    def load(self) -> None:
        """
        Load the FinBERT pipeline. Call once at startup.
        Gracefully degrades to VADER-only if transformers/torch are unavailable.
        """
        if self._pipeline_loaded:
            return
        try:
            from transformers import pipeline  # type: ignore
            logger.info("Loading FinBERT model (%s) …", FINBERT_MODEL)
            self._pipeline = pipeline(
                "text-classification",
                model=FINBERT_MODEL,
                tokenizer=FINBERT_MODEL,
                top_k=None,          # return all classes
                device=-1,           # CPU
            )
            self._pipeline_loaded = True
            logger.info("FinBERT loaded successfully")
        except Exception as exc:
            logger.warning("FinBERT load failed (%s); will use VADER only", exc)
            self._pipeline = None
            self._pipeline_loaded = True   # don't retry on every call

    def _finbert_score(self, texts: list[str]) -> float:
        """
        Run FinBERT on a batch of texts and return aggregate score ∈ [-1, +1].

        Args:
            texts: Truncated headline strings.

        Returns:
            Weighted aggregate: sum(pos) - sum(neg) / n_texts.
        """
        if self._pipeline is None or not texts:
            return 0.0

        # Truncate each text to 512 chars (safe for tokenizer limit)
        truncated = [t[:512] for t in texts if t]
        if not truncated:
            return 0.0

        try:
            results = self._pipeline(truncated, batch_size=8, truncation=True, max_length=512)
        except Exception as exc:
            logger.warning("FinBERT inference error: %s", exc)
            return 0.0

        pos_total = neg_total = 0.0
        n = 0
        for result in results:
            # result is a list of dicts [{label, score}, ...]
            if not isinstance(result, list):
                result = [result]
            label_scores: dict[str, float] = {}
            for item in result:
                label_scores[item["label"].lower()] = float(item["score"])
            pos_total += label_scores.get("positive", 0.0)
            neg_total += label_scores.get("negative", 0.0)
            n += 1

        if n == 0:
            return 0.0
        raw = (pos_total - neg_total) / n
        return float(np.clip(raw, -1.0, 1.0))

    def score(self, pair: str, headlines: list[str]) -> float:
        """
        Return the weighted combined sentiment score for a pair.
        Result is cached for NEWS_CACHE_SECONDS.

        Args:
            pair: Currency pair e.g. "EURUSD".
            headlines: List of raw headline strings (up to 20).

        Returns:
            Combined sentiment score ∈ [-1, +1].
        """
        # Check cache
        if pair in self._cache:
            cached_score, ts = self._cache[pair]
            if (time.time() - ts) < NEWS_CACHE_SECONDS:
                return cached_score

        if not headlines:
            self._cache[pair] = (0.0, time.time())
            return 0.0

        finbert = self._finbert_score(headlines[:20])
        vader = _vader_score(headlines[:20])

        combined = (
            SENTIMENT_FINBERT_WEIGHT * finbert
            + SENTIMENT_VADER_WEIGHT * vader
        )
        combined = float(np.clip(combined, -1.0, 1.0))

        self._cache[pair] = (combined, time.time())
        logger.debug(
            "Sentiment [%s]: finbert=%.3f vader=%.3f combined=%.3f",
            pair, finbert, vader, combined,
        )
        return combined

    def score_all_pairs(
        self,
        headlines_by_pair: dict[str, list[str]],
    ) -> dict[str, float]:
        """
        Score all pairs in one call, reusing cache where possible.

        Args:
            headlines_by_pair: {pair: [headline_str, ...]}

        Returns:
            {pair: sentiment_score}
        """
        return {
            pair: self.score(pair, headlines_by_pair.get(pair, []))
            for pair in PAIRS
        }

    def sentiment_to_class_probs(self, score: float) -> np.ndarray:
        """
        Map a sentiment score ∈ [-1, +1] to a (BUY, HOLD, SELL) probability vector.

        Args:
            score: Combined sentiment score.

        Returns:
            ndarray of shape (3,) summing to 1.0.
        """
        # Soft mapping: positive → BUY tendency, negative → SELL tendency
        buy_prob = float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
        sell_prob = float(np.clip((-score + 1.0) / 2.0, 0.0, 1.0))
        # Normalise so probabilities sum to 1
        total = buy_prob + sell_prob + 1e-9   # HOLD is implicit
        # Carve out HOLD from the centre
        hold_prob = max(0.0, 1.0 - abs(score))
        buy_prob = max(0.0, (score + 1.0) / 2.0 - hold_prob / 2.0)
        sell_prob = max(0.0, (-score + 1.0) / 2.0 - hold_prob / 2.0)
        arr = np.array([buy_prob, hold_prob, sell_prob], dtype=float)
        arr = arr / arr.sum()
        return arr
