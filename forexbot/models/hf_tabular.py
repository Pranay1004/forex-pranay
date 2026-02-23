"""
models/hf_tabular.py — TabPFN tabular transformer for FOREX signal classification.
Uses TabPFNClassifier (Prior-Fitted Networks, no hyperparameter tuning required).
Falls back to a Random Forest if TabPFN is unavailable.
Max 1000 training samples. PCA to 50 features if feature count exceeds limit.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from forexbot.config import (
    MODELS_DIR,
    TABPFN_MAX_FEATURES,
    TABPFN_MAX_SAMPLES,
)

logger = logging.getLogger(__name__)

# Class label mapping: BUY=1, HOLD=0, SELL=-1 → indices 0, 1, 2
_CLASS_INDEX = {1: 0, 0: 1, -1: 2}    # label → probability array index
_INDEX_CLASS = {0: 1, 1: 0, 2: -1}    # probability array index → label


@dataclass
class TabPFNWrapper:
    """
    Wrapper around TabPFNClassifier with PCA dimensionality reduction and fallback.

    Attributes:
        pair: Currency pair this model is trained for.
        classifier: The fitted TabPFN or fallback classifier.
        pca: PCA instance if applied, else None.
        is_fitted: Whether the classifier has been trained.
        n_features_in_: Number of features expected at inference time.
    """

    pair: str
    classifier: Optional[object] = field(default=None, repr=False)
    pca: Optional[PCA] = field(default=None, repr=False)
    is_fitted: bool = False
    n_features_in_: int = 0

    def _apply_pca(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply PCA dimensionality reduction if feature count exceeds limit."""
        if X.shape[1] <= TABPFN_MAX_FEATURES:
            return X
        if fit:
            self.pca = PCA(n_components=TABPFN_MAX_FEATURES, random_state=42)
            return self.pca.fit_transform(X)
        if self.pca is not None:
            return self.pca.transform(X)
        # Fallback: truncate
        return X[:, :TABPFN_MAX_FEATURES]

    def _build_classifier(self):
        """Instantiate TabPFNClassifier, falling back to RandomForestClassifier."""
        try:
            from tabpfn import TabPFNClassifier  # type: ignore
            clf = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
            logger.info("%s: Using TabPFNClassifier", self.pair)
            return clf
        except ImportError:
            logger.warning(
                "%s: tabpfn not installed; falling back to RandomForestClassifier", self.pair
            )
        except Exception as exc:
            logger.warning("%s: TabPFN init error (%s); falling back", self.pair, exc)

        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classifier on training data.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Integer labels: 1=BUY, 0=HOLD, -1=SELL.
        """
        if len(X) == 0:
            logger.warning("%s: Empty training set for TabPFN", self.pair)
            return

        # Subsample to TABPFN_MAX_SAMPLES most recent bars
        if len(X) > TABPFN_MAX_SAMPLES:
            X = X[-TABPFN_MAX_SAMPLES:]
            y = y[-TABPFN_MAX_SAMPLES:]

        # Handle NaN / Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # PCA if needed
        X_red = self._apply_pca(X, fit=True)

        # Convert labels to 0-indexed
        y_mapped = np.array([_CLASS_INDEX.get(int(lbl), 1) for lbl in y])

        self.classifier = self._build_classifier()
        try:
            self.classifier.fit(X_red, y_mapped)
            self.is_fitted = True
            self.n_features_in_ = X.shape[1]
            logger.info("%s: TabPFN fitted on %d samples, %d features (reduced to %d)",
                        self.pair, len(X), X.shape[1], X_red.shape[1])
        except Exception as exc:
            logger.error("%s: TabPFN fit error: %s", self.pair, exc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities for the latest observation.

        Args:
            X: Feature matrix, shape (1, n_features) or (n, n_features).

        Returns:
            Probability array of shape (3,): [p_BUY, p_HOLD, p_SELL].
        """
        default = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)

        if not self.is_fitted or self.classifier is None:
            return default

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Align features
        if X.shape[1] > self.n_features_in_:
            X = X[:, :self.n_features_in_]
        elif X.shape[1] < self.n_features_in_:
            pad = np.zeros((X.shape[0], self.n_features_in_ - X.shape[1]))
            X = np.hstack([X, pad])

        X_red = self._apply_pca(X, fit=False)

        try:
            proba = self.classifier.predict_proba(X_red)  # shape (n, 3)
            # Map back to [BUY, HOLD, SELL] order (classifier outputs 0-indexed)
            # classes_ may not be [0,1,2] if some are missing in training
            classes = list(self.classifier.classes_)
            result = np.zeros(3, dtype=float)
            for i, cls_idx in enumerate(classes):
                result[cls_idx] = float(proba[-1, i])
            # Ensure sums to 1
            total = result.sum()
            if total > 0:
                result /= total
            return result
        except Exception as exc:
            logger.warning("%s: TabPFN predict_proba error: %s", self.pair, exc)
            return default

    def save(self) -> None:
        """Persist the fitted classifier and PCA to disk for the pair."""
        import pickle
        save_dir = MODELS_DIR / self.pair
        save_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "classifier": self.classifier,
            "pca": self.pca,
            "n_features_in_": self.n_features_in_,
        }
        try:
            with open(save_dir / "tabpfn.pkl", "wb") as f:
                pickle.dump(artifact, f)
            logger.debug("%s: TabPFN saved", self.pair)
        except Exception as exc:
            logger.warning("%s: TabPFN save failed: %s", self.pair, exc)

    def load(self) -> bool:
        """
        Load a previously saved classifier from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        import pickle
        path = MODELS_DIR / self.pair / "tabpfn.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                artifact = pickle.load(f)
            self.classifier = artifact["classifier"]
            self.pca = artifact.get("pca")
            self.n_features_in_ = artifact.get("n_features_in_", 0)
            self.is_fitted = True
            logger.info("%s: TabPFN loaded from %s", self.pair, path)
            return True
        except Exception as exc:
            logger.warning("%s: TabPFN load failed: %s", self.pair, exc)
            return False
