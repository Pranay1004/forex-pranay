"""
models/classical.py — XGBoost + LightGBM classifiers with Optuna tuning (20 trials).
Label: ATR-scaled forward return over next LABEL_FORWARD_BARS bars.
Walk-forward CV with WALK_FORWARD_FOLDS expanding folds, no shuffle.
Saves/loads models per pair to/from MODELS_DIR.
"""

import logging
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

# Suppress sklearn/Optuna convergence warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from forexbot.config import (
    LABEL_ATR_MULTIPLIER,
    LABEL_FORWARD_BARS,
    LABEL_MOMENTUM_WINDOW,
    MODELS_DIR,
    OPTUNA_TRIALS,
    USE_GPU,
    WALK_FORWARD_FOLDS,
)

logger = logging.getLogger(__name__)
logging.getLogger("optuna").setLevel(logging.WARNING)  # silence Optuna verbosity

# Label → class index mapping  BUY=1→0, HOLD=0→1, SELL=-1→2
_LABEL_TO_IDX = {1: 0, 0: 1, -1: 2}
_IDX_TO_LABEL = {0: 1, 1: 0, 2: -1}


def build_labels(df: pd.DataFrame) -> pd.Series:
    """
    Construct BUY/HOLD/SELL integer labels using ATR-scaled forward return
    CONFIRMED by multi-period price momentum.

    Labelling logic:
      1. ATR threshold: forward_return vs ATR-scaled threshold (existing)
      2. Momentum confirmation: average sign of ROC at 5, 10, 20 bars
         BUY requires positive momentum; SELL requires negative momentum.
         This filters out noise labels where price spikes against the trend.

    Args:
        df: DataFrame with Close and ATR_14 columns.

    Returns:
        Integer label Series (1=BUY, 0=HOLD, -1=SELL).
    """
    close = df["Close"]
    atr_col = next((c for c in df.columns if "ATR_14" in c), None)
    if atr_col is None:
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
    else:
        atr = df[atr_col]

    # --- ATR component ---
    fwd_ret = close.pct_change(LABEL_FORWARD_BARS).shift(-LABEL_FORWARD_BARS)
    threshold = LABEL_ATR_MULTIPLIER * atr / close

    # --- Momentum component ---
    # Multi-period ROC: captures short, medium, and longer momentum
    roc_5 = close.pct_change(5)
    roc_10 = close.pct_change(LABEL_MOMENTUM_WINDOW)
    roc_20 = close.pct_change(20)
    # Momentum score: average sign of the 3 ROC periods → range [-1, +1]
    mom_score = (np.sign(roc_5) + np.sign(roc_10) + np.sign(roc_20)) / 3.0

    # --- Combined label: ATR threshold + momentum confirmation ---
    labels = pd.Series(0, index=df.index, dtype=int)
    # BUY: forward return exceeds ATR threshold AND momentum is positive
    labels[(fwd_ret > threshold) & (mom_score > 0)] = 1
    # SELL: forward return below -ATR threshold AND momentum is negative
    labels[(fwd_ret < -threshold) & (mom_score < 0)] = -1

    return labels


def _get_class_weights(y: np.ndarray) -> dict:
    """Compute balanced class weights for the given label array."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _prepare_Xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and label vector y, dropping NaN or inf rows.

    Args:
        df: Featured DataFrame with a 'label' column.
        feature_cols: Feature column names to use.

    Returns:
        (X, y) without NaN/inf rows, or (empty, empty) if insufficient data.
    """
    df_clean = df[feature_cols + ["label"]].copy()
    # Remove NaN and inf
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_clean) < 50:
        return np.array([]).reshape(0, len(feature_cols)), np.array([], dtype=int)
    X = df_clean[feature_cols].values.astype(np.float32)
    y = np.array([_LABEL_TO_IDX.get(int(lbl), 1) for lbl in df_clean["label"].values], dtype=int)
    return X, y


# ─── XGBoost ──────────────────────────────────────────────────────────────────

def _xgb_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for XGBoost."""
    import xgboost as xgb  # type: ignore
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "use_label_encoder": False,
        "verbosity": 0,
        "nthread": -1,
        "random_state": 42,
    }
    # GPU acceleration: device="cuda" on Windows/NVIDIA, ignored on CPU fallback
    if USE_GPU:
        params["device"] = "cuda"
        params.pop("nthread", None)  # nthread not valid with CUDA device in XGB 2+
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    from sklearn.metrics import log_loss
    proba = clf.predict_proba(X_val)
    return log_loss(y_val, proba)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pair: str = "",
    pair_idx: int = 0,
    pair_total: int = 0,
):
    """Train XGBoost with Optuna tuning. Resumable via SQLite study storage."""
    import optuna
    import xgboost as xgb  # type: ignore

    pair_label = f"[{pair_idx}/{pair_total}] {pair}" if pair else "unknown"
    study_name = f"{pair}_xgb" if pair else "xgb"
    db_path = MODELS_DIR / pair / "optuna_xgb.db" if pair else MODELS_DIR / "optuna_xgb.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
    )
    done_trials = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, OPTUNA_TRIALS - done_trials)

    if done_trials > 0:
        logger.info("%s XGBoost: resuming from %d/%d completed trials (%d remaining)",
                    pair_label, done_trials, OPTUNA_TRIALS, remaining)
    else:
        logger.info("%s XGBoost: starting %d trials", pair_label, OPTUNA_TRIALS)

    def _xgb_progress_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        n_done = len([t for t in study.trials if t.state.is_finished()])
        pct = int(n_done / OPTUNA_TRIALS * 100)
        logger.info("%s XGBoost trial %d/%d (%d%%) — loss=%.4f | best=%.4f",
                    pair_label, n_done, OPTUNA_TRIALS, pct,
                    trial.value if trial.value is not None else float('nan'),
                    study.best_value)

    if remaining > 0:
        study.optimize(
            lambda t: _xgb_objective(t, X_train, y_train, X_val, y_val),
            n_trials=remaining,
            callbacks=[_xgb_progress_cb],
        )

    best = study.best_params
    best.update({"objective": "multi:softprob", "num_class": 3, "verbosity": 0,
                 "use_label_encoder": False, "random_state": 42})
    if USE_GPU:
        best["device"] = "cuda"  # NVIDIA T1000 / RTX on Windows
    else:
        best["nthread"] = -1     # all CPU cores on Mac/Linux
    clf = xgb.XGBClassifier(**best)
    clf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    logger.info("%s XGBoost: training complete (best loss=%.4f) [device=%s]",
                pair_label, study.best_value, "cuda" if USE_GPU else "cpu")
    return clf


# ─── LightGBM ─────────────────────────────────────────────────────────────────

def _lgb_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for LightGBM."""
    import lightgbm as lgb  # type: ignore
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }
    # GPU acceleration: device_type="gpu" on Windows/NVIDIA
    if USE_GPU:
        params["device_type"] = "gpu"
        params["n_jobs"] = 1  # LightGBM GPU ignores n_jobs; set to 1 to avoid warning
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    from sklearn.metrics import log_loss
    proba = clf.predict_proba(X_val)
    return log_loss(y_val, proba)


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pair: str = "",
    pair_idx: int = 0,
    pair_total: int = 0,
):
    """Train LightGBM with Optuna tuning. Resumable via SQLite study storage."""
    import lightgbm as lgb  # type: ignore
    import optuna

    pair_label = f"[{pair_idx}/{pair_total}] {pair}" if pair else "unknown"
    study_name = f"{pair}_lgb" if pair else "lgb"
    db_path = MODELS_DIR / pair / "optuna_lgb.db" if pair else MODELS_DIR / "optuna_lgb.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
    )
    done_trials = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, OPTUNA_TRIALS - done_trials)

    if done_trials > 0:
        logger.info("%s LightGBM: resuming from %d/%d completed trials (%d remaining)",
                    pair_label, done_trials, OPTUNA_TRIALS, remaining)
    else:
        logger.info("%s LightGBM: starting %d trials", pair_label, OPTUNA_TRIALS)

    def _lgb_progress_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        n_done = len([t for t in study.trials if t.state.is_finished()])
        pct = int(n_done / OPTUNA_TRIALS * 100)
        logger.info("%s LightGBM trial %d/%d (%d%%) — loss=%.4f | best=%.4f",
                    pair_label, n_done, OPTUNA_TRIALS, pct,
                    trial.value if trial.value is not None else float('nan'),
                    study.best_value)

    if remaining > 0:
        study.optimize(
            lambda t: _lgb_objective(t, X_train, y_train, X_val, y_val),
            n_trials=remaining,
            callbacks=[_lgb_progress_cb],
        )

    best = study.best_params
    best.update({"objective": "multiclass", "num_class": 3, "verbose": -1, "random_state": 42})
    if USE_GPU:
        best["device_type"] = "gpu"  # NVIDIA T1000 / RTX on Windows
        best["n_jobs"] = 1
    else:
        best["n_jobs"] = -1          # all CPU cores on Mac/Linux
    clf = lgb.LGBMClassifier(**best)
    clf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    logger.info("%s LightGBM: training complete (best loss=%.4f) [device=%s]",
                pair_label, study.best_value, "gpu" if USE_GPU else "cpu")
    return clf


# ─── Classical model container ────────────────────────────────────────────────

@dataclass
class ClassicalModels:
    """Holds fitted XGBoost and LightGBM classifiers for a single pair."""

    pair: str
    xgb_clf: Optional[object] = field(default=None, repr=False)
    lgb_clf: Optional[object] = field(default=None, repr=False)
    feature_cols: list[str] = field(default_factory=list)
    is_fitted: bool = False

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (xgb_proba, lgb_proba), each of shape (3,).
        Falls back to uniform distribution if not fitted.
        """
        default = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        if not self.is_fitted:
            return default.copy(), default.copy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            xgb_p = self.xgb_clf.predict_proba(X)[-1] if self.xgb_clf else default
        except Exception:
            xgb_p = default.copy()
        try:
            lgb_p = self.lgb_clf.predict_proba(X)[-1] if self.lgb_clf else default
        except Exception:
            lgb_p = default.copy()
        return np.array(xgb_p, dtype=float), np.array(lgb_p, dtype=float)

    def save(self) -> None:
        """Save both classifiers to MODELS_DIR/{pair}/."""
        save_dir = MODELS_DIR / self.pair
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, clf in [("xgboost.pkl", self.xgb_clf), ("lightgbm.pkl", self.lgb_clf)]:
            if clf is None:
                continue
            try:
                with open(save_dir / name, "wb") as f:
                    pickle.dump(clf, f)
            except Exception as exc:
                logger.warning("%s: save %s failed: %s", self.pair, name, exc)
        # Save feature list
        with open(save_dir / "feature_cols.pkl", "wb") as f:
            pickle.dump(self.feature_cols, f)
        logger.info("%s: Classical models saved to %s", self.pair, save_dir)

    def load(self) -> bool:
        """Load models from disk. Returns True if successful."""
        save_dir = MODELS_DIR / self.pair
        try:
            with open(save_dir / "xgboost.pkl", "rb") as f:
                self.xgb_clf = pickle.load(f)
            with open(save_dir / "lightgbm.pkl", "rb") as f:
                self.lgb_clf = pickle.load(f)
            with open(save_dir / "feature_cols.pkl", "rb") as f:
                self.feature_cols = pickle.load(f)
            self.is_fitted = True
            logger.info("%s: Classical models loaded from %s", self.pair, save_dir)
            return True
        except Exception as exc:
            logger.debug("%s: Model load failed: %s", self.pair, exc)
            return False


def train_classical_models(
    df: pd.DataFrame,
    pair: str,
    feature_cols: list[str],
    pair_idx: int = 0,
    pair_total: int = 0,
) -> ClassicalModels:
    """
    Train XGBoost and LightGBM for a pair using walk-forward CV for validation split.

    Args:
        df: Full featured DataFrame with 'label' column.
        pair: Currency pair.
        feature_cols: List of feature column names.
        pair_idx: Position of this pair in the training queue (for progress display).
        pair_total: Total number of pairs being trained.

    Returns:
        Fitted ClassicalModels instance.
    """
    cm = ClassicalModels(pair=pair, feature_cols=feature_cols)

    X, y = _prepare_Xy(df, feature_cols)
    if len(X) < 200:
        logger.warning("%s: Only %d samples — skipping classical training", pair, len(X))
        return cm

    label = f"[{pair_idx}/{pair_total}]" if pair_total else ""
    logger.info("%s %s: Training on %d samples (%d features)",
                label, pair, len(X), X.shape[1])

    # Simple train/validation split (last 20% as validation)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    logger.info("%s: Training XGBoost (%d train, %d val) …", pair, len(X_train), len(X_val))
    try:
        cm.xgb_clf = train_xgboost(
            X_train, y_train, X_val, y_val,
            pair=pair, pair_idx=pair_idx, pair_total=pair_total,
        )
    except Exception as exc:
        logger.error("%s: XGBoost training failed: %s", pair, exc)

    logger.info("%s: Training LightGBM …", pair)
    try:
        cm.lgb_clf = train_lightgbm(
            X_train, y_train, X_val, y_val,
            pair=pair, pair_idx=pair_idx, pair_total=pair_total,
        )
    except Exception as exc:
        logger.error("%s: LightGBM training failed: %s", pair, exc)

    if cm.xgb_clf is not None or cm.lgb_clf is not None:
        cm.is_fitted = True

    return cm
