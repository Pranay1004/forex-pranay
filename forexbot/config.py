"""
config.py — All global constants for ForexBot.
No magic numbers anywhere else in the codebase.
"""

from pathlib import Path
import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

# ─── Pairs (Top 5 by Global FX Volume) ───────────────────────────────────────
PAIRS: list[str] = [
    "EURUSD",   # 28% of daily FX volume
    "USDJPY",   # 13%
    "GBPUSD",   # 11%
    "AUDUSD",   # 5%
    "USDCHF",   # 5%
]

# yfinance suffix map (all pairs append =X)
YFINANCE_SUFFIX = "=X"

# ─── Capital / Risk ──────────────────────────────────────────────────────────
STARTING_BALANCE: float = 10_000.0
MAX_CONCURRENT_POSITIONS: int = 3              # 5 pairs, max 3 open
MAX_POSITIONS_PER_PAIR: int = 1
RISK_PER_TRADE_PCT: float = 0.01          # 1% account risk per trade
ATR_SL_MULTIPLIER: float = 1.5
ATR_TP_MULTIPLIER: float = 2.5
KELLY_FRACTION: float = 0.5              # half-Kelly
MAX_DAILY_DRAWDOWN_PCT: float = 0.03
MAX_TOTAL_DRAWDOWN_PCT: float = 0.10
TIME_EXIT_BARS: int = 24                  # close after 24 H1 bars

# ─── Models ──────────────────────────────────────────────────────────────────
OPTUNA_TRIALS: int = 50                       # thorough hyperparameter search
CONFIDENCE_THRESHOLD: float = 0.55
MIN_DIRECTIONAL_PROB: float = 0.58            # min BUY/SELL class prob to allow entry
MIN_PROB_EDGE: float = 0.07                   # min gap between top-1 and top-2 class probs
PERFORMANCE_GUARD_MIN_WIN_RATE: float = 0.47  # block entries when recent win-rate degrades
PERFORMANCE_GUARD_MIN_RR: float = 1.00        # require avg_win/avg_loss >= this when win-rate weak
TABPFN_MAX_SAMPLES: int = 1000
TABPFN_MAX_FEATURES: int = 50            # PCA target if too many features
WALK_FORWARD_FOLDS: int = 5
LABEL_ATR_MULTIPLIER: float = 0.5        # for BUY/SELL label construction
LABEL_FORWARD_BARS: int = 4
LABEL_MOMENTUM_WINDOW: int = 10           # ROC period for momentum confirmation

# Validation acceptance criteria (realistic for walk-forward H1 forex)
VALIDATION_MIN_WIN_RATE: float = 0.45
VALIDATION_MIN_PROFIT_FACTOR: float = 1.02
VALIDATION_MAX_DRAWDOWN: float = 0.25
VALIDATION_MIN_SHARPE: float = 0.15

# ─── Ensemble weights ────────────────────────────────────────────────────────
ENSEMBLE_WEIGHTS: dict[str, float] = {
    "tabpfn":    0.25,
    "xgboost":   0.30,
    "lightgbm":  0.30,
    "sentiment": 0.15,
}

# ─── Spread simulation (pips per pair) ───────────────────────────────────────
SPREAD_PIPS: dict[str, float] = {
    "EURUSD": 1.0, "USDJPY": 1.2, "GBPUSD": 1.5, "AUDUSD": 1.3, "USDCHF": 1.5,
}
SLIPPAGE_PIPS: float = 0.5               # random uniform up to this value

# Pip sizes
PIP_SIZE: dict[str, float] = {
    "EURUSD": 0.0001, "USDJPY": 0.01, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "USDCHF": 0.0001,
}

# ─── News / Sentiment ────────────────────────────────────────────────────────
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")
TWELVEDATA_API_KEY: str = os.getenv("TWELVEDATA_API_KEY", "")
FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")
NEWS_BLACKOUT_MINUTES_BEFORE: int = 30
NEWS_BLACKOUT_MINUTES_AFTER: int = 20
NEWS_CACHE_SECONDS: int = 3600           # 1 hour
SENTIMENT_FINBERT_WEIGHT: float = 0.6
SENTIMENT_VADER_WEIGHT: float = 0.4
FINBERT_MODEL: str = "ProsusAI/finbert"

# Currency keywords for news filtering
CURRENCY_KEYWORDS: dict[str, list[str]] = {
    "EUR": ["ECB", "Euro", "European", "Lagarde", "eurozone"],
    "USD": ["Fed", "Federal Reserve", "Powell", "US economy", "FOMC", "dollar"],
    "JPY": ["BOJ", "Bank of Japan", "yen", "Ueda", "Japan"],
    "GBP": ["BOE", "Bank of England", "sterling", "Bailey", "Britain", "UK economy"],
    "AUD": ["RBA", "Reserve Bank of Australia", "aussie", "Australia"],
    "CAD": ["BOC", "Bank of Canada", "loonie", "Canada", "oil"],
    "CHF": ["SNB", "Swiss National Bank", "franc", "Switzerland"],
    "NZD": ["RBNZ", "Reserve Bank of NZ", "kiwi", "New Zealand"],
}

# ─── Session windows (UTC hours, inclusive start) ────────────────────────────
# Broad windows covering all liquid sessions so signals are not artificially
# blocked.  Pairs trade across multiple overlapping sessions daily:
#   Tokyo  00:00–09:00 UTC  |  London  07:00–17:00 UTC  |  NY  13:00–22:00 UTC
SESSION_WINDOWS: dict[str, tuple[int, int]] = {
    "EURUSD": (7, 22),   # London open → NY close
    "GBPUSD": (7, 22),   # London open → NY close
    "USDJPY": (0, 22),   # Tokyo open → NY close (most liquid pair globally)
    "AUDUSD": (22, 17),  # Sydney open → London close (wraps midnight)
    "USDCHF": (7, 22),   # London open → NY close
}

# ─── Data / Cache ─────────────────────────────────────────────────────────────
CACHE_DIR: Path = Path("cache/")
MODELS_DIR: Path = Path("models/saved/")
TRADES_FILE: Path = Path("trades/trades.csv")
LOG_FILE: Path = Path("logs/forexbot.log")
REPORTS_DIR: Path = Path("reports/html/")
DATA_YEARS: int = 30                     # default (used for D1/W1 depth)
DATA_YEARS_H1: int = 2                   # yfinance hourly limit ~730 days
DATA_YEARS_D1: int = 30                  # 30 years of daily data for deep training
DATA_YEARS_W1: int = 30                  # 30 years of weekly macro context
H1_INTERVAL: str = "1h"
H4_INTERVAL: str = "4h"                  # resampled from H1 for MTF confirmation
D1_INTERVAL: str = "1d"
W1_INTERVAL: str = "1wk"
TIMEFRAMES: list[str] = ["1h", "1d", "1wk"]

# ─── Multi-Timeframe Confirmation ─────────────────────────────────────────────
MTF_MIN_AGREE: int = 3                   # min TFs agreeing before executing a trade

# ─── HMM Regime ──────────────────────────────────────────────────────────────
HMM_STATES: int = 3
HMM_RECALIBRATE_DAYS: int = 7           # recalibrate weekly
HMM_WINDOW_MONTHS: int = 12            # rolling 12-month training window for more context

REGIME_LABELS: dict[int, str] = {
    0: "RANGING",
    1: "TRENDING_UP",
    2: "TRENDING_DOWN",
}

# ─── Display ─────────────────────────────────────────────────────────────────
DISPLAY_REFRESH_SECONDS: int = 5
RECENT_TRADES_SHOWN: int = 10

# ─── Misc ─────────────────────────────────────────────────────────────────────
VERSION: str = "2.0"
KELLY_WIN_RATE_WINDOW: int = 50         # rolling trade window for Kelly
TRAILING_STOP_ATR_TRIGGER: float = 1.0  # ATR multiples in favour before trailing

# ─── GPU Auto-Detection ───────────────────────────────────────────────────────
# Detects NVIDIA CUDA GPU via nvidia-smi.
#   macOS Apple Silicon (M-series): no CUDA  → USE_GPU = False  (CPU path)
#   Windows + NVIDIA T1000 / RTX / etc:   → USE_GPU = True   (CUDA path)
# XGBoost uses  device="cuda",  LightGBM uses  device_type="gpu"
# Training speedup vs CPU: ~3–5× for Optuna hyperparameter search.
def _detect_gpu() -> bool:
    """Return True if an NVIDIA CUDA GPU is available via nvidia-smi."""
    try:
        kwargs: dict = {"capture_output": True, "timeout": 3}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        result = subprocess.run(["nvidia-smi", "-L"], **kwargs)
        return result.returncode == 0 and len(result.stdout) > 0
    except Exception:
        return False

USE_GPU: bool = _detect_gpu()
