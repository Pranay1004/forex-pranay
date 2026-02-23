"""
main.py — Single entry point for ForexBot.
Runs entirely in the foreground terminal. No background processes.
Usage: python main.py
"""

import logging
import signal
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Suppress NumPy runtime warnings from hmmlearn/sklearn matrix operations
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="hmmlearn")

# ── Bootstrap logging before any other imports ────────────────────────────────
from forexbot.config import (
    CACHE_DIR,
    H1_INTERVAL,
    LOG_FILE,
    MODELS_DIR,
    MTF_MIN_AGREE,
    PAIRS,
    REPORTS_DIR,
    STARTING_BALANCE,
    TRADES_FILE,
    TIMEFRAMES,
    USE_GPU,
)

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("forexbot.main")

# ── Remaining imports ─────────────────────────────────────────────────────────
from rich.console import Console
from rich.live import Live

from forexbot.data.feed import (
    refresh_all_pairs, seconds_to_next_candle, get_latest_close,
    get_mtf_bias, mtf_confirms_signal,
)
from forexbot.data.news import NewsStore, refresh_headlines, refresh_events, get_recent_headlines_text
from forexbot.display.terminal import (
    build_positions_table,
    print_banner,
    print_validation_results,
    render_full_layout,
)
from forexbot.features.regime import RegimeModel, calibrate_regime_model, detect_regime, needs_recalibration
from forexbot.features.statistical import build_cross_pair_features, build_statistical_features
from forexbot.features.technical import build_technical_features, get_feature_columns
from forexbot.models.classical import ClassicalModels, build_labels, train_classical_models
from forexbot.models.ensemble import BUY, HOLD, SELL, run_ensemble
from forexbot.models.hf_sentiment import SentimentEngine
from forexbot.models.hf_tabular import TabPFNWrapper
from forexbot.paper_trading.engine import PaperTradingEngine
from forexbot.paper_trading.portfolio import Portfolio
from forexbot.paper_trading.validator import run_walk_forward_validation
from forexbot.reports.reporter import generate_daily_report
from forexbot.strategy.filters import apply_all_filters
from forexbot.strategy.risk import DrawdownState
from forexbot.strategy.signal import TradeSignal, generate_signal, get_current_atr


def _create_directories() -> None:
    """Create all required directories."""
    for d in [CACHE_DIR, MODELS_DIR, TRADES_FILE.parent, LOG_FILE.parent, REPORTS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("Directories ready.")


def _build_features_for_all(
    ohlcv_data: dict[str, dict],
    force: bool = False,
) -> dict[str, object]:
    """
    Build full feature DataFrames for all pairs on the H1 timeframe.
    Results are cached per-pair by last candle timestamp — the expensive
    indicator build only runs when a new H1 candle has arrived (once/hour).
    Pass force=True to bypass cache (used on first boot).
    """
    global _h1_feature_cache, _h1_last_ts
    featured: dict = {}
    rebuilt_any = False

    for pair in PAIRS:
        df_h1 = ohlcv_data.get(pair, {}).get(H1_INTERVAL)
        if df_h1 is None or df_h1.empty:
            logger.warning("%s: No H1 data — skipping feature build", pair)
            continue

        last_ts = df_h1.index[-1]
        if not force and pair in _h1_feature_cache and _h1_last_ts.get(pair) == last_ts:
            featured[pair] = _h1_feature_cache[pair]
            continue  # cache hit — same candle, no rebuild needed

        try:
            df = build_technical_features(df_h1)
            df = build_statistical_features(df)
            df = df.ffill().dropna()
            if len(df) < 100:
                logger.warning("%s H1: Only %d bars after NaN drop (need ≥100) — skipping", pair, len(df))
                continue
            _h1_feature_cache[pair] = df
            _h1_last_ts[pair] = last_ts
            featured[pair] = df
            rebuilt_any = True
            logger.info("%s H1: %d bars, features built (candle %s)",
                        pair, len(df), str(last_ts)[:16])
        except Exception as exc:
            logger.error("%s: Feature build failed: %s", pair, exc)
            if pair in _h1_feature_cache:
                featured[pair] = _h1_feature_cache[pair]  # fall back to last good

    # Cross-pair features only when at least one pair was rebuilt
    if rebuilt_any:
        for pair in list(featured.keys()):
            try:
                featured[pair] = build_cross_pair_features(featured, pair)
                _h1_feature_cache[pair] = featured[pair]  # update cache with cross-pair
            except Exception as exc:
                logger.warning("%s: Cross-pair features failed: %s", pair, exc)
    return featured


def _build_d1_features(
    ohlcv_data: dict[str, dict],
    force: bool = False,
) -> dict[str, object]:
    """
    Build feature DataFrames on D1 (daily) timeframe.
    Cached by last candle timestamp — rebuilds once per day max.
    """
    global _d1_feature_cache, _d1_last_ts
    from forexbot.config import D1_INTERVAL as _D1
    featured_d1: dict = {}
    for pair in PAIRS:
        df_d1 = ohlcv_data.get(pair, {}).get(_D1)
        if df_d1 is None or df_d1.empty:
            logger.warning("%s: No D1 data — skipping D1 feature build", pair)
            continue

        last_ts = df_d1.index[-1]
        if not force and pair in _d1_feature_cache and _d1_last_ts.get(pair) == last_ts:
            featured_d1[pair] = _d1_feature_cache[pair]
            continue

        try:
            df = build_technical_features(df_d1)
            df = build_statistical_features(df)
            df = df.ffill().dropna()
            if len(df) < 200:
                logger.warning("%s D1: Only %d bars after NaN drop (need ≥200) — skipping", pair, len(df))
                continue
            _d1_feature_cache[pair] = df
            _d1_last_ts[pair] = last_ts
            featured_d1[pair] = df
            logger.info("%s D1: %d bars, features built (~%.0f years)",
                        pair, len(df), len(df) / 252)
        except Exception as exc:
            logger.error("%s: D1 feature build failed: %s", pair, exc)
            if pair in _d1_feature_cache:
                featured_d1[pair] = _d1_feature_cache[pair]
    return featured_d1


def _train_or_load_models(
    featured: dict,
) -> tuple[dict[str, ClassicalModels], dict[str, TabPFNWrapper]]:
    """
    Load saved models if available, otherwise train fresh.
    Returns (classical_models_dict, tabpfn_dict).
    """
    # ── Checkpoint resume note ─────────────────────────────────────────────────
    # Models are saved per-pair to models/saved/{pair}/ as .pkl files.
    # On restart, each pair's models are loaded from disk — Optuna tuning is
    # skipped entirely for pairs with existing checkpoints, saving ~5 min/pair.
    # Optuna trial history is also persisted in optuna_xgb.db / optuna_lgb.db,
    # so even a mid-run crash resumes from whichever trial it was on.
    # To force full retraining: delete models/saved/{pair}/*.pkl
    # To force Optuna restart: delete models/saved/{pair}/*.db
    # ──────────────────────────────────────────────────────────────────────────
    classical_dict: dict[str, ClassicalModels] = {}
    tabpfn_dict: dict[str, TabPFNWrapper] = {}
    pair_total = len(PAIRS)

    for pair_idx, pair in enumerate(PAIRS, start=1):
        df = featured.get(pair)

        # Classical models
        cm = ClassicalModels(pair=pair)
        if not cm.load():
            logger.info("═══ Training pair %d/%d: %s (classical models) ═══",
                        pair_idx, pair_total, pair)
            if df is not None:
                feat_cols = get_feature_columns(df)
                df_labelled = df.copy()
                df_labelled["label"] = build_labels(df_labelled)
                cm = train_classical_models(
                    df_labelled, pair, feat_cols,
                    pair_idx=pair_idx, pair_total=pair_total,
                )
                cm.save()
        else:
            logger.info("Pair %d/%d: %s — classical models loaded from disk (skipping training)",
                        pair_idx, pair_total, pair)
        classical_dict[pair] = cm

        # TabPFN
        tp = TabPFNWrapper(pair=pair)
        if not tp.load():
            logger.info("Pair %d/%d: %s — fitting TabPFN…", pair_idx, pair_total, pair)
            if df is not None:
                feat_cols = get_feature_columns(df)
                df_labelled = df.copy()
                df_labelled["label"] = build_labels(df_labelled)
                clean = df_labelled.dropna(subset=feat_cols + ["label"])
                X = clean[feat_cols].values.astype(np.float32)
                y_raw = clean["label"].values.astype(int)
                if len(X) > 0:
                    tp.fit(X, y_raw)
                    tp.save()
        else:
            logger.info("Pair %d/%d: %s — TabPFN loaded from disk", pair_idx, pair_total, pair)
        tabpfn_dict[pair] = tp

    return classical_dict, tabpfn_dict


def _run_validation(
    featured: dict,
    classical_dict: dict[str, ClassicalModels],
    console: Console,
) -> list[str]:
    """
    Run walk-forward validation for all pairs.
    Returns list of pairs that PASSED validation.
    Requires at least 2 pairs to pass for the bot to start.
    """
    # ── Validation note ────────────────────────────────────────────────────────
    # Walk-forward validation is ALWAYS run fresh against current market data,
    # even when models are loaded from disk checkpoints.
    # It trains lightweight XGBoost (100 trees, no Optuna) on each fold to test
    # whether the strategy logic (labels + features + timeframe) still holds.
    # The full production models (saved .pkl) are NOT used here — this is an
    # independent data-quality gate. Thresholds: Sharpe ≥ 0.15, PF ≥ 1.02,
    # Win Rate ≥ 45%. Pairs that fail are excluded from trading but are not
    # retrained — their .pkl models remain on disk for the next boot.
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("Validation: running walk-forward check on current H1 data "
                "(independent of saved .pkl checkpoints)")
    passed_pairs: list[str] = []
    failed_pairs: list[str] = []
    for pair in PAIRS:
        df = featured.get(pair)
        if df is None or len(df) < 300:
            console.print(f"[yellow]{pair}: insufficient data — skipping validation[/yellow]")
            failed_pairs.append(pair)
            continue

        feat_cols = get_feature_columns(df)

        def _trainer(train_df, p, fc):
            from forexbot.models.classical import train_classical_models as _tcm
            import xgboost as xgb
            from sklearn.preprocessing import LabelEncoder
            # Quick single-model train for validation speed (no Optuna)
            train_df = train_df.copy()
            from forexbot.models.classical import build_labels, _prepare_Xy
            train_df["label"] = build_labels(train_df)
            X, y = _prepare_Xy(train_df, fc)
            if len(X) < 50:
                return None
            xgb_params = {
                "objective": "multi:softprob", "num_class": 3, "n_estimators": 100,
                "verbosity": 0, "use_label_encoder": False, "random_state": 42,
            }
            if USE_GPU:
                xgb_params["device"] = "cuda"  # T1000 / RTX on Windows
            else:
                xgb_params["nthread"] = -1     # all CPU cores on Mac
            clf = xgb.XGBClassifier(**xgb_params)
            clf.fit(X, y)
            return clf

        def _predictor(model, X):
            if model is None:
                return np.ones(len(X), dtype=int)  # all HOLD
            return model.predict(X)

        df_labelled = df.copy()
        df_labelled["label"] = build_labels(df_labelled)
        report = run_walk_forward_validation(df_labelled, pair, feat_cols, _trainer, _predictor)
        print_validation_results(console, report, pair)

        if report.passed:
            passed_pairs.append(pair)
        else:
            failed_pairs.append(pair)

    if failed_pairs:
        console.print(f"[yellow]Failed pairs excluded from trading: {', '.join(failed_pairs)}[/yellow]")
    console.print(f"[cyan]Validated {len(passed_pairs)}/{len(PAIRS)} pairs: {', '.join(passed_pairs) or 'NONE'}[/cyan]")

    return passed_pairs


def _get_current_features(df) -> np.ndarray:
    """Extract the last row of feature matrix for inference."""
    from forexbot.features.technical import get_feature_columns
    feat_cols = get_feature_columns(df)
    last_row = df[feat_cols].iloc[[-1]]
    return last_row.values.astype(np.float32)


# ── Module-level feature caches (keyed by last candle timestamp per pair) ──────
# Features are expensive to rebuild from 12,000 bars. Cache them and only
# rebuild when the last H1/D1 candle timestamp changes (i.e. new bar arrived).
_h1_feature_cache: dict = {}
_d1_feature_cache: dict = {}
_h1_last_ts: dict = {}
_d1_last_ts: dict = {}


def _main() -> None:
    """
    Main execution function — runs the entire foreground trading loop.
    """
    console = Console()
    print_banner(console)

    # ── Step 1-2: Directories ─────────────────────────────────────────────────
    _create_directories()

    # ── Step 3: Download data ─────────────────────────────────────────────────
    console.print("[cyan]Downloading OHLCV data for 5 pairs (H1 2yr + D1 30yr + W1 30yr)…[/cyan]")
    ohlcv_data = refresh_all_pairs()

    # ── Step 4: Build features ────────────────────────────────────────────────
    console.print("[cyan]Building H1 feature matrices…[/cyan]")
    featured = _build_features_for_all(ohlcv_data, force=True)

    console.print("[cyan]Building D1 feature matrices (30yr deep training data)…[/cyan]")
    featured_d1 = _build_d1_features(ohlcv_data, force=True)

    # ── Step 5: Market regime (HMM) — use D1 if available for better regime ──
    console.print("[cyan]Calibrating market regime HMMs…[/cyan]")
    regime_models: dict[str, RegimeModel] = {}
    for pair in PAIRS:
        # Prefer D1 features for regime detection (longer history = better HMM)
        df_regime = featured_d1.get(pair) if featured_d1.get(pair) is not None else featured.get(pair)
        if df_regime is not None:
            regime_models[pair] = calibrate_regime_model(df_regime, pair)

    # ── Step 6: Train / load classical models ─────────────────────────────────
    console.print("[cyan]Training / loading classical models (XGB + LGB)…[/cyan]")
    classical_dict, tabpfn_dict = _train_or_load_models(featured)

    # ── Step 7: Load HuggingFace models ───────────────────────────────────────
    console.print("[cyan]Loading FinBERT sentiment model…[/cyan]")
    sentiment_engine = SentimentEngine()
    sentiment_engine.load()

    # ── Step 8: Fetch news + sentiment ────────────────────────────────────────
    console.print("[cyan]Fetching news and computing initial sentiment…[/cyan]")
    news_store = NewsStore()
    refresh_headlines(news_store)
    refresh_events(news_store)

    # ── Step 9: Walk-forward validation ───────────────────────────────────────
    console.print("[cyan]Running walk-forward validation…[/cyan]")
    validated_pairs = _run_validation(featured, classical_dict, console)
    if len(validated_pairs) < 2:
        console.print("[bold red]Fewer than 2 pairs passed validation — aborting bot startup.[/bold red]")
        sys.exit(1)
    console.print(f"[bold green]{len(validated_pairs)} pairs validated — starting paper trading.[/bold green]\n")

    # ── Initialise paper trading engine ───────────────────────────────────────
    portfolio = Portfolio(balance=STARTING_BALANCE, start_of_day_balance=STARTING_BALANCE)
    drawdown = DrawdownState(
        daily_peak_balance=STARTING_BALANCE,
        total_peak_balance=STARTING_BALANCE,
    )
    engine = PaperTradingEngine(portfolio, drawdown)

    # State tracked each loop iteration
    regimes: dict[str, int] = {p: 0 for p in PAIRS}
    regime_probs: dict[str, np.ndarray] = {}
    signals_display: dict[str, tuple[int, float]] = {p: (HOLD, 0.0) for p in PAIRS}
    current_prices: dict[str, float] = {}
    sentiment_scores: dict[str, float] = {p: 0.0 for p in PAIRS}
    model_probas_by_pair: dict[str, dict[str, list]] = {}
    start_time = datetime.now(tz=timezone.utc)

    # Handle Ctrl+C cleanly
    shutdown_requested = False

    def _handle_sigint(sig, frame):
        nonlocal shutdown_requested
        logger.info("Ctrl+C received — shutting down cleanly…")
        shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Step 10-11: Main loop with Rich Live ───────────────────────────────────
    console.print("[green]Starting main trading loop…[/green]")

    with Live(console=console, refresh_per_second=1 / max(1, 3)) as live:
        last_report_date: str = ""

        while not shutdown_requested:
            loop_start = time.time()

            try:
                # ─ Refresh data ──────────────────────────────────────────────
                ohlcv_data = refresh_all_pairs()
                featured = _build_features_for_all(ohlcv_data)      # cache-aware
                featured_d1 = _build_d1_features(ohlcv_data)        # cache-aware
                refresh_headlines(news_store)
                refresh_events(news_store)
            except Exception as exc:
                logger.error("Data refresh error: %s", exc)

            # Update current prices
            for pair in PAIRS:
                df = featured.get(pair)
                if df is not None and not df.empty:
                    current_prices[pair] = float(df["Close"].iloc[-1])

            # ─ Regime detection (prefer D1 for longer history) ────────────────
            for pair in PAIRS:
                df_regime = featured_d1.get(pair) if featured_d1.get(pair) is not None else featured.get(pair)
                if df_regime is None:
                    continue
                rm = regime_models.get(pair)
                if rm is None or needs_recalibration(rm):
                    regime_models[pair] = calibrate_regime_model(df_regime, pair)
                    rm = regime_models[pair]
                try:
                    regime, probs = detect_regime(df_regime, rm)
                    regimes[pair] = regime
                    regime_probs[pair] = probs
                except Exception as exc:
                    logger.warning("%s: Regime detection error: %s", pair, exc)

            # ─ Signals & paper trades ─────────────────────────────────────────
            for pair in validated_pairs:
                df = featured.get(pair)
                if df is None or df.empty:
                    continue

                price = current_prices.get(pair, 0.0)
                if price <= 0:
                    continue

                regime = regimes.get(pair, 0)
                atr = get_current_atr(df)

                # Pre-trade filters
                natr_col = next((c for c in df.columns if "NATR" in c), None)
                natr = float(df[natr_col].iloc[-1]) if natr_col and not df[natr_col].empty else 0.001
                allowed, filter_reason = apply_all_filters(pair, news_store, natr=natr)

                if not allowed:
                    signals_display[pair] = (HOLD, 0.0)
                    continue

                # Compute ensemble
                try:
                    X_latest = _get_current_features(df)
                    headlines = get_recent_headlines_text(news_store, pair, n=20)
                    sentiment_scores[pair] = sentiment_engine.score(pair, headlines)

                    ensemble_result = run_ensemble(
                        pair=pair,
                        X_latest=X_latest,
                        classical=classical_dict[pair],
                        tabpfn=tabpfn_dict[pair],
                        sentiment_engine=sentiment_engine,
                        headlines=headlines,
                        regime=regime,
                    )
                    signals_display[pair] = (ensemble_result.signal, ensemble_result.confidence)

                    # Store model probas for display
                    model_probas_by_pair[pair] = {
                        k: v.tolist() for k, v in ensemble_result.model_probas.items()
                    }

                    # ─── Multi-Timeframe confirmation ─────────────────────────
                    if ensemble_result.signal != HOLD:
                        pair_ohlcv = ohlcv_data.get(pair, {})
                        biases = get_mtf_bias(pair_ohlcv, pair)
                        if not mtf_confirms_signal(biases, ensemble_result.signal, MTF_MIN_AGREE):
                            logger.info(
                                "%s: MTF filter blocked signal %d (biases=%s)",
                                pair, ensemble_result.signal, biases,
                            )
                            signals_display[pair] = (HOLD, 0.0)
                            continue

                    # Generate trade signal
                    win_rate = portfolio.recent_win_rate()
                    avg_win, avg_loss = portfolio.recent_avg_win_loss()
                    trade_signal = generate_signal(
                        pair=pair,
                        ensemble_result=ensemble_result,
                        current_price=price,
                        atr=atr,
                        account_balance=portfolio.balance,
                        regime=regime,
                        sentiment_score=sentiment_scores[pair],
                        win_rate=win_rate,
                        avg_win=avg_win,
                        avg_loss=avg_loss,
                    )

                    # Try to execute
                    if trade_signal.direction != HOLD:
                        engine.try_open(trade_signal, atr=atr)

                except Exception as exc:
                    logger.error("%s: Trading loop error: %s", pair, exc)

            # ─ Update open positions ──────────────────────────────────────────
            try:
                current_atrs = {
                    p: get_current_atr(featured[p]) for p in PAIRS if p in featured
                }
                engine.update_positions(current_prices, current_atrs, regimes)
            except Exception as exc:
                logger.error("Position update error: %s", exc)

            # ─ Drawdown check ─────────────────────────────────────────────────
            drawdown.update(portfolio.balance)

            # ─ Daily reset check ──────────────────────────────────────────────
            today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            if today_str != last_report_date and last_report_date != "":
                # New day: generate report and reset daily tracking
                try:
                    report_path = generate_daily_report(portfolio)
                    logger.info("Daily report: %s", report_path)
                except Exception as exc:
                    logger.error("Daily report error: %s", exc)
                portfolio.start_of_day_balance = portfolio.balance
                drawdown.reset_daily(portfolio.balance)
            last_report_date = today_str

            # ─ Display update ─────────────────────────────────────────────────
            try:
                secs_to_next = seconds_to_next_candle(H1_INTERVAL)
                focused_pair = PAIRS[0]
                layout = render_full_layout(
                    portfolio=portfolio,
                    start_time=start_time,
                    regimes=regimes,
                    signals_display=signals_display,
                    current_prices=current_prices,
                    sentiment_scores=sentiment_scores,
                    model_probas_by_pair=model_probas_by_pair,
                    seconds_to_next=secs_to_next,
                    focused_pair=focused_pair,
                )
                live.update(layout)
            except Exception as exc:
                logger.error("Display update error: %s", exc)

            # ─ Sleep until next display refresh ───────────────────────────────
            from forexbot.config import DISPLAY_REFRESH_SECONDS
            elapsed = time.time() - loop_start
            sleep_secs = max(0, DISPLAY_REFRESH_SECONDS - elapsed)
            # Sleep in short increments so Ctrl+C is responsive
            for _ in range(max(1, int(sleep_secs))):
                if shutdown_requested:
                    break
                time.sleep(1)

    # ── Shutdown ─────────────────────────────────────────────────────────────
    console.print("\n[bold yellow]Shutting down ForexBot…[/bold yellow]")
    try:
        report_path = generate_daily_report(portfolio)
        console.print(f"[green]Final report saved: {report_path}[/green]")
    except Exception as exc:
        logger.error("Shutdown report error: %s", exc)

    console.print(
        f"[bold cyan]ForexBot stopped cleanly. "
        f"Final balance: ${portfolio.balance:,.2f} | "
        f"Trades: {len(portfolio.closed_trades)}[/bold cyan]"
    )


if __name__ == "__main__":
    _main()
