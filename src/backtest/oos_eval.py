"""
Out-of-Sample Evaluation
========================
Uses the EXISTING raw history files (data/raw/price_history_*.json) — the same
files the ML models were trained on — but filters to markets that resolved AFTER
the training cutoff date (2026-06-01).

This is a TRUE out-of-sample test:
  - Training data:    all markets with endDate < 2026-06-01
  - OOS test data:    markets with endDate >= 2026-06-01  (June 2026)

Three models are compared using a flat $20 bet (fair comparison):
  1. Our ensemble   (best_model_<type>.pkl)
  2. LR baseline    (baseline_logistic.pkl)
  3. Random Forest  (baseline_rf.pkl)

Run from repo root:
    venv/bin/python src/backtest/oos_eval.py
"""

import os
import sys
import json
import math
import glob
import numpy as np
import joblib
from datetime import datetime, timezone
from collections import defaultdict

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.optimizer.pipeline import (
    load_models, MODELS, detect_market_type,
    get_price_features_with_type,
)

# ── Config ────────────────────────────────────────────────────────────────────

# Only include markets that resolved AFTER this date (training cutoff)
CUTOFF_DATE = "2026-06-01"

# Same files as the main backtest
HISTORY_FILES = sorted(glob.glob("data/raw/price_history_*.json"))

# Only these market types have trained ensemble models
ALLOWED_TYPES = {"1hour", "4hour", "1day", "weekly"}

# Market price zone: only bet in the uncertainty zone
PRICE_LO = 0.20
PRICE_HI = 0.80

# Edge thresholds
# MIN_EDGE: skip near-zero edge (model has no view)
# MAX_EDGE: no upper cap in OOS eval — we want to see all predictions,
#           including overconfident ones; in live trading this is capped
#           to 0.15 for risk management, but that cap should not be applied
#           when evaluating model quality on historical data.
MIN_EDGE = 0.02
MAX_EDGE = 1.00   # effectively no upper cap for OOS evaluation

# Minimum price history points required
MIN_HIST_PTS = 3

# Flat bet size — same for ALL models (fair comparison)
FLAT_BET = 20.0

COIN_LIST = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "HYPE"]
coin_idx_map = {c: i for i, c in enumerate(COIN_LIST)}


# ── Ground truth ──────────────────────────────────────────────────────────────

def get_outcome(m: dict):
    """Return 1 (YES), 0 (NO), or None (unresolved/ambiguous)."""
    try:
        op = m.get("outcomePrices")
        if op is None:
            return None
        parsed = json.loads(op) if isinstance(op, str) else op
        if not parsed or len(parsed) != 2:
            return None
        floats = [float(x) for x in parsed]
        if abs(floats[0] - 1.0) < 0.05:
            return 1   # YES resolved
        if abs(floats[0] - 0.0) < 0.05:
            return 0   # NO resolved
        return None    # still trading / ambiguous
    except Exception:
        return None


# ── Entry price (same logic as backtest.py) ───────────────────────────────────

def get_market_price(m: dict):
    """
    Best pre-resolution market price from CLOB history.
    Resolved markets have post-resolution price (0 or 1) in API fields,
    so use the first ~10% of the price trajectory instead.

    Returns (price, entry_idx) where entry_idx is the history index used,
    so the caller can truncate the trajectory to avoid look-ahead bias.
    """
    history = m.get("price_history", [])
    if len(history) >= MIN_HIST_PTS:
        idx = max(0, len(history) // 10)
        p = history[idx].get("p", 0.5)
        if 0.005 < p < 0.995:
            return p, idx
        p = history[0].get("p", 0.5)
        if 0.005 < p < 0.995:
            return p, 0
    for field in ["lastTradePrice", "bestBid", "bestAsk"]:
        raw = m.get(field)
        if raw:
            p = float(raw)
            if 0.005 < p < 0.995:
                return p, None  # no index — API field, use full history
    return None, None


# ── PnL ───────────────────────────────────────────────────────────────────────

def compute_pnl(bet: float, entry_price: float, outcome: int, direction: str) -> float:
    if direction == "YES":
        return bet * (1.0 / entry_price - 1.0) if outcome == 1 else -bet
    else:
        no_price = 1.0 - entry_price
        if no_price < 0.005:
            return 0.0
        return bet * (1.0 / no_price - 1.0) if outcome == 0 else -bet


# ── Model prediction (shared by LR, RF, ensemble) ────────────────────────────

def model_predict(model, m: dict, coin_idx: int, market_type: str,
                  entry_price: float = None, entry_idx: int = None):
    """
    Predict P(YES) for a market using only prices known at entry time.

    Two sources of look-ahead bias are eliminated:
    1. API fields (lastTradePrice, outcomePrices) reflect post-resolution values
       (0 or 1) — injecting entry_price fixes this.
    2. Full price_history includes data AFTER the entry point — truncating to
       entry_idx fixes this. In live trading you only see prices up to now.
    """
    if model is None:
        return None
    try:
        # Truncate history to what was known at entry
        history = m.get("price_history", [])
        if entry_idx is not None and entry_idx < len(history):
            history = history[:entry_idx + 1]
        m = {**m, "price_history": history}
        if entry_price is not None:
            m = {**m, "lastTradePrice": str(entry_price)}
        df, _ = get_price_features_with_type(m, coin_idx, market_type)
        if df is None:
            return None
        try:
            df = df.reindex(columns=model.feature_names_in_, fill_value=0)
        except AttributeError:
            pass
        return float(model.predict_proba(df)[0][1])
    except Exception:
        return None


# ── Aggregate stats ───────────────────────────────────────────────────────────

def _max_drawdown(cum_pnl: np.ndarray) -> float:
    peak, max_dd = (cum_pnl[0] if len(cum_pnl) else 0.0), 0.0
    for v in cum_pnl:
        peak = max(peak, v)
        dd = (peak - v) / (abs(peak) + 1e-9)
        max_dd = max(max_dd, dd)
    return max_dd


def strategy_stats(pnl_arr, bet_arr, correct_arr, label):
    pnl  = np.array([p for p in pnl_arr     if p is not None], dtype=float)
    bets = np.array([b for b in bet_arr      if b is not None], dtype=float)
    corr = np.array([c for c in correct_arr  if c is not None], dtype=float)
    n = len(pnl)
    if n == 0:
        return {"label": label, "n_trades": 0, "total_pnl": 0, "total_bet": 0,
                "roi_pct": 0, "win_rate_pct": 0, "direction_accuracy_pct": 0,
                "sharpe": 0, "max_drawdown_pct": 0}
    total_pnl = float(pnl.sum())
    total_bet = float(bets.sum())
    roi       = (total_pnl / total_bet * 100) if total_bet > 0 else 0.0
    win_rate  = float((pnl > 0).mean() * 100)
    sharpe    = float(pnl.mean() / pnl.std() * math.sqrt(252)) if pnl.std() > 0 else 0.0
    max_dd    = float(_max_drawdown(np.cumsum(pnl)) * 100)
    dir_acc   = float(corr.mean() * 100) if len(corr) > 0 else win_rate
    return {
        "label": label, "n_trades": n,
        "total_pnl": round(total_pnl, 2), "total_bet": round(total_bet, 2),
        "roi_pct": round(roi, 2), "win_rate_pct": round(win_rate, 2),
        "direction_accuracy_pct": round(dir_acc, 2),
        "sharpe": round(sharpe, 3), "max_drawdown_pct": round(max_dd, 2),
    }


# ── Main OOS evaluation ───────────────────────────────────────────────────────

def run_oos_eval(save_path: str = "data/oos_results.json") -> dict:
    print("\n" + "=" * 60)
    print("  OUT-OF-SAMPLE EVALUATION")
    print(f"  OOS window: endDate >= {CUTOFF_DATE}")
    print(f"  Source:     data/raw/price_history_*.json (same as training)")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    load_models()
    print(f"  Ensemble models available: {list(MODELS.keys())}")

    lr_model = None
    try:
        lr_model = joblib.load("src/model/baseline_logistic.pkl")
        print("  LR: loaded")
    except FileNotFoundError:
        print("  LR: NOT FOUND")

    rf_model = None
    try:
        rf_model = joblib.load("src/model/baseline_rf.pkl")
        print("  RF: loaded")
    except FileNotFoundError:
        print("  RF: NOT FOUND")

    trades = []
    skipped = defaultdict(int)
    oos_seen = 0

    print(f"\nScanning {len(HISTORY_FILES)} history files for June+ markets...")

    for fpath in HISTORY_FILES:
        market_type_from_file = (
            os.path.basename(fpath)
            .replace("price_history_", "")
            .replace(".json", "")
        )
        if market_type_from_file not in ALLOWED_TYPES:
            continue

        with open(fpath) as fp:
            file_data = json.load(fp)

        file_oos = 0

        for coin, markets in file_data.items():
            coin_idx = coin_idx_map.get(coin, 0)

            for m in markets:
                # ── OOS filter: only June+ markets ──
                end_date = (m.get("endDate") or "")[:10]
                if end_date < CUTOFF_DATE:
                    continue

                oos_seen += 1
                file_oos += 1

                # ── Ground truth ──
                outcome = get_outcome(m)
                if outcome is None:
                    skipped["no_outcome"] += 1
                    continue

                # ── History check ──
                if len(m.get("price_history", [])) < MIN_HIST_PTS:
                    skipped["no_hist"] += 1
                    continue

                # ── Entry price ──
                entry_price, entry_idx = get_market_price(m)
                if entry_price is None:
                    skipped["no_price"] += 1
                    continue

                if not (PRICE_LO <= entry_price <= PRICE_HI):
                    skipped["price_zone"] += 1
                    continue

                # ── Detect market type (may differ from filename) ──
                question_str = m.get("question", "")
                market_type = detect_market_type(question_str, m) or market_type_from_file
                if market_type not in ALLOWED_TYPES:
                    skipped["wrong_type"] += 1
                    continue

                # ── Pick the right ensemble model for this market type ──
                ensemble_model = MODELS.get(market_type) or MODELS.get("all")
                if ensemble_model is None:
                    skipped["no_model"] += 1
                    continue

                # ── Ensemble prediction ──
                # Pass entry_idx so model only sees prices known at entry time
                our_estimate = model_predict(
                    ensemble_model, m, coin_idx, market_type, entry_price, entry_idx
                )
                if our_estimate is None:
                    skipped["no_features"] += 1
                    continue

                our_edge  = our_estimate - entry_price
                direction = "YES" if our_edge >= 0 else "NO"
                abs_edge  = abs(our_edge)

                if abs_edge < MIN_EDGE:
                    skipped["edge_too_low"] += 1
                    continue
                if abs_edge > MAX_EDGE:
                    skipped["edge_too_high"] += 1
                    continue

                # Our model PnL (flat $20 bet)
                our_corr = int(
                    (direction == "YES" and outcome == 1) or
                    (direction == "NO"  and outcome == 0)
                )
                our_pnl = compute_pnl(FLAT_BET, entry_price, outcome, direction)

                # ── LR baseline ──
                lr_est  = model_predict(lr_model, m, coin_idx, market_type, entry_price, entry_idx)
                lr_pnl  = lr_corr = None
                if lr_est is not None:
                    lr_dir  = "YES" if (lr_est - entry_price) >= 0 else "NO"
                    lr_pnl  = compute_pnl(FLAT_BET, entry_price, outcome, lr_dir)
                    lr_corr = int(
                        (lr_dir == "YES" and outcome == 1) or
                        (lr_dir == "NO"  and outcome == 0)
                    )

                # ── RF baseline ──
                rf_est  = model_predict(rf_model, m, coin_idx, market_type, entry_price, entry_idx)
                rf_pnl  = rf_corr = None
                if rf_est is not None:
                    rf_dir  = "YES" if (rf_est - entry_price) >= 0 else "NO"
                    rf_pnl  = compute_pnl(FLAT_BET, entry_price, outcome, rf_dir)
                    rf_corr = int(
                        (rf_dir == "YES" and outcome == 1) or
                        (rf_dir == "NO"  and outcome == 0)
                    )

                trades.append({
                    "coin":         coin,
                    "question":     m.get("question", "")[:80],
                    "market_type":  market_type,
                    "end_date":     end_date,
                    "entry_price":  round(entry_price, 4),
                    "outcome":      outcome,
                    "our_estimate": round(our_estimate, 4),
                    "our_edge":     round(our_edge, 4),
                    "direction":    direction,
                    "our_pnl":      round(our_pnl, 4),
                    "our_correct":  our_corr,
                    "lr_estimate":  round(lr_est, 4) if lr_est is not None else None,
                    "lr_pnl":       round(lr_pnl, 4) if lr_pnl is not None else None,
                    "lr_correct":   lr_corr,
                    "rf_estimate":  round(rf_est, 4) if rf_est is not None else None,
                    "rf_pnl":       round(rf_pnl, 4) if rf_pnl is not None else None,
                    "rf_correct":   rf_corr,
                })

        if file_oos:
            print(f"  {os.path.basename(fpath)}: {file_oos} OOS markets found")

    print(f"\n  Total OOS markets seen:  {oos_seen}")
    print(f"  Valid trades:            {len(trades)}")
    print(f"  Skipped breakdown:       {dict(skipped)}")

    if not trades:
        print("\nNo valid OOS trades — cannot compute stats.")
        return None

    # ── Aggregate stats ───────────────────────────────────────────────────────
    our_pnls  = [r["our_pnl"]    for r in trades]
    our_bets  = [FLAT_BET]       * len(trades)
    our_corrs = [r["our_correct"] for r in trades]

    lr_pnls  = [r["lr_pnl"]    for r in trades]
    lr_bets  = [FLAT_BET if r["lr_pnl"] is not None else 0 for r in trades]
    lr_corrs = [r["lr_correct"] for r in trades]

    rf_pnls  = [r["rf_pnl"]    for r in trades]
    rf_bets  = [FLAT_BET if r["rf_pnl"] is not None else 0 for r in trades]
    rf_corrs = [r["rf_correct"] for r in trades]

    uni_pnls = [
        compute_pnl(FLAT_BET, r["entry_price"], r["outcome"], "YES")
        for r in trades
    ]

    stats_our = strategy_stats(our_pnls, our_bets, our_corrs, "Our Ensemble (OOS)")
    stats_lr  = strategy_stats(lr_pnls,  lr_bets,  lr_corrs,  "LR Baseline (OOS)")
    stats_rf  = strategy_stats(rf_pnls,  rf_bets,  rf_corrs,  "Random Forest (OOS)")
    stats_uni = strategy_stats(uni_pnls, our_bets, [],         "Always-YES (OOS)")

    result = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cutoff_date":  CUTOFF_DATE,
            "n_oos_seen":   oos_seen,
            "n_trades":     len(trades),
            "flat_bet":     FLAT_BET,
            "note": (
                "True out-of-sample: models trained on pre-June data, "
                "evaluated on June 2026 markets from the same raw files."
            ),
        },
        "summary": {
            "our_model":   stats_our,
            "lr_baseline": stats_lr,
            "rf_baseline": stats_rf,
            "uniform":     stats_uni,
        },
        "trades": trades,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved → {save_path}")

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  OUT-OF-SAMPLE RESULTS  (flat $20 bet per trade)")
    print("=" * 60)
    for s in [stats_our, stats_lr, stats_rf, stats_uni]:
        sign = "+" if s["roi_pct"] >= 0 else ""
        print(f"\n  {s['label']}")
        print(f"    Trades:    {s['n_trades']}")
        print(f"    Total PnL: ${s['total_pnl']:+.2f}")
        print(f"    ROI:       {sign}{s['roi_pct']:.2f}%")
        print(f"    Win Rate:  {s['win_rate_pct']:.1f}%")
        print(f"    Dir. Acc:  {s['direction_accuracy_pct']:.1f}%")
        print(f"    Sharpe:    {s['sharpe']:.3f}")
        print(f"    Max DD:    {s['max_drawdown_pct']:.1f}%")
    print("=" * 60)

    return result


if __name__ == "__main__":
    run_oos_eval()
