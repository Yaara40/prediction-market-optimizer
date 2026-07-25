"""
Backtester for the Kelly-Markowitz prediction market optimizer.

Uses the pre-fetched closed market data in data/raw/price_history_*.json
(the same files the ML models were trained on) to simulate four strategies:

  1. Our ensemble model  + Half-Kelly sizing
  2. LR baseline         + uniform flat bet
  3. Random Forest       + uniform flat bet
  4. Always-YES          + uniform flat bet  (naive benchmark)

Outputs per-trade results, cumulative PnL curves, and aggregate stats.
"""

import os
import sys
import json
import math
import glob
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.optimizer.pipeline import (
    load_models, MODELS, detect_market_type,
    get_price_features_with_type,
)
from src.optimizer.kelly_markowitz import KELLY_FRACTION

# ── Config ────────────────────────────────────────────────────────────────────

# Which history files to include (all by default)
HISTORY_FILES = sorted(glob.glob("data/raw/price_history_*.json"))

# Market types to include — skip very short windows (model noise > signal)
ALLOWED_TYPES = {"1hour", "4hour", "1day", "weekly"}

# Market price zone: only bet in the uncertainty zone (mirrors live pipeline)
PRICE_LO = 0.20
PRICE_HI = 0.80

# Edge thresholds (mirrors live pipeline)
MIN_EDGE = 0.02
MAX_EDGE = 0.15

# Minimum price history points required
MIN_HIST_PTS = 3

# Minimum volume filter
MIN_VOLUME = 10.0

# Flat bet size for LR and uniform strategies
FLAT_BET = 20.0

# Half-Kelly max fraction of session bankroll per trade
SESSION_BANKROLL = 500.0

COIN_LIST = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "HYPE"]
coin_idx_map = {c: i for i, c in enumerate(COIN_LIST)}


# ── Ground truth extraction ───────────────────────────────────────────────────

def get_outcome(m: dict):
    """Return 1 (YES resolved), 0 (NO resolved), or None (unresolved/multi-outcome)."""
    try:
        op = m.get("outcomePrices")
        if op is None:
            return None
        parsed = json.loads(op) if isinstance(op, str) else op
        if not parsed or len(parsed) != 2:
            return None          # skip multi-outcome markets
        floats = [float(x) for x in parsed]
        if abs(floats[0] - 1.0) < 0.05:
            return 1             # YES resolved
        if abs(floats[0] - 0.0) < 0.05:
            return 0             # NO resolved
        return None              # still trading / ambiguous
    except Exception:
        return None


# ── Pre-resolution market price ───────────────────────────────────────────────

def get_market_price(m: dict):
    """
    Best estimate of the market price at the START of the trading window
    (i.e. before resolution).  For resolved markets the API fields hold the
    post-resolution price (0 or 1), so we use the first point of the CLOB
    history instead.

    Returns (price, entry_idx) where entry_idx is the history index used for
    entry price — callers should truncate history[:entry_idx+1] before feeding
    the market to any ML model to avoid look-ahead bias.
    """
    history = m.get("price_history", [])
    if len(history) >= 3:
        # Use early-window price (first 10% of trajectory) as the entry price
        idx = max(0, len(history) // 10)
        p = history[idx].get("p", 0.5)
        if 0.005 < p < 0.995:
            return p, idx
        # Fallback: first point
        p = history[0].get("p", 0.5)
        if 0.005 < p < 0.995:
            return p, 0

    # No history — try API price fields (works for non-intraday markets)
    for field in ["lastTradePrice", "bestBid", "bestAsk"]:
        raw = m.get(field)
        if raw:
            p = float(raw)
            if 0.005 < p < 0.995:
                return p, None  # no index — full history is fine (it's empty anyway)
    return None, None


# ── Half-Kelly sizing ─────────────────────────────────────────────────────────

def kelly_bet(edge: float, entry_price: float) -> float:
    """Return Half-Kelly dollar bet size (capped at 50% of SESSION_BANKROLL)."""
    denom = 1.0 - entry_price if entry_price < 1.0 else 1e-6
    raw_kelly = edge / denom
    half_kelly = KELLY_FRACTION * raw_kelly
    half_kelly = max(0.0, min(half_kelly, 0.50))
    return round(SESSION_BANKROLL * half_kelly, 2)


# ── PnL ───────────────────────────────────────────────────────────────────────

def compute_pnl(bet: float, entry_price: float, outcome: int, direction: str) -> float:
    if direction == "YES":
        return bet * (1.0 / entry_price - 1.0) if outcome == 1 else -bet
    else:
        no_price = 1.0 - entry_price
        if no_price < 0.005:
            return 0.0
        return bet * (1.0 / no_price - 1.0) if outcome == 0 else -bet


# ── LR baseline prediction ────────────────────────────────────────────────────

def lr_predict(baseline_model, market: dict, coin_idx: int, market_type: str,
               entry_idx: int = None):
    if baseline_model is None:
        return None
    try:
        # Truncate history to entry point to avoid look-ahead bias
        if entry_idx is not None:
            history = market.get("price_history", [])
            market = {**market, "price_history": history[:entry_idx + 1]}
        df, _ = get_price_features_with_type(market, coin_idx, market_type)
        if df is None:
            return None
        try:
            df = df.reindex(columns=baseline_model.feature_names_in_, fill_value=0)
        except AttributeError:
            pass
        return float(baseline_model.predict_proba(df)[0][1])
    except Exception:
        return None


# ── Aggregate stats helper ────────────────────────────────────────────────────

def _max_drawdown(cum_pnl: np.ndarray) -> float:
    peak, max_dd = cum_pnl[0] if len(cum_pnl) else 0.0, 0.0
    for v in cum_pnl:
        peak = max(peak, v)
        dd = (peak - v) / (abs(peak) + 1e-9)
        max_dd = max(max_dd, dd)
    return max_dd


def strategy_stats(pnl_arr, bet_arr, correct_arr, label):
    pnl  = np.array([p for p in pnl_arr if p is not None], dtype=float)
    bets = np.array([b for b in bet_arr  if b is not None], dtype=float)
    corr = np.array([c for c in correct_arr if c is not None], dtype=float)
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


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(save_path: str = "data/backtest_results.json") -> dict:
    print("\n=== BACKTEST START ===\n")

    print("Loading ML models...")
    load_models()
    print(f"  {len(MODELS)} models loaded")

    baseline_model = None
    try:
        baseline_model = joblib.load("src/model/baseline_logistic.pkl")
        print("  baseline_logistic.pkl loaded")
    except FileNotFoundError:
        print("  WARNING: baseline_logistic.pkl not found — LR column will be empty")

    rf_model = None
    try:
        rf_model = joblib.load("src/model/baseline_rf.pkl")
        print("  baseline_rf.pkl loaded")
    except FileNotFoundError:
        print("  WARNING: baseline_rf.pkl not found — RF column will be empty")

    trades = []
    skipped = {"no_outcome": 0, "no_price": 0, "price_zone": 0, "volume": 0,
               "no_hist": 0, "no_model": 0, "no_features": 0, "edge_too_low": 0,
               "edge_too_high": 0, "wrong_type": 0}

    # reuse the same prediction function for both LR and RF
    def baseline_predict(model, m, cidx, mtype, entry_idx=None):
        return lr_predict(model, m, cidx, mtype, entry_idx)

    print(f"\nProcessing {len(HISTORY_FILES)} history files...")

    for fpath in HISTORY_FILES:
        market_type_from_file = os.path.basename(fpath).replace("price_history_", "").replace(".json", "")
        if market_type_from_file not in ALLOWED_TYPES:
            print(f"  Skipping {os.path.basename(fpath)} (type {market_type_from_file} not in ALLOWED_TYPES)")
            continue

        with open(fpath) as fp:
            file_data = json.load(fp)

        file_trades = 0
        file_seen   = 0

        for coin, markets in file_data.items():
            coin_idx = coin_idx_map.get(coin, 0)

            for m in markets:
                file_seen += 1

                # ── Ground truth ──
                outcome = get_outcome(m)
                if outcome is None:
                    skipped["no_outcome"] += 1
                    continue

                # ── Volume ──
                vol = float(m.get("volumeNum") or m.get("volume") or 0)
                if vol < MIN_VOLUME:
                    skipped["volume"] += 1
                    continue

                # ── Price history ──
                history = m.get("price_history", [])
                if len(history) < MIN_HIST_PTS:
                    skipped["no_hist"] += 1
                    continue

                # ── Entry price (early in the window) ──
                entry_price, entry_idx = get_market_price(m)
                if entry_price is None:
                    skipped["no_price"] += 1
                    continue

                # ── Price zone filter ──
                if not (PRICE_LO <= entry_price <= PRICE_HI):
                    skipped["price_zone"] += 1
                    continue

                # ── Market type ──
                question    = m.get("question", "")
                market_type = detect_market_type(question, m)
                if market_type not in ALLOWED_TYPES:
                    skipped["wrong_type"] += 1
                    continue

                # ── Our model ──
                model = MODELS.get(market_type) or MODELS.get("all")
                if model is None:
                    skipped["no_model"] += 1
                    continue

                # Truncate price history to entry point — no look-ahead bias
                if entry_idx is not None:
                    m_truncated = {**m, "price_history": history[:entry_idx + 1]}
                else:
                    m_truncated = m

                features_df, _ = get_price_features_with_type(m_truncated, coin_idx, market_type)
                if features_df is None:
                    skipped["no_features"] += 1
                    continue

                try:
                    aligned = features_df.reindex(columns=model.feature_names_in_, fill_value=0)
                except AttributeError:
                    aligned = features_df

                try:
                    our_estimate = float(model.predict_proba(aligned)[0][1])
                except Exception:
                    skipped["no_features"] += 1
                    continue

                our_edge = our_estimate - entry_price
                direction = "YES" if our_edge >= 0 else "NO"
                abs_edge  = abs(our_edge)

                if abs_edge < MIN_EDGE:
                    skipped["edge_too_low"] += 1
                    continue
                if abs_edge > MAX_EDGE:
                    skipped["edge_too_high"] += 1
                    continue

                # ── LR baseline ──
                lr_estimate = lr_predict(baseline_model, m, coin_idx, market_type, entry_idx)
                lr_edge     = (lr_estimate - entry_price) if lr_estimate is not None else None

                # ── Random Forest baseline ──
                rf_estimate = baseline_predict(rf_model, m, coin_idx, market_type, entry_idx)
                rf_edge     = (rf_estimate - entry_price) if rf_estimate is not None else None

                # ── Kelly sizing helper for this trade ──
                bet_price = entry_price if direction == "YES" else (1 - entry_price)
                def _kelly_bet_frac(frac: float) -> float:
                    """Dollar bet for a given Kelly fraction (0 = uniform FLAT_BET)."""
                    if frac == 0.0:
                        return FLAT_BET
                    denom = 1.0 - bet_price if bet_price < 1.0 else 1e-6
                    raw = abs_edge / denom
                    scaled = min(frac * raw, 0.50)
                    return round(SESSION_BANKROLL * scaled, 2)

                # ── Strategy 1: Our model + Full Kelly (1.0) ──
                full_bet = _kelly_bet_frac(1.0)
                full_pnl = compute_pnl(full_bet, entry_price, outcome, direction)

                # ── Strategy 2: Our model + Half-Kelly (0.5) — primary ──
                our_bet  = _kelly_bet_frac(0.5)
                our_pnl  = compute_pnl(our_bet, entry_price, outcome, direction)
                our_corr = int((direction == "YES" and outcome == 1) or
                               (direction == "NO"  and outcome == 0))

                # ── Strategy 3: Our model + Quarter-Kelly (0.25) ──
                qtr_bet = _kelly_bet_frac(0.25)
                qtr_pnl = compute_pnl(qtr_bet, entry_price, outcome, direction)

                # ── Strategy 4: LR + flat bet ──
                lr_pnl = lr_corr = None
                if lr_edge is not None:
                    lr_dir  = "YES" if lr_edge >= 0 else "NO"
                    lr_pnl  = compute_pnl(FLAT_BET, entry_price, outcome, lr_dir)
                    lr_corr = int((lr_dir == "YES" and outcome == 1) or
                                  (lr_dir == "NO"  and outcome == 0))

                # ── Strategy 5: Random Forest + flat bet ──
                rf_pnl = rf_corr = None
                if rf_edge is not None:
                    rf_dir  = "YES" if rf_edge >= 0 else "NO"
                    rf_pnl  = compute_pnl(FLAT_BET, entry_price, outcome, rf_dir)
                    rf_corr = int((rf_dir == "YES" and outcome == 1) or
                                  (rf_dir == "NO"  and outcome == 0))

                # ── Strategy 6: Our model + flat bet (fair comparison) ──
                our_flat_pnl = compute_pnl(FLAT_BET, entry_price, outcome, direction)

                # ── Strategy 7: Always YES + flat bet (naive benchmark) ──
                uni_pnl = compute_pnl(FLAT_BET, entry_price, outcome, "YES")

                # ── End date for charting ──
                end_date = (m.get("endDate") or "")[:10]

                trades.append({
                    "coin":         coin,
                    "question":     question[:80],
                    "market_type":  market_type,
                    "end_date":     end_date,
                    "market_price": round(entry_price, 4),
                    "our_estimate": round(our_estimate, 4),
                    "our_edge":     round(our_edge, 4),
                    "lr_estimate":  round(lr_estimate, 4) if lr_estimate is not None else None,
                    "lr_edge":      round(lr_edge, 4) if lr_edge is not None else None,
                    "rf_estimate":  round(rf_estimate, 4) if rf_estimate is not None else None,
                    "rf_edge":      round(rf_edge, 4) if rf_edge is not None else None,
                    "outcome":      outcome,
                    "direction":    direction,
                    "volume":       round(vol, 0),
                    # Half-Kelly (primary)
                    "our_bet":      our_bet,
                    "our_pnl":      round(our_pnl, 4),
                    "our_correct":  our_corr,
                    # Our model + flat bet (fair comparison)
                    "our_flat_bet": FLAT_BET,
                    "our_flat_pnl": round(our_flat_pnl, 4),
                    # Full Kelly
                    "full_bet":     full_bet,
                    "full_pnl":     round(full_pnl, 4),
                    # Quarter-Kelly
                    "qtr_bet":      qtr_bet,
                    "qtr_pnl":      round(qtr_pnl, 4),
                    # LR baseline
                    "lr_bet":       FLAT_BET if lr_edge is not None else 0,
                    "lr_pnl":       round(lr_pnl, 4) if lr_pnl is not None else None,
                    "lr_correct":   lr_corr,
                    # Random Forest baseline
                    "rf_bet":       FLAT_BET if rf_edge is not None else 0,
                    "rf_pnl":       round(rf_pnl, 4) if rf_pnl is not None else None,
                    "rf_correct":   rf_corr,
                    # Naive benchmark
                    "uniform_bet":  FLAT_BET,
                    "uniform_pnl":  round(uni_pnl, 4),
                })
                file_trades += 1

        print(f"  {os.path.basename(fpath)}: {file_seen} markets → {file_trades} valid trades")

    print(f"\n=== {len(trades)} valid trades total ===")
    print("Skip reasons:", {k: v for k, v in skipped.items() if v > 0})

    if not trades:
        print("\nNo trades found — cannot compute stats.")
        return {"trades": [], "summary": {}, "meta": {}}

    # ── Aggregate ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(trades)
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df = df.sort_values("end_date").reset_index(drop=True)

    stats_our      = strategy_stats(df["our_pnl"],      df["our_bet"],      df["our_correct"],  "Our Model (Half-Kelly)")
    stats_our_flat = strategy_stats(df["our_flat_pnl"], df["our_flat_bet"], df["our_correct"],  "Our Model (Flat $20)")
    stats_lr       = strategy_stats(df["lr_pnl"],       df["lr_bet"],       df["lr_correct"],   "LR Baseline (Flat $20)")
    stats_rf       = strategy_stats(df["rf_pnl"],       df["rf_bet"],       df["rf_correct"],   "Random Forest (Flat $20)")
    stats_uni      = strategy_stats(df["uniform_pnl"],  df["uniform_bet"],  [None]*len(df),     "Always-YES (Flat $20)")

    # Kelly fraction comparison — same model predictions, different sizing
    stats_full = strategy_stats(df["full_pnl"], df["full_bet"], df["our_correct"], "Full Kelly (1.0×)")
    stats_half = strategy_stats(df["our_pnl"],  df["our_bet"],  df["our_correct"], "Half Kelly (0.5×)")
    stats_qtr  = strategy_stats(df["qtr_pnl"],  df["qtr_bet"],  df["our_correct"], "Quarter Kelly (0.25×)")
    stats_flat = strategy_stats(df["uniform_pnl"], df["uniform_bet"], [None]*len(df), "Uniform (0×)")

    kelly_comparison = [stats_full, stats_half, stats_qtr, stats_flat]

    # Cumulative PnL by date — strategy comparison chart
    daily_our       = df.groupby("end_date")["our_pnl"].sum().cumsum()
    daily_our_flat  = df.groupby("end_date")["our_flat_pnl"].sum().cumsum()
    daily_lr        = df[df["lr_pnl"].notna()].groupby("end_date")["lr_pnl"].sum().cumsum()
    daily_rf        = df[df["rf_pnl"].notna()].groupby("end_date")["rf_pnl"].sum().cumsum()
    daily_uni       = df.groupby("end_date")["uniform_pnl"].sum().cumsum()

    all_dates  = sorted(set(daily_our.index) | set(daily_our_flat.index) | set(daily_lr.index) | set(daily_rf.index) | set(daily_uni.index))
    chart = []
    for d in all_dates:
        chart.append({
            "date":      str(d)[:10],
            "our":       round(float(daily_our[d]),      2) if d in daily_our.index      else None,
            "our_flat":  round(float(daily_our_flat[d]), 2) if d in daily_our_flat.index else None,
            "lr":        round(float(daily_lr[d]),       2) if d in daily_lr.index       else None,
            "rf":        round(float(daily_rf[d]),       2) if d in daily_rf.index       else None,
            "uniform":   round(float(daily_uni[d]),      2) if d in daily_uni.index      else None,
        })

    # Cumulative PnL by date — Kelly fraction chart
    daily_full = df.groupby("end_date")["full_pnl"].sum().cumsum()
    daily_half = df.groupby("end_date")["our_pnl"].sum().cumsum()
    daily_qtr  = df.groupby("end_date")["qtr_pnl"].sum().cumsum()
    daily_flat = df.groupby("end_date")["uniform_pnl"].sum().cumsum()

    kelly_dates = sorted(set(daily_full.index) | set(daily_half.index) | set(daily_qtr.index) | set(daily_flat.index))
    kelly_chart = []
    for d in kelly_dates:
        kelly_chart.append({
            "date":    str(d)[:10],
            "full":    round(float(daily_full[d]), 2) if d in daily_full.index else None,
            "half":    round(float(daily_half[d]), 2) if d in daily_half.index else None,
            "quarter": round(float(daily_qtr[d]),  2) if d in daily_qtr.index  else None,
            "uniform": round(float(daily_flat[d]), 2) if d in daily_flat.index else None,
        })

    # Per-coin breakdown
    coin_breakdown = []
    for coin, grp in df.groupby("coin"):
        coin_breakdown.append({
            "coin":      coin,
            "n_trades":  len(grp),
            "our_pnl":   round(grp["our_pnl"].sum(), 2),
            "win_rate":  round((grp["our_pnl"] > 0).mean() * 100, 1),
            "mean_edge": round(grp["our_edge"].abs().mean() * 100, 2),
        })

    result = {
        "summary": {
            "our_model":      stats_our,
            "our_model_flat": stats_our_flat,
            "lr_baseline":    stats_lr,
            "rf_baseline":    stats_rf,
            "uniform":        stats_uni,
        },
        "kelly_comparison": kelly_comparison,
        "kelly_chart":      kelly_chart,
        "cumulative_chart": chart,
        "coin_breakdown":   coin_breakdown,
        "trades":           trades,
        "meta": {
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "n_markets_seen":  sum(skipped.values()) + len(trades),
            "n_valid_trades":  len(trades),
            "kelly_fraction":  KELLY_FRACTION,
            "session_bankroll": SESSION_BANKROLL,
            "price_filter":    [PRICE_LO, PRICE_HI],
            "edge_filter":     [MIN_EDGE, MAX_EDGE],
            "allowed_types":   list(ALLOWED_TYPES),
            "data_source":     "data/raw/price_history_*.json",
        },
    }

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved → {save_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("BACKTEST SUMMARY")
    print("="*55)
    for s in [stats_our, stats_our_flat, stats_lr, stats_rf, stats_uni]:
        print(f"\n{s['label']}")
        print(f"  Trades:     {s['n_trades']}")
        print(f"  Total PnL:  ${s['total_pnl']:+.2f}")
        print(f"  ROI:        {s['roi_pct']:+.2f}%")
        print(f"  Win rate:   {s['win_rate_pct']:.1f}%")
        print(f"  Dir. Acc:   {s['direction_accuracy_pct']:.1f}%")
        print(f"  Sharpe:     {s['sharpe']:.3f}")
        print(f"  Max DD:     {s['max_drawdown_pct']:.1f}%")
    print("="*55)
    print("\nKELLY FRACTION COMPARISON (same model, different sizing)")
    print("="*55)
    for s in kelly_comparison:
        print(f"\n{s['label']}")
        print(f"  Total PnL:  ${s['total_pnl']:+.2f}   ROI: {s['roi_pct']:+.2f}%   Sharpe: {s['sharpe']:.3f}   Max DD: {s['max_drawdown_pct']:.1f}%")
    print("="*55)

    return result


if __name__ == "__main__":
    run_backtest()
