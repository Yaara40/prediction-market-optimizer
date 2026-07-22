# Prediction Market Optimizer — Full Project Report

## Overview

A full-stack ML system that scans live Polymarket crypto prediction markets, generates probability estimates using trained gradient-boosting models, and allocates capital across markets using Kelly Criterion + Markowitz portfolio theory. The system comprises a Python FastAPI backend and a React/TypeScript frontend.

---

## Phase 1: Data Collection

### What we scraped

Polymarket hosts hundreds of binary prediction markets for crypto assets (BTC, ETH, SOL, XRP, BNB, DOGE, HYPE). Each market asks a yes/no question like:

- "Bitcoin above $65,000 on July 18, 11AM ET?" (1-hour market)
- "ETH up or down — July 18, 10AM-2PM ET?" (4-hour market)
- "What price will BTC hit July 13–19?" (weekly market)

For each resolved market we collected:
- **Price trajectory** — the market's YES price over time (CLOB order book history), sampled into 10 evenly-spaced snapshots
- **Outcome** — did YES resolve (1) or NO (0)?
- **Metadata** — volume, market type, start/end dates

### Market types discovered

| Type | Duration | Example |
|------|----------|---------|
| 5min | ≤ 5 min | "BTC above X at 10:55AM–11:00AM" |
| 15min | ≤ 15 min | "BTC up or down 10:45–11:00AM" |
| 1hour | ~1 hr | "BTC above X on July 18, 11AM ET" |
| 4hour | ~4 hr | "ETH up or down - July 18, 10AM–2PM ET" |
| 1day | ~1 day | "BTC above X on July 20?" |
| weekly | ~1 week | "What price will BTC hit July 13–19?" |
| ~~monthly~~ | ~~~1 month~~ | ~~Removed — model poorly calibrated~~ |

### Raw data files

```
data/raw/
  price_history_1hour.json   — 1,551 resolved 1-hour markets
  price_history_4hour.json   — 490 resolved 4-hour markets
  price_history_1day.json    — 1,642 resolved daily markets
  price_history_5min.json    — 2,442 resolved 5-min markets
  price_history_15min.json   — 3,072 resolved 15-min markets
  price_history_weekly.json  — 5,167 resolved weekly markets
  oos_1hour.json             — June 2026 1-hour markets (out-of-sample)
  oos_4hour.json             — June 2026 4-hour markets (out-of-sample)
```

Total: ~13,000+ resolved markets across all timeframes.

---

## Phase 2: Feature Engineering

For each resolved market we extracted 18 features:

| Feature | Description |
|---------|-------------|
| `price_t0` … `price_t9` | YES price at 10 evenly-spaced timestamps (0%–100% through window) |
| `price_early` | Price at 5% of window (opening signal) |
| `price_mid` | Price at 50% (mid-session) |
| `price_late` | Price at 95% (near-close signal) |
| `crossings_05` | Number of times the price crossed 0.5 (volatility proxy) |
| `duration_minutes` | Total market window length |
| `volume` | Trading volume |
| `coin` | Integer-encoded coin (BTC=0, ETH=1, …) |
| `market_type` | Integer-encoded timeframe (0=5min … 7=all) |

**Label**: 1 if YES resolved, 0 if NO resolved.

The trajectory features (price_t0–price_t9) are the most informative — they capture whether the market is trending toward YES or NO and how much movement occurred. The `price_late` feature (near the close) is typically the strongest single predictor.

---

## Phase 3: Model Training

### Approach: one specialized model per timeframe

We trained **four candidate classifiers** on each timeframe's dataset:
- XGBoost
- LightGBM
- CatBoost
- Random Forest

The best model per timeframe was selected by lowest **Brier score** (probability calibration loss — penalizes confident wrong predictions more than unsure wrong ones).

### Dataset sizes (after feature engineering)

| Timeframe | Training rows |
|-----------|--------------|
| 5min | 2,442 |
| 15min | 3,072 |
| 1hour | 1,551 |
| 4hour | 490 |
| 1day | 1,642 |
| weekly | 5,167 |
| all (combined) | 14,364 |

### Why separate models per timeframe?

1-hour markets look completely different from weekly markets. A 1-hour price trajectory is almost purely driven by momentum and micro-liquidity, while a weekly market trajectory reflects multi-day supply/demand. One model trained on all timeframes blends these signals and becomes mediocre at both. Separate models specialize.

### Model files

```
src/model/
  best_model_1hour.pkl   (LightGBM or CatBoost, selected by Brier score)
  best_model_4hour.pkl
  best_model_1day.pkl
  best_model_weekly.pkl
  best_model_5min.pkl
  best_model_15min.pkl
  best_model_all.pkl     (combined model, fallback)
```

---

## Phase 4: Post-Hoc Calibration (Isotonic Regression)

### The problem: raw GBM outputs are not true probabilities

Gradient Boosting models output scores, not calibrated probabilities. In our analysis, when the model predicted "0.6 (60% YES)", the actual YES rate in the data was only ~30%. The model was systematically overconfident about YES outcomes.

**Calibration errors (ECE) before and after:**

| Model | Pre-calibration ECE | Post-calibration ECE |
|-------|--------------------|--------------------|
| 5min | 0.0911 | ~0.000 |
| 15min | 0.0596 | ~0.000 |
| 1hour | 0.0818 | ~0.000 |
| 4hour | 0.1514 | ~0.000 |
| 1day | 0.1569 | ~0.000 |
| weekly | 0.1128 | ~0.000 |
| all | 0.0212 | ~0.000 |

### How isotonic calibration works

1. Take the trained model and a held-out calibration set (20% of training data, not used during training)
2. Get raw model probability scores for each calibration example
3. Fit `IsotonicRegression(raw_scores → actual_YES_frequencies)` — this learns a monotone piecewise-constant lookup table that maps e.g. raw 0.62 → empirical 0.31
4. Save the calibrator as `calibrator_1hour.pkl` alongside the model
5. At prediction time: `calibrated_prob = calibrator.predict([raw_score])[0]`

The calibrator is now applied in `predict_probabilities()` in `pipeline.py`.

---

## Phase 5: Backtesting

### In-sample backtest (historical data)

We replayed all 1,342 resolved markets using our model's predictions and compared four strategies, all betting **flat $20 per trade** for fair comparison:

| Strategy | Trades | ROI | Win Rate | Sharpe | Max Drawdown |
|----------|--------|-----|----------|--------|-------------|
| **Our Model (Half-Kelly)** | 1,342 | **+16.66%** | 56.11% | 2.41 | 234% |
| Our Model (Flat $20) | 1,342 | +11.93% | 56.11% | 1.91 | 206% |
| LR Baseline (Flat $20) | 1,342 | -14.60% | 42.55% | -2.32 | 1,875% |
| Random Forest (Flat $20) | 1,342 | -26.11% | 36.66% | -4.24 | 6,362% |
| Always-YES (Flat $20) | 1,342 | -44.82% | 27.57% | -7.92 | 10,829% |

Key observation: The LR and RF baselines have **inverted** signal — they systematically pick the wrong direction. Our model's 56.1% win rate is modest but consistent.

### Out-of-sample evaluation (June 2026)

True OOS: models trained on pre-June data, evaluated on 2,290 June 2026 markets (never seen during training).

| Strategy | Trades | ROI | Win Rate | Sharpe | Max Drawdown |
|----------|--------|-----|----------|--------|-------------|
| **Our Model** | 2,290 | **+25.22%** | 62.66% | 4.00 | 33.82% |
| LR Baseline | 2,290 | +20.88% | 60.57% | 3.29 | 90.25% |
| Random Forest | 2,290 | +19.91% | 59.87% | 3.11 | 82.39% |
| Always-YES | 2,290 | -6.47% | 46.46% | -1.01 | 14,708% |

In OOS, all ML models perform well — but our ensemble leads with +25.22% ROI and the lowest max drawdown (33.82% vs 90%+ for baselines). This is the most honest performance number.

---

## Phase 6: Kelly-Markowitz Portfolio Optimizer

### The Kelly Criterion

For each market we compute:

```
edge = our_estimate − market_price
f* = edge / (1 − market_price)   [raw Kelly fraction]
```

This tells us what fraction of our bankroll to bet on each market to maximize long-run growth. We scale it by a user-chosen **Kelly fraction**:
- Risk 1–3: Quarter-Kelly (0.25×) — conservative
- Risk 4–6: Half-Kelly (0.50×) — balanced
- Risk 7–8: ¾-Kelly (0.75×) — aggressive
- Risk 9–10: Full-Kelly (1.00×) — maximum growth, high variance

### Markowitz Correlation Penalty

We penalize allocating capital to multiple assets that are highly correlated (e.g., BTC and ETH move together). The penalty is:

```
penalty = λ × (weights · Correlation_Matrix · weights)
```

At low risk levels, λ is large (force diversification). At high risk, λ → 0 (concentrate into best edge).

Correlation comes from 90-day CoinGecko price histories.

### Multiple markets per coin

Previously the system only kept ONE market per coin (the "best" one). Now ALL eligible markets per coin pass through. A coin with both a 1-hour and a 4-hour market with positive edge contributes two separate positions. Same-coin markets get a high correlation (0.95) in the matrix — the Markowitz penalty naturally limits over-concentration.

### Eligibility filters

Markets must pass:
- Edge > 2% (our model beats market by at least 2%)
- Market price between 20% and 80% (avoid near-certain outcomes)
- Edge ≤ 15% (above this is model error, not real alpha)
- Not monthly (model poorly calibrated for 1-month horizons)

---

## Phase 7: Live System Architecture

```
Polymarket API ──► fetch_active_markets()
                       │  (7 coins × N markets each)
                       ▼
              enrich_markets_with_history()
                       │  (CLOB order book → price trajectory)
                       ▼
              predict_probabilities()
                       │  (per-timeframe model + isotonic calibrator)
                       ▼
              Eligibility filter (edge, price range, type)
                       │
                       ▼
           Kelly-Markowitz optimizer
                       │
                       ▼
              FastAPI /api/optimize
                       │
                       ▼
           React frontend (OptimizerPage)
```

### Tech stack

**Backend**: Python 3.14, FastAPI, LightGBM/CatBoost/XGBoost, scikit-learn, joblib, scipy, numpy, pandas

**Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Recharts, react-router-dom

---

## Phase 8: Frontend Pages

### Dashboard
- Live stats: active markets, opportunities (positive-edge count), avg edge, model accuracy
- Top 5 opportunities by edge with EdgeBar visualization
- "How It Works" section explaining the algorithm

### Optimizer
- Risk slider (1–10) and amount presets ($100/$500/$1k/$5k)
- Real-time Kelly-Markowitz portfolio allocation
- Allocation chart (pie) + table (position, edge, confidence, market question)
- Live model vs baseline comparison

### Markets
- All current live markets across 7 coins
- Filter by coin and by timeframe (1hr / 4hr / daily / weekly)
- Sortable columns: price, model estimate, edge, volume

### History (Backtest)
- Cumulative PnL chart comparing 5 strategies
- Kelly fraction comparison (Full / Half / Quarter / Uniform)
- Per-coin breakdown
- Out-of-sample evaluation section with June 2026 results

---

## Key Numbers Summary

| Metric | Value |
|--------|-------|
| Total resolved markets scraped | ~13,000 |
| In-sample trades backtested | 1,342 |
| OOS trades (June 2026) | 2,290 |
| In-sample ROI (Half-Kelly) | +16.66% |
| OOS ROI (Flat $20) | +25.22% |
| OOS Sharpe ratio | 4.00 |
| OOS Max drawdown | 33.82% |
| OOS Win rate | 62.66% |
| Coins tracked live | 7 (BTC, ETH, SOL, XRP, BNB, DOGE, HYPE) |
| Market types supported | 6 (5min, 15min, 1hr, 4hr, daily, weekly) |

---

## What We Tried and What Didn't Work

| Approach | Outcome |
|----------|---------|
| Single model for all timeframes | Mediocre — 1hr and weekly have different dynamics |
| Monthly markets | Dropped — model output ~0.99 for everything, useless |
| Raw GBM probabilities (no calibration) | YES bias: model predicted 58.9% YES but only 27.6% resolved YES |
| Picking only 1 market per coin | Missed diversification opportunities; 3 markets for same coin with different edges were all discarded |
| LR baseline as primary model | Negative ROI — LR can't capture nonlinear price momentum patterns |
| Isotonic calibration | Fixed YES bias, improved probability estimates across all timeframes |
