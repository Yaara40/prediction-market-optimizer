# Prediction Market Optimizer — Full Project Report

**Authors:** Yaara Yizchaki Tzabari, Omer Cohen
**Institution:** Ono Academic College — Computer Science, Final Project, 2026
**Repositories:** [ML + API](https://github.com/Yaara40/prediction-market-optimizer) · [Frontend](https://github.com/Yaara40/robo-advisor-frontend)

---

## 1. Introduction and Motivation

Binary prediction markets let participants bet on yes/no outcomes — for example, "Will Bitcoin be above $65,000 on July 18 at 11AM ET?" The market price of a YES share is the crowd's implied probability that the outcome resolves YES. If a share trades at 0.62, the crowd believes there is a 62% chance of YES.

These markets are supposedly efficient — the crowd aggregates information and prices should reflect true probabilities. But efficiency is never perfect, especially in short-horizon crypto markets where liquidity is thin, participants are unsophisticated, and prices are driven by momentum and anchoring biases rather than fundamental analysis.

Our hypothesis was simple: **the price trajectory during a market's trading window contains predictive signal that the crowd systematically underweights.** A market that opened at 0.50, drifted to 0.72, and then pulled back to 0.65 has a very different profile from one that opened at 0.65 and never moved. A machine learning model trained on thousands of resolved markets could learn these patterns and find systematic mispricings.

The goal of this project was to build that system end-to-end — from data collection to live portfolio allocation — and validate it on real out-of-sample data.

---

## 2. Where We Started

The project began with a single question: can ML beat the Polymarket crowd on crypto prediction markets?

We started by writing a basic scraper to download resolved markets from the Polymarket Gamma API. The earliest commits (`add polymarket data fetching script`, `added more labels to train the data`) show us exploring what data was even available and what format it came in. At this stage we had no feature engineering, no models, and no clear sense of whether there was any signal at all.

The first attempt trained a TabPFN model on a small dataset of 12h–7day markets (`add trained TabPFN model and training results`). TabPFN is a prior-fitted neural network designed for small tabular datasets. It produced predictions but was slow and could not be easily deployed in a live system. We removed it (`remove large model files`, `updates on pipeline and train - removing tabpfn`) and switched to standard gradient boosting.

---

## 3. Phase 1 — Data Collection

### 3.1 What We Scraped

Polymarket exposes a public REST API (`gamma-api.polymarket.com`) with endpoints for resolved and active markets. We built dedicated scraping scripts for each market type:

- `fetch_1hour_markets.py` — markets with ~1 hour windows ("Bitcoin above $X on July 18, 11AM ET?")
- `fetch_4hour_markets.py` — "Up or Down" markets over 4-hour windows
- `fetch_daily_markets.py` — next-day price targets
- `fetch_weekly_markets.py` — weekly price range buckets

For each resolved market we also fetched its **CLOB (Central Limit Order Book) price history** — the tick-by-tick YES price over the entire trading window — from `clob.polymarket.com/prices-history`. This is the core signal.

### 3.2 Market Types Discovered

One of the first things we had to figure out was that Polymarket markets are not uniform. They come in wildly different structures:

| Type | Duration | Example Question |
|------|----------|-----------------|
| 5min | ≤ 5 minutes | "BTC above X at 10:55AM–11:00AM ET" |
| 15min | ≤ 15 minutes | "BTC up or down 10:45–11:00AM" |
| 1hour | ~1 hour | "BTC above $65k on July 18, 11AM ET?" |
| 4hour | ~4 hours | "ETH up or down — July 18, 10AM–2PM ET?" |
| 1day | ~1 day | "Will BTC be above $64k on July 20?" |
| weekly | ~1 week | "What price will BTC hit July 13–19?" |
| monthly | ~1 month | Dropped — model was poorly calibrated |

The commit `add 4hour and 1hour fetch scripts, confirm no 1hour market type exists` captures an important discovery: Polymarket does not use a consistent `market_type` field. We had to reverse-engineer the type from the question text and duration using regex parsing.

### 3.3 Dataset Scale

After scraping, the raw dataset across all timeframes:

| Timeframe | Resolved Markets |
|-----------|----------------|
| 5min | 2,442 |
| 15min | 3,072 |
| 1hour | 1,551 |
| 4hour | 490 |
| 1day | 1,642 |
| weekly | 5,167 |
| **Total** | **~13,138** |

Additionally, a separate out-of-sample (OOS) set was collected for June 2026 — markets that resolved after the training cutoff, never seen during model development.

---

## 4. Phase 2 — Feature Engineering

### 4.1 The Core Insight: Price Trajectory as Signal

The most important engineering decision was how to represent the price history. A raw time series of variable length cannot be fed directly to a gradient boosting model. We needed fixed-length features.

Our solution: **sample the price trajectory at 10 evenly-spaced timestamps** across the trading window (0%, 11%, 22%, ..., 100%). This gives 10 features (`price_t0` through `price_t9`) that capture the shape of the price movement regardless of how long the market ran.

Additionally we extracted:
- `price_early` — price at 5% of window (opening signal)
- `price_mid` — price at 50% (mid-session)
- `price_late` — price at 95% (near-close signal, typically the strongest predictor)
- `crossings_05` — number of times price crossed 0.5 (volatility/uncertainty proxy)
- `duration_minutes` — length of the trading window
- `volume` — trading volume
- `coin` — integer-encoded coin identity (BTC=0, ETH=1, SOL=2, ...)
- `market_type` — integer-encoded timeframe

**Total: 18 features per market. Label: 1 if YES resolved, 0 if NO.**

### 4.2 Why `price_late` Matters Most

Intuitively this makes sense: the market price near the close of the window already incorporates most available information. If BTC is at $64,800 and the question asks "above $65,000?" with 5 minutes to go, the YES price will naturally be very low. The trajectory features give the model the full arc — not just where the price ended up, but how it got there.

---

## 5. Phase 3 — Model Training

### 5.1 The Problem with One Model

Our first working model (`working pipeline with catboost model and kelly markowitz optimizer`) was a single CatBoost model trained on all timeframes combined. It worked but was mediocre. The reason is fundamental: a 1-hour market and a weekly market have completely different dynamics.

A 1-hour price trajectory is driven almost entirely by momentum and micro-liquidity — the crowd reacts to a single price move. A weekly market reflects multi-day supply/demand, news events, and macro sentiment. A single model blends these signals and becomes mediocre at both.

### 5.2 Per-Timeframe Specialization

The commit `train separate models for 5min and 15min with price history features` was the pivot. We then extended this across all timeframes (`train 7 specialized models for all market duration types`, `complete pipeline with 8 models, 7 coins, cached correlation matrix`).

For each timeframe we trained **four candidate classifiers**:
- **XGBoost** — fast, well-regularized
- **LightGBM** — efficient on larger datasets
- **CatBoost** — handles categorical features natively
- **Random Forest** — ensemble baseline

Each classifier was evaluated by **Brier score** (probability calibration loss, not accuracy), and the best model per timeframe was saved. Brier score penalizes confident wrong predictions heavily — critical for a system that uses these probabilities to size bets.

### 5.3 Model Selection Results

CatBoost and LightGBM dominated across timeframes. Random Forest consistently had the worst Brier score due to its tendency to produce extreme probability outputs.

---

## 6. Phase 4 — Post-Hoc Calibration

### 6.1 The Problem: GBMs Are Overconfident

After training, we noticed something concerning: when the model predicted 0.62 (62% YES), the actual YES rate in the data was only ~30%. The model was systematically overconfident about YES outcomes. This is a well-known property of gradient boosting models — they output scores, not true probabilities.

If we used these raw scores as probabilities in the Kelly formula, we would massively overbet. A 62% estimate → 20% edge calculation → large bet, when the true edge is near zero.

**Calibration errors (ECE — Expected Calibration Error) before calibration:**

| Timeframe | Pre-Calibration ECE |
|-----------|-------------------|
| 5min | 0.091 |
| 15min | 0.060 |
| 1hour | 0.082 |
| 4hour | 0.151 |
| 1day | 0.157 |
| weekly | 0.113 |

### 6.2 Isotonic Regression Calibration

We applied **isotonic regression** post-hoc:

1. Hold out 20% of the training data as a calibration set (never used for training)
2. Get raw model scores for each calibration example
3. Fit `IsotonicRegression(raw_scores → actual YES frequencies)` — a monotone piecewise-constant lookup table
4. Save as `calibrator_{timeframe}.pkl`
5. At prediction time: raw score → calibrator → calibrated probability

**After calibration: ECE ≈ 0.000 across all timeframes.** The model's 62% prediction now truly reflects a ~62% empirical YES rate in the calibration data.

This step was essential for the Kelly formula to produce sensible bet sizes.

---

## 7. Phase 5 — Backtesting and the Look-Ahead Bias Problem

### 7.1 Initial (Inflated) Results

Our first backtest showed spectacular numbers: +16.66% ROI in-sample, 56.1% win rate, Sharpe 2.41. The OOS results were even more impressive: +25.22% ROI, 62.66% win rate, Sharpe 4.00.

These numbers felt too good. We audited the code.

### 7.2 The Look-Ahead Bias Bug

The bug was in how we determined the **entry price** — the market price at the moment we would theoretically place a bet. The function `get_market_price()` was using the price at a point early in the trading window, but the `predict_probabilities()` function was receiving the **full price history** including all prices after that entry point.

In other words: the model was being asked "what is the probability of YES?" while having access to everything that happened in the market after we would have placed our bet. This is look-ahead bias — using future information to make past predictions.

**The fix** (applied to both `backtest.py` and `oos_eval.py`):

1. `get_market_price()` was changed to return `(price, entry_idx)` — both the entry price and the index in the price history at which we would enter
2. Before calling any model, we truncate the price history: `history[:entry_idx + 1]`
3. The model never sees any price data that occurred after our entry point

### 7.3 Real Results After Fix

The numbers dropped substantially — as expected when you remove the ability to see the future:

**In-Sample (1,505 trades, Half-Kelly sizing):**

| Strategy | ROI | Win Rate | Sharpe |
|----------|-----|----------|--------|
| **Our Model (Half-Kelly)** | **+2.26%** | 46.4% | 0.318 |
| Our Model (Flat $20) | −8.56% | 46.4% | −1.378 |
| LR Baseline (Flat $20) | −26.93% | 37.5% | −4.515 |
| Random Forest (Flat $20) | −35.36% | 32.7% | −6.018 |
| Always-YES (Flat $20) | −43.47% | 28.5% | −7.663 |

**Out-of-Sample — June 2026 (2,209 trades):**

| Strategy | ROI | Win Rate | Sharpe |
|----------|-----|----------|--------|
| **Our Model (Half-Kelly)** | **+0.49%** | 52.7% | 0.081 |
| LR Baseline | −2.09% | 51.6% | −0.347 |
| Random Forest | −3.44% | 49.3% | −0.550 |
| Always-YES | −5.20% | — | — |

These are the honest numbers. The model wins because Kelly sizing is asymmetric: when edge is large, the bet is large; when edge is small, the bet is small. Even at a sub-50% win rate in-sample, the wins are systematically larger than the losses.

The OOS win rate jumps to 52.7% — the model generalizes better to June 2026 data than to the in-sample period, suggesting the in-sample period may have had harder market conditions.

---

## 8. Phase 6 — Kelly-Markowitz Portfolio Optimizer

### 8.1 The Kelly Criterion

For each market the formula is:

```
edge   = our_estimate − market_price
f*     = edge / (1 − market_price)    [raw Kelly fraction]
f_bet  = f* × kelly_multiplier        [scaled by risk level]
```

The user controls risk through a slider (1–10):
- Risk 1–3: Quarter-Kelly (0.25×) — very conservative
- Risk 4–6: Half-Kelly (0.50×) — balanced default
- Risk 7–8: ¾-Kelly (0.75×) — aggressive
- Risk 9–10: Full-Kelly (1.00×) — maximum growth, high variance

### 8.2 The Markowitz Correlation Penalty

Simply maximizing Kelly fractions independently ignores correlation between markets. If BTC has a 1-hour market and a 4-hour market both with positive edge, betting full Kelly on both concentrates heavily in BTC's direction — effectively doubling down on a single coin's movement.

We penalize this using a Markowitz-style quadratic penalty:

```
minimize: −expected_return + λ × (weights · CorrelationMatrix · weights)
```

The correlation matrix has three types of entries:
- Same-coin, different timeframe: 0.95 (highly correlated — same underlying asset)
- Different coins: pulled from 90-day CoinGecko price correlation
- Unknown pairs: 0.50 (conservative default)

At low risk levels λ is large → the optimizer forces diversification. At high risk levels λ → 0 → the optimizer concentrates into the single best edge.

### 8.3 Eligibility Filters

Not every market with a positive model estimate is traded. Four filters are applied:

- **Edge > 2%**: our model must beat the market price by at least 2 percentage points
- **Market price 20%–80%**: avoid near-certain outcomes where the market is already very sure
- **Edge ≤ 15%**: edges above 15% almost certainly indicate model miscalibration, not real alpha
- **Not monthly**: the monthly model was poorly calibrated (outputs near 1.0 for everything) — excluded entirely

---

## 9. Phase 7 — Live System

### 9.1 Architecture

The live system runs as a FastAPI backend that the React frontend communicates with:

```
Polymarket Gamma API
    │
    ▼
fetch_active_markets()          — 7 coins, all eligible timeframes
    │
    ▼
enrich_markets_with_history()   — CLOB price history per market (2-min cache)
    │
    ▼
predict_probabilities()         — per-timeframe GBM + isotonic calibrator
    │
    ▼
Eligibility filter              — edge, price range, type, no look-ahead
    │
    ▼
kelly_markowitz optimizer       — SLSQP constrained optimization
    │
    ▼
FastAPI /api/optimize           — JSON response to frontend
    │
    ▼
React OptimizerPage             — allocation table, pie chart, report
```

### 9.2 Caching Strategy

Fetching all live markets and enriching them with CLOB history takes 10–30 seconds. We implemented a two-layer cache:

- **Markets cache**: 5-minute TTL — active markets change slowly
- **CLOB history cache**: 2-minute TTL — prices update frequently during trading windows
- **Active markets cache**: 1-minute TTL — used during live trading

The `/api/optimize` endpoint always force-refreshes (bypasses cache) to get the freshest possible signal for portfolio decisions.

### 9.3 Key Engineering Challenges

**Parsing market types from question text:** Polymarket doesn't provide a clean `market_type` field. We built regex parsers to classify each market by its question structure. `_is_short_window()` identifies 5-min and 15-min markets from time patterns like "10:55AM-11:00AM". Market duration is computed from the parsed start/end times.

**Multiple markets per coin:** Early versions kept only one market per coin. Later versions (`complete pipeline with 8 models, 7 coins`) pass all eligible markets through. The Markowitz penalty handles concentration automatically — two BTC markets get correlation 0.95, which penalizes over-weighting BTC heavily.

**Look-ahead prevention in live mode:** The live system naturally avoids look-ahead because we only have price history up to the current moment. The bug only manifested in backtesting where the full history was available. After the fix, `get_market_price()` returns `(price, entry_idx)` and all model calls truncate `history[:entry_idx+1]`.

---

## 10. Phase 8 — Frontend

The React + TypeScript frontend (`robo-advisor-frontend`) was built alongside the backend and provides four main pages:

### Dashboard
Live overview: active market count, opportunity count (positive-edge markets), average edge across opportunities, and model win rate (pulled dynamically from the latest backtest results JSON — not hardcoded). The bottom section shows comparison metrics between our model and baselines, updated to reflect the post-look-ahead-bias numbers.

### Optimizer
The core use case. Users set a risk level (1–10 slider) and an amount to allocate ($100 to $5,000 presets). The backend runs the full pipeline and returns a Kelly-Markowitz allocation. The frontend shows a pie chart of allocations, a table with position details (edge, confidence, market question, dollar amount), and a written reasoning report generated by the API.

### Markets
Table of all live markets across 7 coins with filtering by coin and timeframe. Shows market price, model estimate, edge, and volume. Sorted by edge descending so the best opportunities appear at the top.

### History (Performance)
Cumulative PnL chart comparing 5 strategies over time: Our Model (Kelly), Our Model (Flat $20), LR Baseline, Random Forest, Always-YES. Strategy comparison table with ROI, win rate, Sharpe, and drawdown. The Kelly line is shown as a solid bright green line; flat-bet strategies are dashed. Numbers shown are the real post-fix values.

---

## 11. What We Tried That Didn't Work

| Approach | What Happened | What We Did Instead |
|----------|--------------|-------------------|
| TabPFN as primary model | Slow, couldn't be deployed live, no clear improvement over GBM | Switched to CatBoost/LightGBM/XGBoost |
| Single model for all timeframes | Mediocre Brier score — 1h and weekly have completely different signal profiles | Per-timeframe specialized models |
| Monthly markets | Model output ~0.99 for nearly everything — useless and dangerous in Kelly | Dropped monthly entirely |
| Raw GBM probabilities (no calibration) | YES bias: model predicted 58.9% YES but only 27.6% resolved YES → overbetting | Isotonic regression calibration → ECE ≈ 0 |
| Only one market per coin | Missed diversification — 3 BTC markets with different edges all discarded | All eligible markets pass through, Markowitz handles correlation |
| Keeping look-ahead in backtest | Inflated results (+16.66% in-sample, +25.22% OOS) — model could see future prices | Fixed: truncate history to entry point before any model call |
| Hardcoded performance numbers in UI | Numbers went stale after look-ahead fix, dashboard showed wrong stats | Numbers pulled dynamically from backtest results JSON |

---

## 12. Key Design Decisions and Why

**Why Kelly Criterion?** Kelly maximizes the geometric growth rate of the bankroll over many bets. It is the mathematically optimal bet sizing strategy when you have a calibrated probability estimate. Flat betting (same amount per trade) treats all edges equally — a 3% edge and a 10% edge get the same bet. Kelly sizes proportionally, so larger edges get larger bets.

**Why Markowitz penalty?** Without correlation penalty, the optimizer would concentrate entirely into whichever single market has the highest Kelly fraction. This creates catastrophic drawdown risk if that market resolves against us. The Markowitz penalty forces the optimizer to spread across multiple markets in proportion to how independently they move.

**Why per-timeframe models?** The price trajectory of a 1-hour market tells a fundamentally different story than a weekly market. In a 1-hour market, the trajectory is almost entirely driven by momentum — the price is either running toward YES or drifting away. In a weekly market, multi-day patterns, macro events, and range-trading dominate. A single model confuses these signals.

**Why isotonic regression (not Platt scaling)?** Platt scaling assumes a sigmoid relationship between raw scores and probabilities. GBM outputs can have arbitrary non-monotone relationships with true probabilities. Isotonic regression is non-parametric and only assumes monotonicity — a weaker and safer assumption for tree-based models.

**Why 2% minimum edge filter?** Market prices include a spread (the difference between best bid and best ask). On thin markets this can be 2–4%. An edge below 2% would frequently be consumed by the spread, making the trade unprofitable in practice.

---

## 13. Results Summary

| Metric | Value |
|--------|-------|
| Total resolved markets scraped | 13,138 |
| In-sample trades (Half-Kelly) | 1,505 |
| In-sample ROI | **+2.26%** |
| In-sample Sharpe | 0.318 |
| In-sample win rate | 46.4% |
| OOS trades (June 2026) | 2,209 |
| OOS ROI (Half-Kelly) | **+0.49%** |
| OOS win rate | 52.7% |
| OOS Sharpe | 0.081 |
| LR Baseline OOS ROI | −2.09% |
| RF Baseline OOS ROI | −3.44% |
| Always-YES OOS ROI | −5.20% |
| Coins tracked live | 7 (BTC, ETH, SOL, XRP, BNB, DOGE, HYPE) |
| Market types supported | 4 live (1h, 4h, 1day, weekly) |
| ECE after calibration | ~0.000 |

---

## 14. Conclusions

This project demonstrates that machine learning can find systematic mispricings in short-horizon crypto prediction markets — but the edge is modest and requires careful implementation to be real.

The look-ahead bias audit was the most important moment in the project. It is easy to build a system that looks profitable because it accidentally uses future information. Removing that bias dropped in-sample ROI from +16.66% to +2.26% — a humbling but honest result. The positive OOS ROI (+0.49%) is the most meaningful number: it confirms that the edge generalizes to markets the model never saw.

The Kelly sizing result tells a clear story about bet sizing. The same model, the same predictions, the same 1,505 trades: flat betting loses −8.56%, Half-Kelly wins +2.26%. The difference is entirely in position sizing. This is the core insight of the Kelly Criterion applied in practice.

The system is fully live and deployed — it fetches real Polymarket data, runs real model inference, and produces real portfolio allocations. Whether those allocations would be profitable at scale with real money is a question that would require longer live deployment to answer definitively.

---

## 15. Repository Structure

**prediction-market-optimizer** (ML + API):
```
api/main.py                     FastAPI server, all endpoints
src/optimizer/pipeline.py       Feature extraction, inference, model routing
src/optimizer/kelly_markowitz.py  Kelly + Markowitz allocation
src/backtest/backtest.py        In-sample backtest with look-ahead fix
src/backtest/oos_eval.py        OOS evaluation with look-ahead fix
src/model/best_model_*.pkl      Trained GBM models (one per timeframe)
src/model/calibrator_*.pkl      Isotonic calibrators (one per timeframe)
src/data/fetch_*_markets.py     Market-type-specific scrapers
data/raw/price_history_*.json   13,138 resolved markets with CLOB history
data/backtest_results.json      Latest backtest output
data/oos_results.json           Latest OOS evaluation output
```

**robo-advisor-frontend** (React):
```
src/pages/DashboardPage.tsx     Live stats, top opportunities
src/pages/OptimizerPage.tsx     Risk slider, allocation table, report
src/pages/MarketsPage.tsx       All live markets, filterable
src/pages/HistoryPage.tsx       Backtest charts, strategy comparison
src/components/layout/Sidebar.tsx  Navigation, live model accuracy
src/components/shared/LoadingSpinner.tsx  Animated step-by-step loader
```
