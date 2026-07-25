import json
import joblib
import re
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from src.data.coingecko import get_correlation_matrix
from src.optimizer.kelly_markowitz import optimize_portfolio

# Numeric encoding used during training (must match train.py TYPE_MAP)
MARKET_TYPE_INT = {"5min": 0, "15min": 1, "1hour": 2, "4hour": 3, "1day": 4, "weekly": 5, "all": 7}

MODEL_MAP = {
    "5min":   "src/model/best_model_5min.pkl",
    "15min":  "src/model/best_model_15min.pkl",
    "1hour":  "src/model/best_model_1hour.pkl",
    "4hour":  "src/model/best_model_4hour.pkl",
    "1day":   "src/model/best_model_1day.pkl",
    "weekly": "src/model/best_model_weekly.pkl",
    "all":    "src/model/best_model_all.pkl",
}

CALIBRATOR_MAP = {
    "5min":   "src/model/calibrator_5min.pkl",
    "15min":  "src/model/calibrator_15min.pkl",
    "1hour":  "src/model/calibrator_1hour.pkl",
    "4hour":  "src/model/calibrator_4hour.pkl",
    "1day":   "src/model/calibrator_1day.pkl",
    "weekly": "src/model/calibrator_weekly.pkl",
    "all":    "src/model/calibrator_all.pkl",
}

MODELS = {}
CALIBRATORS = {}

def load_models():
    for name, path in MODEL_MAP.items():
        try:
            MODELS[name] = joblib.load(path)
        except FileNotFoundError:
            print(f"model not found: {path}")
    for name, path in CALIBRATOR_MAP.items():
        try:
            CALIBRATORS[name] = joblib.load(path)
        except FileNotFoundError:
            pass  # calibrators are optional; fall back to raw model output

def detect_market_type(question: str, market: dict = None) -> str:
    """Classify a market's timeframe.

    Priority:
    1. Explicit time-range in question text (e.g. "10:55AM-11:00AM") → 5min/15min/1hour/4hour
    2. startDate/endDate duration when available — most reliable for daily/weekly/monthly
    3. Question-text patterns as fallback
    """
    # 1. Explicit time-range window: "10:55AM-11:00AM"
    match = re.search(r'(\d+):(\d+)(AM|PM)-(\d+):(\d+)(AM|PM)', question, re.IGNORECASE)
    if match:
        h1, m1 = int(match.group(1)), int(match.group(2))
        h2, m2 = int(match.group(4)), int(match.group(5))
        p1, p2 = match.group(3).upper(), match.group(6).upper()
        if p1 == "PM" and h1 != 12: h1 += 12
        if p1 == "AM" and h1 == 12: h1 = 0
        if p2 == "PM" and h2 != 12: h2 += 12
        if p2 == "AM" and h2 == 12: h2 = 0
        mins = (h2 * 60 + m2) - (h1 * 60 + m1)
        if mins <= 0: mins += 24 * 60
        if mins <= 5: return "5min"
        if mins <= 15: return "15min"
        if mins <= 60: return "1hour"
        if mins <= 240: return "4hour"
        return "1day"

    # 2. Question-text patterns that identify single-time-point markets (before date check)
    # "Up or Down - July 18, 11AM ET" — resolves at a specific hour
    if re.search(r'up or down - .+, \d+[AP]M ET', question, re.IGNORECASE):
        return "1hour"
    # "Bitcoin above 62,000 on July 18, 3AM ET?" — specific-hour price target
    if re.search(r'(above|dip to|reach)[\s\S]{1,30}on \w+ \d+, \d+[AP]M', question, re.IGNORECASE):
        return "1hour"

    # Multi-outcome weekly format: "What price will Bitcoin hit July 13-19?" or
    # "What price will Bitcoin hit July 13-July 19?" — date range in question
    if re.search(r'what price will .+ hit .+\w+ \d{1,2}[-–]\w*\s*\d{1,2}', question, re.IGNORECASE):
        return "weekly"

    # "Coin Up or Down on July 25?" — daily binary
    if re.search(r'up or down on \w+ \d+\?', question, re.IGNORECASE):
        return "1day"
    # "Will coin reach/dip to $X on July 25?" — daily price target (fetch_daily_markets)
    if re.search(r'will .{1,20}(reach|dip to).{1,30}on \w+ \d+\?', question, re.IGNORECASE):
        return "1day"
    # "Will the price of coin be above/less than $X on July 25?" — weekly multi-strike
    if re.search(r'will the price of .{1,20}(above|less than|between).{1,30}on \w+ \d+\?', question, re.IGNORECASE):
        return "weekly"

    # 3. Use startDate/endDate duration for everything else — most reliable for daily/weekly/monthly
    if market:
        try:
            start_str = market.get("startDate") or market.get("start_date")
            end_str   = market.get("endDate")   or market.get("end_date")
            if start_str and end_str:
                start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                end_dt   = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                duration_hours = (end_dt - start_dt).total_seconds() / 3600
                if duration_hours <= 5:     return "1hour"
                if duration_hours <= 20:    return "4hour"
                if duration_hours <= 50:    return "1day"
                return "weekly"
        except Exception:
            pass

    # 4. Text-only fallback (no date info available)
    if re.search(r'up or down on \w+ \d+\?', question, re.IGNORECASE):
        return "1day"

    # "this week" / "by <day of week>" / "by <month> <date>" — weekly
    if re.search(r'\bthis week\b|\bby (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', question, re.IGNORECASE):
        return "weekly"
    if re.search(r'\bby \w+ \d{1,2}\b', question, re.IGNORECASE):
        return "weekly"

    return "all"

def get_model_for_market(question: str, market: dict = None):
    market_type = detect_market_type(question, market)
    model = MODELS.get(market_type) or MODELS.get("all")
    return model, market_type

def load_active_markets():
    with open("data/active_markets.json") as f:
        return json.load(f)

def get_price_features(market: dict, coin_idx: int):
    # --- Collect all available price signals ---
    ltp = float(market.get("lastTradePrice") or 0)
    bid = float(market.get("bestBid") or 0)
    ask = float(market.get("bestAsk") or 0)

    # Parse outcomePrices.
    # Binary markets: ["0.52", "0.48"] — index 0 is YES.
    # Multi-outcome (weekly/monthly) markets: ["0.001", "0.025", "0.97", "0.001", ...]
    # For multi-outcome markets pick the price closest to 0.5 — it's the most
    # uncertain bucket and gives the strongest signal for ML.
    outcome_yes = None
    try:
        op = market.get("outcomePrices")
        if op:
            import json as _json
            parsed = _json.loads(op) if isinstance(op, str) else op
            if parsed and len(parsed) >= 1:
                floats = [float(x) for x in parsed]
                if len(floats) == 2:
                    # Standard binary: index 0 is YES
                    outcome_yes = floats[0]
                else:
                    # Multi-outcome: pick the one closest to 0.5
                    outcome_yes = min(floats, key=lambda x: abs(x - 0.5))
    except Exception:
        pass

    # Best single-point estimate of the current YES price:
    # prefer lastTradePrice if it's a real mid-market value (not 0 or 1),
    # then outcomePrices YES (or best bucket for multi-outcome),
    # then mid(bid,ask), then ask alone
    def valid(p):
        return p is not None and 0.005 < p < 0.995

    if valid(ltp):
        market_price = ltp
    elif valid(outcome_yes):
        market_price = outcome_yes
    elif valid(bid) and valid(ask):
        market_price = (bid + ask) / 2.0
    elif valid(bid):
        market_price = bid
    elif valid(ask):
        market_price = ask
    else:
        return None, None

    # --- Build price trajectory ---
    # Prefer real CLOB history if the market was enriched with it.
    # Fall back to a synthetic trajectory only when no history is available.
    price_history = market.get("price_history", [])

    if len(price_history) >= 3:
        # Use the real trajectory — same extraction as build_dataset scripts
        prices = [h["p"] for h in price_history]
        timestamps = [h["t"] for h in price_history]
        duration = timestamps[-1] - timestamps[0]
        n = 10

        def _snap(i):
            target = timestamps[0] + duration * i / max(n - 1, 1)
            closest = min(range(len(timestamps)), key=lambda j: abs(timestamps[j] - target))
            return max(0.01, min(0.99, prices[closest]))

        trajectory = [_snap(i) for i in range(n)]

        crossings = sum(
            1 for i in range(1, len(prices))
            if (prices[i - 1] < 0.5) != (prices[i] < 0.5)
        )

        # Use price at 5/50/95% of the window for early/mid/late
        def _at_pct(pct):
            target = timestamps[0] + duration * pct
            closest = min(range(len(timestamps)), key=lambda j: abs(timestamps[j] - target))
            return max(0.01, min(0.99, prices[closest]))

        price_early = _at_pct(0.05)
        price_mid   = _at_pct(0.50)
        price_late  = _at_pct(0.95)

    else:
        # Synthetic fallback: flat line at market_price.
        # Note: predict_probabilities() skips the ML model entirely when
        # price_history is absent, so this trajectory is only used as a
        # feature vector sanity-check and is not fed to the classifier.
        mp_clamped = max(0.01, min(0.99, market_price))
        trajectory = [mp_clamped] * 10
        crossings = 0
        price_early = mp_clamped
        price_mid   = mp_clamped
        price_late  = mp_clamped

    try:
        start = market.get("startDate")
        end = market.get("endDate")
        if start and end:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            duration_minutes = (end_dt - start_dt).total_seconds() / 60
        else:
            duration_minutes = 1440
    except Exception:
        duration_minutes = 1440

    volume = float(market.get("volumeNum") or market.get("volume") or 0)

    features = {
        "price_t0": trajectory[0],
        "price_t1": trajectory[1],
        "price_t2": trajectory[2],
        "price_t3": trajectory[3],
        "price_t4": trajectory[4],
        "price_t5": trajectory[5],
        "price_t6": trajectory[6],
        "price_t7": trajectory[7],
        "price_t8": trajectory[8],
        "price_t9": trajectory[9],
        "crossings_05": crossings,
        "price_early": price_early,
        "price_mid": price_mid,
        "price_late": price_late,
        "duration_minutes": duration_minutes,
        "volume": volume,
        "coin": coin_idx,
        # market_type as integer for the 'all' model (0=5min,1=15min,2=1hour,3=4hour,4=1day,5=weekly,6=monthly,7=all)
        "market_type": MARKET_TYPE_INT.get("all", 7),
    }

    return pd.DataFrame([features]), market_price


def get_price_features_with_type(market: dict, coin_idx: int, market_type: str):
    """Same as get_price_features but injects the correct market_type integer."""
    df, mp = get_price_features(market, coin_idx)
    if df is not None:
        df["market_type"] = MARKET_TYPE_INT.get(market_type, 7)
    return df, mp

def predict_probabilities(markets: dict) -> dict:
    results = {}

    for coin_idx, (coin, coin_markets) in enumerate(markets.items()):
        results[coin] = []
        for market in coin_markets:
            question = market.get("question", "")
            model, market_type = get_model_for_market(question, market)

            if model is None:
                continue

            features, market_price = get_price_features_with_type(market, coin_idx, market_type)
            if features is None:
                continue

            has_real_history = len(market.get("price_history", [])) >= 3

            if not has_real_history:
                # No CLOB history → the synthetic flat trajectory gives the
                # model no usable signal. Fall back to market price so
                # edge = 0 and we don't surface false opportunities.
                our_estimate = market_price
            else:
                try:
                    model_cols = model.feature_names_in_
                    features = features.reindex(columns=model_cols, fill_value=0)
                except AttributeError:
                    pass

                try:
                    raw_estimate = float(model.predict_proba(features)[0][1])
                    # Apply isotonic calibration if available for this market type
                    calibrator = CALIBRATORS.get(market_type) or CALIBRATORS.get("all")
                    if calibrator is not None:
                        our_estimate = float(calibrator.predict([raw_estimate])[0])
                    else:
                        our_estimate = raw_estimate
                except:
                    our_estimate = market_price

            results[coin].append({
                "id": market.get("id"),
                "question": question,
                "endDate": market.get("endDate"),
                "market_type": market_type,
                "market_price": round(market_price, 4),
                "our_estimate": round(our_estimate, 4),
                "edge": round(our_estimate - market_price, 4),
                "volume": float(market.get("volumeNum") or 0),
            })

    return results

def run_pipeline(risk_level: int = 5):
    print("loading models...")
    load_models()
    print(f"loaded {len(MODELS)} models")

    print("loading active markets...")
    markets = load_active_markets()

    print("predicting probabilities...")
    predictions = predict_probabilities(markets)

    best_markets = {}
    for coin, coin_preds in predictions.items():
        positive_edge = [p for p in coin_preds if p["edge"] > 0.02]
        if positive_edge:
            best = max(positive_edge, key=lambda x: x["edge"])
            best_markets[coin] = best
            print(f"{coin}: {best['question'][:50]} | type={best['market_type']} | edge={best['edge']}")

    if len(best_markets) < 2:
        print("not enough markets with positive edge")
        return None

    estimates = {coin: data["our_estimate"] for coin, data in best_markets.items()}
    market_prices = {coin: data["market_price"] for coin, data in best_markets.items()}

    print("\nfetching correlation matrix...")
    full_corr, corr_coins = get_correlation_matrix()

    available_coins = [c for c in best_markets.keys() if c in corr_coins]
    if len(available_coins) < 2:
        print("not enough known coins for correlation matrix")
        return None

    indices = [corr_coins.index(c) for c in available_coins]
    correlation_matrix = full_corr[np.ix_(indices, indices)]
    estimates = {c: estimates[c] for c in available_coins}
    market_prices = {c: market_prices[c] for c in available_coins}

    print("\nrunning optimizer...")
    allocations = optimize_portfolio(estimates, market_prices, correlation_matrix, risk_level)

    print("\nportfolio allocations:")
    for coin, data in allocations.items():
        print(f"  {coin}: {data['weight']*100:.1f}% | edge: {data['edge']:.3f} | our estimate: {data['our_estimate']:.3f} | market: {data['market_price']:.3f}")

    return allocations, best_markets

if __name__ == "__main__":
    run_pipeline(risk_level=5)