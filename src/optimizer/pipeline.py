import json
import joblib
import re
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from src.data.coingecko import get_correlation_matrix
from src.optimizer.kelly_markowitz import optimize_portfolio

# Numeric encoding used during training (must match train.py TYPE_MAP)
MARKET_TYPE_INT = {"5min": 0, "15min": 1, "1hour": 2, "4hour": 3, "1day": 4, "weekly": 5, "monthly": 6, "all": 7}

MODEL_MAP = {
    "5min":    "src/model/best_model_5min.pkl",
    "15min":   "src/model/best_model_15min.pkl",
    "1hour":   "src/model/best_model_1hour.pkl",
    "4hour":   "src/model/best_model_4hour.pkl",
    "1day":    "src/model/best_model_1day.pkl",
    "weekly":  "src/model/best_model_weekly.pkl",
    "monthly": "src/model/best_model_monthly.pkl",
    "all":     "src/model/best_model_all.pkl",
}

MODELS = {}

def load_models():
    for name, path in MODEL_MAP.items():
        try:
            MODELS[name] = joblib.load(path)
        except FileNotFoundError:
            print(f"model not found: {path}")

def detect_market_type(question: str) -> str:
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

    if re.search(r'up or down - .+, \d+[AP]M ET$', question, re.IGNORECASE):
        return "1hour"

    if re.search(r'up or down on \w+ \d+\?', question, re.IGNORECASE):
        return "1day"

    return "all"

def get_model_for_market(question: str):
    market_type = detect_market_type(question)
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

    # Parse outcomePrices: ["0.52", "0.48"] — index 0 is YES
    outcome_yes = None
    try:
        op = market.get("outcomePrices")
        if op:
            import json as _json
            parsed = _json.loads(op) if isinstance(op, str) else op
            if parsed and len(parsed) >= 1:
                outcome_yes = float(parsed[0])
    except Exception:
        pass

    # Best single-point estimate of the current YES price:
    # prefer lastTradePrice if it's a real mid-market value (not 0 or 1),
    # then outcomePrices YES, then mid(bid,ask), then ask alone
    def valid(p):
        return p is not None and 0.01 < p < 0.99

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
        # Synthetic fallback: linear drift from ~0.5 to current market_price
        p_open = max(0.01, min(0.99, 0.5 + (market_price - 0.5) * 0.1))
        trajectory = [
            max(0.01, min(0.99, p_open + (market_price - p_open) * i / 9.0))
            for i in range(10)
        ]
        crossings = 0
        price_early = trajectory[0]
        price_mid   = trajectory[4]
        price_late  = trajectory[9]

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
            model, market_type = get_model_for_market(question)

            if model is None:
                continue

            features, market_price = get_price_features_with_type(market, coin_idx, market_type)
            if features is None:
                continue

            try:
                model_cols = model.feature_names_in_
                features = features.reindex(columns=model_cols, fill_value=0)
            except AttributeError:
                pass

            try:
                our_estimate = float(model.predict_proba(features)[0][1])
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