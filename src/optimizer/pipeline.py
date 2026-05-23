import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.data.coingecko import get_correlation_matrix
from src.optimizer.kelly_markowitz import optimize_portfolio

def load_model():
    """load the trained TabPFN model"""
    return joblib.load("src/model/best_model_tabpfn_cpu.pkl")

def load_active_markets():
    """load active markets from saved json"""
    with open("data/active_markets.json") as f:
        return json.load(f)

def predict_probabilities(model, markets: dict) -> dict:
    """
    for each active market, predict corrected probability using the model
    markets: {"BTC": [...], "ETH": [...], "SOL": [...]}
    returns: {"BTC": [{"id": ..., "question": ..., "market_price": ..., "our_estimate": ...}]}
    """
    results = {}

    for coin_idx, (coin, coin_markets) in enumerate(markets.items()):
        results[coin] = []
        for market in coin_markets:
            market_price = market.get("lastTradePrice")
            if market_price is None:
                market_price = market.get("bestAsk", 0.5)
            if market_price is None:
                continue

            try:
                start = market.get("startDate") or market.get("start_date")
                end = market.get("endDate") or market.get("end_date")
                if start and end:
                    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                    duration_hours = (end_dt - start_dt).total_seconds() / 3600
                else:
                    duration_hours = 24
            except:
                duration_hours = 24

            volume = market.get("volumeNum") or market.get("volume") or 0
            spread = market.get("spread") or 0.01

            features = pd.DataFrame([{
                "best_bid": market_price,
                "volume": float(volume) if volume else 0,
                "spread": float(spread),
                "duration_hours": duration_hours,
                "coin": coin_idx
            }])

            try:
                our_estimate = float(model.predict_proba(features)[0][1])
            except:
                our_estimate = market_price

            results[coin].append({
                "id": market.get("id"),
                "question": market.get("question"),
                "endDate": market.get("endDate"),
                "market_price": round(market_price, 4),
                "our_estimate": round(our_estimate, 4),
                "edge": round(our_estimate - market_price, 4),
                "volume": volume,
            })

    return results

def run_pipeline(risk_level: int = 5, top_n: int = 3):
    """
    run the full pipeline and return portfolio allocations
    risk_level: 1-10
    top_n: how many markets per coin to consider
    """
    print("loading model...")
    model = load_model()

    print("loading active markets...")
    markets = load_active_markets()

    print("predicting probabilities...")
    predictions = predict_probabilities(model, markets)

    # for each coin pick the market with highest positive edge
    best_markets = {}
    for coin, coin_preds in predictions.items():
        positive_edge = [p for p in coin_preds if p["edge"] > 0.02]
        if positive_edge:
            best = max(positive_edge, key=lambda x: x["edge"])
            best_markets[coin] = best
            print(f"{coin}: best market = {best['question'][:50]} | edge={best['edge']}")

    if len(best_markets) < 2:
        print("not enough markets with positive edge")
        return None

    # build inputs for optimizer
    estimates = {coin: data["our_estimate"] for coin, data in best_markets.items()}
    market_prices = {coin: data["market_price"] for coin, data in best_markets.items()}

    print("\nfetching correlation matrix...")
    coins = list(estimates.keys())
    full_corr = get_correlation_matrix()
    coin_order = ["BTC", "ETH", "SOL"]
    indices = [coin_order.index(c) for c in coins]
    correlation_matrix = full_corr[np.ix_(indices, indices)]

    print("\nrunning optimizer...")
    allocations = optimize_portfolio(estimates, market_prices, correlation_matrix, risk_level)

    print("\n=== portfolio allocations ===")
    for coin, data in allocations.items():
        print(f"{coin}: {data['weight']*100:.1f}% | edge: {data['edge']:.3f} | our estimate: {data['our_estimate']:.3f} | market: {data['market_price']:.3f}")

    return allocations, best_markets

if __name__ == "__main__":
    run_pipeline(risk_level=5)