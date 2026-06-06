import requests
import numpy as np
import pandas as pd
import time
import json
import os

COIN_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "BNB": "binancecoin",
}

CACHE_FILE = "data/correlation_cache.json"

def get_historical_prices(coin_id: str, days: int = 90) -> list:
    response = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": days, "interval": "daily"}
    )
    response.raise_for_status()
    data = response.json()
    return [price[1] for price in data["prices"]]

def get_correlation_matrix(coins: list = None) -> tuple:
    if coins is None:
        coins = list(COIN_IDS.keys())

    # use cache if less than 24 hours old
    if os.path.exists(CACHE_FILE):
        age = time.time() - os.path.getmtime(CACHE_FILE)
        if age < 86400:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            print("using cached correlation matrix")
            return np.array(cache["corr"]), cache["coins"]

    print("fetching historical prices from CoinGecko...")

    prices = {}
    for coin in coins:
        coin_id = COIN_IDS.get(coin)
        if not coin_id:
            print(f"  unknown coin: {coin} — skipping")
            continue
        try:
            prices[coin] = get_historical_prices(coin_id, days=90)
            print(f"  {coin}: {len(prices[coin])} days fetched")
            time.sleep(5)
        except Exception as e:
            print(f"  failed to fetch {coin}: {e}")

    available_coins = list(prices.keys())
    if len(available_coins) < 2:
        raise ValueError("not enough coins to compute correlation matrix")

    min_len = min(len(v) for v in prices.values())
    df = pd.DataFrame({c: prices[c][:min_len] for c in available_coins})

    returns = df.pct_change().dropna()
    corr = returns.corr().values

    print("correlation matrix:")
    for i, c1 in enumerate(available_coins):
        for j, c2 in enumerate(available_coins):
            if j > i:
                print(f"  {c1}-{c2}: {corr[i][j]:.3f}")

    # save cache
    with open(CACHE_FILE, "w") as f:
        json.dump({"corr": corr.tolist(), "coins": available_coins}, f)
    print("saved → data/correlation_cache.json")

    return corr, available_coins

if __name__ == "__main__":
    corr, coins = get_correlation_matrix()
    print("\ncoins:", coins)
    print("matrix shape:", corr.shape)