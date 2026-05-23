import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

def get_historical_prices(coin_id: str, days: int = 90) -> list:
    """fetch daily closing prices for a coin from CoinGecko"""
    response = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": days, "interval": "daily"}
    )
    response.raise_for_status()
    data = response.json()
    # prices is a list of [timestamp, price]
    return [price[1] for price in data["prices"]]

def get_correlation_matrix() -> np.ndarray:
    """compute real correlation matrix between BTC, ETH, SOL"""
    print("fetching historical prices from CoinGecko...")
    
    btc = get_historical_prices("bitcoin", days=90)
    eth = get_historical_prices("ethereum", days=90)
    sol = get_historical_prices("solana", days=90)

    # make sure all same length
    min_len = min(len(btc), len(eth), len(sol))
    df = pd.DataFrame({
        "BTC": btc[:min_len],
        "ETH": eth[:min_len],
        "SOL": sol[:min_len]
    })

    # compute correlation matrix from daily returns
    returns = df.pct_change().dropna()
    corr = returns.corr().values

    print(f"correlation matrix:")
    print(f"  BTC-ETH: {corr[0][1]:.3f}")
    print(f"  BTC-SOL: {corr[0][2]:.3f}")
    print(f"  ETH-SOL: {corr[1][2]:.3f}")

    return corr

if __name__ == "__main__":
    corr = get_correlation_matrix()
    print("\nfull matrix:")
    print(corr)