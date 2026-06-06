import json
import requests
import re
import time

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

KEYWORDS = {
    "BTC": [r"\bbitcoin\b", r"\bbtc\b"],
    "ETH": [r"\bethereum\b", r"\beth\b"],
    "SOL": [r"\bsolana\b", r"\bsol\b"],
    "XRP": [r"\bxrp\b", r"\bripple\b"],
    "BNB": [r"\bbnb\b", r"\bbinance\b"],
    "DOGE": [r"\bdogecoin\b", r"\bdoge\b"],
    "HYPE": [r"\bhype\b", r"\bhyperliquid\b"]
}

def matches_coin(text: str, keywords: list) -> bool:
    return any(re.search(k, text, re.IGNORECASE) for k in keywords)

def fetch_active_markets(max_pages: int = 20) -> dict:
    all_markets = []
    after_cursor = None

    for page in range(max_pages):
        params = {
            "limit": 100,
            "order": "startDate",
            "ascending": "false",
            "active": "true",
            "closed": "false"
        }
        if after_cursor:
            params["after_cursor"] = after_cursor

        retries = 3
        response = None
        for attempt in range(retries):
            try:
                response = requests.get(
                    f"{GAMMA_BASE_URL}/markets/keyset",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                break
            except Exception as e:
                print(f"page {page+1} attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    response = None

        if response is None:
            break

        data = response.json()
        markets = data.get("markets", [])
        if not markets:
            break

        all_markets.extend(markets)
        print(f"page {page+1}: fetched {len(all_markets)} active markets so far")

        after_cursor = data.get("next_cursor")
        if not after_cursor:
            break

        time.sleep(0.1)

    filtered = {coin: [] for coin in KEYWORDS}
    for market in all_markets:
        text = (
            str(market.get("question", "")) + " " +
            str(market.get("slug", ""))
        ).lower()
        for coin, keywords in KEYWORDS.items():
            if matches_coin(text, keywords):
                filtered[coin].append({
                    "id": market.get("id"),
                    "question": market.get("question"),
                    "startDate": market.get("startDate"),
                    "endDate": market.get("endDate"),
                    "lastTradePrice": market.get("lastTradePrice"),
                    "bestBid": market.get("bestBid"),
                    "bestAsk": market.get("bestAsk"),
                    "spread": market.get("spread"),
                    "volume": market.get("volume"),
                    "volumeNum": market.get("volumeNum"),
                    "liquidity": market.get("liquidity"),
                    "outcomes": market.get("outcomes"),
                    "outcomePrices": market.get("outcomePrices"),
                    "clobTokenIds": market.get("clobTokenIds"),
                })

    return filtered

if __name__ == "__main__":
    print("fetching active markets...")
    markets = fetch_active_markets()
    total = sum(len(v) for v in markets.values())
    print(f"\ndone: {total} active markets")
    for coin, ms in markets.items():
        print(f"  {coin}: {len(ms)}")

    with open("data/active_markets.json", "w") as f:
        json.dump(markets, f, indent=2)
    print("saved → data/active_markets.json")