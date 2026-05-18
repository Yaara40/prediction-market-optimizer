import requests
import re
import json
from datetime import datetime, timezone, timedelta

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

KEYWORDS = {
    "BTC": [r"\bbitcoin\b", r"\bbtc\b"],
    "ETH": [r"\bethereum\b", r"\beth\b"],
    "SOL": [r"\bsolana\b", r"\bsol\b"]
}

def matches_coin(text: str, keywords: list) -> bool:
    return any(re.search(k, text, re.IGNORECASE) for k in keywords)

def is_valid_timeframe(end_date_str: str) -> bool:
    if not end_date_str:
        return False
    try:
        if "T" in end_date_str:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        else:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_until_end = (end_date - now).days
        return 0 <= days_until_end <= 30
    except:
        return False

def get_active_markets():
    """fetch active crypto markets using keyset pagination newest first"""
    all_markets = []
    after_cursor = None
    max_pages = 20

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

        response = requests.get(f"{GAMMA_BASE_URL}/markets/keyset", params=params)
        response.raise_for_status()
        data = response.json()

        markets = data.get("markets", [])
        if not markets:
            break

        all_markets.extend(markets)
        print(f"page {page+1}: fetched {len(all_markets)} markets so far...")

        after_cursor = data.get("next_cursor")
        if not after_cursor:
            break

    filtered = {"BTC": [], "ETH": [], "SOL": []}
    for market in all_markets:
        text = (
            str(market.get("question", "")) + " " +
            str(market.get("slug", ""))
        ).lower()
        for coin, keywords in KEYWORDS.items():
            if matches_coin(text, keywords):
                if is_valid_timeframe(market.get("endDate")):
                    filtered[coin].append({
                        "id": market.get("id"),
                        "question": market.get("question"),
                        "endDate": market.get("endDate"),
                        "lastTradePrice": market.get("lastTradePrice"),
                    })

    return filtered

def get_resolved_markets():
    """fetch resolved crypto markets using keyset pagination"""
    all_markets = []
    after_cursor = None
    max_pages = 50

    for page in range(max_pages):
        params = {
            "limit": 100,
            "order": "startDate",
            "ascending": "false",
            "closed": "true"
        }
        if after_cursor:
            params["after_cursor"] = after_cursor

        response = requests.get(f"{GAMMA_BASE_URL}/markets/keyset", params=params)
        response.raise_for_status()
        data = response.json()

        markets = data.get("markets", [])
        if not markets:
            break

        all_markets.extend(markets)
        print(f"page {page+1}: fetched {len(all_markets)} resolved markets so far...")

        after_cursor = data.get("next_cursor")
        if not after_cursor:
            break

    filtered = {"BTC": [], "ETH": [], "SOL": []}
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
    "outcomePrices": market.get("outcomePrices"),
    "outcomes": market.get("outcomes"),
    "volume": market.get("volume"),
    "volumeNum": market.get("volumeNum"),
    "liquidity": market.get("liquidity"),
    "lastTradePrice": market.get("lastTradePrice"),
    "bestBid": market.get("bestBid"),
    "bestAsk": market.get("bestAsk"),
    "spread": market.get("spread"),
    "oneDayPriceChange": market.get("oneDayPriceChange"),
    "closed": market.get("closed"),
    "acceptingOrders": market.get("acceptingOrders"),
})

    return filtered

if __name__ == "__main__":
    print("--- fetching active markets ---")
    active = get_active_markets()
    total_active = sum(len(v) for v in active.values())
    print(f"found {total_active} active crypto markets")

    print("\n--- fetching resolved markets ---")
    resolved = get_resolved_markets()
    total_resolved = sum(len(v) for v in resolved.values())
    print(f"found {total_resolved} resolved crypto markets")

    # save to disk
    with open("data/active_markets.json", "w") as f:
        json.dump(active, f, indent=2)
    print("\nsaved → data/active_markets.json")

    with open("data/resolved_markets.json", "w") as f:
        json.dump(resolved, f, indent=2)
    print("saved → data/resolved_markets.json")