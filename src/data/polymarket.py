import json
import requests
import re
import time
from datetime import datetime, timezone

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
        return 0 <= days_until_end <= 180
    except:
        return False

def get_active_markets():
    """fetch active crypto markets using keyset pagination newest first"""
    all_markets = []
    after_cursor = None

    for page in range(20):
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
                response = requests.get(f"{GAMMA_BASE_URL}/markets/keyset", params=params)
                response.raise_for_status()
                break
            except Exception as e:
                print(f"page {page+1}: attempt {attempt+1} failed — {e}")
                if attempt < retries - 1:
                    print("retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    response = None
                    break

        if response is None:
            break

        data = response.json()
        markets = data.get("markets", [])
        if not markets:
            break

        all_markets.extend(markets)
        print(f"page {page+1}: fetched {len(all_markets)} active markets so far...")

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
                    })

    return filtered

def get_resolved_markets_by_timeframe(min_hours: int, max_hours: int):
    """fetch resolved crypto markets that lasted between min and max hours"""
    all_markets = []
    after_cursor = None

    for page in range(100):
        params = {
            "limit": 100,
            "order": "startDate",
            "ascending": "false",
            "closed": "true"
        }
        if after_cursor:
            params["after_cursor"] = after_cursor

        retries = 3
        response = None
        for attempt in range(retries):
            try:
                response = requests.get(f"{GAMMA_BASE_URL}/markets/keyset", params=params)
                response.raise_for_status()
                break
            except Exception as e:
                print(f"page {page+1}: attempt {attempt+1} failed — {e}")
                if attempt < retries - 1:
                    print("retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("giving up on this page, stopping.")
                    response = None
                    break

        if response is None:
            break

        data = response.json()
        markets = data.get("markets", [])
        if not markets:
            break

        for market in markets:
            start = market.get("startDate")
            end = market.get("endDate")
            if not start or not end:
                continue
            try:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                duration_hours = (end_dt - start_dt).total_seconds() / 3600
                if min_hours <= duration_hours <= max_hours:
                    all_markets.append(market)
            except:
                continue

        print(f"page {page+1}: fetched {len(all_markets)} matching markets so far...")

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
                    "closed": market.get("closed"),
                })

    return filtered

if __name__ == "__main__":
    print("--- fetching active markets ---")
    active = get_active_markets()
    total_active = sum(len(v) for v in active.values())
    print(f"found {total_active} active crypto markets")
    with open("data/active_markets.json", "w") as f:
        json.dump(active, f, indent=2)
    print("saved → data/active_markets.json")

    print("\n--- fetching 12h-7day resolved markets ---")
    resolved = get_resolved_markets_by_timeframe(min_hours=12, max_hours=168)
    total = sum(len(v) for v in resolved.values())
    print(f"found {total} resolved crypto markets")
    with open("data/resolved_markets_12h_7days.json", "w") as f:
        json.dump(resolved, f, indent=2)
    print("saved → data/resolved_markets_12h_7days.json")