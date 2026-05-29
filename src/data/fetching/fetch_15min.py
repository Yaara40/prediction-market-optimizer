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
    "BNB": [r"\bbnb\b", r"\bbinance\b"]
}

def matches_coin(text: str, keywords: list) -> bool:
    return any(re.search(k, text, re.IGNORECASE) for k in keywords)

def get_duration_minutes(question: str) -> float | None:
    match = re.search(r"(\d+):(\d+)(AM|PM)-(\d+):(\d+)(AM|PM)", question, re.IGNORECASE)
    if not match:
        return None
    h1, m1 = int(match.group(1)), int(match.group(2))
    h2, m2 = int(match.group(4)), int(match.group(5))
    p1, p2 = match.group(3).upper(), match.group(6).upper()
    if p1 == "PM" and h1 != 12: h1 += 12
    if p1 == "AM" and h1 == 12: h1 = 0
    if p2 == "PM" and h2 != 12: h2 += 12
    if p2 == "AM" and h2 == 12: h2 = 0
    mins = (h2 * 60 + m2) - (h1 * 60 + m1)
    if mins <= 0: mins += 24 * 60
    return mins

def fetch(max_pages: int = 1125, sample_every: int = 2) -> dict:
    after_cursor = None
    collected = {"BTC": [], "ETH": [], "SOL": [], "XRP": [], "BNB": []}
    market_counter = 0

    for page in range(max_pages):
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

        for market in markets:
            duration = get_duration_minutes(market.get("question", ""))
            if duration != 15:
                continue

            market_counter += 1
            if market_counter % sample_every != 0:
                continue

            text = (str(market.get("question", "")) + " " + str(market.get("slug", ""))).lower()
            for coin, keywords in KEYWORDS.items():
                if matches_coin(text, keywords):
                    collected[coin].append({
                        "id": market.get("id"),
                        "question": market.get("question"),
                        "startDate": market.get("startDate"),
                        "endDate": market.get("endDate"),
                        "outcomePrices": market.get("outcomePrices"),
                        "outcomes": market.get("outcomes"),
                        "volume": market.get("volume"),
                        "volumeNum": market.get("volumeNum"),
                        "lastTradePrice": market.get("lastTradePrice"),
                        "bestBid": market.get("bestBid"),
                        "bestAsk": market.get("bestAsk"),
                        "spread": market.get("spread"),
                        "closed": market.get("closed"),
                        "clobTokenIds": market.get("clobTokenIds"),
                        "duration_minutes": duration,
                    })

        total = sum(len(v) for v in collected.values())
        if page % 50 == 0:
            print(f"page {page+1}: {total} markets collected")

        after_cursor = data.get("next_cursor")
        if not after_cursor:
            break

        time.sleep(0.1)

    return collected

if __name__ == "__main__":
    print("fetching 15-minute markets (~3000 target, every 2nd)...")
    markets = fetch()
    total = sum(len(v) for v in markets.values())
    print(f"\ndone: {total} markets")
    for coin, ms in markets.items():
        print(f"  {coin}: {len(ms)}")
        if ms:
            print(f"    sample: {ms[0]['question']} | price: {ms[0]['lastTradePrice']}")
    with open("data/raw/markets_15min.json", "w") as f:
        json.dump(markets, f, indent=2)
    print("saved to data/raw/markets_15min.json")