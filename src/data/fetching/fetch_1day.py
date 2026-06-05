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

def is_daily_market(question: str) -> bool:
    return bool(re.search(r"up or down on \w+ \d+\?", question, re.IGNORECASE))

def fetch_page(after_slug, retries=5):
    params = {
        "limit": 100,
        "order": "startDate",
        "ascending": "false",
        "closed": "true",
        "tag_slug": "daily"
    }
    if after_slug:
        params["after_cursor"] = after_slug

    for attempt in range(retries):
        try:
            response = requests.get(
                f"{GAMMA_BASE_URL}/events",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  attempt {attempt+1}/{retries} failed: {e} — retrying in {wait}s")
            time.sleep(wait)
    return None

def fetch(max_pages: int = 500) -> dict:
    after_slug = None
    collected = {"BTC": [], "ETH": [], "SOL": [], "XRP": [], "BNB": [], "DOGE": [], "HYPE": []}
    consecutive_failures = 0

    for page in range(max_pages):
        events = fetch_page(after_slug)

        if events is None:
            consecutive_failures += 1
            print(f"page {page+1}: failed — skipping")
            if consecutive_failures >= 3:
                print("3 consecutive failures — stopping")
                break
            continue

        consecutive_failures = 0

        if not events:
            break

        for event in events:
            for market in event.get("markets", []):
                question = market.get("question", "")
                if not is_daily_market(question):
                    continue

                text = question.lower()
                for coin, keywords in KEYWORDS.items():
                    if matches_coin(text, keywords):
                        collected[coin].append({
                            "id": market.get("id"),
                            "question": question,
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
                            "duration_minutes": 1440,
                        })

        after_slug = events[-1].get("slug") if events else None

        if (page + 1) % 10 == 0:
            total = sum(len(v) for v in collected.values())
            print(f"page {page+1}: {total} markets collected")

        if not after_slug:
            break

        time.sleep(0.1)

    return collected

if __name__ == "__main__":
    print("fetching 1-day markets via events API...")
    markets = fetch()
    total = sum(len(v) for v in markets.values())
    print(f"\ndone: {total} markets")
    for coin, ms in markets.items():
        print(f"  {coin}: {len(ms)}")
        if ms:
            print(f"    sample: {ms[0]['question']} | price: {ms[0]['lastTradePrice']}")
    with open("data/raw/markets_1day.json", "w") as f:
        json.dump(markets, f, indent=2)
    print("saved to data/raw/markets_1day.json")