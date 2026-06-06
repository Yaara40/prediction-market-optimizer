import json
import requests
import re
import time
from datetime import datetime, timezone

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

def get_duration_minutes(start_date: str, end_date: str) -> float | None:
    try:
        start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        return (end - start).total_seconds() / 60
    except:
        return None

def fetch(max_pages: int = 500) -> dict:
    offset = 0
    collected = {"BTC": [], "ETH": [], "SOL": [], "XRP": [], "BNB": [], "DOGE": [], "HYPE": []}
    seen_ids = set()
    consecutive_failures = 0

    for page in range(max_pages):
        params = {
            "limit": 100,
            "order": "startDate",
            "ascending": "false",
            "closed": "true",
            "tag_slug": "monthly",
            "offset": offset
        }

        events = None
        for attempt in range(5):
            try:
                response = requests.get(f"{GAMMA_BASE_URL}/events", params=params, timeout=30)
                response.raise_for_status()
                events = response.json()
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  attempt {attempt+1}/5 failed: {e} — retrying in {wait}s")
                time.sleep(wait)

        if events is None:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print("3 consecutive failures — stopping")
                break
            continue

        consecutive_failures = 0

        if not events:
            break

        for event in events:
            for market in event.get("markets", []):
                mid = market.get("id")
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)

                question = market.get("question", "")
                start_date = market.get("startDate", "")
                end_date = market.get("endDate", "")
                duration = get_duration_minutes(start_date, end_date)

                text = question.lower()
                for coin, keywords in KEYWORDS.items():
                    if matches_coin(text, keywords):
                        collected[coin].append({
                            "id": mid,
                            "question": question,
                            "startDate": start_date,
                            "endDate": end_date,
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

        offset += 100

        if (page + 1) % 10 == 0:
            total = sum(len(v) for v in collected.values())
            print(f"page {page+1} (offset {offset}): {total} markets collected")

        if not events or len(events) < 100:
            break

        time.sleep(0.1)

    return collected

if __name__ == "__main__":
    print("fetching monthly markets via events API...")
    markets = fetch()
    total = sum(len(v) for v in markets.values())
    print(f"\ndone: {total} markets")
    for coin, ms in markets.items():
        print(f"  {coin}: {len(ms)}")
        if ms:
            print(f"    sample: {ms[0]['question']} | price: {ms[0]['lastTradePrice']}")
    with open("data/raw/markets_monthly.json", "w") as f:
        json.dump(markets, f, indent=2)
    print("saved to data/raw/markets_monthly.json")