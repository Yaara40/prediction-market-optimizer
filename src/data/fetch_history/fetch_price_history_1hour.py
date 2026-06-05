import json
import requests
import time
from datetime import datetime, timezone, timedelta
import re

CLOB_BASE_URL = "https://clob.polymarket.com"

def parse_trading_window(market: dict) -> tuple:
    question = market.get("question", "")
    end_date = market.get("endDate", "")

    if not end_date:
        return None, None

    try:
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except:
        return None, None

    # hourly: "Bitcoin Up or Down - May 30, 1AM ET"
    match = re.search(r'(\d+)(AM|PM) ET$', question, re.IGNORECASE)
    if match:
        h, period = int(match.group(1)), match.group(2).upper()
        if period == "PM" and h != 12: h += 12
        if period == "AM" and h == 12: h = 0

        market_date = end_dt.date()
        end_trading = datetime(market_date.year, market_date.month, market_date.day,
                              h, 0, tzinfo=timezone.utc) + timedelta(hours=4)
        start_trading = end_trading - timedelta(hours=1)

        return int(start_trading.timestamp()), int(end_trading.timestamp())

    return None, None

def fetch_history(token_id: str, start_ts: int, end_ts: int) -> list:
    try:
        response = requests.get(
            f"{CLOB_BASE_URL}/prices-history",
            params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 1}
        )
        response.raise_for_status()
        return response.json().get("history", [])
    except:
        return []

if __name__ == "__main__":
    with open("data/raw/markets_1hour.json") as f:
        raw = json.load(f)

    results = {}
    total = sum(len(v) for v in raw.values())
    processed = 0
    failed = 0

    for coin, markets in raw.items():
        results[coin] = []
        for m in markets:
            processed += 1

            clob_ids = m.get("clobTokenIds")
            if not clob_ids:
                failed += 1
                continue

            try:
                token_ids = json.loads(clob_ids) if isinstance(clob_ids, str) else clob_ids
                token_id = token_ids[0]
            except:
                failed += 1
                continue

            start_ts, end_ts = parse_trading_window(m)
            if not start_ts or not end_ts:
                failed += 1
                continue

            history = fetch_history(token_id, start_ts, end_ts)

            results[coin].append({
                **m,
                "price_history": history
            })

            if processed % 100 == 0:
                print(f"processed {processed}/{total} — failed: {failed}")

            time.sleep(0.05)

    with open("data/raw/price_history_1hour.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\ndone: {processed - failed}/{total} saved to data/raw/price_history_1hour.json")
    print(f"failed: {failed}")