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

    # 4hour: "Bitcoin Up or Down - May 29, 4:00AM-8:00AM ET"
    match = re.search(r'(\d+):(\d+)(AM|PM)-(\d+):(\d+)(AM|PM)', question, re.IGNORECASE)
    if match:
        h1, m1, period1 = int(match.group(1)), int(match.group(2)), match.group(3).upper()
        h2, m2, period2 = int(match.group(4)), int(match.group(5)), match.group(6).upper()

        if period1 == "PM" and h1 != 12: h1 += 12
        if period1 == "AM" and h1 == 12: h1 = 0
        if period2 == "PM" and h2 != 12: h2 += 12
        if period2 == "AM" and h2 == 12: h2 = 0

        market_date = end_dt.date()
        start_trading = datetime(market_date.year, market_date.month, market_date.day,
                                h1, m1, tzinfo=timezone.utc) + timedelta(hours=4)
        end_trading = datetime(market_date.year, market_date.month, market_date.day,
                              h2, m2, tzinfo=timezone.utc) + timedelta(hours=4)

        if end_trading <= start_trading:
            end_trading += timedelta(days=1)

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
    with open("data/raw/markets_4hour.json") as f:
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

    with open("data/raw/price_history_4hour.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\ndone: {processed - failed}/{total} saved to data/raw/price_history_4hour.json")
    print(f"failed: {failed}")