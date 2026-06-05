import json
import requests
import time
from datetime import datetime, timezone, timedelta
import re

CLOB_BASE_URL = "https://clob.polymarket.com"

def parse_trading_window(market: dict) -> tuple:
    """1day market: 'Bitcoin Up or Down on May 30?' — runs from noon to noon next day"""
    end_date = market.get("endDate", "")

    if not end_date:
        return None, None

    try:
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        # end is at noon ET (16:00 UTC), start is 24 hours before
        end_ts = int(end_dt.timestamp())
        start_ts = end_ts - (24 * 3600)
        return start_ts, end_ts
    except:
        return None, None

def fetch_history(token_id: str, start_ts: int, end_ts: int) -> list:
    try:
        response = requests.get(
            f"{CLOB_BASE_URL}/prices-history",
            params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 60}
        )
        response.raise_for_status()
        return response.json().get("history", [])
    except:
        return []

if __name__ == "__main__":
    with open("data/raw/markets_1day.json") as f:
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

    with open("data/raw/price_history_1day.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\ndone: {processed - failed}/{total} saved to data/raw/price_history_1day.json")
    print(f"failed: {failed}")