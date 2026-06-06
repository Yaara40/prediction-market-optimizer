import json
import requests
import time
from datetime import datetime, timezone

CLOB_BASE_URL = "https://clob.polymarket.com"

def fetch_history(token_id: str, start_ts: int, end_ts: int) -> list:
    try:
        response = requests.get(
            f"{CLOB_BASE_URL}/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": 720}
        )
        response.raise_for_status()
        return response.json().get("history", [])
    except:
        return []

if __name__ == "__main__":
    with open("data/raw/markets_monthly.json") as f:
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

            try:
                start_ts = int(datetime.fromisoformat(
                    m["startDate"].replace("Z", "+00:00")
                ).timestamp())
                end_ts = int(datetime.fromisoformat(
                    m["endDate"].replace("Z", "+00:00")
                ).timestamp())
            except:
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

    with open("data/raw/price_history_monthly.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\ndone: {processed - failed}/{total} saved to data/raw/price_history_monthly.json")
    print(f"failed: {failed}")