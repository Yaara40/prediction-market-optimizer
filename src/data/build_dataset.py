import json
import re
import time
import pandas as pd
from datetime import datetime, timezone
from src.data.price_history import get_market_features

def get_market_type(question: str) -> str:
    match = re.search(r'(\d+):(\d+)[AP]M-(\d+):(\d+)[AP]M', question)
    if match:
        m1 = int(match.group(2))
        m2 = int(match.group(4))
        diff = abs(m2 - m1)
        if diff == 5 or diff == 55:
            return '5min'
        elif diff == 15 or diff == 45:
            return '15min'
    if re.search(r'\d+[AP]M ET$', question):
        return '1day'
    return 'event'

def build_dataset(market_type: str = '15min'):
    with open("data/resolved_markets_12h_7days.json") as f:
        raw = json.load(f)

    rows = []
    total_checked = 0
    failed = 0

    for coin_idx, (coin, markets) in enumerate(raw.items()):
        for m in markets:
            if get_market_type(m['question']) != market_type:
                continue

            total_checked += 1

            outcome_prices = m.get("outcomePrices")
            if not outcome_prices:
                failed += 1
                continue

            try:
                prices = json.loads(outcome_prices)
                label = 1 if prices[0] == "1" else 0
            except:
                failed += 1
                continue

            try:
                start_dt = datetime.fromisoformat(m["startDate"].replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00"))
                duration_hours = (end_dt - start_dt).total_seconds() / 3600
            except:
                failed += 1
                continue

            features = get_market_features(m)
            if not features:
                failed += 1
                continue

            rows.append({
                **features,
                "duration_hours": duration_hours,
                "volume": float(m.get("volumeNum") or 0),
                "coin": coin_idx,
                "label": label
            })

            if len(rows) % 50 == 0:
                print(f"processed {len(rows)} markets so far...")

            time.sleep(0.05)

    df = pd.DataFrame(rows)
    output_path = f"data/dataset_{market_type}.csv"
    df.to_csv(output_path, index=False)
    print(f"\ndone. {len(rows)}/{total_checked} markets saved to {output_path}")
    print(f"failed to fetch history: {failed}")
    return df

if __name__ == "__main__":
    print("building dataset for 5-minute markets...")
    df = build_dataset(market_type='15min')
    print(df.describe())
