import json
import pandas as pd

def extract_features(market: dict, n_snapshots: int = 10) -> dict | None:
    history = market.get("price_history", [])
    if len(history) < 3:
        return None

    prices = [h["p"] for h in history]
    timestamps = [h["t"] for h in history]
    duration = timestamps[-1] - timestamps[0]

    if duration == 0:
        return None

    price_snapshots = {}
    for i in range(n_snapshots):
        target_ts = timestamps[0] + (duration * i / max(n_snapshots - 1, 1))
        closest = min(range(len(timestamps)), key=lambda j: abs(timestamps[j] - target_ts))
        price_snapshots[f"price_t{i}"] = prices[closest]

    crossings = 0
    for i in range(1, len(prices)):
        if (prices[i-1] < 0.5 and prices[i] >= 0.5) or \
           (prices[i-1] >= 0.5 and prices[i] < 0.5):
            crossings += 1

    def price_at(pct):
        target_ts = timestamps[0] + duration * pct
        closest = min(range(len(timestamps)), key=lambda j: abs(timestamps[j] - target_ts))
        return prices[closest]

    return {
        **price_snapshots,
        "crossings_05": crossings,
        "price_early": price_at(0.05),
        "price_mid": price_at(0.50),
        "price_late": price_at(0.95),
    }

def build(input_file: str, output_file: str, n_snapshots: int = 10):
    with open(input_file) as f:
        raw = json.load(f)

    rows = []
    total = sum(len(v) for v in raw.values())
    failed = 0

    for coin_idx, (coin, markets) in enumerate(raw.items()):
        for m in markets:
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

            features = extract_features(m, n_snapshots=n_snapshots)
            if not features:
                failed += 1
                continue

            rows.append({
                **features,
                "duration_minutes": m.get("duration_minutes", 1440),
                "volume": float(m.get("volumeNum") or 0),
                "coin": coin_idx,
                "label": label
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"done: {len(rows)}/{total} saved to {output_file} (failed: {failed})")
    return df

if __name__ == "__main__":
    print("building 1-day dataset...")
    df = build(
        input_file="data/raw/price_history_1day.json",
        output_file="data/raw/dataset_1day.csv",
        n_snapshots=10
    )
    print(df.describe())