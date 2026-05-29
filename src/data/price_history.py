import requests
import json
import re
import numpy as np
from datetime import datetime, timezone, timedelta

CLOB_BASE_URL = "https://clob.polymarket.com"

def parse_trading_window(market: dict) -> tuple:
    """extract actual trading start and end timestamps from question title"""
    question = market.get("question", "")
    end_date = market.get("endDate", "")

    if not end_date:
        return None, None

    # parse the end date to get the date part
    try:
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except:
        return None, None

    # match pattern like "1:50AM-1:55AM" or "9:00AM-9:15AM"
    match = re.search(r'(\d+):(\d+)(AM|PM)-(\d+):(\d+)(AM|PM)', question, re.IGNORECASE)
    if match:
        h1, m1, period1 = int(match.group(1)), int(match.group(2)), match.group(3).upper()
        h2, m2, period2 = int(match.group(4)), int(match.group(5)), match.group(6).upper()

        # convert to 24h
        if period1 == "PM" and h1 != 12:
            h1 += 12
        if period1 == "AM" and h1 == 12:
            h1 = 0
        if period2 == "PM" and h2 != 12:
            h2 += 12
        if period2 == "AM" and h2 == 12:
            h2 = 0

        # use end_date's date as reference, converting from ET to UTC (ET = UTC-4 or UTC-5)
        # polymarket uses ET so we add 4 hours for EDT
        market_date = end_dt.date()
        start_trading = datetime(market_date.year, market_date.month, market_date.day,
                                h1, m1, tzinfo=timezone.utc) + timedelta(hours=4)
        end_trading = datetime(market_date.year, market_date.month, market_date.day,
                              h2, m2, tzinfo=timezone.utc) + timedelta(hours=4)

        # handle midnight crossover
        if end_trading <= start_trading:
            end_trading += timedelta(days=1)

        return int(start_trading.timestamp()), int(end_trading.timestamp())

    # match 1-day pattern like "4AM ET"
    match = re.search(r'(\d+)(AM|PM) ET$', question, re.IGNORECASE)
    if match:
        h, period = int(match.group(1)), match.group(2).upper()
        if period == "PM" and h != 12:
            h += 12
        if period == "AM" and h == 12:
            h = 0
        market_date = end_dt.date()
        start_trading = datetime(market_date.year, market_date.month, market_date.day,
                                h, 0, tzinfo=timezone.utc) + timedelta(hours=4)
        end_trading = start_trading + timedelta(hours=24)
        return int(start_trading.timestamp()), int(end_trading.timestamp())

    # fallback to full startDate-endDate
    try:
        start_ts = int(datetime.fromisoformat(
            market["startDate"].replace("Z", "+00:00")
        ).timestamp())
        return start_ts, int(end_dt.timestamp())
    except:
        return None, None

def fetch_price_history(token_id: str, start_ts: int, end_ts: int) -> list:
    """fetch price history for a token between two timestamps"""
    try:
        response = requests.get(
            f"{CLOB_BASE_URL}/prices-history",
            params={
                "market": token_id,
                "startTs": start_ts,
                "endTs": end_ts,
                "fidelity": 1
            }
        )
        response.raise_for_status()
        return response.json().get("history", [])
    except:
        return []

def extract_features(history: list, start_ts: int, end_ts: int) -> dict:
    """extract normalized features from price history"""
    if len(history) < 8:
        return None

    prices = [h["p"] for h in history]
    timestamps = [h["t"] for h in history]
    duration = end_ts - start_ts

    # 10 normalized price snapshots
    price_snapshots = {}
    for i in range(10):
        target_ts = start_ts + (duration * i / 9)
        closest = min(range(len(timestamps)), key=lambda j: abs(timestamps[j] - target_ts))
        price_snapshots[f"price_t{i}"] = prices[closest]

    # number of times price crossed 0.5
    crossings = 0
    for i in range(1, len(prices)):
        if (prices[i-1] < 0.5 and prices[i] >= 0.5) or \
           (prices[i-1] >= 0.5 and prices[i] < 0.5):
            crossings += 1

    # price at 5%, 50%, 95% of market lifetime
    t_early = start_ts + duration * 0.05
    t_mid = start_ts + duration * 0.50
    t_late = start_ts + duration * 0.95

    def price_at(target):
        closest = min(range(len(timestamps)), key=lambda j: abs(timestamps[j] - target))
        return prices[closest]

    return {
        **price_snapshots,
        "crossings_05": crossings,
        "price_early": price_at(t_early),
        "price_mid": price_at(t_mid),
        "price_late": price_at(t_late),
    }

def get_market_features(market: dict) -> dict:
    """get all features for one market"""
    clob_ids = market.get("clobTokenIds")
    if not clob_ids:
        return None

    try:
        token_ids = json.loads(clob_ids) if isinstance(clob_ids, str) else clob_ids
        token_id = token_ids[0]
    except:
        return None

    start_ts, end_ts = parse_trading_window(market)
    if not start_ts or not end_ts:
        return None

    history = fetch_price_history(token_id, start_ts, end_ts)
    return extract_features(history, start_ts, end_ts)

if __name__ == "__main__":
    import json
    with open("data/resolved_markets_12h_7days.json") as f:
        data = json.load(f)

    targets = [
        'Bitcoin Up or Down - May 28, 1:50AM-1:55AM ET',
        'Bitcoin Up or Down - May 26, 5:35PM-5:40PM ET'
    ]

    for m in data["BTC"]:
        if m["question"] in targets:
            print("market:", m["question"])
            start_ts, end_ts = parse_trading_window(m)
            print("trading window:", start_ts, "->", end_ts)
            features = get_market_features(m)
            if features:
                for k, v in features.items():
                    print(f"  {k}: {v:.3f}")
            else:
                print("no features")
            print()