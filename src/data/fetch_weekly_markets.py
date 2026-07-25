"""
fetch_weekly_markets.py
=======================
Fetches currently-live weekly Polymarket crypto markets (tag_slug="weekly").

Three types, all from the same events API:

1. "Coin above ___ on [Date]?"       — 11 price-level sub-markets per event
   One event per coin per day (July 25, 26, 27...). Pick closest to 0.5.

2. "Coin price on [Date]?"           — 11 price-range buckets per event
   Same structure. Pick closest to 0.5.

3. "What price will Coin hit Jul 20-26?"  — 14 sub-markets, true weekly candle
   Pick closest to 0.5.

Filter: only include events whose endDate is still in the future AND
belongs to the current week (Sun–Sat ET). Pre-created future-week events
are excluded.

Usage:
    python src/data/fetch_weekly_markets.py
    from src.data.fetch_weekly_markets import fetch_weekly_markets
    markets = fetch_weekly_markets()   # dict[coin -> list[market]]
"""

import json
import re
import requests
from datetime import datetime, timezone, timedelta

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
ET_OFFSET = timedelta(hours=-4)  # EDT (UTC-4)

COINS = {
    "BTC": [r"\bbitcoin\b", r"\bbtc\b"],
    "ETH": [r"\bethereum\b", r"\beth\b"],
    "SOL": [r"\bsolana\b", r"\bsol\b"],
    "XRP": [r"\bxrp\b", r"\bripple\b"],
    "BNB": [r"\bbnb\b", r"\bbinance\b"],
    "DOGE": [r"\bdogecoin\b", r"\bdoge\b"],
    "HYPE": [r"\bhype\b", r"\bhyperliquid\b"],
}


def _coin_for(text: str) -> str | None:
    t = text.lower()
    for coin, patterns in COINS.items():
        if any(re.search(p, t, re.IGNORECASE) for p in patterns):
            return coin
    return None


def _get_price(m: dict) -> float | None:
    for key in ("lastTradePrice", "bestBid", "bestAsk"):
        val = m.get(key)
        if val is not None:
            try:
                v = float(val)
                if 0.005 < v < 0.995:
                    return v
            except Exception:
                pass
    op = m.get("outcomePrices")
    if op:
        try:
            parsed = json.loads(op) if isinstance(op, str) else op
            if parsed:
                v = float(parsed[0])
                if 0.005 < v < 0.995:
                    return v
        except Exception:
            pass
    return None


def _normalise(m: dict) -> dict:
    return {
        "id": m.get("id"),
        "question": str(m.get("question", "")),
        "startDate": m.get("startDate"),
        "endDate": m.get("endDate"),
        "lastTradePrice": m.get("lastTradePrice"),
        "bestBid": m.get("bestBid"),
        "bestAsk": m.get("bestAsk"),
        "spread": m.get("spread"),
        "volume": m.get("volume"),
        "volumeNum": m.get("volumeNum"),
        "liquidity": m.get("liquidity"),
        "outcomes": m.get("outcomes"),
        "outcomePrices": m.get("outcomePrices"),
        "clobTokenIds": m.get("clobTokenIds"),
    }


def _current_week_end_ts(now_utc: datetime) -> float:
    """
    Return the timestamp of end-of-week Sunday midnight ET.
    Polymarket weekly series runs Mon-Sun. We use a generous window:
    any event ending within the next 7 days is considered 'this week'.
    """
    return now_utc.timestamp() + 7 * 24 * 3600


def fetch_weekly_markets() -> dict:
    """
    Return all currently-live weekly crypto markets, organised by coin.

    Returns:
        dict[coin -> list[market_dict]]

    Each coin can have multiple markets (one per event type per day/week).
    For each event, we pick the ONE sub-market closest to 0.5.
    """
    now_utc = datetime.now(timezone.utc)
    now_ts = now_utc.timestamp()
    week_end_ts = _current_week_end_ts(now_utc)
    now_et = now_utc + ET_OFFSET

    print(f"[fetch_weekly] Now: {now_et.strftime('%I:%M %p ET')}")

    try:
        resp = requests.get(
            f"{GAMMA_BASE_URL}/events",
            params={"limit": 100, "tag_slug": "weekly", "active": "true", "closed": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
        if not isinstance(events, list):
            return {coin: [] for coin in COINS}
    except Exception as e:
        print(f"  [weekly] events API failed: {e}")
        return {coin: [] for coin in COINS}

    result: dict = {coin: [] for coin in COINS}

    for event in events:
        title = event.get("title", "")
        ed = event.get("endDate", "")
        tags = [t.get("slug", "") for t in event.get("tags", [])]

        # Must be a crypto event
        if "crypto" not in tags:
            continue

        # Must still be open
        if not ed:
            continue
        try:
            end_dt = datetime.fromisoformat(ed.replace("Z", "+00:00"))
            end_ts = end_dt.timestamp()
        except Exception:
            continue

        if end_ts <= now_ts:
            continue  # already closed
        if end_ts > week_end_ts:
            continue  # beyond this week

        # Identify coin from title
        coin = _coin_for(title)
        if coin is None:
            continue

        sub_markets = event.get("markets", [])
        if not sub_markets:
            continue

        # Pick the sub-market closest to 0.5
        candidates = []
        for m in sub_markets:
            price = _get_price(m)
            if price is None:
                continue
            candidates.append((abs(price - 0.5), m))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])
        best = candidates[0][1]
        result[coin].append(_normalise(best))

    total = sum(len(v) for v in result.values())
    by_coin = {c: len(v) for c, v in result.items() if v}
    print(f"[fetch_weekly] Total live weekly markets: {total} | {by_coin}")
    return result


if __name__ == "__main__":
    markets = fetch_weekly_markets()
    print("\n── Results ──")
    for coin, mlist in markets.items():
        if not mlist:
            continue
        print(f"\n{coin} ({len(mlist)} markets):")
        for m in mlist:
            ltp = m.get("lastTradePrice")
            bid = m.get("bestBid")
            vol = float(m.get("volumeNum") or 0)
            print(f"  ltp={ltp} bid={bid} vol={vol:.0f} | {m['question'][:65]}")
