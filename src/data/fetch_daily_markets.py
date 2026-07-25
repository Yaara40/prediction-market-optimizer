"""
fetch_daily_markets.py
======================
Fetches ONLY the currently-live daily Polymarket crypto markets.

Two types of daily markets (both from tag_slug="daily", tag "crypto"):

1. "Coin Up or Down on [Date]?"
   - 7 coins: BTC, ETH, SOL, XRP, BNB, DOGE, HYPE
   - 1 binary sub-market per event
   - Ends at midnight ET (e.g. 2026-07-25T16:00:00Z = 12:00 AM ET next day)

2. "What price will Coin hit on [Date]?"
   - 4 coins: BTC, ETH, SOL, XRP
   - 10-16 price-level sub-markets per event
   - Ends ~12h later than the Up/Down (e.g. 2026-07-26T04:00:00Z)
   - We pick the ONE sub-market closest to 0.5 (most uncertain bucket)

Both types resolve for today's ET date. Tomorrow's pre-created sessions
are excluded by checking that the event title contains today's ET date.

Usage:
    python src/data/fetch_daily_markets.py
    from src.data.fetch_daily_markets import fetch_daily_markets
    markets = fetch_daily_markets()   # dict[coin -> list[market]]
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


def fetch_daily_markets() -> dict:
    """
    Return all currently-live daily crypto markets, organised by coin.

    Returns:
        dict[coin -> list[market_dict]]

    Each coin can have up to 2 markets:
      - "Up or Down on [today]?" binary
      - "What price will coin hit on [today]?" closest-to-0.5 bucket
    """
    now_utc = datetime.now(timezone.utc)
    now_ts = now_utc.timestamp()
    now_et = now_utc + ET_OFFSET

    # Today's ET date string for title matching, e.g. "July 25"
    today_et_str = now_et.strftime("%-d")  # day without leading zero
    today_et_month = now_et.strftime("%B")  # full month name
    today_label = f"{today_et_month} {today_et_str}"  # e.g. "July 25"

    print(f"[fetch_daily] Now: {now_et.strftime('%I:%M %p ET')} | today: {today_label}")

    try:
        resp = requests.get(
            f"{GAMMA_BASE_URL}/events",
            params={"limit": 100, "tag_slug": "daily", "active": "true", "closed": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
        if not isinstance(events, list):
            return {coin: [] for coin in COINS}
    except Exception as e:
        print(f"  [daily] events API failed: {e}")
        return {coin: [] for coin in COINS}

    updown: dict = {}    # coin -> market_dict
    price_hit: dict = {} # coin -> market_dict (best bucket)

    for event in events:
        title = event.get("title", "")
        ed = event.get("endDate", "")
        tags = [t.get("slug", "") for t in event.get("tags", [])]

        # Must be a crypto event
        if "crypto" not in tags:
            continue

        # Must be for today's ET date
        if today_label not in title:
            continue

        # Must still be open
        if ed:
            try:
                end_dt = datetime.fromisoformat(ed.replace("Z", "+00:00"))
                if end_dt.timestamp() <= now_ts:
                    continue
            except Exception:
                continue

        sub_markets = event.get("markets", [])
        if not sub_markets:
            continue

        # Determine which type based on title
        is_updown = bool(re.search(r'up or down on', title, re.IGNORECASE))
        is_pricehit = bool(re.search(r'what price will', title, re.IGNORECASE))

        if is_updown:
            # Single binary sub-market
            for m in sub_markets:
                q = str(m.get("question", ""))
                coin = _coin_for(title)  # use title for coin detection (more reliable)
                if coin is None:
                    coin = _coin_for(q)
                if coin is None:
                    continue
                price = _get_price(m)
                if price is None:
                    continue
                if coin not in updown:
                    updown[coin] = _normalise(m)

        elif is_pricehit:
            # Multiple price-level sub-markets — pick closest to 0.5
            coin = _coin_for(title)
            if coin is None:
                continue
            candidates = []
            for m in sub_markets:
                price = _get_price(m)
                if price is None:
                    continue
                candidates.append((abs(price - 0.5), price, m))
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[0])
            best_m = candidates[0][2]
            if coin not in price_hit:
                price_hit[coin] = _normalise(best_m)

    # Merge into per-coin lists
    result: dict = {coin: [] for coin in COINS}
    for coin, m in updown.items():
        result[coin].append(m)
    for coin, m in price_hit.items():
        result[coin].append(m)

    total = sum(len(v) for v in result.values())
    ud_coins = list(updown.keys())
    ph_coins = list(price_hit.keys())
    print(f"[fetch_daily] Total: {total} | Up/Down: {ud_coins} | Price-hit: {ph_coins}")
    return result


if __name__ == "__main__":
    markets = fetch_daily_markets()
    print("\n── Results ──")
    for coin, mlist in markets.items():
        if not mlist:
            continue
        print(f"\n{coin}:")
        for m in mlist:
            ltp = m.get("lastTradePrice")
            bid = m.get("bestBid")
            ask = m.get("bestAsk")
            vol = float(m.get("volumeNum") or 0)
            ed = m.get("endDate", "")
            print(f"  [{ed}] ltp={ltp} bid={bid} ask={ask} vol={vol:.1f}")
            print(f"  {m['question']}")
