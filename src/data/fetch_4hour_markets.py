"""
fetch_4hour_markets.py
======================
Fetches ONLY the currently-live 4-hour Polymarket markets.

Market type: "Coin Up or Down - [Date], [H:MM]AM-[H:MM]AM ET"
Source: Polymarket events API (tag_slug="4h")
One market per coin per 4-hour block. 7 coins total.

4-hour blocks (ET):
  12:00AM–4:00AM  →  ends 04:00 ET = 08:00 UTC
   4:00AM–8:00AM  →  ends 08:00 ET = 12:00 UTC
   8:00AM–12:00PM →  ends 12:00 ET = 16:00 UTC
  12:00PM–4:00PM  →  ends 16:00 ET = 20:00 UTC
   4:00PM–8:00PM  →  ends 20:00 ET = 00:00 UTC
   8:00PM–12:00AM →  ends 00:00 ET = 04:00 UTC

Usage:
    python src/data/fetch_4hour_markets.py
    from src.data.fetch_4hour_markets import fetch_4hour_markets
    markets = fetch_4hour_markets()   # dict[coin -> list[market]]
"""

import json
import re
import time
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
    """Best single-point YES price estimate for a market."""
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


def _current_4h_window(now_utc: datetime) -> tuple[float, float]:
    """
    Return (window_start_ts, window_end_ts) for the current 4-hour ET block.

    4h blocks are aligned to 0, 4, 8, 12, 16, 20 ET.
    Example at 4:37 AM ET → block = 4AM–8AM ET → end = 8AM ET = 12:00 UTC.
    """
    now_et = now_utc + ET_OFFSET
    # Floor to nearest 4-hour boundary
    block_hour = (now_et.hour // 4) * 4
    block_start_et = now_et.replace(hour=block_hour, minute=0, second=0, microsecond=0)
    block_end_et = block_start_et + timedelta(hours=4)

    def et_to_utc(dt_et: datetime) -> datetime:
        return dt_et - ET_OFFSET

    return et_to_utc(block_start_et).timestamp(), et_to_utc(block_end_et).timestamp()


def fetch_4hour_markets() -> dict:
    """
    Return all currently-live 4-hour "Up or Down" markets, organised by coin.

    Returns:
        dict[coin -> list[market_dict]]

    Each coin has at most 1 market (the current 4h block binary).
    """
    now_utc = datetime.now(timezone.utc)
    now_ts = now_utc.timestamp()
    _, window_end_ts = _current_4h_window(now_utc)

    window_end_et = datetime.fromtimestamp(window_end_ts, tz=timezone.utc) + ET_OFFSET
    now_et = now_utc + ET_OFFSET
    print(f"[fetch_4hour] Now: {now_et.strftime('%I:%M %p ET')} | current window ends: {window_end_et.strftime('%I:%M %p ET')}")

    try:
        resp = requests.get(
            f"{GAMMA_BASE_URL}/events",
            params={"limit": 100, "tag_slug": "4h", "active": "true", "closed": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
        if not isinstance(events, list):
            return {coin: [] for coin in COINS}
    except Exception as e:
        print(f"  [4h] events API failed: {e}")
        return {coin: [] for coin in COINS}

    result: dict = {}

    for event in events:
        ed = event.get("endDate", "")
        if not ed:
            continue
        try:
            end_dt = datetime.fromisoformat(ed.replace("Z", "+00:00"))
            end_ts = end_dt.timestamp()
        except Exception:
            continue

        # Only the current 4h window: endDate must match within 5-min tolerance
        if end_ts <= now_ts:
            continue  # already closed
        if abs(end_ts - window_end_ts) > 300:
            continue  # different block (next session, etc.)

        sub_markets = event.get("markets", [])
        if not sub_markets:
            continue

        for m in sub_markets:
            q = str(m.get("question", ""))
            coin = _coin_for(q)
            if coin is None:
                continue
            price = _get_price(m)
            if price is None:
                continue  # no price signal at all
            if coin not in result:
                result[coin] = _normalise(m)

    out = {coin: ([result[coin]] if coin in result else []) for coin in COINS}
    total = sum(len(v) for v in out.values())
    found = [c for c, v in out.items() if v]
    print(f"[fetch_4hour] Total live 4h markets: {total} | coins: {found}")
    return out


if __name__ == "__main__":
    markets = fetch_4hour_markets()
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
