"""
fetch_1hour_markets.py
======================
Fetches ONLY the currently-live 1-hour Polymarket markets.

Two types of 1-hour markets:

1. "Coin Up or Down - [Date], [H]AM ET"
   Source: Polymarket events API (tag_slug="1h")
   One market per coin per hour. The single sub-market per event is the binary
   "Up or Down" result for that coin in that ET hour candle.

2. "Coin above $X on [Date], [H]AM ET?"
   Source: Polymarket markets keyset API
   Many price-level variants per coin per hour (e.g. BTC above 62k, 63k, 64k …).
   We pick the ONE with lastTradePrice closest to 0.5 — the most uncertain bucket,
   which gives the ML model the most useful signal.

Timezone: Polymarket 1h markets are defined in ET (EDT = UTC-4 in summer).
  "4AM ET" session ends at 5AM ET = 09:00 UTC.

Usage:
    python src/data/fetch_1hour_markets.py      # prints what's live right now
    from src.data.fetch_1hour_markets import fetch_1hour_markets
    markets = fetch_1hour_markets()             # returns dict[coin -> list[market]]
"""

import json
import re
import time
import requests
from datetime import datetime, timezone, timedelta

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
ET_OFFSET = timedelta(hours=-4)  # EDT (UTC-4) — adjust to -5 in winter

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
    """Return the coin key that matches the market question, or None."""
    t = text.lower()
    for coin, patterns in COINS.items():
        if any(re.search(p, t, re.IGNORECASE) for p in patterns):
            return coin
    return None


def _get_price(m: dict) -> float | None:
    """Best single-point YES price estimate for a market."""
    ltp = m.get("lastTradePrice")
    if ltp is not None:
        try:
            v = float(ltp)
            if 0.005 < v < 0.995:
                return v
        except Exception:
            pass
    bid = m.get("bestBid")
    if bid is not None:
        try:
            v = float(bid)
            if 0.005 < v < 0.995:
                return v
        except Exception:
            pass
    ask = m.get("bestAsk")
    if ask is not None:
        try:
            v = float(ask)
            if 0.005 < v < 0.995:
                return v
        except Exception:
            pass
    op = m.get("outcomePrices")
    if op:
        try:
            parsed = json.loads(op) if isinstance(op, str) else op
            if parsed:
                return float(parsed[0])
        except Exception:
            pass
    return None


def _normalise(m: dict) -> dict:
    """Return a normalised market dict compatible with the pipeline."""
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


def _current_hour_window(now_utc: datetime) -> tuple[float, float]:
    """
    Return (window_start_ts, window_end_ts) for the current ET hour.

    Example at 4:18 AM ET:
      → current ET hour = 4  (started 4:00 AM, ends 5:00 AM)
      → window_end = 5:00 AM ET = 09:00 UTC
      → window_start = 4:00 AM ET = 08:00 UTC
    """
    now_et = now_utc + ET_OFFSET
    # Floor to current ET hour
    hour_start_et = now_et.replace(minute=0, second=0, microsecond=0)
    hour_end_et = hour_start_et + timedelta(hours=1)

    # Convert ET → UTC (subtract offset since ET = UTC + offset where offset is negative)
    def et_to_utc(dt_et: datetime) -> datetime:
        return dt_et - ET_OFFSET

    start_ts = et_to_utc(hour_start_et).timestamp()
    end_ts = et_to_utc(hour_end_et).timestamp()
    return start_ts, end_ts


# ─────────────────────────────────────────────────────────────────────────────
# Source 1: "Up or Down" binary events (tag_slug="1h")
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_updown_markets(now_utc: datetime) -> dict:
    """
    Fetch "Coin Up or Down - [Date], [H]AM ET" markets for the CURRENT ET hour.

    Returns dict[coin -> market_dict] (at most 1 per coin).
    """
    now_ts = now_utc.timestamp()
    _, window_end_ts = _current_hour_window(now_utc)

    try:
        resp = requests.get(
            f"{GAMMA_BASE_URL}/events",
            params={"limit": 100, "tag_slug": "1h", "active": "true", "closed": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
        if not isinstance(events, list):
            return {}
    except Exception as e:
        print(f"  [1h] events API failed: {e}")
        return {}

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

        # Only events in the CURRENT 1-hour window:
        # endDate must match the current hour-end (within a small tolerance)
        # and must still be in the future.
        if end_ts <= now_ts:
            continue  # already closed
        # Allow a 5-minute tolerance so we still grab it near the end of the hour
        if abs(end_ts - window_end_ts) > 300:
            continue  # different hour window (future or past)

        sub_markets = event.get("markets", [])
        if not sub_markets:
            continue

        # Each event has exactly 1 sub-market for the "Up or Down" binary
        for m in sub_markets:
            q = str(m.get("question", ""))
            coin = _coin_for(q)
            if coin is None:
                continue
            price = _get_price(m)
            if price is None:
                continue  # no real price signal yet (market hasn't opened)
            if coin not in result:
                result[coin] = _normalise(m)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: "Coin above $X on [Date], [H]AM ET?" from keyset
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_above_markets(now_utc: datetime, max_pages: int = 10) -> dict:
    """
    Fetch "Coin above $X on [Date], [H]AM ET?" markets for the CURRENT ET hour.

    Multiple price levels exist per coin. We pick the ONE whose lastTradePrice
    is closest to 0.5 (most uncertain = best ML signal).

    Returns dict[coin -> market_dict].
    """
    now_ts = now_utc.timestamp()
    _, window_end_ts = _current_hour_window(now_utc)

    # Tolerance: markets whose endDate is within ±10 min of the current hour-end
    # (Polymarket sometimes has them at :00 vs :01)
    WINDOW_TOL = 600  # 10 minutes

    # Gather all "above" candidates for the current hour
    candidates: dict[str, list[dict]] = {coin: [] for coin in COINS}
    cursor = None

    for page in range(max_pages):
        params = {
            "limit": 100,
            "order": "startDate",
            "ascending": "false",
            "active": "true",
            "closed": "false",
        }
        if cursor:
            params["after_cursor"] = cursor

        try:
            resp = requests.get(
                f"{GAMMA_BASE_URL}/markets/keyset",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [above] keyset page {page+1} failed: {e}")
            break

        page_markets = data.get("markets", [])
        if not page_markets:
            break

        found_any_current = False
        for m in page_markets:
            q = str(m.get("question", ""))
            if not re.search(r'\babove\b', q, re.IGNORECASE):
                continue

            ed = m.get("endDate", "")
            if not ed:
                continue
            try:
                end_dt = datetime.fromisoformat(ed.replace("Z", "+00:00"))
                end_ts = end_dt.timestamp()
            except Exception:
                continue

            if end_ts <= now_ts:
                continue  # already closed

            # Is this in the current 1-hour window?
            if abs(end_ts - window_end_ts) > WINDOW_TOL:
                continue

            found_any_current = True
            coin = _coin_for(q)
            if coin is None:
                continue

            price = _get_price(m)
            if price is None:
                continue  # no trade data

            candidates[coin].append({**m, "_price": price})

        cursor = data.get("next_cursor")
        if not cursor:
            break

        # If we've scanned several pages and found no current-window markets,
        # stop early — keyset is sorted by startDate desc so older markets follow.
        if page >= 3 and not found_any_current:
            break

        time.sleep(0.05)

    # For each coin, pick the market closest to 0.5
    result: dict = {}
    for coin, mlist in candidates.items():
        if not mlist:
            continue
        best = min(mlist, key=lambda m: abs(m["_price"] - 0.5))
        # Remove the internal _price field before returning
        entry = {k: v for k, v in best.items() if k != "_price"}
        result[coin] = _normalise(entry)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_1hour_markets() -> dict:
    """
    Return all currently-live 1-hour markets, organised by coin.

    Returns:
        dict[coin -> list[market_dict]]

    Each coin can have up to 2 markets:
      - "Up or Down" binary (from events API)
      - "above $X" price-level (best bucket from keyset, closest to 0.5)
    """
    now_utc = datetime.now(timezone.utc)
    ET_OFFSET_NEG = timedelta(hours=-4)
    now_et = now_utc + ET_OFFSET_NEG
    _, window_end_ts = _current_hour_window(now_utc)
    window_end_et = datetime.fromtimestamp(window_end_ts, tz=timezone.utc) + ET_OFFSET_NEG

    print(f"[fetch_1hour] Now: {now_et.strftime('%I:%M %p ET')} | current window ends: {window_end_et.strftime('%I:%M %p ET')}")

    print("[fetch_1hour] Fetching 'Up or Down' markets (events API)...")
    updown = _fetch_updown_markets(now_utc)
    print(f"  → {len(updown)} coins found: {list(updown.keys())}")

    print("[fetch_1hour] Fetching 'above $X' markets (keyset)...")
    above = _fetch_above_markets(now_utc)
    print(f"  → {len(above)} coins found: {list(above.keys())}")

    # Merge into per-coin lists
    result: dict = {coin: [] for coin in COINS}
    for coin, m in updown.items():
        result[coin].append(m)
    for coin, m in above.items():
        result[coin].append(m)

    total = sum(len(v) for v in result.values())
    print(f"[fetch_1hour] Total live 1h markets: {total}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    markets = fetch_1hour_markets()
    print("\n── Results ──")
    for coin, mlist in markets.items():
        if not mlist:
            continue
        print(f"\n{coin}:")
        for m in mlist:
            q = m.get("question", "")
            ltp = m.get("lastTradePrice")
            bid = m.get("bestBid")
            ask = m.get("bestAsk")
            vol = m.get("volumeNum") or 0
            ed = m.get("endDate", "")
            print(f"  [{ed}] ltp={ltp} bid={bid} ask={ask} vol={vol:.1f}")
            print(f"  {q}")
