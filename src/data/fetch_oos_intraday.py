"""
Fetch June 2026 OOS markets for 1hour and 4hour timeframes.

Two-step pipeline per type:
  1. Fetch closed markets from Gamma API (filtered to June 2026)
  2. Enrich each market with CLOB price history

Results are APPENDED to the existing price_history_1hour.json and
price_history_4hour.json files so the OOS evaluator picks them up
automatically (it reads those files and filters by endDate >= 2026-06-01).

Run from repo root:
    venv/bin/python src/data/fetch_oos_intraday.py
"""

import json
import os
import re
import sys
import time
import requests
from datetime import datetime, timezone, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"

OOS_START = "2026-06-01"
OOS_END   = "2026-06-30"

COIN_LIST = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "HYPE"]
KEYWORDS = {
    "BTC":  [r"\bbitcoin\b", r"\bbtc\b"],
    "ETH":  [r"\bethereum\b", r"\beth\b"],
    "SOL":  [r"\bsolana\b",   r"\bsol\b"],
    "XRP":  [r"\bxrp\b",      r"\bripple\b"],
    "BNB":  [r"\bbnb\b",      r"\bbinance\b"],
    "DOGE": [r"\bdogecoin\b", r"\bdoge\b"],
    "HYPE": [r"\bhype\b",     r"\bhyperliquid\b"],
}


def matches_coin(text: str, kws: list) -> bool:
    return any(re.search(k, text, re.IGNORECASE) for k in kws)


# ── Step 1: Fetch closed markets from Gamma ───────────────────────────────────

def _is_1hour(question: str) -> bool:
    """True if question matches the 1-hour "Up or Down - Date, HH[AP]M ET" format."""
    return bool(re.search(r"up or down - .+, \d+[AP]M ET$", question, re.IGNORECASE))


def _get_4hour_minutes(question: str):
    """Return window duration in minutes if this is a 4-hour market, else None."""
    m = re.search(r"(\d+):(\d+)(AM|PM)-(\d+):(\d+)(AM|PM)", question, re.IGNORECASE)
    if not m:
        return None
    h1, mi1 = int(m.group(1)), int(m.group(2))
    h2, mi2 = int(m.group(4)), int(m.group(5))
    p1, p2  = m.group(3).upper(), m.group(6).upper()
    if p1 == "PM" and h1 != 12: h1 += 12
    if p1 == "AM" and h1 == 12: h1 = 0
    if p2 == "PM" and h2 != 12: h2 += 12
    if p2 == "AM" and h2 == 12: h2 = 0
    mins = (h2 * 60 + mi2) - (h1 * 60 + mi1)
    if mins <= 0: mins += 24 * 60
    return mins if mins == 240 else None


def fetch_june_markets() -> tuple[dict, dict]:
    """
    Page through closed markets (newest first) and collect June 2026
    1hour and 4hour markets.  Stop paging once endDate goes before June.
    Returns (markets_1hour, markets_4hour) — same structure as the raw files.
    """
    print(f"\nFetching closed 1h/4h markets for {OOS_START} – {OOS_END} ...")
    m1h = {c: [] for c in COIN_LIST}
    m4h = {c: [] for c in COIN_LIST}
    after_cursor = None
    page = 0
    seen_1h = seen_4h = 0
    stop = False

    while not stop:
        params = {
            "limit": 100,
            "order": "startDate",
            "ascending": "false",
            "closed": "true",
        }
        if after_cursor:
            params["after_cursor"] = after_cursor

        for attempt in range(5):
            try:
                resp = requests.get(f"{GAMMA}/markets/keyset", params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  page {page+1} attempt {attempt+1} failed: {e} — retry in {wait}s")
                time.sleep(wait)
        else:
            print("  too many failures — stopping")
            break

        markets = data.get("markets", [])
        if not markets:
            break

        for market in markets:
            question = market.get("question", "")
            end_date = (market.get("endDate") or "")[:10]

            if end_date and end_date > OOS_END:
                continue  # future of our OOS window — keep going
            if end_date and end_date < OOS_START:
                stop = True  # gone past June — done
                break

            text = (question + " " + str(market.get("slug", ""))).lower()
            record = {
                "id":             market.get("id"),
                "question":       question,
                "startDate":      market.get("startDate"),
                "endDate":        market.get("endDate"),
                "outcomePrices":  market.get("outcomePrices"),
                "outcomes":       market.get("outcomes"),
                "volume":         market.get("volume"),
                "volumeNum":      market.get("volumeNum"),
                "lastTradePrice": market.get("lastTradePrice"),
                "bestBid":        market.get("bestBid"),
                "bestAsk":        market.get("bestAsk"),
                "spread":         market.get("spread"),
                "closed":         market.get("closed"),
                "clobTokenIds":   market.get("clobTokenIds"),
            }

            if _is_1hour(question):
                seen_1h += 1
                record["duration_minutes"] = 60
                for coin, kws in KEYWORDS.items():
                    if matches_coin(text, kws):
                        m1h[coin].append(record)
                        break

            elif _get_4hour_minutes(question) == 240:
                seen_4h += 1
                record["duration_minutes"] = 240
                for coin, kws in KEYWORDS.items():
                    if matches_coin(text, kws):
                        m4h[coin].append(record)
                        break

        page += 1
        if (page % 10) == 0:
            t1 = sum(len(v) for v in m1h.values())
            t4 = sum(len(v) for v in m4h.values())
            print(f"  page {page}: 1h={t1}  4h={t4}")

        after_cursor = data.get("next_cursor")
        if not after_cursor:
            break

        time.sleep(0.1)

    t1 = sum(len(v) for v in m1h.values())
    t4 = sum(len(v) for v in m4h.values())
    print(f"  done — 1hour: {t1} markets ({seen_1h} seen)  |  4hour: {t4} markets ({seen_4h} seen)")
    return m1h, m4h


# ── Step 2: Enrich with CLOB price history ────────────────────────────────────

def _parse_1hour_window(market: dict):
    question = market.get("question", "")
    end_date = market.get("endDate", "")
    if not end_date:
        return None, None
    try:
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except Exception:
        return None, None
    match = re.search(r"(\d+)(AM|PM) ET$", question, re.IGNORECASE)
    if match:
        h, period = int(match.group(1)), match.group(2).upper()
        if period == "PM" and h != 12: h += 12
        if period == "AM" and h == 12: h = 0
        market_date = end_dt.date()
        end_ts   = int(datetime(market_date.year, market_date.month, market_date.day,
                                h, 0, tzinfo=timezone.utc).timestamp()) + 4 * 3600
        start_ts = end_ts - 3600
        return start_ts, end_ts
    return None, None


def _parse_4hour_window(market: dict):
    question = market.get("question", "")
    end_date = market.get("endDate", "")
    if not end_date:
        return None, None
    try:
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except Exception:
        return None, None
    m = re.search(r"(\d+):(\d+)(AM|PM)-(\d+):(\d+)(AM|PM)", question, re.IGNORECASE)
    if m:
        h1, mi1 = int(m.group(1)), int(m.group(2))
        h2, mi2 = int(m.group(4)), int(m.group(5))
        p1, p2  = m.group(3).upper(), m.group(6).upper()
        if p1 == "PM" and h1 != 12: h1 += 12
        if p1 == "AM" and h1 == 12: h1 = 0
        if p2 == "PM" and h2 != 12: h2 += 12
        if p2 == "AM" and h2 == 12: h2 = 0
        market_date = end_dt.date()
        start_ts = int(datetime(market_date.year, market_date.month, market_date.day,
                                h1, mi1, tzinfo=timezone.utc).timestamp()) + 4 * 3600
        end_ts   = int(datetime(market_date.year, market_date.month, market_date.day,
                                h2, mi2, tzinfo=timezone.utc).timestamp()) + 4 * 3600
        if end_ts <= start_ts:
            end_ts += 86400
        return start_ts, end_ts
    return None, None


def _fetch_clob(token_id: str, start_ts: int, end_ts: int) -> list:
    try:
        resp = requests.get(
            f"{CLOB}/prices-history",
            params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 1},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("history", [])
    except Exception:
        return []


def enrich(markets: dict, parse_window_fn) -> dict:
    enriched = {c: [] for c in markets}
    total = sum(len(v) for v in markets.values())
    done = ok = 0

    for coin, ms in markets.items():
        for m in ms:
            done += 1
            clob_ids = m.get("clobTokenIds")
            if not clob_ids:
                enriched[coin].append({**m, "price_history": []})
                continue
            try:
                ids = json.loads(clob_ids) if isinstance(clob_ids, str) else clob_ids
                token_id = ids[0]
            except Exception:
                enriched[coin].append({**m, "price_history": []})
                continue

            start_ts, end_ts = parse_window_fn(m)
            if not start_ts:
                enriched[coin].append({**m, "price_history": []})
                continue

            history = _fetch_clob(token_id, start_ts, end_ts)
            enriched[coin].append({**m, "price_history": history})
            if history:
                ok += 1

            if done % 50 == 0:
                print(f"  enriched {done}/{total} — {ok} with CLOB history")
            time.sleep(0.05)

    print(f"  done: {done} markets enriched, {ok} with real CLOB history")
    return enriched


# ── Step 3: Merge into existing price_history files ───────────────────────────

def merge_into_file(new_markets: dict, filepath: str):
    """
    Append new_markets to the existing file, deduplicating by market id.
    """
    if os.path.exists(filepath):
        with open(filepath) as f:
            existing = json.load(f)
    else:
        existing = {c: [] for c in COIN_LIST}

    added = 0
    for coin, ms in new_markets.items():
        existing_ids = {m.get("id") for m in existing.get(coin, [])}
        for m in ms:
            if m.get("id") not in existing_ids:
                existing.setdefault(coin, []).append(m)
                existing_ids.add(m.get("id"))
                added += 1

    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"  {added} new markets appended to {filepath}")
    return added


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: fetch
    markets_1h, markets_4h = fetch_june_markets()

    t1 = sum(len(v) for v in markets_1h.values())
    t4 = sum(len(v) for v in markets_4h.values())

    if t1 == 0 and t4 == 0:
        print("No June 1h/4h markets found — exiting.")
        sys.exit(0)

    # Step 2: enrich with CLOB history
    if t1 > 0:
        print(f"\nEnriching {t1} 1-hour markets with CLOB history...")
        enriched_1h = enrich(markets_1h, _parse_1hour_window)
        with open("data/raw/oos_1hour.json", "w") as f:
            json.dump(enriched_1h, f, indent=2)
        print(f"  saved to data/raw/oos_1hour.json")

        print("\nMerging into data/raw/price_history_1hour.json ...")
        merge_into_file(enriched_1h, "data/raw/price_history_1hour.json")
    else:
        print("No 1-hour June markets collected.")

    if t4 > 0:
        print(f"\nEnriching {t4} 4-hour markets with CLOB history...")
        enriched_4h = enrich(markets_4h, _parse_4hour_window)
        with open("data/raw/oos_4hour.json", "w") as f:
            json.dump(enriched_4h, f, indent=2)
        print(f"  saved to data/raw/oos_4hour.json")

        print("\nMerging into data/raw/price_history_4hour.json ...")
        merge_into_file(enriched_4h, "data/raw/price_history_4hour.json")
    else:
        print("No 4-hour June markets collected.")

    print("\nDone. Now re-run the OOS evaluator:")
    print("  venv/bin/python src/backtest/oos_eval.py")
