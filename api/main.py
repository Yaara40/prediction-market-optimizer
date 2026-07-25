import os
import sys
import json
import random
import time
from datetime import datetime, timedelta

# MUST be first — resolves relative paths used by pipeline (src/model/, data/)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import re
import joblib
import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.optimizer.pipeline import (
    predict_probabilities,
    load_models,
    MODELS,
    get_price_features_with_type,
    get_model_for_market,
)
from src.data.coingecko import get_correlation_matrix
from src.optimizer.kelly_markowitz import optimize_portfolio, kelly_fraction_for_risk
from src.data.fetch_1hour_markets import fetch_1hour_markets
from src.data.fetch_4hour_markets import fetch_4hour_markets
from src.data.fetch_daily_markets import fetch_daily_markets
from src.data.fetch_weekly_markets import fetch_weekly_markets

app = FastAPI(title="Prediction Market Optimizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Baseline models for live comparison ──────────────────────────────────────

BASELINE_MODEL = None  # Logistic Regression trained on combined dataset

def load_baseline():
    global BASELINE_MODEL
    try:
        BASELINE_MODEL = joblib.load("src/model/baseline_logistic.pkl")
        print("Loaded baseline_logistic.pkl")
    except FileNotFoundError:
        print("baseline_logistic.pkl not found — live comparison unavailable")


# ── Startup: load models once ────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    print("Loading ML models...")
    load_models()
    print(f"Loaded {len(MODELS)} models.")
    load_baseline()

# ── Live Polymarket fetch with 5-min in-memory cache ─────────────────────────

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

KEYWORDS = {
    "BTC": [r"\bbitcoin\b", r"\bbtc\b"],
    "ETH": [r"\bethereum\b", r"\beth\b"],
    "SOL": [r"\bsolana\b", r"\bsol\b"],
    "XRP": [r"\bxrp\b", r"\bripple\b"],
    "BNB": [r"\bbnb\b", r"\bbinance\b"],
    "DOGE": [r"\bdogecoin\b", r"\bdoge\b"],
    "HYPE": [r"\bhype\b", r"\bhyperliquid\b"],
}

_markets_cache: dict = {}
_cache_ts: float = 0.0
CACHE_TTL = 300  # 5 minutes


def _matches_coin(text: str, keywords: list) -> bool:
    return any(re.search(k, text, re.IGNORECASE) for k in keywords)


def fetch_live_markets(max_pages: int = 20) -> dict:
    """Fetch active crypto markets from Polymarket Gamma API, sorted by startDate (newest first).
    Used for historical data collection. For live trading, use fetch_active_markets()."""
    global _markets_cache, _cache_ts

    now = time.time()
    if _markets_cache and (now - _cache_ts) < CACHE_TTL:
        print(f"Using cached markets (age: {int(now - _cache_ts)}s)")
        return _markets_cache

    print("Fetching live markets from Polymarket...")
    all_markets = []
    after_cursor = None

    for page in range(max_pages):
        params = {
            "limit": 100,
            "order": "startDate",
            "ascending": "false",
            "active": "true",
            "closed": "false",
        }
        if after_cursor:
            params["after_cursor"] = after_cursor

        response = None
        for attempt in range(3):
            try:
                response = requests.get(
                    f"{GAMMA_BASE_URL}/markets/keyset",
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                break
            except Exception as e:
                print(f"  page {page+1} attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(3)
                else:
                    response = None

        if response is None:
            break

        data = response.json()
        markets = data.get("markets", [])
        if not markets:
            break

        all_markets.extend(markets)
        print(f"  page {page+1}: {len(all_markets)} markets fetched")

        after_cursor = data.get("next_cursor")
        if not after_cursor:
            break

        time.sleep(0.1)

    filtered: dict = {coin: [] for coin in KEYWORDS}
    for market in all_markets:
        text = (
            str(market.get("question", "")) + " " + str(market.get("slug", ""))
        ).lower()
        for coin, keywords in KEYWORDS.items():
            if _matches_coin(text, keywords):
                filtered[coin].append({
                    "id": market.get("id"),
                    "question": market.get("question"),
                    "startDate": market.get("startDate"),
                    "endDate": market.get("endDate"),
                    "lastTradePrice": market.get("lastTradePrice"),
                    "bestBid": market.get("bestBid"),
                    "bestAsk": market.get("bestAsk"),
                    "spread": market.get("spread"),
                    "volume": market.get("volume"),
                    "volumeNum": market.get("volumeNum"),
                    "liquidity": market.get("liquidity"),
                    "outcomes": market.get("outcomes"),
                    "outcomePrices": market.get("outcomePrices"),
                    "clobTokenIds": market.get("clobTokenIds"),
                })

    total = sum(len(v) for v in filtered.values())
    print(f"  done: {total} crypto markets")

    _markets_cache = filtered
    _cache_ts = now
    return filtered


_active_cache: dict = {}
_active_cache_ts: float = 0.0
ACTIVE_CACHE_TTL = 60  # 1 minute — refresh often during live trading


def _extract_market_dict(market: dict) -> dict:
    """Normalise a raw market object into the dict shape used throughout the pipeline."""
    return {
        "id": market.get("id"),
        "question": str(market.get("question", "")),
        "startDate": market.get("startDate"),
        "endDate": market.get("endDate"),
        "lastTradePrice": market.get("lastTradePrice"),
        "bestBid": market.get("bestBid"),
        "bestAsk": market.get("bestAsk"),
        "spread": market.get("spread"),
        "volume": market.get("volume"),
        "volumeNum": market.get("volumeNum"),
        "liquidity": market.get("liquidity"),
        "outcomes": market.get("outcomes"),
        "outcomePrices": market.get("outcomePrices"),
        "clobTokenIds": market.get("clobTokenIds"),
    }


def _fetch_events_by_tag(tag_slug: str, now_ts: float) -> list:
    """
    Fetch Polymarket events by tag and return a flat list of individual
    sub-markets that are currently active (endDate in the future).

    For each event we pick the ONE sub-market whose lastTradePrice is
    closest to 0.5 — that bucket has the most uncertainty and the most
    useful ML signal.
    """
    try:
        resp = requests.get(
            f"{GAMMA_BASE_URL}/events",
            params={"limit": 100, "active": "true", "closed": "false", "tag_slug": tag_slug},
            timeout=30,
        )
        resp.raise_for_status()
        events = resp.json()
        if not isinstance(events, list):
            return []
    except Exception as e:
        print(f"  events fetch ({tag_slug}) failed: {e}")
        return []

    picked = []
    for event in events:
        ed = event.get("endDate", "")
        if not ed:
            continue
        try:
            end_dt = datetime.fromisoformat(ed.replace("Z", "+00:00"))
            if end_dt.timestamp() < now_ts:
                continue
        except Exception:
            continue

        sub_markets = event.get("markets", [])
        if not sub_markets:
            continue

        def _price(m):
            ltp = m.get("lastTradePrice")
            if ltp is not None:
                try:
                    return float(ltp)
                except Exception:
                    pass
            bid = m.get("bestBid")
            if bid is not None:
                try:
                    return float(bid)
                except Exception:
                    pass
            op = m.get("outcomePrices")
            if op:
                try:
                    parsed = json.loads(op) if isinstance(op, str) else op
                    return float(parsed[0])
                except Exception:
                    pass
            return None

        valid_markets = [m for m in sub_markets if _price(m) is not None and 0.001 < (_price(m) or 0) < 0.999]
        if not valid_markets:
            continue

        best = min(valid_markets, key=lambda m: abs((_price(m) or 0.5) - 0.5))
        picked.append(_extract_market_dict(best))

    return picked


def fetch_active_markets(max_pages: int = 15, force_refresh: bool = False) -> dict:
    """Fetch active crypto markets from Polymarket.

    Two sources are combined:
    1. /markets/keyset sorted by startDate desc — captures today's intraday
       (1hour, 4hour, 1day) markets.
    2. /events?tag_slug=weekly — captures weekly grouped-event price-range
       markets that started weeks ago and don't appear in the keyset endpoint.
       Monthly markets are excluded — the monthly model is poorly calibrated.
    """
    global _active_cache, _active_cache_ts

    now = time.time()
    if not force_refresh and _active_cache and (now - _active_cache_ts) < ACTIVE_CACHE_TTL:
        print(f"Using cached active markets (age: {int(now - _active_cache_ts)}s)")
        return _active_cache

    print("Fetching intraday markets from Polymarket (startDate desc)...")
    all_markets: list = []
    after_cursor = None

    for page in range(max_pages):
        params = {
            "limit": 100,
            "order": "startDate",
            "ascending": "false",
            "active": "true",
            "closed": "false",
        }
        if after_cursor:
            params["after_cursor"] = after_cursor

        response = None
        for attempt in range(3):
            try:
                response = requests.get(
                    f"{GAMMA_BASE_URL}/markets/keyset",
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                break
            except Exception as e:
                print(f"  page {page+1} attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(3)
                else:
                    response = None

        if response is None:
            break

        data = response.json()
        page_markets = data.get("markets", [])
        if not page_markets:
            break

        all_markets.extend(page_markets)
        print(f"  page {page+1}: {len(all_markets)} markets fetched")

        after_cursor = data.get("next_cursor")
        if not after_cursor:
            break

        time.sleep(0.1)

    print("Fetching 1h markets (Up or Down + above $X)...")
    onehour_by_coin = fetch_1hour_markets()
    onehour_raw = [m for mlist in onehour_by_coin.values() for m in mlist]
    print(f"  1h markets found: {len(onehour_raw)}")

    print("Fetching 4h markets (Up or Down per coin)...")
    fourhour_by_coin = fetch_4hour_markets()
    fourhour_raw = [m for mlist in fourhour_by_coin.values() for m in mlist]
    print(f"  4h markets found: {len(fourhour_raw)}")

    print("Fetching daily markets (Up or Down + price targets)...")
    daily_by_coin = fetch_daily_markets()
    daily_raw = [m for mlist in daily_by_coin.values() for m in mlist]
    print(f"  daily markets found: {len(daily_raw)}")

    print("Fetching weekly markets (above/price-range/candle)...")
    weekly_by_coin = fetch_weekly_markets()
    weekly_raw = [m for mlist in weekly_by_coin.values() for m in mlist]
    print(f"  weekly markets found: {len(weekly_raw)}")

    # Pattern for 5min/15min markets: "10:55AM-11:00AM"
    def _is_short_window(question: str) -> bool:
        full = re.search(r'(\d+):(\d+)(AM|PM)-(\d+):(\d+)(AM|PM)', question, re.IGNORECASE)
        if not full:
            return False
        h1, m1, p1 = int(full.group(1)), int(full.group(2)), full.group(3).upper()
        h2, m2, p2 = int(full.group(4)), int(full.group(5)), full.group(6).upper()
        if p1 == "PM" and h1 != 12: h1 += 12
        if p1 == "AM" and h1 == 12: h1 = 0
        if p2 == "PM" and h2 != 12: h2 += 12
        if p2 == "AM" and h2 == 12: h2 = 0
        mins = (h2 * 60 + m2) - (h1 * 60 + m1)
        if mins <= 0: mins += 24 * 60
        return mins <= 15  # skip 5min and 15min markets

    filtered: dict = {coin: [] for coin in KEYWORDS}
    seen_ids: set = set()
    skipped_short = 0

    def _add_market(market: dict, from_keyset: bool = False):
        nonlocal skipped_short
        question = str(market.get("question", ""))
        if _is_short_window(question):
            skipped_short += 1
            return
        # Markets from dedicated fetchers (1h, 4h, daily) are pre-filtered and
        # correct — pass them straight through with no extra checks.
        # Only apply noise filters to keyset and weekly raw sources.
        if from_keyset:
            # "above $X on [Date], [H]AM ET?" — handled by fetch_1hour_markets
            if re.search(r'\babove\b.{1,30}on \w+ \d+, \d+[AP]M ET', question, re.IGNORECASE):
                return
            # "Will …  on [Date]?" — daily/weekly price targets, handled by dedicated fetchers
            if re.search(r'\bwill\b.{1,40}\bon \w+ \d+\?', question, re.IGNORECASE):
                return
            # "Up or Down" from keyset: only keep current 1h/4h windows (end ≤ 4.5h)
            # Daily "Up or Down on July 25?" comes from daily_raw, not keyset
            if re.search(r'up or down', question, re.IGNORECASE):
                ed = market.get("endDate", "")
                if ed:
                    try:
                        end_dt = datetime.fromisoformat(ed.replace("Z", "+00:00"))
                        if end_dt.timestamp() > now + 4.5 * 3600:
                            return
                    except Exception:
                        pass
        mid = market.get("id")
        if mid and mid in seen_ids:
            return
        if mid:
            seen_ids.add(mid)
        text = (question + " " + str(market.get("slug", ""))).lower()
        for coin, keywords in KEYWORDS.items():
            if _matches_coin(text, keywords):
                filtered[coin].append(_extract_market_dict(market))
                break

    for market in all_markets:
        _add_market(market, from_keyset=True)

    for market in onehour_raw + fourhour_raw + daily_raw + weekly_raw:
        _add_market(market)

    total = sum(len(v) for v in filtered.values())
    print(f"  done: {total} crypto markets (skipped {skipped_short} ≤15min markets)")

    _active_cache = filtered
    _active_cache_ts = now
    return filtered


CLOB_BASE_URL = "https://clob.polymarket.com"

# Cache for CLOB history so repeated calls don't re-fetch every market
_clob_cache: dict = {}  # token_id -> price_history list
_clob_cache_ts: float = 0.0
CLOB_CACHE_TTL = 120  # 2 minutes — short so we get fresh data


def _parse_trading_window(market: dict):
    """Return (start_ts, end_ts) integers for the market's trading window.

    Supports two question formats:
    - "Bitcoin Up or Down - July 18, 10:55AM-11:00AM ET"  (explicit range)
    - "Bitcoin above 63,000 on July 17, 11AM ET?"          (single end time)

    Falls back to startDate/endDate from the API if no time pattern found.
    """
    import re as _re
    from datetime import timezone as _tz
    ET_OFFSET = timedelta(hours=-4)  # EDT (UTC-4)

    question = market.get("question", "")
    end_date = market.get("endDate", "")
    start_date = market.get("startDate", "")
    if not end_date:
        return None, None
    try:
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except Exception:
        return None, None

    # Pattern 1: explicit range like "10:55AM-11:00AM ET"
    range_match = _re.search(r'(\d+):?(\d*)(AM|PM)-(\d+):?(\d*)(AM|PM)', question, _re.IGNORECASE)
    if range_match:
        h1 = int(range_match.group(1))
        m1 = int(range_match.group(2)) if range_match.group(2) else 0
        p1 = range_match.group(3).upper()
        h2 = int(range_match.group(4))
        m2 = int(range_match.group(5)) if range_match.group(5) else 0
        p2 = range_match.group(6).upper()
        if p1 == "PM" and h1 != 12: h1 += 12
        if p1 == "AM" and h1 == 12: h1 = 0
        if p2 == "PM" and h2 != 12: h2 += 12
        if p2 == "AM" and h2 == 12: h2 = 0
        # Get ET date from endDate (endDate is UTC, convert to ET for local date)
        end_et = end_dt.astimezone(_tz(ET_OFFSET))
        market_date = end_et.date()
        start_dt = datetime(market_date.year, market_date.month, market_date.day,
                            h1, m1, tzinfo=_tz(ET_OFFSET)).astimezone(_tz.utc)
        end_dt2 = datetime(market_date.year, market_date.month, market_date.day,
                           h2, m2, tzinfo=_tz(ET_OFFSET)).astimezone(_tz.utc)
        if end_dt2 <= start_dt:
            end_dt2 += timedelta(days=1)
        return int(start_dt.timestamp()), int(end_dt2.timestamp())

    # Pattern 2: single end time like "11AM ET" — use startDate as window start
    single_match = _re.search(r'(\d+):?(\d*)\s*(AM|PM)\s+ET', question, _re.IGNORECASE)
    if single_match and start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            return int(start_dt.timestamp()), int(end_dt.timestamp())
        except Exception:
            pass

    # Fallback: use startDate/endDate directly
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            return int(start_dt.timestamp()), int(end_dt.timestamp())
        except Exception:
            pass

    return None, None


def _fetch_clob_history(token_id: str, start_ts: int, end_ts: int) -> list:
    """Fetch price history from CLOB API for a given token and time window."""
    global _clob_cache, _clob_cache_ts
    now = time.time()

    # Invalidate whole cache if too old
    if now - _clob_cache_ts > CLOB_CACHE_TTL:
        _clob_cache = {}
        _clob_cache_ts = now

    cache_key = f"{token_id}:{start_ts}:{end_ts}"
    if cache_key in _clob_cache:
        return _clob_cache[cache_key]

    try:
        resp = requests.get(
            f"{CLOB_BASE_URL}/prices-history",
            params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 1},
            timeout=8,
        )
        resp.raise_for_status()
        history = resp.json().get("history", [])
    except Exception:
        history = []

    _clob_cache[cache_key] = history
    return history


def enrich_markets_with_history(markets: dict, top_n: int = 20) -> dict:
    """
    For each coin, select the top_n most promising markets (by volume + having a
    real price signal), try to fetch CLOB history for markets currently within
    their trading window, and include all candidates regardless.

    Markets without real history get price_history=[] and the pipeline falls back
    to a synthetic trajectory — still better than nothing for feature extraction.
    """
    enriched = {coin: [] for coin in markets}
    now_ts = int(time.time())

    for coin, coin_markets in markets.items():
        # Score each market: prefer ones with volume and non-trivial prices
        def score(m):
            vol = float(m.get("volumeNum") or 0)
            ltp = float(m.get("lastTradePrice") or 0)
            bid = float(m.get("bestBid") or 0)
            ask = float(m.get("bestAsk") or 0)
            has_price = 0.01 < ltp < 0.99 or 0.01 < bid < 0.99 or 0.01 < ask < 0.99
            return vol * 10 + (100 if has_price else 0)

        candidates = sorted(coin_markets, key=score, reverse=True)[:top_n]

        RECENTLY_CLOSED_WINDOW = 7200  # 2 hours grace after close

        for m in candidates:
            start_ts, end_ts = _parse_trading_window(m)

            if end_ts is None:
                continue
            if end_ts <= now_ts:
                continue  # already closed
            if end_ts > now_ts + 26 * 3600:
                # endDate > 26h away: either a future intraday session (pre-created
                # by Polymarket) or a weekly market (runs 7 days).
                # Weekly markets are kept — they have real prices and pass the
                # has_real_price check below.
                # Future intraday sessions have no trades yet (price ≈ 0.5, volume 0)
                # and are caught by has_real_price → filtered out automatically.
                pass  # let it fall through to the price-signal check

            # Skip markets with no real price signal (still at default 0.5 or zero)
            ltp = float(m.get("lastTradePrice") or 0)
            bid = float(m.get("bestBid") or 0)
            ask = float(m.get("bestAsk") or 0)
            outcome_yes = None
            try:
                op = m.get("outcomePrices")
                if op:
                    parsed = json.loads(op) if isinstance(op, str) else op
                    if parsed:
                        outcome_yes = float(parsed[0])
            except Exception:
                pass

            def _real(p):
                # Accept prices with real signal. Wider range (0.005–0.995) so
                # weekly/monthly markets (where one range bucket trades near 0.01)
                # are not silently dropped.
                return p is not None and 0.005 < p < 0.995

            # For multi-outcome markets (weekly/monthly), outcomePrices is an array
            # of prices for each bucket. Pick the one closest to 0.5 as the
            # representative price — it has the most uncertainty and best signal.
            best_outcome_price = None
            try:
                op = m.get("outcomePrices")
                if op:
                    parsed = json.loads(op) if isinstance(op, str) else op
                    if parsed and len(parsed) > 1:
                        floats = [float(x) for x in parsed]
                        # Pick the one closest to 0.5 (most uncertain = best for ML)
                        best_outcome_price = min(floats, key=lambda x: abs(x - 0.5))
            except Exception:
                pass

            has_real_price = (
                _real(ltp) or _real(bid) or _real(ask)
                or _real(outcome_yes) or _real(best_outcome_price)
            )
            if not has_real_price:
                continue

            # Fetch CLOB history for markets still within (or just past) their window
            history = []
            if now_ts <= end_ts + RECENTLY_CLOSED_WINDOW:
                clob_ids = m.get("clobTokenIds")
                if clob_ids:
                    try:
                        ids = json.loads(clob_ids) if isinstance(clob_ids, str) else clob_ids
                        token_id = ids[0]
                        # For "Up or Down" binary markets that just opened, the CLOB
                        # may have very few trades. Widen the fetch window to 30min
                        # before market start so we capture pre-market price discovery
                        # and any early trades — gives the model a real trajectory.
                        is_up_or_down = bool(re.search(r'up or down', m.get("question", ""), re.IGNORECASE))
                        fetch_start = start_ts - 1800 if is_up_or_down else start_ts
                        history = _fetch_clob_history(token_id, fetch_start, end_ts)

                        # If still empty for an Up/Down market, try fetching the last
                        # 2 hours from now — catches markets mid-session with activity
                        if not history and is_up_or_down:
                            history = _fetch_clob_history(token_id, now_ts - 7200, now_ts)
                    except Exception:
                        history = []

            enriched[coin].append({**m, "price_history": history})

        time.sleep(0.02)  # be gentle with the CLOB API

    with_hist = sum(1 for ms in enriched.values() for m in ms if len(m.get("price_history", [])) >= 3)
    total = sum(len(ms) for ms in enriched.values())
    print(f"  enriched: {total} live markets with real prices, {with_hist} with CLOB history")
    return enriched


def run_pipeline_live(risk_level: int = 5):
    """Run the full pipeline using live Polymarket data with real CLOB history.

    Each eligible (coin, market) pair is treated as a separate optimizer asset,
    keyed as "COIN|TYPE" (e.g. "BTC|1hour").  Multiple high-edge markets for the
    same coin are all passed through — the Markowitz penalty handles correlation
    between them (same-coin markets get correlation=0.95; cross-coin pairs use
    the CoinGecko price correlation).
    """
    markets = fetch_active_markets(force_refresh=True)  # always fresh on optimize
    print("Enriching with CLOB price histories (active/recently-closed markets only)...")
    markets = enrich_markets_with_history(markets)

    total_live = sum(len(ms) for ms in markets.values())
    if total_live == 0:
        print("No live markets with real prices found — trading windows may be between sessions.")
        return None

    predictions = predict_probabilities(markets)

    # Collect ALL eligible markets across all coins as (asset_key, prediction, coin) tuples.
    # asset_key = "COIN|TYPE" so same coin + different timeframe → two distinct assets.
    all_eligible: list[tuple[str, dict, str]] = []
    for coin, coin_preds in predictions.items():
        eligible = [
            p for p in coin_preds
            if p["edge"] > 0.02
            and p["market_type"] != "monthly"
            and 0.20 <= p["market_price"] <= 0.80   # exclude extreme OTM / near-resolved
            and p["edge"] <= 0.15                    # edges > 15% = model miscalibration
        ]
        print(f"  {coin}: {len(eligible)} eligible markets (from {len(coin_preds)} predictions)")
        for p in eligible:
            in_ideal = "IDEAL" if 0.35 <= p["market_price"] <= 0.65 else "wide"
            print(f"    [{in_ideal}] price={p['market_price']:.2f} edge={p['edge']:+.3f} type={p['market_type']} | {p['question'][:60]}")
            key = f"{coin}|{p['market_type']}"
            all_eligible.append((key, p, coin))

    if len(all_eligible) < 2:
        print(f"Only {len(all_eligible)} eligible market(s) found — need at least 2.")
        return None

    # Deduplicate: if two predictions share the same key, keep the one with higher edge
    seen: dict[str, tuple[dict, str]] = {}
    for key, pred, coin in all_eligible:
        if key not in seen or pred["edge"] > seen[key][0]["edge"]:
            seen[key] = (pred, coin)

    asset_keys = list(seen.keys())
    asset_preds = {k: seen[k][0] for k in asset_keys}
    asset_coins = {k: seen[k][1] for k in asset_keys}

    estimates = {k: asset_preds[k]["our_estimate"] for k in asset_keys}
    market_prices_map = {k: asset_preds[k]["market_price"] for k in asset_keys}

    # Build correlation matrix for (coin, market) assets:
    #   • Same coin, different timeframe → 0.95 (highly correlated outcomes)
    #   • Different coins               → use CoinGecko price correlation
    full_corr, corr_coins = get_correlation_matrix()
    n = len(asset_keys)
    correlation_matrix = np.eye(n)
    for i, ki in enumerate(asset_keys):
        for j, kj in enumerate(asset_keys):
            if i == j:
                continue
            ci, cj = asset_coins[ki], asset_coins[kj]
            if ci == cj:
                correlation_matrix[i, j] = 0.95
            elif ci in corr_coins and cj in corr_coins:
                ii, jj = corr_coins.index(ci), corr_coins.index(cj)
                correlation_matrix[i, j] = full_corr[ii, jj]
            else:
                correlation_matrix[i, j] = 0.5  # conservative default

    # Force exact symmetry — numerical noise in full_corr can make the matrix
    # slightly asymmetric, which would cause SLSQP to produce incorrect results.
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2.0

    allocations = optimize_portfolio(estimates, market_prices_map, correlation_matrix, risk_level)

    # best_markets keyed by asset_key for the endpoint to use
    best_markets = {k: asset_preds[k] for k in asset_keys}
    # Attach the parent coin to each prediction entry
    for k in asset_keys:
        best_markets[k]["_coin"] = asset_coins[k]

    return allocations, best_markets, markets


# ── Request bodies ────────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    risk_level: int = 5
    amount: float = 1000.0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    if not 1 <= req.risk_level <= 10:
        raise HTTPException(status_code=400, detail="risk_level must be between 1 and 10")

    result = run_pipeline_live(risk_level=req.risk_level)
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="No active markets with real prices right now. Polymarket trading windows run ~10:00AM-4:00PM ET on weekdays. Try again when a window is active.",
        )

    allocations, best_markets, enriched_markets = result
    # asset_keys are "COIN|TYPE" strings (e.g. "BTC|1hour", "BTC|4hour")
    asset_keys = list(allocations.keys())

    # Weights from the optimizer may sum to < 1.0 (cash buffer from Half-Kelly /
    # inequality constraint).  Do NOT renormalise — the undeployed fraction is
    # intentional: low-edge or highly correlated markets leave more in cash.
    raw_weights = np.array([allocations[k]["weight"] for k in asset_keys], dtype=float)
    raw_weights = np.clip(raw_weights, 0.0, None)
    total_w = raw_weights.sum()

    # Safety: if the optimizer somehow returns all-zero weights, fall back to
    # equal allocation (shouldn't happen but guards against solver failures).
    if total_w < 1e-6:
        raw_weights = np.ones(len(asset_keys)) / len(asset_keys)
        total_w = 1.0

    # Cap total deployed at 100% (rounding noise can push it fractionally over).
    if total_w > 1.0:
        raw_weights = raw_weights / total_w
        total_w = 1.0

    # Dollar amounts: each asset gets weight * req.amount.
    # Cash remainder = req.amount * (1 - total_w) — reported in the report.
    dollar_amounts = [round(float(w) * req.amount, 2) for w in raw_weights]

    # Absorb rounding cents into the largest position only if total deployed
    # would otherwise exceed req.amount (never add to cash-remainder coins).
    allocated = sum(dollar_amounts)
    rounding_gap = round(req.amount * total_w - allocated, 2)
    if rounding_gap != 0 and any(d > 0 for d in dollar_amounts):
        largest_idx = int(np.argmax(raw_weights))
        dollar_amounts[largest_idx] = round(dollar_amounts[largest_idx] + rounding_gap, 2)

    def _confidence(edge: float) -> float:
        # Scale edge to confidence, but cap at 0.75 — the models have known
        # calibration issues (training on historical resolved markets) so we
        # never claim more than 75% confidence even on large edges.
        # Formula: softly saturates — 10% edge ≈ 33%, 20% edge ≈ 50%, 40% ≈ 67%
        return round(min(abs(edge) / (abs(edge) + 0.2), 0.75), 3)

    # allocation_data is keyed by asset_key ("COIN|TYPE") for internal use;
    # the frontend receives this and uses _coin for display grouping.
    allocation_data = {
        key: {
            "weight": round(float(raw_weights[i]), 4),
            "dollar_amount": dollar_amounts[i],
            "edge": allocations[key]["edge"],
            "our_estimate": allocations[key]["our_estimate"],
            "market_price": allocations[key]["market_price"],
            "confidence": _confidence(allocations[key]["edge"]),
            "market_type": best_markets[key].get("market_type", "all"),
            "coin": best_markets[key].get("_coin", key.split("|")[0]),
            "market": {k: v for k, v in best_markets[key].items() if k != "_coin"},
        }
        for i, key in enumerate(asset_keys)
    }

    # coins list for response — unique coins that have at least one allocation
    coins = sorted({d["coin"] for d in allocation_data.values()})

    # ── Live baseline comparison ───────────────────────────────────────────────
    # Run Logistic Regression on the same feature vectors used by our ensemble,
    # so the frontend can display real per-run numbers (not hardcoded).
    model_comparison = None
    if BASELINE_MODEL is not None:
        try:
            coin_idx_map = {c: i for i, c in enumerate(list(MODELS.keys())[:7])}
            our_estimates, lr_estimates, market_prices_list, coins_used = [], [], [], []

            # Build a fast lookup: market_id → enriched market dict (has full price fields)
            enriched_by_id: dict = {}
            for coin_markets in enriched_markets.values():
                for m in coin_markets:
                    if m.get("id"):
                        enriched_by_id[m["id"]] = m

            for key, d in allocation_data.items():
                best_pred = best_markets[key]  # prediction dict (has "id" and "question")
                # Use the enriched market (full price fields) for feature extraction
                market = enriched_by_id.get(best_pred.get("id"), best_pred)
                _, market_type = get_model_for_market(market.get("question", ""), market)
                asset_coin = d["coin"]
                coin_idx = coin_idx_map.get(asset_coin, 0)
                features_df, _ = get_price_features_with_type(market, coin_idx, market_type)
                if features_df is None:
                    continue

                # Align columns to what the baseline was trained on.
                # Handle both plain estimators (feature_names_in_) and Pipelines.
                try:
                    bl_cols = BASELINE_MODEL.feature_names_in_
                    features_aligned = features_df.reindex(columns=bl_cols, fill_value=0)
                except AttributeError:
                    # Pipeline or estimator without feature_names_in_ — use raw df
                    features_aligned = features_df

                try:
                    lr_prob = float(BASELINE_MODEL.predict_proba(features_aligned)[0][1])
                except Exception as inner_e:
                    print(f"  baseline predict failed for {key}: {inner_e}")
                    continue

                our_estimates.append(d["our_estimate"])
                lr_estimates.append(lr_prob)
                market_prices_list.append(d["market_price"])
                coins_used.append(d["coin"])

            print(f"Live comparison: {len(our_estimates)} assets — our={our_estimates} lr={lr_estimates}")

            if our_estimates:
                our_arr = np.array(our_estimates)
                lr_arr  = np.array(lr_estimates)
                mp_arr  = np.array(market_prices_list)

                our_edges = our_arr - mp_arr  # always positive (we filtered for edge > 0.02)
                lr_edges  = lr_arr  - mp_arr  # may be negative (LR disagrees with our direction)

                # Use signed mean edge — our model's edges are always positive here because
                # we only include markets where we found positive edge. The LR's signed edge
                # on the same markets shows how much it agrees. Positive advantage means
                # our model sees more alpha on these markets than LR does.
                our_mean_edge = float(np.mean(our_edges))
                lr_mean_edge  = float(np.mean(lr_edges))
                edge_advantage = round((our_mean_edge - lr_mean_edge) * 100, 2)

                # Agreement: correlation between the two models' predictions
                if len(our_arr) > 1:
                    corr = float(np.corrcoef(our_arr, lr_arr)[0, 1])
                else:
                    corr = 1.0

                model_comparison = {
                    "our_mean_edge_pct":  round(our_mean_edge * 100, 2),
                    "lr_mean_edge_pct":   round(lr_mean_edge * 100, 2),
                    "edge_advantage_pct": edge_advantage,   # positive = our model sees bigger edge than LR on the same markets
                    "prediction_correlation": round(corr, 3),
                    "n_markets": len(our_estimates),
                    "per_coin": [
                        {
                            "coin": coins_used[i],
                            "our_estimate": round(our_estimates[i], 4),
                            "lr_estimate":  round(lr_estimates[i], 4),
                            "market_price": round(market_prices_list[i], 4),
                            "our_edge": round(our_estimates[i] - market_prices_list[i], 4),
                            "lr_edge":  round(lr_estimates[i]  - market_prices_list[i], 4),
                        }
                        for i in range(len(our_estimates))
                    ],
                }
                print(f"  model_comparison built: our={model_comparison['our_mean_edge_pct']}% lr={model_comparison['lr_mean_edge_pct']}%")
        except Exception as e:
            import traceback
            print(f"Live comparison failed: {e}")
            traceback.print_exc()

    # Build reasoning report — only include assets with actual dollar allocation
    sorted_by_weight = sorted(allocation_data.items(), key=lambda x: x[1]["weight"], reverse=True)
    active_positions = [(key, d) for key, d in sorted_by_weight if d["dollar_amount"] > 0]
    skipped_positions = [(key, d) for key, d in sorted_by_weight if d["dollar_amount"] == 0]
    top_key, top_data = active_positions[0] if active_positions else sorted_by_weight[0]

    risk_label = {1:"Very Conservative",2:"Conservative",3:"Moderate",4:"Moderate+",
                  5:"Balanced",6:"Moderate Aggressive",7:"Aggressive",8:"Very Aggressive",
                  9:"High Risk",10:"Max Risk"}.get(req.risk_level, "Balanced")

    # Kelly fraction scales with risk level; Markowitz shapes the split across correlated assets.
    kf = kelly_fraction_for_risk(req.risk_level)
    kelly_label = {0.25: "Quarter-Kelly", 0.50: "Half-Kelly", 0.75: "¾-Kelly", 1.00: "Full-Kelly"}.get(kf, f"{kf}×-Kelly")
    total_deployed_pct = sum(d["weight"] for d in allocation_data.values()) * 100
    cash_pct = max(0.0, 100.0 - total_deployed_pct)
    cash_dollars = round(req.amount * cash_pct / 100, 2)

    coin_lines = []
    total_expected_gain = 0.0
    for key, d in active_positions:
        raw_kelly = d["edge"] / (1 - d["market_price"]) if d["market_price"] < 1 else 0
        scaled_kelly = raw_kelly * kf
        direction = "YES" if d["our_estimate"] > d["market_price"] else "NO"
        horizon_label = {"1hour": "1-hour", "4hour": "4-hour", "1day": "daily", "weekly": "weekly"}.get(d.get("market_type", ""), "")
        expected_gain = d["dollar_amount"] * d["edge"] / d["market_price"] if d["market_price"] > 0 else 0
        total_expected_gain += expected_gain
        display_label = f"{d['coin']} ({horizon_label})" if horizon_label else d["coin"]
        coin_lines.append(
            f"• {display_label} — ${d['dollar_amount']:.2f} ({d['weight']*100:.1f}%) → est. gain ${expected_gain:+.2f}"
        )
        coin_lines.append(
            f"  Market price: {d['market_price']*100:.1f}% | Model estimate: {d['our_estimate']*100:.1f}% | Edge: {d['edge']*100:+.1f}%"
        )
        coin_lines.append(
            f"  Full Kelly: {raw_kelly*100:.1f}% → {kelly_label}: {scaled_kelly*100:.1f}% → final: {d['weight']*100:.1f}% | {direction} | Confidence: {d['confidence']*100:.0f}%"
        )
        coin_lines.append("")

    skipped_line = ""
    if skipped_positions:
        skipped_reasons = []
        for k, d in skipped_positions:
            label = f"{d['coin']} ({d.get('market_type', '')})"
            if d["edge"] <= 0.02:
                skipped_reasons.append(f"{label} (edge {d['edge']*100:+.1f}% below 2% threshold)")
            else:
                skipped_reasons.append(f"{label} (weight driven to 0 by correlation penalty)")
        skipped_line = f"Excluded: {'; '.join(skipped_reasons)}."

    expected_roi = (total_expected_gain / req.amount * 100) if req.amount > 0 else 0

    cash_line = (f"Cash reserve: ${cash_dollars:.2f} ({cash_pct:.1f}%) — {kelly_label} risk buffer."
                 if cash_pct > 0.5 else "")

    top_display = f"{top_data['coin']} ({top_data.get('market_type', '')})"
    report_lines = [
        f"KELLY-MARKOWITZ PORTFOLIO — Risk level {req.risk_level}/10 ({risk_label}) · {kelly_label} ({kf:.2f}×)",
        "",
        "━━━ This Run ━━━",
        f"Markets with positive edge found: {len(active_positions)} positions across {len(coins)} coin(s).",
        *(([skipped_line, ""]) if skipped_line else []),
        *(([cash_line, ""]) if cash_line else []),
        "Position breakdown:",
        "",
        *coin_lines,
        f"EXPECTED GAIN  ${total_expected_gain:+.2f}  ({expected_roi:+.1f}% on ${req.amount:.0f} deployed)",
        "",
        "━━━ Key Driver ━━━",
        f"{top_display} received the largest position ({top_data['weight']*100:.1f}%) because it has the highest",
        f"{kelly_label}-weighted edge ({top_data['edge']*100:+.1f}%) relative to its correlation with the other selected assets.",
        ("At this risk level, the optimizer concentrates into the top edge regardless of correlation."
         if req.risk_level >= 7
         else "At this risk level, the optimizer diversifies: high correlation between assets is penalised, so even a slightly lower-edge market may receive significant weight if it moves independently."),
    ]

    return {
        "coins": coins,
        "risk_level": req.risk_level,
        "amount": req.amount,
        "allocations": allocation_data,
        "report": "\n".join(report_lines),
        "model_comparison": model_comparison,
    }


@app.get("/api/markets")
def get_markets():
    markets = fetch_active_markets()
    markets = enrich_markets_with_history(markets)
    predictions = predict_probabilities(markets)

    result = []
    for coin, preds in predictions.items():
        for p in preds:
            # Skip short-window markets from the display table
            if p["market_type"] in ("5min", "15min"):
                continue
            # Skip markets with no real model signal — these have no CLOB history so
            # our_estimate falls back to market_price, producing edge ≈ 0 and volume = 0.
            # They are pre-created future sessions that slipped past the time filter.
            if p["volume"] == 0 and abs(p["edge"]) < 0.01:
                continue
            result.append({
                "coin": coin,
                "id": p["id"],
                "question": p["question"],
                "endDate": p.get("endDate"),
                "market_type": p["market_type"],
                "market_price": p["market_price"],
                "our_estimate": p["our_estimate"],
                "edge": p["edge"],
                "volume": p["volume"],
                "confidence": round(min(abs(p["edge"]) / (abs(p["edge"]) + 0.2), 0.75), 3),
            })

    result.sort(key=lambda x: x["edge"], reverse=True)
    return {"markets": result, "total": len(result)}


@app.get("/api/dashboard")
def get_dashboard():
    markets = fetch_active_markets()
    markets = enrich_markets_with_history(markets)
    predictions = predict_probabilities(markets)

    all_preds = [p for preds in predictions.values() for p in preds]
    opportunities = [p for p in all_preds if p["edge"] > 0.02]

    avg_edge = (
        round(sum(p["edge"] for p in opportunities) / len(opportunities), 4)
        if opportunities
        else 0.0
    )

    top = sorted(opportunities, key=lambda x: x["edge"], reverse=True)[:10]

    coin_lookup: dict = {}
    for coin, preds in predictions.items():
        for p in preds:
            coin_lookup[p["id"]] = coin

    top_with_coin = [
        {**p, "coin": coin_lookup.get(p["id"], "?")}
        for p in top
    ]

    coin_stats = {}
    for coin, preds in predictions.items():
        if not preds:
            continue
        coin_opps = [p for p in preds if p["edge"] > 0.02]
        best_edge = max((p["edge"] for p in coin_opps), default=0.0)
        coin_stats[coin] = {
            "opportunities": len(coin_opps),
            "best_edge": round(best_edge, 4),
            "total_markets": len(preds),
        }

    # Load real model accuracy from the latest backtest results if available
    model_accuracy = 0.464  # fallback: real post-fix win rate
    try:
        if os.path.exists(BACKTEST_RESULTS_PATH):
            with open(BACKTEST_RESULTS_PATH) as _f:
                _bt = json.load(_f)
            model_accuracy = round(
                _bt.get("summary", {}).get("our_model", {}).get("direction_accuracy_pct", 46.4) / 100, 4
            )
    except Exception:
        pass

    return {
        "active_markets": len(all_preds),
        "opportunities": len(opportunities),
        "avg_edge": avg_edge,
        "model_accuracy": model_accuracy,
        "coins": list(predictions.keys()),
        "coin_stats": coin_stats,
        "top_opportunities": top_with_coin,
        "cache_age_seconds": int(time.time() - _cache_ts),
    }


@app.get("/api/refresh")
def refresh_markets():
    """Force a fresh fetch from Polymarket (bypasses the 5-min cache)."""
    global _cache_ts
    _cache_ts = 0.0
    markets = fetch_live_markets()
    total = sum(len(v) for v in markets.values())
    return {"status": "refreshed", "total_markets": total}


BACKTEST_RESULTS_PATH = "data/backtest_results.json"


@app.get("/api/history")
def get_history():
    """
    Returns real backtest results from the last run of the backtester.
    Run the backtester first:  python -m src.backtest.backtest
    Falls back to an empty result if no backtest file exists yet.
    """
    if not os.path.exists(BACKTEST_RESULTS_PATH):
        raise HTTPException(
            status_code=404,
            detail="No backtest results found. Run: python -m src.backtest.backtest",
        )

    with open(BACKTEST_RESULTS_PATH) as f:
        data = json.load(f)

    return data


OOS_RESULTS_PATH = "data/oos_results.json"


@app.get("/api/history/oos")
def get_oos_history():
    """
    Returns out-of-sample evaluation results.
    Run the OOS evaluator first:  python -m src.backtest.oos_eval
    """
    if not os.path.exists(OOS_RESULTS_PATH):
        raise HTTPException(
            status_code=404,
            detail="No OOS results found. Run: python -m src.backtest.oos_eval",
        )

    with open(OOS_RESULTS_PATH) as f:
        data = json.load(f)

    return data


@app.post("/api/backtest/run")
def trigger_backtest():
    """
    Runs the full backtester in-process and saves results to data/backtest_results.json.
    This can take several minutes (fetches closed markets + CLOB history).
    """
    from src.backtest.backtest import run_backtest
    try:
        result = run_backtest(save_path=BACKTEST_RESULTS_PATH)
        return {
            "status": "ok",
            "n_trades": result["meta"]["n_valid_trades"],
            "summary": result["summary"],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
