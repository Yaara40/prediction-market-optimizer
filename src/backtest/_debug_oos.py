"""Debug script: find a June 1-hour market and check CLOB history."""
import json, requests, re, time
from datetime import datetime, timezone, timedelta

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"

after_cursor = None
found = None
for page in range(200):
    params = {"limit": 100, "order": "startDate", "ascending": "false", "closed": "true"}
    if after_cursor:
        params["after_cursor"] = after_cursor
    resp = requests.get(f"{GAMMA}/markets/keyset", params=params, timeout=30)
    data = resp.json()
    markets = data.get("markets", [])
    if not markets:
        break
    for m in markets:
        q  = m.get("question", "")
        ed = (m.get("endDate") or "")[:10]
        if re.search(r"up or down - .+, \d+[AP]M ET", q, re.IGNORECASE) and ed >= "2026-06-01":
            found = m
            break
    if found:
        break
    after_cursor = data.get("next_cursor")
    if not after_cursor:
        break
    time.sleep(0.05)

if not found:
    print("No June 1-hour market found")
    exit()

print("Question:", found.get("question"))
print("endDate:", found.get("endDate"))
print("outcomePrices:", found.get("outcomePrices"))
print("lastTradePrice:", found.get("lastTradePrice"))

ids = found.get("clobTokenIds", "[]")
token_ids = json.loads(ids) if isinstance(ids, str) else ids
token_id = token_ids[0] if token_ids else None

q = found.get("question", "")
end_date = found.get("endDate", "")
end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
match = re.search(r"(\d+)(AM|PM) ET", q, re.IGNORECASE)
if match:
    h, period = int(match.group(1)), match.group(2).upper()
    if period == "PM" and h != 12:
        h += 12
    if period == "AM" and h == 12:
        h = 0
    market_date = end_dt.date()
    end_t   = datetime(market_date.year, market_date.month, market_date.day, h, 0, tzinfo=timezone.utc) + timedelta(hours=4)
    start_t = end_t - timedelta(hours=1)
    print("window:", start_t, "->", end_t)

    hist_resp = requests.get(f"{CLOB}/prices-history",
        params={"market": token_id, "startTs": int(start_t.timestamp()), "endTs": int(end_t.timestamp()), "fidelity": 1},
        timeout=15)
    print("CLOB status:", hist_resp.status_code)
    hist_data = hist_resp.json()
    history = hist_data.get("history", [])
    print("history length:", len(history))
    if history:
        print("first:", history[0])
        print("last:", history[-1])
    else:
        print("full response:", str(hist_data)[:400])
