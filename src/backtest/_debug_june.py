"""Check what June 1-hour markets look like - specifically outcomePrices."""
import json, requests, re, time

GAMMA = "https://gamma-api.polymarket.com"
after_cursor = None
samples = []

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
        op = m.get("outcomePrices")
        if re.search(r"up or down - .+, \d+[AP]M ET", q, re.IGNORECASE) and "2026-06" <= ed <= "2026-07-01":
            samples.append({"q": q[:70], "endDate": ed, "outcomePrices": op, "lastTradePrice": m.get("lastTradePrice")})
            if len(samples) >= 10:
                break
    if len(samples) >= 10:
        break
    after_cursor = data.get("next_cursor")
    if not after_cursor:
        break
    time.sleep(0.05)

for s in samples:
    print(s)
