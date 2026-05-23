import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
from tabpfn import TabPFNClassifier

def load_data():
    with open("data/resolved_markets_12h_7days.json") as f:
        raw = json.load(f)

    rows = []
    for coin_idx, (coin, markets) in enumerate(raw.items()):
        for m in markets:
            best_bid = m.get("bestBid")
            volume = m.get("volumeNum")
            spread = m.get("spread")
            start = m.get("startDate")
            end = m.get("endDate")
            outcome_prices = m.get("outcomePrices")

            if best_bid is None or outcome_prices is None:
                continue

            try:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                duration_hours = (end_dt - start_dt).total_seconds() / 3600
            except:
                continue

            try:
                prices = json.loads(outcome_prices)
                label = 1 if prices[0] == "1" else 0
            except:
                continue

            rows.append({
                "best_bid": best_bid,
                "volume": volume if volume else 0,
                "spread": spread if spread else 0,
                "duration_hours": duration_hours,
                "coin": coin_idx,
                "label": label
            })

    return pd.DataFrame(rows)

def train_models(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(n_estimators=100, random_state=42, verbose=0),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)
        results[name] = {"model": model, "accuracy": acc, "brier": brier}
        print(f"{name}: accuracy={acc:.3f} | brier={brier:.3f}")

    # TabPFN separately with limited data
    print("TabPFN: training on 500 samples (designed for small datasets)...")
    from tabpfn import TabPFNClassifier
    tabpfn = TabPFNClassifier(device="cpu", n_estimators=4)
    X_train_small = X_train[:500]
    y_train_small = y_train[:500]
    tabpfn.fit(X_train_small, y_train_small)
    y_pred = tabpfn.predict(X_test)
    y_prob = tabpfn.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    results["TabPFN"] = {"model": tabpfn, "accuracy": acc, "brier": brier}
    print(f"TabPFN: accuracy={acc:.3f} | brier={brier:.3f}")

    best_name = min(results, key=lambda x: results[x]["brier"])
    print(f"\nbest model: {best_name}")
    return results[best_name]["model"], best_name
if __name__ == "__main__":
    print("loading data...")
    df = load_data()
    print(f"total samples: {len(df)}")
    print(f"label distribution: {df['label'].value_counts().to_dict()}")

    print("\ntraining models...")
    best_model, best_name = train_models(df)

    joblib.dump(best_model, f"src/model/best_model.pkl")
    print(f"\nsaved best model ({best_name}) → src/model/best_model.pkl")