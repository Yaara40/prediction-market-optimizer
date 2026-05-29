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

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def train_models(df: pd.DataFrame, dataset_name: str):
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

    print(f"\n{dataset_name} — {len(df)} samples, {len(X.columns)} features")
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)
        results[name] = {"model": model, "accuracy": acc, "brier": brier}
        print(f"  {name}: accuracy={acc:.3f} | brier={brier:.3f}")

    best_name = min(results, key=lambda x: results[x]["brier"])
    print(f"  best: {best_name}")
    return results[best_name]["model"], best_name

if __name__ == "__main__":
    datasets = {
        "5min": "data/raw/dataset_5min.csv",
        "15min": "data/raw/dataset_15min.csv",
    }

    for name, path in datasets.items():
        print(f"\nloading {name} dataset...")
        df = load_data(path)
        best_model, best_name = train_models(df, name)
        joblib.dump(best_model, f"src/model/best_model_{name}.pkl")
        print(f"  saved → src/model/best_model_{name}.pkl")