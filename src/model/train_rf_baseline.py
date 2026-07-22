"""
Train and save Random Forest baseline model for backtest comparison.

Run from the repo root:
    python src/model/train_rf_baseline.py
"""

import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

DATASETS = {
    "5min":    "data/datasets/dataset_5min.csv",
    "15min":   "data/datasets/dataset_15min.csv",
    "1hour":   "data/datasets/dataset_1hour.csv",
    "4hour":   "data/datasets/dataset_4hour.csv",
    "1day":    "data/datasets/dataset_1day.csv",
    "weekly":  "data/datasets/dataset_weekly.csv",
    "monthly": "data/datasets/dataset_monthly.csv",
}

TYPE_MAP = {"5min": 0, "15min": 1, "1hour": 2, "4hour": 3, "1day": 4, "weekly": 5, "monthly": 6}

SAVE_PATH = "src/model/baseline_rf.pkl"


def load_combined_dataset():
    dfs = []
    for name, path in DATASETS.items():
        try:
            df = pd.read_csv(path)
            df["market_type"] = TYPE_MAP[name]
            dfs.append(df)
        except FileNotFoundError:
            print(f"  skipping {name} — file not found")

    common = set(dfs[0].columns)
    for df in dfs[1:]:
        common &= set(df.columns)

    combined = pd.concat([df[list(common)] for df in dfs], ignore_index=True)
    return combined


def main():
    print("Loading combined dataset...")
    df = load_combined_dataset()
    print(f"  Total samples: {len(df):,}  |  Features: {df.shape[1] - 1}")

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest baseline...")
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"  Accuracy: {acc:.3f}  |  AUC-ROC: {auc:.3f}")

    joblib.dump(rf, SAVE_PATH)
    print(f"  Saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
