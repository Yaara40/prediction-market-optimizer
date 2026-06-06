import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib

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

def train_models(df: pd.DataFrame, name: str):
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "XGBoost":      XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        "LightGBM":     LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        "CatBoost":     CatBoostClassifier(n_estimators=100, random_state=42, verbose=0),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)
        results[model_name] = {"model": model, "accuracy": acc, "brier": brier}
        print(f"  {model_name}: accuracy={acc:.3f} | brier={brier:.3f}")

    best_name = min(results, key=lambda x: results[x]["brier"])
    print(f"  best: {best_name}")
    return results[best_name]["model"], best_name

def build_combined_dataset():
    dfs = []
    for name, path in DATASETS.items():
        try:
            df = pd.read_csv(path)
            df["market_type"] = TYPE_MAP[name]
            dfs.append(df)
        except FileNotFoundError:
            print(f"  skipping {name} - file not found")
            continue

    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)

    print(f"  common features: {len(common_cols)}")
    combined = pd.concat([df[list(common_cols)] for df in dfs], ignore_index=True)
    return combined

if __name__ == "__main__":
    summary = []

    # phase 1: train specialized models
    print("phase 1: training specialized models...")
    for dataset_name, path in DATASETS.items():
        print(f"\ntraining {dataset_name}...")
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"  file not found: {path} — skipping")
            continue

        print(f"  samples: {len(df)} | features: {len(df.columns)-1}")
        best_model, best_name = train_models(df, dataset_name)

        joblib.dump(best_model, f"src/model/best_model_{dataset_name}.pkl")
        print(f"  saved → src/model/best_model_{dataset_name}.pkl")

        summary.append({
            "dataset": dataset_name,
            "samples": len(df),
            "best_model": best_name,
        })

    # phase 2: train combined model
    print("\nphase 2: training combined model...")
    combined_df = build_combined_dataset()
    print(f"  total samples: {len(combined_df)} | features: {len(combined_df.columns)-1}")
    best_model, best_name = train_models(combined_df, "combined")
    joblib.dump(best_model, "src/model/best_model_all.pkl")
    print(f"  saved → src/model/best_model_all.pkl")

    print("\nsummary:")
    for s in summary:
        print(f"  {s['dataset']}: {s['samples']} samples | best: {s['best_model']}")
    print(f"  combined: {len(combined_df)} samples | best: {best_name}")