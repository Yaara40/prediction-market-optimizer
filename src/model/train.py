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

if __name__ == "__main__":
    summary = []

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

    print("\nsummary:")
    for s in summary:
        print(f"  {s['dataset']}: {s['samples']} samples | best: {s['best_model']}")