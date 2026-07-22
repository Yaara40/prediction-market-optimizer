"""
Isotonic calibration for prediction market models.

The GBM/RF models output raw probability scores that are NOT well-calibrated:
a model output of 0.6 does not mean 60% of outcomes resolve YES.
Isotonic regression fixes this by learning a monotone mapping from raw scores
to empirical YES frequencies on a held-out calibration set.

Usage:
    cd /path/to/prediction-market-optimizer
    python src/model/calibrate.py

This overwrites src/model/calibrator_<type>.pkl for each timeframe.
"""

import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

DATASETS = {
    "5min":   "data/datasets/dataset_5min.csv",
    "15min":  "data/datasets/dataset_15min.csv",
    "1hour":  "data/datasets/dataset_1hour.csv",
    "4hour":  "data/datasets/dataset_4hour.csv",
    "1day":   "data/datasets/dataset_1day.csv",
    "weekly": "data/datasets/dataset_weekly.csv",
    "all":    None,  # combined dataset built on the fly
}

TYPE_MAP = {"5min": 0, "15min": 1, "1hour": 2, "4hour": 3, "1day": 4, "weekly": 5, "all": 7}


def load_combined_dataset():
    """Build the combined dataset the same way train.py does."""
    dfs = []
    for name, path in DATASETS.items():
        if path is None:
            continue
        try:
            df = pd.read_csv(path)
            df["market_type"] = TYPE_MAP[name]
            dfs.append(df)
        except FileNotFoundError:
            continue
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)
    return pd.concat([df[list(common_cols)] for df in dfs], ignore_index=True)


def calibrate_model(model_type: str, df: pd.DataFrame):
    """Fit and save an isotonic calibrator for one market-type model."""
    model_path = f"src/model/best_model_{model_type}.pkl"
    calib_path = f"src/model/calibrator_{model_type}.pkl"

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"  [{model_type}] model not found at {model_path} — skipping")
        return

    X = df.drop("label", axis=1)
    y = df["label"].values

    # 80/20 split: train the calibrator on 20% hold-out
    # (same random_state=42 as train.py — calibration set ≈ the original test set)
    _, X_cal, _, y_cal = train_test_split(X, y, test_size=0.2, random_state=42)

    # Align feature columns to what the model expects
    try:
        model_cols = model.feature_names_in_
        X_cal = X_cal.reindex(columns=model_cols, fill_value=0)
    except AttributeError:
        pass

    # Raw probability from the model on the calibration set
    raw_probs = model.predict_proba(X_cal)[:, 1]

    # Fit isotonic regression: maps raw_prob → empirical YES frequency
    # out_of_bounds='clip' clips any future predictions outside [0,1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_cal)

    # Diagnostics
    cal_probs = calibrator.predict(raw_probs)
    # Mean calibration error in 10 bins
    n_bins = 10
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_cal, raw_probs, n_bins=n_bins
    )
    raw_cal_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

    fraction_of_positives_cal, mean_predicted_cal = calibration_curve(
        y_cal, cal_probs, n_bins=n_bins
    )
    post_cal_error = np.mean(np.abs(fraction_of_positives_cal - mean_predicted_cal))

    print(
        f"  [{model_type:8s}] n_cal={len(y_cal):5d} | "
        f"pre-calib ECE={raw_cal_error:.4f} → post-calib ECE={post_cal_error:.4f} "
        f"({'improved' if post_cal_error < raw_cal_error else 'no change'})"
    )

    joblib.dump(calibrator, calib_path)
    print(f"             calibrator saved → {calib_path}")


if __name__ == "__main__":
    print("=== Isotonic Calibration ===\n")
    print("How it works:")
    print("  The model outputs raw scores (e.g. 0.62) that are NOT true probabilities.")
    print("  On the historical data, a model score of 0.62 might correspond to only 30%")
    print("  actual YES outcomes. Isotonic regression learns a monotone lookup table")
    print("  that maps raw scores → empirical YES frequencies, fixing this bias.\n")

    for name, path in DATASETS.items():
        print(f"\nCalibrating {name} model...")
        if path is None:
            # Combined model — build dataset on the fly
            df = load_combined_dataset()
            # Inject market_type = 7 (all) for the all-model
            if "market_type" not in df.columns:
                df["market_type"] = 7
        else:
            try:
                df = pd.read_csv(path)
                df["market_type"] = TYPE_MAP.get(name, 7)
            except FileNotFoundError:
                print(f"  dataset not found: {path} — skipping")
                continue

        calibrate_model(name, df)

    print("\n=== Done ===")
    print("Calibrators saved to src/model/calibrator_*.pkl")
    print("Update pipeline.py to load and apply them (see predict_probabilities).")
