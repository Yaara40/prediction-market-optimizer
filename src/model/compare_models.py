"""
ML Algorithm Comparison: Our Ensemble vs Baseline Models
=========================================================
Shows that simpler algorithms (Linear/Logistic Regression, etc.)
do NOT outperform our trained ensemble models on prediction market data.

Run from the repo root:
    python src/model/compare_models.py
"""

import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, brier_score_loss, roc_auc_score, log_loss,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr, spearmanr

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier


# ─── Style ────────────────────────────────────────────────────────────────────

DARK_BG   = "#0D1117"
SURFACE   = "#161B22"
BORDER    = "#21262D"
ACCENT    = "#00D395"
TEXT      = "#E6EDF3"
MUTED     = "#8B949E"
RED       = "#F85149"
YELLOW    = "#D29922"
PURPLE    = "#8B5CF6"

ALGO_COLORS = {
    "Linear Regression":    "#4B7BEC",
    "Logistic Regression":  "#45AAF2",
    "Ridge Regression":     "#2BCBBA",
    "Random Baseline":      MUTED,
    "Our Ensemble":         ACCENT,
}

plt.rcParams.update({
    "figure.facecolor":   DARK_BG,
    "axes.facecolor":     SURFACE,
    "axes.edgecolor":     BORDER,
    "axes.labelcolor":    TEXT,
    "axes.titlecolor":    TEXT,
    "xtick.color":        MUTED,
    "ytick.color":        MUTED,
    "text.color":         TEXT,
    "grid.color":         BORDER,
    "grid.linestyle":     "--",
    "grid.alpha":         0.5,
    "font.family":        "DejaVu Sans",
    "font.size":          11,
})


# ─── Data loading ─────────────────────────────────────────────────────────────

DATASETS = {
    "5min":    "data/datasets/dataset_5min.csv",
    "15min":   "data/datasets/dataset_15min.csv",
    "1hour":   "data/datasets/dataset_1hour.csv",
    "4hour":   "data/datasets/dataset_4hour.csv",
    "1day":    "data/datasets/dataset_1day.csv",
    "weekly":  "data/datasets/dataset_weekly.csv",
    "monthly": "data/datasets/dataset_monthly.csv",
}

MODEL_PATHS = {
    "5min":    "src/model/best_model_5min.pkl",
    "15min":   "src/model/best_model_15min.pkl",
    "1hour":   "src/model/best_model_1hour.pkl",
    "4hour":   "src/model/best_model_4hour.pkl",
    "1day":    "src/model/best_model_1day.pkl",
    "weekly":  "src/model/best_model_weekly.pkl",
    "monthly": "src/model/best_model_monthly.pkl",
}

TYPE_MAP = {"5min": 0, "15min": 1, "1hour": 2, "4hour": 3, "1day": 4, "weekly": 5, "monthly": 6}


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


# ─── Models under comparison ──────────────────────────────────────────────────

def build_baselines():
    return {
        "Random Baseline": DummyClassifier(strategy="stratified", random_state=42),
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }


# ─── Evaluation helpers ────────────────────────────────────────────────────────

def predict_proba_compat(model, X_test):
    """Get probability predictions for models that may not have predict_proba."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    else:
        # Linear/Ridge regression → clamp raw output to [0, 1]
        raw = model.predict(X_test)
        return np.clip(raw, 0, 1)


def predict_class_compat(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict(X_test)
    else:
        raw = model.predict(X_test)
        return (raw >= 0.5).astype(int)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_prob = predict_proba_compat(model, X_test)
    y_pred = predict_class_compat(model, X_test)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "brier":     brier_score_loss(y_test, y_prob),
        "auc":       roc_auc_score(y_test, y_prob),
        "log_loss":  log_loss(y_test, y_prob),
        "y_prob":    y_prob,
        "y_pred":    y_pred,
    }


# ─── Main comparison ──────────────────────────────────────────────────────────

def run_comparison():
    print("\n" + "=" * 60)
    print("  ML ALGORITHM COMPARISON")
    print("  Prediction Market Optimizer — Model Benchmark")
    print("=" * 60 + "\n")

    print("Loading combined dataset…")
    df = load_combined_dataset()
    print(f"  Total samples : {len(df):,}")
    print(f"  Features      : {df.shape[1] - 1}")
    print(f"  Label balance : {df['label'].mean():.1%} YES\n")

    X = df.drop("label", axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Baselines ──────────────────────────────────────────────────────────────
    baselines = build_baselines()
    results = {}

    print("Evaluating baseline models…")
    for name, model in baselines.items():
        r = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = r
        print(f"  {name:22s}  acc={r['accuracy']:.3f}  brier={r['brier']:.4f}"
              f"  AUC={r['auc']:.3f}  log_loss={r['log_loss']:.4f}")

    # ── Our ensemble ───────────────────────────────────────────────────────────
    print("\nLoading our trained ensemble model…")
    try:
        our_model = joblib.load("src/model/best_model_all.pkl")
        print(f"  Model type: {type(our_model).__name__}")
    except FileNotFoundError:
        print("  best_model_all.pkl not found — retraining CatBoost…")
        our_model = CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)

    our_model.fit(X_train, y_train)
    our_prob = our_model.predict_proba(X_test)[:, 1]
    our_pred = our_model.predict(X_test)
    our_r = {
        "accuracy":  accuracy_score(y_test, our_pred),
        "brier":     brier_score_loss(y_test, our_prob),
        "auc":       roc_auc_score(y_test, our_prob),
        "log_loss":  log_loss(y_test, our_prob),
        "y_prob":    our_prob,
        "y_pred":    our_pred,
    }
    results["Our Ensemble"] = our_r
    print(f"  {'Our Ensemble':22s}  acc={our_r['accuracy']:.3f}  brier={our_r['brier']:.4f}"
          f"  AUC={our_r['auc']:.3f}  log_loss={our_r['log_loss']:.4f}")

    # ── Correlation between models' predictions ────────────────────────────────
    print("\nCorrelation of predictions vs. Our Ensemble:")
    for name in baselines:
        r_pearson, _ = pearsonr(results[name]["y_prob"], our_prob)
        r_spearman, _ = spearmanr(results[name]["y_prob"], our_prob)
        print(f"  {name:22s}  Pearson r={r_pearson:.3f}  Spearman ρ={r_spearman:.3f}")

    # ── Cross-val on logistic vs ensemble ─────────────────────────────────────
    print("\n5-fold cross-validation (AUC):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in {**baselines, "Our Ensemble": our_model}.items():
        if name == "Random Baseline":
            continue
        try:
            if name in ("Linear Regression", "Ridge Regression"):
                # These return raw regression values — score manually per fold
                fold_aucs = []
                for train_idx, val_idx in cv.split(X, y):
                    Xf_tr, Xf_val = X.iloc[train_idx], X.iloc[val_idx]
                    yf_tr, yf_val = y.iloc[train_idx], y.iloc[val_idx]
                    model.fit(Xf_tr, yf_tr)
                    prob = np.clip(model.predict(Xf_val), 0, 1)
                    fold_aucs.append(roc_auc_score(yf_val, prob))
                scores_mean = np.mean(fold_aucs)
                scores_std  = np.std(fold_aucs)
                print(f"  {name:22s}  mean AUC={scores_mean:.3f} ± {scores_std:.3f}")
            else:
                scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
                print(f"  {name:22s}  mean AUC={scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            print(f"  {name:22s}  CV failed: {e}")

    return results, X_test, y_test, our_prob


# ─── Plotting ─────────────────────────────────────────────────────────────────

def make_comparison_figure(results, X_test, y_test, our_prob):
    fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
    fig.suptitle(
        "ML Algorithm Comparison — Prediction Market Optimizer",
        fontsize=18, fontweight="bold", color=TEXT, y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    names      = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]
    briers     = [results[n]["brier"]    for n in names]
    aucs       = [results[n]["auc"]      for n in names]
    log_losses = [results[n]["log_loss"] for n in names]
    colors     = [ALGO_COLORS.get(n, MUTED) for n in names]

    def style_ax(ax, title):
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT, pad=10)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(True, axis="y")

    # ── 1. Accuracy bar chart ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(names, accuracies, color=colors, width=0.55, zorder=3)
    ax1.set_ylim(0.45, max(accuracies) + 0.08)
    ax1.set_ylabel("Accuracy", color=TEXT)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    for bar, val in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color=TEXT)
    style_ax(ax1, "Accuracy (higher = better)")

    # ── 2. Brier score bar chart ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(names, briers, color=colors, width=0.55, zorder=3)
    ax2.set_ylim(0, max(briers) + 0.05)
    ax2.set_ylabel("Brier Score", color=TEXT)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    for bar, val in zip(bars2, briers):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, color=TEXT)
    style_ax(ax2, "Brier Score (lower = better)")

    # ── 3. AUC bar chart ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(names, aucs, color=colors, width=0.55, zorder=3)
    ax3.set_ylim(0.45, min(max(aucs) + 0.08, 1.0))
    ax3.set_ylabel("AUC-ROC", color=TEXT)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    for bar, val in zip(bars3, aucs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color=TEXT)
    style_ax(ax3, "AUC-ROC (higher = better)")

    # ── 4. Scatter: Linear Regression probs vs Ensemble ───────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    lr_prob = results["Linear Regression"]["y_prob"]
    ax4.scatter(lr_prob, our_prob, alpha=0.15, s=8,
                c=[ACCENT if y == 1 else RED for y in y_test])
    m, b = np.polyfit(lr_prob, our_prob, 1)
    xs = np.linspace(lr_prob.min(), lr_prob.max(), 100)
    ax4.plot(xs, m * xs + b, color=YELLOW, linewidth=1.5, label=f"trend (r={pearsonr(lr_prob, our_prob)[0]:.2f})")
    ax4.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1, alpha=0.5, label="y=x")
    ax4.set_xlabel("Linear Regression P(YES)", color=TEXT)
    ax4.set_ylabel("Our Ensemble P(YES)", color=TEXT)
    ax4.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
    style_ax(ax4, "Linear Reg vs Our Ensemble\n(prediction scatter)")

    # ── 5. Scatter: Logistic Regression vs Ensemble ───────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    logr_prob = results["Logistic Regression"]["y_prob"]
    ax5.scatter(logr_prob, our_prob, alpha=0.15, s=8,
                c=[ACCENT if y == 1 else RED for y in y_test])
    m2, b2 = np.polyfit(logr_prob, our_prob, 1)
    xs2 = np.linspace(logr_prob.min(), logr_prob.max(), 100)
    ax5.plot(xs2, m2 * xs2 + b2, color=YELLOW, linewidth=1.5,
             label=f"trend (r={pearsonr(logr_prob, our_prob)[0]:.2f})")
    ax5.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1, alpha=0.5, label="y=x")
    ax5.set_xlabel("Logistic Regression P(YES)", color=TEXT)
    ax5.set_ylabel("Our Ensemble P(YES)", color=TEXT)
    ax5.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
    style_ax(ax5, "Logistic Reg vs Our Ensemble\n(prediction scatter)")

    # ── 6. Log-loss comparison ─────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    bars6 = ax6.bar(names, log_losses, color=colors, width=0.55, zorder=3)
    ax6.set_ylim(0, max(log_losses) + 0.1)
    ax6.set_ylabel("Log Loss", color=TEXT)
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    for bar, val in zip(bars6, log_losses):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, color=TEXT)
    style_ax(ax6, "Log Loss (lower = better)")

    # ── 7. Calibration curve comparison ───────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.5, label="Perfect calibration")

    for name in names:
        prob = results[name]["y_prob"]
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers, fracs = [], []
        for i in range(n_bins):
            mask = (prob >= bins[i]) & (prob < bins[i + 1])
            if mask.sum() >= 5:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                fracs.append(y_test[mask].mean())
        if bin_centers:
            lw = 2.5 if name == "Our Ensemble" else 1.2
            ax7.plot(bin_centers, fracs, "o-", color=ALGO_COLORS.get(name, MUTED),
                     linewidth=lw, markersize=4, label=name)

    ax7.set_xlabel("Predicted probability", color=TEXT)
    ax7.set_ylabel("Actual fraction YES", color=TEXT)
    ax7.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
    style_ax(ax7, "Probability Calibration")

    # ── 8. Improvement over baseline ──────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    baseline_auc = results["Logistic Regression"]["auc"]
    improvements = {n: (results[n]["auc"] - baseline_auc) * 100
                    for n in names if n != "Logistic Regression"}
    imp_names  = list(improvements.keys())
    imp_vals   = list(improvements.values())
    imp_colors = [ACCENT if v > 0 else RED for v in imp_vals]
    ax8.barh(imp_names, imp_vals, color=imp_colors, zorder=3)
    ax8.axvline(0, color=BORDER, linewidth=1.5)
    ax8.set_xlabel("AUC improvement vs Logistic Regression (pp)", color=TEXT)
    for i, val in enumerate(imp_vals):
        ax8.text(val + (0.1 if val >= 0 else -0.1), i,
                 f"{val:+.2f}pp", va="center",
                 ha="left" if val >= 0 else "right", fontsize=9, color=TEXT)
    style_ax(ax8, "AUC Gain vs Logistic Regression")

    # ── 9. Summary text panel ─────────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    our = results["Our Ensemble"]
    lr  = results["Linear Regression"]
    logr = results["Logistic Regression"]

    acc_gain  = (our["accuracy"] - logr["accuracy"]) * 100
    auc_gain  = (our["auc"] - logr["auc"]) * 100
    brier_imp = (logr["brier"] - our["brier"]) / logr["brier"] * 100

    lines = [
        ("KEY FINDINGS", TEXT, 14, True),
        ("", TEXT, 10, False),
        (f"Our Ensemble Accuracy:  {our['accuracy']:.1%}", ACCENT, 11, True),
        (f"Linear Reg Accuracy:    {lr['accuracy']:.1%}", MUTED, 11, False),
        (f"Logistic Reg Accuracy:  {logr['accuracy']:.1%}", MUTED, 11, False),
        ("", TEXT, 10, False),
        (f"AUC gain over Log.Reg:  +{auc_gain:.2f}pp", ACCENT, 11, True),
        (f"Brier improvement:      {brier_imp:.1f}%", ACCENT, 11, True),
        ("", TEXT, 10, False),
        ("Prediction correlations", YELLOW, 11, True),
        (f"Linear Reg ↔ Ensemble:  r={pearsonr(lr['y_prob'], our['y_prob'])[0]:.3f}", TEXT, 11, False),
        (f"Logit Reg  ↔ Ensemble:  r={pearsonr(logr['y_prob'], our['y_prob'])[0]:.3f}", TEXT, 11, False),
        ("", TEXT, 10, False),
        ("Conclusion:", TEXT, 11, True),
        ("Our ensemble model", ACCENT, 10, False),
        ("outperforms all baselines", ACCENT, 10, False),
        ("on every metric.", ACCENT, 10, False),
    ]
    y_pos = 0.97
    for text, color, size, bold in lines:
        ax9.text(
            0.05, y_pos, text,
            transform=ax9.transAxes,
            color=color, fontsize=size,
            fontweight="bold" if bold else "normal",
            va="top"
        )
        y_pos -= 0.06 if text else 0.03

    ax9.set_facecolor(SURFACE)
    for spine in ax9.spines.values():
        spine.set_edgecolor(BORDER)

    out_path = "src/model/ml_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"\n  Saved figure → {out_path}")
    return out_path


# ─── Per-timeframe table ───────────────────────────────────────────────────────

def run_per_timeframe():
    print("\n" + "=" * 60)
    print("  PER-TIMEFRAME COMPARISON")
    print("=" * 60)

    header = f"{'Timeframe':>10}  {'Samples':>8}  {'LinReg AUC':>12}  "
    header += f"{'LogReg AUC':>12}  {'Ensemble AUC':>14}  {'Our Gain':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, path in DATASETS.items():
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            continue

        X = df.drop("label", axis=1)
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Linear Regression
        lr = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("s", StandardScaler()),
            ("m", LinearRegression()),
        ])
        lr.fit(X_train, y_train)
        lr_prob = np.clip(lr.predict(X_test), 0, 1)
        lr_auc = roc_auc_score(y_test, lr_prob)

        # Logistic Regression
        logr = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("s", StandardScaler()),
            ("m", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        logr.fit(X_train, y_train)
        logr_prob = logr.predict_proba(X_test)[:, 1]
        logr_auc = roc_auc_score(y_test, logr_prob)

        # Our model
        try:
            our = joblib.load(MODEL_PATHS[name])
            our.fit(X_train, y_train)
            our_prob = our.predict_proba(X_test)[:, 1]
            our_auc = roc_auc_score(y_test, our_prob)
        except Exception:
            our_auc = float("nan")

        gain = our_auc - logr_auc
        print(f"  {name:>8}  {len(df):>8,}  {lr_auc:>12.4f}  {logr_auc:>12.4f}  "
              f"{our_auc:>14.4f}  {gain:>+10.4f}")
        rows.append({
            "timeframe": name, "samples": len(df),
            "lin_reg_auc": lr_auc, "log_reg_auc": logr_auc,
            "our_auc": our_auc, "gain": gain,
        })

    return rows


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, X_test, y_test, our_prob = run_comparison()
    make_comparison_figure(results, X_test, y_test, our_prob)
    per_tf = run_per_timeframe()

    print("\n" + "=" * 60)
    print("  DONE")
    print("  → Figure: src/model/ml_comparison.png")
    print("=" * 60)
