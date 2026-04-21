"""
Scientific diagnostics for the best model (Cell C: medium arch + E1+E2).
Answers: is this model overfitting? are results reproducible? are they real?

Tests:
  1. Train R² vs Test R²  — gap reveals overfitting
  2. Residual analysis     — random residuals = good fit, patterns = bad
  3. K-Fold CV (k=5)       — unbiased estimate without fixed-split luck
  4. Learning curve        — performance vs training set size
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from scipy import stats
from pathlib import Path

from utils.data_loader import load_dataset, DATASETS, INPUT_FEATURES, OUTPUT_FEATURE, TEMP_LABEL
from utils.metrics import evaluate

SEED = 0
RESULTS_DIR = Path("results/diagnostics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_model(n=2):
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(n,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
]

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Train R² vs Test R²
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("TEST 1 — Train R² vs Test R² (gap > 0.1 = overfitting concern)")
print("=" * 65)
print(f"{'Dataset':<14} {'Train R²':>9} {'Test R²':>9} {'Gap':>7} {'Status'}")
print("-" * 55)

test1_results = {}
for i in DATASETS:
    df = load_dataset(i)
    X  = df[INPUT_FEATURES].values
    y  = df[OUTPUT_FEATURE].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, shuffle=False)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    tf.random.set_seed(SEED)
    model = build_model()
    model.fit(X_train, y_train, epochs=300, batch_size=32,
              validation_split=0.2, callbacks=CALLBACKS, verbose=0)

    train_r2 = float(r2_score(y_train, model.predict(X_train, verbose=0)))
    test_r2  = float(r2_score(y_test,  model.predict(X_test,  verbose=0)))
    gap = train_r2 - test_r2
    status = "⚠️ overfit" if gap > 0.1 else "✅ ok"

    print(f"ds{i} ({TEMP_LABEL[i]})   {train_r2:>9.4f} {test_r2:>9.4f} {gap:>7.4f}  {status}")
    test1_results[i] = {"train_r2": train_r2, "test_r2": test_r2, "gap": gap}

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Residual analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("TEST 2 — Residual Analysis")
print("  Shapiro-Wilk p > 0.05 → residuals are normal (good fit)")
print("  Correlation |r| < 0.1 → residuals uncorrelated with pred (good)")
print("=" * 65)
print(f"{'Dataset':<14} {'Shapiro p':>10} {'Normal?':>9} {'|corr|':>8} {'Uncorr?'}")
print("-" * 55)

fig, axes = plt.subplots(len(DATASETS), 3, figsize=(15, 4 * len(DATASETS)))

test2_results = {}
for j, i in enumerate(DATASETS):
    df = load_dataset(i)
    X  = df[INPUT_FEATURES].values
    y  = df[OUTPUT_FEATURE].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, shuffle=False)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    tf.random.set_seed(SEED)
    model = build_model()
    model.fit(X_train, y_train, epochs=300, batch_size=32,
              validation_split=0.2, callbacks=CALLBACKS, verbose=0)

    y_pred    = model.predict(X_test, verbose=0).flatten()
    residuals = y_test - y_pred

    # Shapiro-Wilk (sample 5000 max — test requires finite sample)
    sample = residuals[:5000] if len(residuals) > 5000 else residuals
    _, p_shapiro = stats.shapiro(sample)
    corr = float(np.corrcoef(y_pred, residuals)[0, 1])

    normal_ok = "✅" if p_shapiro > 0.05 else "❌"
    corr_ok   = "✅" if abs(corr) < 0.1  else "❌"
    print(f"ds{i} ({TEMP_LABEL[i]})   {p_shapiro:>10.4f} {normal_ok:>9} {abs(corr):>8.4f}  {corr_ok}")
    test2_results[i] = {"shapiro_p": p_shapiro, "residual_corr": corr}

    # Residual plots
    x_ax = np.arange(len(residuals))
    axes[j, 0].scatter(y_pred, residuals, s=1, alpha=0.3, color=f"C{j}")
    axes[j, 0].axhline(0, color="red", lw=0.8, linestyle="--")
    axes[j, 0].set_xlabel("Predicted"); axes[j, 0].set_ylabel("Residual")
    axes[j, 0].set_title(f"ds{i} — Residuals vs Predicted")

    axes[j, 1].hist(residuals, bins=50, color=f"C{j}", alpha=0.8)
    axes[j, 1].set_xlabel("Residual"); axes[j, 1].set_ylabel("Count")
    axes[j, 1].set_title(f"ds{i} — Residual Distribution (p={p_shapiro:.3f})")

    axes[j, 2].scatter(range(len(y_test)), y_test,  s=1, alpha=0.4, label="Truth")
    axes[j, 2].scatter(range(len(y_pred)), y_pred,  s=1, alpha=0.4, label="Pred")
    r2 = r2_score(y_test, y_pred)
    axes[j, 2].set_title(f"ds{i} — Pred vs Truth | R²={r2:.4f}")
    axes[j, 2].legend(fontsize=7, markerscale=3)

fig.suptitle("Residual Analysis — Medium ANN (Cell C)", fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "residual_analysis.png", dpi=150)
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: K-Fold Cross-Validation (k=5)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("TEST 3 — 5-Fold Cross-Validation (unbiased, no fixed-split luck)")
print("=" * 65)
print(f"{'Dataset':<14} {'R² mean':>9} {'±std':>7} {'min':>7} {'max':>7} {'Status'}")
print("-" * 60)

kf = KFold(n_splits=5, shuffle=False)
test3_results = {}

for i in DATASETS:
    df = load_dataset(i)
    X  = df[INPUT_FEATURES].values
    y  = df[OUTPUT_FEATURE].values

    fold_r2s = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        tf.random.set_seed(SEED + fold_idx)
        model = build_model()
        model.fit(X_train, y_train, epochs=300, batch_size=32,
                  validation_split=0.2, callbacks=CALLBACKS, verbose=0)

        r2 = float(r2_score(y_test, model.predict(X_test, verbose=0)))
        fold_r2s.append(r2)

    mean_r2, std_r2 = np.mean(fold_r2s), np.std(fold_r2s)
    status = "✅ stable" if std_r2 < 0.05 else ("⚠️ unstable" if std_r2 < 0.15 else "❌ very unstable")
    print(f"ds{i} ({TEMP_LABEL[i]})   {mean_r2:>9.4f} {std_r2:>7.4f} "
          f"{min(fold_r2s):>7.4f} {max(fold_r2s):>7.4f}  {status}")
    test3_results[i] = {"fold_r2s": fold_r2s, "mean": mean_r2, "std": std_r2}

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Learning curve (performance vs training size)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("TEST 4 — Learning Curve (R² vs training set size)")
print("  Converging curves = model learns, not memorises")
print("=" * 65)

TRAIN_FRACS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 4), sharey=True)

for j, i in enumerate(DATASETS):
    df = load_dataset(i)
    X  = df[INPUT_FEATURES].values
    y  = df[OUTPUT_FEATURE].values

    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, shuffle=False)

    scaler_full = MinMaxScaler().fit(X_full_train)
    X_test_sc   = scaler_full.transform(X_test)

    train_r2s, test_r2s, sizes = [], [], []

    for frac in TRAIN_FRACS:
        n = max(64, int(len(X_full_train) * frac))
        X_sub, y_sub = X_full_train[:n], y_full_train[:n]

        scaler = MinMaxScaler()
        X_sub_sc = scaler.fit_transform(X_sub)
        X_test_sc2 = scaler.transform(X_test)

        tf.random.set_seed(SEED)
        model = build_model()
        model.fit(X_sub_sc, y_sub, epochs=300, batch_size=32,
                  validation_split=0.2, callbacks=CALLBACKS, verbose=0)

        train_r2s.append(r2_score(y_sub,  model.predict(X_sub_sc, verbose=0)))
        test_r2s.append (r2_score(y_test, model.predict(X_test_sc2, verbose=0)))
        sizes.append(n)
        print(f"  ds{i} | n={n:>6} | train R²={train_r2s[-1]:.3f} | test R²={test_r2s[-1]:.3f}")

    axes[j].plot(sizes, train_r2s, "o-", label="Train R²", lw=1.5)
    axes[j].plot(sizes, test_r2s,  "s--", label="Test R²",  lw=1.5)
    axes[j].set_title(f"ds{i} ({TEMP_LABEL[i]})", fontsize=9)
    axes[j].set_xlabel("Training samples")
    axes[j].legend(fontsize=7)
    axes[j].set_ylim(-0.1, 1.05)
    axes[j].axhline(0, color="black", lw=0.5, linestyle="--")

axes[0].set_ylabel("R²")
fig.suptitle("Learning Curves — Medium ANN (Cell C)", fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "learning_curves.png", dpi=150)
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("DIAGNOSTIC SUMMARY")
print("=" * 65)
print(f"{'Dataset':<14} {'T1 gap':>8} {'T2 normal':>11} {'T3 CV R²':>10} {'T3 std':>8}")
print("-" * 55)
for i in DATASETS:
    gap    = test1_results[i]["gap"]
    normal = "✅" if test2_results[i]["shapiro_p"] > 0.05 else "❌"
    cv_r2  = test3_results[i]["mean"]
    cv_std = test3_results[i]["std"]
    print(f"ds{i} ({TEMP_LABEL[i]})   {gap:>8.4f} {normal:>11} {cv_r2:>10.4f} {cv_std:>8.4f}")

report = {
    "test1_train_vs_test": {str(i): test1_results[i] for i in DATASETS},
    "test2_residuals":     {str(i): test2_results[i] for i in DATASETS},
    "test3_kfold_cv":      {str(i): test3_results[i] for i in DATASETS},
}
with open(RESULTS_DIR / "diagnostic_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n[Diagnostics complete] → {RESULTS_DIR}")
