"""
Phase 7 — CNN+LSTM Hybrid
Motivation from previous phases:
  - ANN (P4):    best mean R²=0.846, but loses on ds1/ds2 vs LSTM
  - LSTM (P5):   best on ds1/ds2, loses on ds3/ds4
  - CNN-1D (P6): standalone fails — overfit, unstable
  Hypothesis: CNN extracts local features → LSTM captures long-range dependencies
  → take the best of both, avoid weaknesses of each standalone

Architecture:
  Conv1D(64,k=3) → Conv1D(32,k=3) → LSTM(64) → Dense(16) → Dense(1)
  (CNN keeps sequence structure for LSTM — no GlobalAvgPool)

Input:  data/dataset{1-4}.csv — E1, E2; window=20 → sequences (n, 20, 2)
Output: results/phase7_cnn_lstm/
          loss_curves.png          — train vs val loss (seed=0)
          all_phases_comparison.png — R² across all 5 models (Mentor→Hybrid)
          cnn_lstm_metrics.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from pathlib import Path

from utils.data_loader import load_dataset, DATASETS, INPUT_FEATURES, OUTPUT_FEATURE, TEMP_LABEL
from utils.metrics import evaluate

SEEDS       = [0, 42, 123]
WIN         = 20
RESULTS_DIR = Path("results/phase7_cnn_lstm")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# All previous phase results (mean R², 3 seeds, 70/30 split)
PREV = {
    "Mentor":  {1: 0.628, 2: 0.342, 3: 0.856, 4: 0.720},
    "ANN P4":  {1: 0.908, 2: 0.835, 3: 0.910, 4: 0.731},
    "LSTM P5": {1: 0.937, 2: 0.859, 3: 0.843, 4: 0.663},
    "CNN P6":  {1: 0.900, 2: 0.782, 3: 0.838, 4: 0.647},
}


def make_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_cnn_lstm(window, n_features=2):
    """CNN feature extraction → LSTM temporal modelling."""
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(window, n_features)),
        # CNN block — extract local features, keep sequence shape
        tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        # LSTM block — model temporal dependencies across CNN features
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="cnn_lstm")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
]

# ─────────────────────────────────────────────────────────────────────────────
# Train — 3 seeds
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print(f"PHASE 7 — CNN+LSTM Hybrid | window={WIN} | 3 seeds | ds1–ds4")
print("=" * 65)

all_results   = {i: [] for i in DATASETS}
all_histories = {}

for seed in SEEDS:
    tf.random.set_seed(seed); np.random.seed(seed)
    for i in DATASETS:
        df    = load_dataset(i)
        X_raw = df[INPUT_FEATURES].values
        y_raw = df[OUTPUT_FEATURE].values
        split = int(len(X_raw) * 0.7)

        X_tr_sc = MinMaxScaler().fit(X_raw[:split]).transform(X_raw[:split])
        X_te_sc = MinMaxScaler().fit(X_raw[:split]).transform(X_raw[split:])
        # proper: fit scaler on train only
        sc = MinMaxScaler().fit(X_raw[:split])
        X_tr_sc = sc.transform(X_raw[:split])
        X_te_sc = sc.transform(X_raw[split:])
        y_tr, y_te = y_raw[:split], y_raw[split:]

        X_tr_seq, y_tr_seq = make_sequences(X_tr_sc, y_tr, WIN)
        X_te_seq, y_te_seq = make_sequences(X_te_sc, y_te, WIN)

        tf.random.set_seed(seed)
        model = build_cnn_lstm(WIN)
        history = model.fit(
            X_tr_seq, y_tr_seq,
            epochs=300, batch_size=32,
            validation_split=0.2,
            callbacks=CALLBACKS, verbose=0,
        )

        y_pred = model.predict(X_te_seq, verbose=0).flatten()
        m = evaluate(y_te_seq, y_pred)
        m["train_r2"]      = float(r2_score(y_tr_seq, model.predict(X_tr_seq, verbose=0)))
        m["residual_corr"] = float(np.corrcoef(y_pred, y_te_seq - y_pred)[0, 1])
        m["epochs"]        = len(history.history["loss"])
        m["seed"]          = seed
        all_results[i].append(m)
        if seed == 0:
            all_histories[i] = history.history

# ─────────────────────────────────────────────────────────────────────────────
# Full comparison — all phases
# ─────────────────────────────────────────────────────────────────────────────

hybrid_r2 = {i: np.mean([r["r2"] for r in all_results[i]]) for i in DATASETS}
hybrid_std = {i: np.std ([r["r2"] for r in all_results[i]]) for i in DATASETS}

print(f"\n{'':14}" + "".join(f" {k:>9}" for k in PREV) + "  {'Hybrid':>9}  {'±std':>6}")
print("-" * 75)
for i in DATASETS:
    row = f"ds{i} ({TEMP_LABEL[i]}) "
    for k in PREV:
        row += f" {PREV[k][i]:>9.3f}"
    row += f"  {hybrid_r2[i]:>9.4f}  {hybrid_std[i]:>6.4f}"
    # mark best
    all_vals = {k: PREV[k][i] for k in PREV}
    all_vals["Hybrid"] = hybrid_r2[i]
    best = max(all_vals, key=all_vals.get)
    row += f"  ← {best}" if best == "Hybrid" else ""
    print(row)

means = {k: np.mean(list(PREV[k].values())) for k in PREV}
means["Hybrid"] = np.mean(list(hybrid_r2.values()))
print("-" * 75)
row = f"{'Mean':<14}"
for k in PREV:
    row += f" {means[k]:>9.3f}"
row += f"  {means['Hybrid']:>9.4f}"
best_overall = max(means, key=means.get)
row += f"  ← BEST" if best_overall == "Hybrid" else f"  (best: {best_overall})"
print(row)

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- OVERFITTING (seed=0) ---")
print(f"{'Dataset':<14} {'Train R²':>9} {'Test R²':>9} {'Gap':>7}  Status")
print("-" * 50)
for i in DATASETS:
    r   = all_results[i][0]
    gap = r["train_r2"] - r["r2"]
    print(f"ds{i} ({TEMP_LABEL[i]})   {r['train_r2']:>9.4f} {r['r2']:>9.4f} "
          f"{gap:>7.4f}  {'⚠️' if gap > 0.1 else '✅'}")

print(f"\n--- RESIDUAL CORRELATION vs ALL PHASES (seed=0) ---")
ANN_C  = {1: 0.169, 2: 0.436, 3: 0.276, 4: 0.177}
LSTM_C = {1: 0.131, 2: 0.714, 3: 0.469, 4: 0.327}
CNN_C  = {1: 0.292, 2:-0.383, 3: 0.311, 4: 0.327}
print(f"{'Dataset':<14} {'ANN':>7} {'LSTM':>7} {'CNN':>7} {'Hybrid':>8}")
print("-" * 50)
for i in DATASETS:
    rc = all_results[i][0]["residual_corr"]
    print(f"ds{i} ({TEMP_LABEL[i]})  {ANN_C[i]:>7.3f} {LSTM_C[i]:>7.3f} "
          f"{CNN_C[i]:>7.3f} {rc:>8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# TimeSeriesSplit CV
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- TIME-SERIES CV (5 folds) ---")
print(f"{'Dataset':<14} {'CV R²':>7} {'±std':>7} {'min':>7} {'max':>7}  Status")
print("-" * 58)

tscv = TimeSeriesSplit(n_splits=5)
cv_summary = {}
for i in DATASETS:
    df    = load_dataset(i)
    X_sc  = MinMaxScaler().fit_transform(df[INPUT_FEATURES].values)
    X_seq, y_seq = make_sequences(X_sc, df[OUTPUT_FEATURE].values, WIN)

    fold_r2s = []
    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X_seq)):
        X_tr, X_te = X_seq[tr_idx], X_seq[te_idx]
        y_tr, y_te = y_seq[tr_idx], y_seq[te_idx]
        sc   = MinMaxScaler()
        X_tr = sc.fit_transform(X_tr.reshape(-1, 2)).reshape(X_tr.shape)
        X_te = sc.transform(X_te.reshape(-1, 2)).reshape(X_te.shape)

        tf.random.set_seed(SEEDS[0] + fold_idx)
        m = build_cnn_lstm(WIN)
        m.fit(X_tr, y_tr, epochs=300, batch_size=32,
              validation_split=0.2, callbacks=CALLBACKS, verbose=0)
        fold_r2s.append(float(r2_score(y_te, m.predict(X_te, verbose=0))))

    mn, sd = np.mean(fold_r2s), np.std(fold_r2s)
    status = "✅" if sd < 0.05 else ("⚠️" if sd < 0.15 else "❌")
    print(f"ds{i} ({TEMP_LABEL[i]})   {mn:>7.4f} {sd:>7.4f} "
          f"{min(fold_r2s):>7.4f} {max(fold_r2s):>7.4f}  {status}")
    cv_summary[i] = {"mean": round(mn,4), "std": round(sd,4), "folds": fold_r2s}

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

# Loss curves
fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 4))
for j, i in enumerate(DATASETS):
    h  = all_histories[i]
    ax = axes[j]
    ax.plot(h["loss"],     lw=0.9, label="Train")
    ax.plot(h["val_loss"], lw=0.9, label="Val")
    ax.set_title(f"CNN+LSTM ds{i} ({TEMP_LABEL[i]}) | R²={all_results[i][0]['r2']:.3f}", fontsize=9)
    ax.legend(fontsize=7); ax.set_xlabel("Epoch")
fig.suptitle("Phase 7 — Train vs Val Loss (seed=0)", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
plt.close(fig)

# Full comparison heatmap-style bar chart
all_models = list(PREV.keys()) + ["Hybrid"]
all_means  = [means[k] for k in all_models]
colors     = ["gray", "C0", "C1", "C2", "C3"]

fig, axes = plt.subplots(1, len(DATASETS), figsize=(20, 5), sharey=True)
x = np.arange(len(all_models))
for j, i in enumerate(DATASETS):
    ax   = axes[j]
    vals = [PREV[k][i] if k != "Hybrid" else hybrid_r2[i] for k in all_models]
    errs = [0, 0, 0, 0, hybrid_std[i]]
    bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(all_models, fontsize=7, rotation=15)
    ax.set_title(f"ds{i} ({TEMP_LABEL[i]})", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("R²")
    ax.axhline(0.9, color="green", lw=0.7, linestyle="--", alpha=0.5)

fig.suptitle("All Phases — R² Comparison (green line = R²=0.90 target)", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "all_phases_comparison.png", dpi=150)
plt.close(fig)

with open(RESULTS_DIR / "cnn_lstm_metrics.json", "w") as f:
    json.dump({"r2": {str(i): {"mean": hybrid_r2[i], "std": hybrid_std[i]} for i in DATASETS},
               "tscv": cv_summary}, f, indent=2)

print(f"\n[Phase 7 complete] → {RESULTS_DIR}")
