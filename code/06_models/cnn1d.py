"""
Phase 6 — CNN-1D
CNN extracts local temporal patterns via convolution kernels.
Compared to LSTM: faster to train, fewer parameters, no vanishing gradient.
Compared to ANN: sees temporal context (same window=20 as LSTM).

Architecture: Conv1D(64,3) → Conv1D(32,3) → GlobalAvgPool → Dense(16) → Dense(1)
Same window, same features (E1+E2), same split as Phase 5 — fair 3-way comparison.
Built-in diagnostics: overfitting check, TimeSeriesSplit CV, 3 seeds.

Input:  data/dataset{1-4}.csv — E1, E2; window=20 → sequences (n, 20, 2)
Output: results/phase6_cnn1d/
          loss_curves.png    — CNN-1D train vs val loss (seed=0)
          r2_comparison.png  — 3-way bar: ANN P4 vs LSTM P5 vs CNN-1D
          cnn1d_metrics.json
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
RESULTS_DIR = Path("results/phase6_cnn1d")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reference results from previous phases
PHASE4_R2 = {1: 0.908, 2: 0.835, 3: 0.910, 4: 0.731}
PHASE5_R2 = {1: 0.937, 2: 0.859, 3: 0.843, 4: 0.663}


def make_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_cnn1d(window, n_features=2):
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(window, n_features)),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="cnn1d")
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
print(f"PHASE 6 — CNN-1D | window={WIN} | 3 seeds | ds1–ds4")
print("=" * 65)

all_results   = {i: [] for i in DATASETS}
all_histories = {}

for seed in SEEDS:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    for i in DATASETS:
        df     = load_dataset(i)
        X_raw  = df[INPUT_FEATURES].values
        y_raw  = df[OUTPUT_FEATURE].values
        split  = int(len(X_raw) * 0.7)

        X_tr_raw, X_te_raw = X_raw[:split], X_raw[split:]
        y_tr_raw, y_te_raw = y_raw[:split], y_raw[split:]

        scaler  = MinMaxScaler()
        X_tr_sc = scaler.fit_transform(X_tr_raw)
        X_te_sc = scaler.transform(X_te_raw)

        X_tr_seq, y_tr_seq = make_sequences(X_tr_sc, y_tr_raw, WIN)
        X_te_seq, y_te_seq = make_sequences(X_te_sc, y_te_raw, WIN)

        tf.random.set_seed(seed)
        model = build_cnn1d(WIN)
        history = model.fit(
            X_tr_seq, y_tr_seq,
            epochs=300, batch_size=32,
            validation_split=0.2,
            callbacks=CALLBACKS,
            verbose=0,
        )

        y_pred   = model.predict(X_te_seq, verbose=0).flatten()
        m        = evaluate(y_te_seq, y_pred)
        m["train_r2"]      = float(r2_score(y_tr_seq, model.predict(X_tr_seq, verbose=0)))
        m["residual_corr"] = float(np.corrcoef(y_pred, y_te_seq - y_pred)[0, 1])
        m["epochs"]        = len(history.history["loss"])
        m["seed"]          = seed
        all_results[i].append(m)

        if seed == 0:
            all_histories[i] = history.history

# ─────────────────────────────────────────────────────────────────────────────
# 3-way comparison
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'Dataset':<14} {'ANN':>7} {'LSTM':>7} {'CNN-1D':>8} {'±std':>6}  {'Best'}")
print("-" * 55)

summary, cnn_means = {}, []
for i in DATASETS:
    r2s   = [r["r2"] for r in all_results[i]]
    m, s  = np.mean(r2s), np.std(r2s)
    cnn_means.append(m)
    scores = {"ANN": PHASE4_R2[i], "LSTM": PHASE5_R2[i], "CNN-1D": m}
    best   = max(scores, key=scores.get)
    marker = {"ANN": "", "LSTM": "", "CNN-1D": " ✅"}
    print(f"ds{i} ({TEMP_LABEL[i]})   "
          f"{PHASE4_R2[i]:>7.3f} {PHASE5_R2[i]:>7.3f} {m:>8.4f} {s:>6.4f}  {best}{marker.get(best,'')}")
    summary[i] = {"r2_mean": round(m, 4), "r2_std": round(s, 4)}

ann_mean  = np.mean(list(PHASE4_R2.values()))
lstm_mean = np.mean(list(PHASE5_R2.values()))
cnn_mean  = np.mean(cnn_means)
overall_best = max({"ANN": ann_mean, "LSTM": lstm_mean, "CNN-1D": cnn_mean}, key=lambda k: {"ANN": ann_mean, "LSTM": lstm_mean, "CNN-1D": cnn_mean}[k])
print(f"{'Mean':<14} {ann_mean:>7.3f} {lstm_mean:>7.3f} {cnn_mean:>8.4f}{'':>7}  {overall_best}")

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- OVERFITTING (seed=0, gap = train R² - test R²) ---")
print(f"{'Dataset':<14} {'Train R²':>9} {'Test R²':>9} {'Gap':>7}  {'Status'}")
print("-" * 50)
for i in DATASETS:
    r    = all_results[i][0]
    gap  = r["train_r2"] - r["r2"]
    flag = "⚠️" if gap > 0.1 else "✅"
    print(f"ds{i} ({TEMP_LABEL[i]})   {r['train_r2']:>9.4f} {r['r2']:>9.4f} {gap:>7.4f}  {flag}")

print(f"\n--- RESIDUAL CORRELATION (seed=0) ---")
print(f"{'Dataset':<14} {'CNN-1D corr':>13}  {'vs ANN':>8}  {'vs LSTM':>8}")
print("-" * 50)
ANN_CORR  = {1: 0.169, 2: 0.436, 3: 0.276, 4: 0.177}
LSTM_CORR = {1: 0.131, 2: 0.714, 3: 0.469, 4: 0.327}
for i in DATASETS:
    rc = all_results[i][0]["residual_corr"]
    da = rc - ANN_CORR[i]
    dl = rc - LSTM_CORR[i]
    print(f"ds{i} ({TEMP_LABEL[i]})   {rc:>13.4f}  {da:>+8.4f}  {dl:>+8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# TimeSeriesSplit CV
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- TIME-SERIES CV (5 folds) ---")
print(f"{'Dataset':<14} {'CV R² mean':>11} {'±std':>7} {'min':>7} {'max':>7}  {'Status'}")
print("-" * 60)

tscv      = TimeSeriesSplit(n_splits=5)
cv_summary = {}
for i in DATASETS:
    df    = load_dataset(i)
    X_raw = df[INPUT_FEATURES].values
    y_raw = df[OUTPUT_FEATURE].values
    X_sc  = MinMaxScaler().fit_transform(X_raw)
    X_seq, y_seq = make_sequences(X_sc, y_raw, WIN)

    fold_r2s = []
    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X_seq)):
        X_tr, X_te = X_seq[tr_idx], X_seq[te_idx]
        y_tr, y_te = y_seq[tr_idx], y_seq[te_idx]

        sc   = MinMaxScaler()
        flat = X_tr.reshape(-1, X_tr.shape[-1])
        X_tr = sc.fit_transform(flat).reshape(X_tr.shape)
        X_te = sc.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)

        tf.random.set_seed(SEEDS[0] + fold_idx)
        m = build_cnn1d(WIN)
        m.fit(X_tr, y_tr, epochs=300, batch_size=32,
              validation_split=0.2, callbacks=CALLBACKS, verbose=0)
        fold_r2s.append(float(r2_score(y_te, m.predict(X_te, verbose=0))))

    mean_r2, std_r2 = np.mean(fold_r2s), np.std(fold_r2s)
    status = "✅ stable" if std_r2 < 0.05 else ("⚠️" if std_r2 < 0.15 else "❌")
    print(f"ds{i} ({TEMP_LABEL[i]})   {mean_r2:>11.4f} {std_r2:>7.4f} "
          f"{min(fold_r2s):>7.4f} {max(fold_r2s):>7.4f}  {status}")
    cv_summary[i] = {"mean": round(mean_r2, 4), "std": round(std_r2, 4)}

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 4))
for j, i in enumerate(DATASETS):
    h  = all_histories[i]
    r2 = all_results[i][0]["r2"]
    ax = axes[j]
    ax.plot(h["loss"],     lw=0.9, label="Train")
    ax.plot(h["val_loss"], lw=0.9, label="Val")
    ax.set_title(f"CNN-1D ds{i} ({TEMP_LABEL[i]}) | R²={r2:.3f}", fontsize=9)
    ax.legend(fontsize=7); ax.set_xlabel("Epoch")
fig.suptitle("Phase 6 — Train vs Val Loss (seed=0)", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
plt.close(fig)

# 3-way bar chart
fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 4), sharey=True)
for j, i in enumerate(DATASETS):
    ax    = axes[j]
    r2s   = [r["r2"] for r in all_results[i]]
    vals  = [PHASE4_R2[i], PHASE5_R2[i], np.mean(r2s)]
    errs  = [0, 0, np.std(r2s)]
    colors = ["C0", "C1", "C2"]
    ax.bar(range(3), vals, yerr=errs, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(range(3)); ax.set_xticklabels(["ANN", "LSTM", "CNN-1D"], fontsize=8)
    ax.set_title(f"ds{i} ({TEMP_LABEL[i]})", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("R²")
fig.suptitle("Phase 6 — 3-way R² Comparison", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "r2_comparison.png", dpi=150)
plt.close(fig)

with open(RESULTS_DIR / "cnn1d_metrics.json", "w") as f:
    json.dump({"comparison": summary, "tscv": cv_summary}, f, indent=2)

print(f"\n[Phase 6 complete] → {RESULTS_DIR}")
