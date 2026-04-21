"""
Phase 5 — LSTM
Scientific justification: ANN residuals have temporal structure (corr 0.13–0.60),
meaning the signal carries information across time steps that ANN ignores.

Design:
  - Sliding window sequences (window=20 steps)
  - Input: [E1_t-19, ..., E1_t], [E2_t-19, ..., E2_t]  shape=(20, 2)
  - Output: Strain_t
  - TimeSeriesSplit CV (correct for sequential data — no future leakage)
  - Comparison baseline: Cell C (medium ANN, same features)
  - 3 seeds, built-in diagnostics
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

SEEDS      = [0, 42, 123]
WIN        = 20          # sequence length (window size)
RESULTS_DIR = Path("results/phase5_lstm")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Sequence builder ──────────────────────────────────────────────────────────

def make_sequences(X: np.ndarray, y: np.ndarray, window: int):
    """Sliding window: X shape (n-window, window, features), y shape (n-window,)"""
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ── Model builders ────────────────────────────────────────────────────────────

def build_ann_cell_c(n_features=2):
    """Phase 4 Cell C — fair comparison baseline."""
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="ann_cell_c")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def build_lstm(window, n_features=2):
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(window, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="lstm")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Train both models, collect diagnostics
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print(f"PHASE 5 — LSTM vs ANN Cell C | window={WIN} | 3 seeds | ds1–ds4")
print("=" * 65)

all_results = {"ann": {i: [] for i in DATASETS},
               "lstm": {i: [] for i in DATASETS}}
all_histories = {"ann": {}, "lstm": {}}   # seed=0 only

for seed in SEEDS:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    for i in DATASETS:
        df = load_dataset(i)
        X_raw = df[INPUT_FEATURES].values
        y_raw = df[OUTPUT_FEATURE].values

        # ── 70/30 split (shuffle=False — keep temporal order) ─────────────
        split = int(len(X_raw) * 0.7)
        X_tr_raw, X_te_raw = X_raw[:split], X_raw[split:]
        y_tr_raw, y_te_raw = y_raw[:split], y_raw[split:]

        scaler = MinMaxScaler()
        X_tr_sc = scaler.fit_transform(X_tr_raw)
        X_te_sc = scaler.transform(X_te_raw)

        # ── ANN (flat input) ──────────────────────────────────────────────
        tf.random.set_seed(seed)
        ann = build_ann_cell_c()
        h_ann = ann.fit(X_tr_sc, y_tr_raw, epochs=300, batch_size=32,
                        validation_split=0.2, callbacks=CALLBACKS, verbose=0)
        y_pred_ann = ann.predict(X_te_sc, verbose=0).flatten()
        m_ann = evaluate(y_te_raw, y_pred_ann)
        m_ann["train_r2"] = float(r2_score(y_tr_raw, ann.predict(X_tr_sc, verbose=0)))
        m_ann["seed"] = seed
        all_results["ann"][i].append(m_ann)
        if seed == 0:
            all_histories["ann"][i] = h_ann.history

        # ── LSTM (sequence input) ─────────────────────────────────────────
        X_tr_seq, y_tr_seq = make_sequences(X_tr_sc, y_tr_raw, WIN)
        X_te_seq, y_te_seq = make_sequences(X_te_sc, y_te_raw, WIN)

        tf.random.set_seed(seed)
        lstm = build_lstm(WIN)
        h_lstm = lstm.fit(X_tr_seq, y_tr_seq, epochs=300, batch_size=32,
                          validation_split=0.2, callbacks=CALLBACKS, verbose=0)
        y_pred_lstm = lstm.predict(X_te_seq, verbose=0).flatten()
        m_lstm = evaluate(y_te_seq, y_pred_lstm)
        m_lstm["train_r2"] = float(r2_score(y_tr_seq[WIN:] if len(y_tr_seq) > WIN else y_tr_seq,
                                            lstm.predict(X_tr_seq, verbose=0)[WIN:] if len(X_tr_seq) > WIN else lstm.predict(X_tr_seq, verbose=0)))
        m_lstm["residual_corr"] = float(
            np.corrcoef(y_pred_lstm, y_te_seq - y_pred_lstm)[0, 1])
        m_lstm["seed"] = seed
        all_results["lstm"][i].append(m_lstm)
        if seed == 0:
            all_histories["lstm"][i] = h_lstm.history

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'Dataset':<14} {'ANN R²':>8} {'±std':>6} {'LSTM R²':>8} {'±std':>6} {'Δ R²':>7} {'Winner'}")
print("-" * 65)

summary = {}
ann_means, lstm_means = [], []
for i in DATASETS:
    ann_r2s  = [r["r2"] for r in all_results["ann"][i]]
    lstm_r2s = [r["r2"] for r in all_results["lstm"][i]]
    a_m, a_s = np.mean(ann_r2s),  np.std(ann_r2s)
    l_m, l_s = np.mean(lstm_r2s), np.std(lstm_r2s)
    delta  = l_m - a_m
    winner = "LSTM ✅" if delta > 0.005 else ("ANN" if delta < -0.005 else "tie")
    ann_means.append(a_m); lstm_means.append(l_m)
    print(f"ds{i} ({TEMP_LABEL[i]})   {a_m:>8.4f} {a_s:>6.4f} {l_m:>8.4f} {l_s:>6.4f} {delta:>+7.4f}  {winner}")
    summary[i] = {"ann": {"r2_mean": round(a_m,4), "r2_std": round(a_s,4)},
                  "lstm": {"r2_mean": round(l_m,4), "r2_std": round(l_s,4)}}

print(f"{'Mean':<14} {np.mean(ann_means):>8.4f} {'':>6} {np.mean(lstm_means):>8.4f} {'':>6} "
      f"{np.mean(lstm_means)-np.mean(ann_means):>+7.4f}  "
      f"{'LSTM ✅' if np.mean(lstm_means) > np.mean(ann_means) else 'ANN'}")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'--- OVERFITTING CHECK (seed=0) ---':}")
print(f"{'Dataset':<14} {'Model':<8} {'Train R²':>9} {'Test R²':>9} {'Gap':>7} {'Status'}")
print("-" * 55)
for i in DATASETS:
    for model_name in ["ann", "lstm"]:
        r = all_results[model_name][i][0]
        gap = r["train_r2"] - r["r2"]
        status = "⚠️" if gap > 0.1 else "✅"
        print(f"ds{i} ({TEMP_LABEL[i]})   {model_name:<8} {r['train_r2']:>9.4f} {r['r2']:>9.4f} {gap:>7.4f}  {status}")

print(f"\n{'--- RESIDUAL CORRELATION (LSTM, seed=0) ---':}")
print(f"{'Dataset':<14} {'ANN corr':>10} {'LSTM corr':>11} {'Improved?'}")
print("-" * 45)
for i in DATASETS:
    ann_r  = all_results["ann"][i][0]
    lstm_r = all_results["lstm"][i][0]
    ann_corr = float(np.corrcoef(
        [r["r2"] for r in all_results["ann"][i]],
        [r["r2"] for r in all_results["ann"][i]])[0,1])
    # recalculate ANN residual corr from seed=0 predictions
    df = load_dataset(i)
    X_raw = df[INPUT_FEATURES].values
    y_raw = df[OUTPUT_FEATURE].values
    split = int(len(X_raw) * 0.7)
    X_tr, X_te = X_raw[:split], X_raw[split:]
    y_tr, y_te = y_raw[:split], y_raw[split:]
    sc = MinMaxScaler().fit(X_tr)
    X_te_sc = sc.transform(X_te)
    tf.random.set_seed(0)
    ann_m = build_ann_cell_c()
    ann_m.fit(sc.transform(X_tr), y_tr, epochs=300, batch_size=32,
              validation_split=0.2, callbacks=CALLBACKS, verbose=0)
    yp_ann = ann_m.predict(X_te_sc, verbose=0).flatten()
    ann_rc = float(np.corrcoef(yp_ann, y_te - yp_ann)[0,1])
    lstm_rc = lstm_r.get("residual_corr", float("nan"))
    improved = "✅ better" if abs(lstm_rc) < abs(ann_rc) else "❌ worse"
    print(f"ds{i} ({TEMP_LABEL[i]})   {ann_rc:>10.4f} {lstm_rc:>11.4f}  {improved}")

# ─────────────────────────────────────────────────────────────────────────────
# TimeSeriesSplit CV (LSTM only — the real test)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'--- TIME-SERIES CV (LSTM, 5 folds) ---':}")
print(f"{'Dataset':<14} {'CV R² mean':>11} {'±std':>7} {'min':>7} {'max':>7} {'Status'}")
print("-" * 55)

tscv = TimeSeriesSplit(n_splits=5)
cv_summary = {}
for i in DATASETS:
    df = load_dataset(i)
    X_raw = df[INPUT_FEATURES].values
    y_raw = df[OUTPUT_FEATURE].values

    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X_raw)
    X_seq, y_seq = make_sequences(X_sc, y_raw, WIN)

    fold_r2s = []
    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X_seq)):
        X_tr, X_te = X_seq[tr_idx], X_seq[te_idx]
        y_tr, y_te = y_seq[tr_idx], y_seq[te_idx]

        # refit scaler on train fold only
        sc_fold = MinMaxScaler()
        flat_tr = X_tr.reshape(-1, X_tr.shape[-1])
        sc_fold.fit(flat_tr)
        X_tr = sc_fold.transform(flat_tr).reshape(X_tr.shape)
        flat_te = X_te.reshape(-1, X_te.shape[-1])
        X_te = sc_fold.transform(flat_te).reshape(X_te.shape)

        tf.random.set_seed(SEEDS[0] + fold_idx)
        m = build_lstm(WIN)
        m.fit(X_tr, y_tr, epochs=300, batch_size=32,
              validation_split=0.2, callbacks=CALLBACKS, verbose=0)
        fold_r2s.append(float(r2_score(y_te, m.predict(X_te, verbose=0))))

    mean_r2, std_r2 = np.mean(fold_r2s), np.std(fold_r2s)
    status = "✅ stable" if std_r2 < 0.05 else ("⚠️" if std_r2 < 0.15 else "❌")
    print(f"ds{i} ({TEMP_LABEL[i]})   {mean_r2:>11.4f} {std_r2:>7.4f} "
          f"{min(fold_r2s):>7.4f} {max(fold_r2s):>7.4f}  {status}")
    cv_summary[i] = {"mean": round(mean_r2,4), "std": round(std_r2,4), "folds": fold_r2s}

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

# Loss curves
fig, axes = plt.subplots(2, len(DATASETS), figsize=(18, 7))
for j, i in enumerate(DATASETS):
    for row, model_name in enumerate(["ann", "lstm"]):
        h  = all_histories[model_name][i]
        ax = axes[row, j]
        ax.plot(h["loss"],     lw=0.9, label="Train")
        ax.plot(h["val_loss"], lw=0.9, label="Val")
        r2 = all_results[model_name][i][0]["r2"]
        ax.set_title(f"{model_name.upper()} ds{i} | R²={r2:.3f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("Epoch")
axes[0,0].set_ylabel("ANN Loss")
axes[1,0].set_ylabel("LSTM Loss")
fig.suptitle("Phase 5 — Train vs Val Loss (seed=0)", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
plt.close(fig)

# R² comparison bar chart
fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 4), sharey=True)
x = np.array([0, 1])
for j, i in enumerate(DATASETS):
    ax = axes[j]
    ann_r2s  = [r["r2"] for r in all_results["ann"][i]]
    lstm_r2s = [r["r2"] for r in all_results["lstm"][i]]
    ax.bar(0, np.mean(ann_r2s),  yerr=np.std(ann_r2s),  capsize=5,
           label="ANN (Cell C)", color="C0", alpha=0.85)
    ax.bar(1, np.mean(lstm_r2s), yerr=np.std(lstm_r2s), capsize=5,
           label="LSTM", color="C1", alpha=0.85)
    ax.set_xticks([0,1]); ax.set_xticklabels(["ANN","LSTM"])
    ax.set_title(f"ds{i} ({TEMP_LABEL[i]})", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("R²")
    ax.legend(fontsize=7)

fig.suptitle("Phase 5 — ANN vs LSTM (mean±std, 3 seeds)", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "r2_comparison.png", dpi=150)
plt.close(fig)

# Save
with open(RESULTS_DIR / "lstm_metrics.json", "w") as f:
    json.dump({"comparison": summary, "tscv": cv_summary}, f, indent=2)

print(f"\n[Phase 5 complete] → {RESULTS_DIR}")
