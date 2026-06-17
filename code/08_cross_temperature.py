"""
Phase 8 — Cross-temperature Generalization
Key question: can the model predict strain at temperatures never seen in training?

Two test setups:
  1. Leave-one-out: train on 3 datasets, test on the remaining 1 (4 combinations)
  2. Extrapolation: train on lower temps (ds1+ds2+ds3), test on highest (ds4)
     — hardest case, most realistic deployment scenario

Models tested: ANN P4 Cell C vs LSTM P5 (the two best models)

Input:  data/dataset{1-4}.csv — combined across datasets for cross-temp training
Output: results/phase8_cross_temp/
          cross_temp_results.png — LOO bar + within vs cross gap chart
          cross_temp_metrics.json — LOO results + extrapolation results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from pathlib import Path
from itertools import combinations

from utils.data_loader import load_dataset, DATASETS, INPUT_FEATURES, OUTPUT_FEATURE, TEMP_LABEL
from utils.metrics import evaluate

SEEDS       = [0, 42, 123]
WIN         = 20
RESULTS_DIR = Path("results/phase8_cross_temp")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_ann():
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="ann_p4")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def build_lstm():
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(WIN, 2)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="lstm_p5")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
]


def load_combined(train_ids):
    """Concatenate raw X and y arrays from multiple datasets for cross-temp training."""
    dfs = [load_dataset(i) for i in train_ids]
    X = np.vstack([df[INPUT_FEATURES].values for df in dfs])
    y = np.concatenate([df[OUTPUT_FEATURE].values for df in dfs])
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Setup 1 — Leave-One-Out
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("PHASE 8 — Cross-temperature Generalization")
print("=" * 65)
print("\n[Setup 1] Leave-One-Out: train on 3 datasets, test on 1 unseen")
print(f"{'Test DS':<14} {'Train on':^20} {'ANN R²':>8} {'±std':>6} {'LSTM R²':>8} {'±std':>6} {'Winner'}")
print("-" * 72)

loo_results = {}

for test_id in DATASETS:
    train_ids = [i for i in DATASETS if i != test_id]
    train_label = "+".join([f"ds{i}" for i in train_ids])

    # Load test data (full dataset)
    df_test = load_dataset(test_id)
    X_test_raw = df_test[INPUT_FEATURES].values
    y_test     = df_test[OUTPUT_FEATURE].values

    # Load train data (combined)
    X_train_raw, y_train = load_combined(train_ids)

    ann_r2s, lstm_r2s = [], []

    for seed in SEEDS:
        tf.random.set_seed(seed); np.random.seed(seed)

        scaler = MinMaxScaler().fit(X_train_raw)
        X_tr_sc = scaler.transform(X_train_raw)
        X_te_sc = scaler.transform(X_test_raw)

        # ANN
        tf.random.set_seed(seed)
        ann = build_ann()
        ann.fit(X_tr_sc, y_train, epochs=300, batch_size=64,
                validation_split=0.1, callbacks=CALLBACKS, verbose=0)
        ann_r2s.append(float(r2_score(y_test, ann.predict(X_te_sc, verbose=0))))

        # LSTM
        X_tr_seq, y_tr_seq = make_sequences(X_tr_sc, y_train, WIN)
        X_te_seq, y_te_seq = make_sequences(X_te_sc, y_test,  WIN)
        tf.random.set_seed(seed)
        lstm = build_lstm()
        lstm.fit(X_tr_seq, y_tr_seq, epochs=300, batch_size=64,
                 validation_split=0.1, callbacks=CALLBACKS, verbose=0)
        lstm_r2s.append(float(r2_score(y_te_seq, lstm.predict(X_te_seq, verbose=0))))

    a_m, a_s = np.mean(ann_r2s),  np.std(ann_r2s)
    l_m, l_s = np.mean(lstm_r2s), np.std(lstm_r2s)
    winner = "ANN ✅" if a_m > l_m + 0.005 else ("LSTM ✅" if l_m > a_m + 0.005 else "tie")
    print(f"ds{test_id} ({TEMP_LABEL[test_id]})   {train_label:^20} {a_m:>8.4f} {a_s:>6.4f} "
          f"{l_m:>8.4f} {l_s:>6.4f}  {winner}")
    loo_results[test_id] = {"train_on": train_ids,
                             "ann":  {"mean": round(a_m,4), "std": round(a_s,4)},
                             "lstm": {"mean": round(l_m,4), "std": round(l_s,4)}}

# ─────────────────────────────────────────────────────────────────────────────
# Setup 2 — Temperature Extrapolation (hardest case)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[Setup 2] Extrapolation: train ds1+ds2+ds3 (33–52°C) → test ds4 (62°C)")
print("This is the hardest case — model must generalize to highest unseen temperature")
print(f"{'Model':<8} {'R² mean':>9} {'±std':>7} {'min':>7} {'max':>7}  {'Verdict'}")
print("-" * 55)

extrap_results = {}
train_ids_extrap = [1, 2, 3]
test_id_extrap   = 4

df_test = load_dataset(test_id_extrap)
X_test_raw = df_test[INPUT_FEATURES].values
y_test_ext = df_test[OUTPUT_FEATURE].values
X_train_raw_ext, y_train_ext = load_combined(train_ids_extrap)

for model_name, build_fn, use_seq in [("ANN", build_ann, False), ("LSTM", build_lstm, True)]:
    r2s = []
    for seed in SEEDS:
        tf.random.set_seed(seed); np.random.seed(seed)
        scaler = MinMaxScaler().fit(X_train_raw_ext)
        X_tr_sc = scaler.transform(X_train_raw_ext)
        X_te_sc = scaler.transform(X_test_raw)

        tf.random.set_seed(seed)
        model = build_fn()
        if use_seq:
            X_tr_s, y_tr_s = make_sequences(X_tr_sc, y_train_ext, WIN)
            X_te_s, y_te_s = make_sequences(X_te_sc, y_test_ext, WIN)
            model.fit(X_tr_s, y_tr_s, epochs=300, batch_size=64,
                      validation_split=0.1, callbacks=CALLBACKS, verbose=0)
            r2s.append(float(r2_score(y_te_s, model.predict(X_te_s, verbose=0))))
        else:
            model.fit(X_tr_sc, y_train_ext, epochs=300, batch_size=64,
                      validation_split=0.1, callbacks=CALLBACKS, verbose=0)
            r2s.append(float(r2_score(y_test_ext, model.predict(X_te_sc, verbose=0))))

    m, s = np.mean(r2s), np.std(r2s)
    verdict = "✅ generalizes" if m > 0.8 else ("⚠️ partial" if m > 0.6 else "❌ fails")
    print(f"{model_name:<8} {m:>9.4f} {s:>7.4f} {min(r2s):>7.4f} {max(r2s):>7.4f}  {verdict}")
    extrap_results[model_name] = {"r2_mean": round(m,4), "r2_std": round(s,4), "runs": r2s}

# ─────────────────────────────────────────────────────────────────────────────
# Compare: within-dataset vs cross-dataset R²
# ─────────────────────────────────────────────────────────────────────────────

WITHIN_ANN  = {1: 0.908, 2: 0.835, 3: 0.910, 4: 0.731}
WITHIN_LSTM = {1: 0.937, 2: 0.859, 3: 0.843, 4: 0.663}

print(f"\n[Generalization Gap] Within-dataset vs Cross-dataset R²")
print(f"{'Dataset':<14} {'ANN within':>11} {'ANN cross':>10} {'gap':>6} | "
      f"{'LSTM within':>11} {'LSTM cross':>10} {'gap':>6}")
print("-" * 75)
for test_id in DATASETS:
    a_cross = loo_results[test_id]["ann"]["mean"]
    l_cross = loo_results[test_id]["lstm"]["mean"]
    a_gap   = WITHIN_ANN[test_id]  - a_cross
    l_gap   = WITHIN_LSTM[test_id] - l_cross
    print(f"ds{test_id} ({TEMP_LABEL[test_id]})   "
          f"{WITHIN_ANN[test_id]:>11.3f} {a_cross:>10.4f} {a_gap:>+6.3f} | "
          f"{WITHIN_LSTM[test_id]:>11.3f} {l_cross:>10.4f} {l_gap:>+6.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LOO bar chart
ax = axes[0]
x  = np.arange(len(DATASETS))
w  = 0.35
ann_means  = [loo_results[i]["ann"]["mean"]  for i in DATASETS]
lstm_means = [loo_results[i]["lstm"]["mean"] for i in DATASETS]
ann_stds   = [loo_results[i]["ann"]["std"]   for i in DATASETS]
lstm_stds  = [loo_results[i]["lstm"]["std"]  for i in DATASETS]
ax.bar(x - w/2, ann_means,  w, yerr=ann_stds,  capsize=4, label="ANN P4",  color="C0", alpha=0.85)
ax.bar(x + w/2, lstm_means, w, yerr=lstm_stds, capsize=4, label="LSTM P5", color="C1", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"ds{i}\n({TEMP_LABEL[i]})" for i in DATASETS], fontsize=8)
ax.set_ylabel("R²"); ax.set_ylim(0, 1.05)
ax.axhline(0.8, color="green", lw=0.8, linestyle="--", alpha=0.6, label="R²=0.8")
ax.set_title("Leave-One-Out: ANN vs LSTM", fontsize=10)
ax.legend(fontsize=8)

# Within vs cross comparison
ax = axes[1]
x  = np.arange(len(DATASETS))
ax.bar(x - w/2, list(WITHIN_ANN.values()),  w, label="ANN within",  color="C0", alpha=0.5)
ax.bar(x - w/2, ann_means,                  w, label="ANN cross",   color="C0", alpha=0.85)
ax.bar(x + w/2, list(WITHIN_LSTM.values()), w, label="LSTM within", color="C1", alpha=0.5)
ax.bar(x + w/2, lstm_means,                 w, label="LSTM cross",  color="C1", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"ds{i}\n({TEMP_LABEL[i]})" for i in DATASETS], fontsize=8)
ax.set_ylabel("R²"); ax.set_ylim(0, 1.05)
ax.set_title("Within vs Cross-dataset R²\n(light=within, dark=cross)", fontsize=10)
ax.legend(fontsize=7)

fig.suptitle("Phase 8 — Cross-temperature Generalization", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "cross_temp_results.png", dpi=150)
plt.close(fig)

with open(RESULTS_DIR / "cross_temp_metrics.json", "w") as f:
    json.dump({"loo": loo_results, "extrapolation": extrap_results}, f, indent=2)

print(f"\n[Phase 8 complete] → {RESULTS_DIR}")
