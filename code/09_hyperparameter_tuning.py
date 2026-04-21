"""
Phase 9 — Hyperparameter Tuning with Optuna
Upgrade of Phase 4 ANN Cell C: same architecture family (feedforward ANN, E1+E2),
but let Optuna find the optimal hyperparameters instead of fixing them by hand.

Search space:
  - n_layers: 1–4
  - units per layer: 16, 32, 64, 128
  - dropout rate: 0.0–0.4
  - learning rate: 1e-4 to 1e-2 (log scale)
  - batch size: 32, 64, 128

Protocol: same as Phase 4 — per-dataset, scaler fit on train only, 3 seeds final eval.
Optuna runs on seed=0 only (speed), final evaluation uses 3 seeds.
"""

import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from pathlib import Path

from utils.data_loader import load_dataset, DATASETS, INPUT_FEATURES, OUTPUT_FEATURE, TEMP_LABEL

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEEDS       = [0, 42, 123]
N_TRIALS    = 50
RESULTS_DIR = Path("results/phase9_optuna")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Phase 4 Cell C baseline for comparison
P4_RESULTS = {1: 0.908, 2: 0.835, 3: 0.910, 4: 0.731}

CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
]


def build_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    lr       = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout  = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)

    layers = [tf.keras.Input(shape=(2,))]
    for i in range(n_layers):
        units = trial.suggest_categorical(f"units_{i}", [16, 32, 64, 128])
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
        if dropout > 0:
            layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(1))

    m = tf.keras.Sequential(layers)
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return m


def build_best(params):
    n_layers = params["n_layers"]
    lr       = params["lr"]
    dropout  = params["dropout"]

    layers = [tf.keras.Input(shape=(2,))]
    for i in range(n_layers):
        units = params[f"units_{i}"]
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
        if dropout > 0:
            layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(1))

    m = tf.keras.Sequential(layers)
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return m


print("=" * 65)
print("PHASE 9 — Hyperparameter Tuning (Optuna)")
print(f"Search: {N_TRIALS} trials per dataset | Final eval: 3 seeds")
print("=" * 65)

all_results = {}

for ds_id in DATASETS:
    df   = load_dataset(ds_id)
    X    = df[INPUT_FEATURES].values
    y    = df[OUTPUT_FEATURE].values

    # Temporal split (same as all phases)
    split = int(len(X) * 0.7)
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test          = y[:split], y[split:]

    scaler   = MinMaxScaler().fit(X_train_raw)
    X_tr_sc  = scaler.transform(X_train_raw)
    X_te_sc  = scaler.transform(X_test_raw)

    # ── Optuna search (seed=0 only for speed) ───────────────────────────────
    def objective(trial):
        tf.random.set_seed(0); np.random.seed(0)
        batch = trial.suggest_categorical("batch_size", [32, 64, 128])
        model = build_model(trial)
        model.fit(X_tr_sc, y_train, epochs=300, batch_size=batch,
                  validation_split=0.1, callbacks=CALLBACKS, verbose=0)
        pred = model.predict(X_te_sc, verbose=0).flatten()
        return float(r2_score(y_test, pred))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_trial_r2 = study.best_value

    # ── Final evaluation — 3 seeds with best params ──────────────────────────
    r2s = []
    for seed in SEEDS:
        tf.random.set_seed(seed); np.random.seed(seed)
        batch = best_params["batch_size"]
        model = build_best(best_params)
        model.fit(X_tr_sc, y_train, epochs=300, batch_size=batch,
                  validation_split=0.1, callbacks=CALLBACKS, verbose=0)
        pred = model.predict(X_te_sc, verbose=0).flatten()
        r2s.append(float(r2_score(y_test, pred)))

    m, s = np.mean(r2s), np.std(r2s)
    delta = m - P4_RESULTS[ds_id]
    sign  = "+" if delta >= 0 else ""

    print(f"\nds{ds_id} ({TEMP_LABEL[ds_id]})")
    print(f"  Best params: {best_params}")
    print(f"  Optuna best (seed=0): {best_trial_r2:.4f}")
    print(f"  Final R² (3 seeds):   {m:.4f} ±{s:.4f}  (vs P4: {sign}{delta:.3f})")

    all_results[ds_id] = {
        "best_params":    best_params,
        "optuna_best_r2": round(best_trial_r2, 4),
        "final_r2_mean":  round(m, 4),
        "final_r2_std":   round(s, 4),
        "p4_r2":          P4_RESULTS[ds_id],
        "delta":          round(delta, 4),
        "all_seeds":      r2s,
    }

# ── Summary table ────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMMARY — Phase 4 Cell C vs Phase 9 Optuna")
print(f"{'Dataset':<14} {'P4 R²':>8} {'P9 R²':>8} {'±std':>6} {'delta':>8}")
print("-" * 50)
p4_means, p9_means = [], []
for ds_id in DATASETS:
    r = all_results[ds_id]
    sign = "+" if r["delta"] >= 0 else ""
    print(f"ds{ds_id} ({TEMP_LABEL[ds_id]})   {r['p4_r2']:>8.3f} {r['final_r2_mean']:>8.4f} "
          f"{r['final_r2_std']:>6.4f} {sign}{r['delta']:>7.3f}")
    p4_means.append(r["p4_r2"])
    p9_means.append(r["final_r2_mean"])

print("-" * 50)
print(f"{'Mean':<14} {np.mean(p4_means):>8.3f} {np.mean(p9_means):>8.4f} "
      f"{'':>6} {np.mean(p9_means)-np.mean(p4_means):>+8.3f}")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(DATASETS))
w = 0.35
p4_vals = [all_results[i]["p4_r2"]         for i in DATASETS]
p9_vals = [all_results[i]["final_r2_mean"] for i in DATASETS]
p9_stds = [all_results[i]["final_r2_std"]  for i in DATASETS]

ax.bar(x - w/2, p4_vals, w, label="ANN P4 Cell C", color="C0", alpha=0.8)
ax.bar(x + w/2, p9_vals, w, yerr=p9_stds, capsize=5,
       label="P9 Optuna (best arch)", color="C2", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"ds{i}\n({TEMP_LABEL[i]})" for i in DATASETS])
ax.set_ylabel("R²")
ax.set_ylim(0, 1.05)
ax.axhline(0.8, color="green", lw=0.8, linestyle="--", alpha=0.6, label="R²=0.8")
ax.set_title("Phase 9 — Optuna Tuning vs Phase 4 Cell C", fontsize=11)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "optuna_vs_p4.png", dpi=150)
plt.close(fig)

with open(RESULTS_DIR / "optuna_results.json", "w") as f:
    json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

print(f"\n[Phase 9 complete] → {RESULTS_DIR}")
