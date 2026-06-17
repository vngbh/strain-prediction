"""
Phase 4 — Deep ANN Ablation Study (revised)
2×2 ablation: {mentor features, all features} × {mentor arch, medium arch}
Each cell = 3 seeds → mean ± std.
Scaler fit on X_train only. ds1–ds4 only.

Input:  data/dataset{1-4}.csv — E1, E2 (+ derived features for some cells)
Output: results/phase4_deep_ann/
          loss_curves.png     — train vs val loss, all 4 cells × 4 datasets (seed=0)
          ablation_r2.png     — R² bar chart per cell per dataset
          ablation_metrics.json

Cells:
  A — mentor arch (4 units) + mentor features (E1+E2),  50 epochs, no callbacks
  B — mentor arch            + all features (5),         50 epochs, no callbacks
  C — medium arch (64-32-16) + mentor features,         300 epochs, EarlyStopping
  D — medium arch            + all features (5),        300 epochs, EarlyStopping
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

from utils.data_loader import load_dataset, DATASETS, INPUT_FEATURES, OUTPUT_FEATURE, TEMP_LABEL
from utils.metrics import evaluate

SEEDS = [0, 42, 123]
RESULTS_DIR = Path("results/phase4_deep_ann")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MENTOR_FEATURES = INPUT_FEATURES
ALL_FEATURES    = INPUT_FEATURES + ["Temperature", "delta_R", "R_ratio"]

CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
]


def add_features(df):
    df = df.copy()
    df["delta_R"] = df["Electrode1_resistance"] - df["Electrode2_resistance"]
    df["R_ratio"]  = df["Electrode1_resistance"] / df["Electrode2_resistance"]
    return df


def build_mentor_arch(n):
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(n,)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="mentor_arch")
    m.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
    return m


def build_medium_arch(n):
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(n,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="medium_arch")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


# 2×2 ablation cells
CELLS = {
    "A: mentor-arch + mentor-feat": (build_mentor_arch, MENTOR_FEATURES, 50,  False),
    "B: mentor-arch + all-feat":    (build_mentor_arch, ALL_FEATURES,    50,  False),
    "C: medium-arch + mentor-feat": (build_medium_arch, MENTOR_FEATURES, 300, True),
    "D: medium-arch + all-feat":    (build_medium_arch, ALL_FEATURES,    300, True),
}

all_runs     = {cell: {i: [] for i in DATASETS} for cell in CELLS}
all_histories= {cell: {i: [] for i in DATASETS} for cell in CELLS}

print("=" * 70)
print("PHASE 4 — 2×2 Ablation | ds1–ds4 | 3 seeds")
print("Rows: architecture  |  Columns: features")
print("=" * 70)

for seed in SEEDS:
    for i in DATASETS:
        df = add_features(load_dataset(i))
        y  = df[OUTPUT_FEATURE].values

        for cell_name, (build_fn, features, epochs, use_cb) in CELLS.items():
            X = df[features].values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed, shuffle=False)

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            tf.random.set_seed(seed)
            model = build_fn(len(features))
            callbacks = CALLBACKS if use_cb else []
            history = model.fit(
                X_train, y_train, epochs=epochs, batch_size=32,
                validation_split=0.2, callbacks=callbacks, verbose=0)

            y_pred = model.predict(X_test, verbose=0).flatten()
            m = evaluate(y_test, y_pred)
            m["seed"] = seed
            m["train_loss_final"] = float(history.history["loss"][-1])
            m["val_loss_final"]   = float(history.history["val_loss"][-1])
            all_runs[cell_name][i].append(m)
            if seed == 0:
                all_histories[cell_name][i] = history.history

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'Cell':<36}" + "".join(f"  ds{i}" for i in DATASETS) + "   mean")
print("-" * 70)

summary = {}
for cell_name, ds_runs in all_runs.items():
    row = f"{cell_name:<36}"
    means, per_ds = [], {}
    for i in DATASETS:
        r2s = [r["r2"] for r in ds_runs[i]]
        m, s = np.mean(r2s), np.std(r2s)
        per_ds[i] = {"r2_mean": round(m, 4), "r2_std": round(s, 4)}
        means.append(m)
        row += f"  {m:.3f}"
    row += f"   {np.mean(means):.3f}"
    print(row)
    summary[cell_name] = {"per_dataset": per_ds, "overall_mean": round(np.mean(means), 4)}

# ── Overfitting check ─────────────────────────────────────────────────────────

print(f"\n{'Cell':<36}  train_loss   val_loss   gap   overfit?")
print("-" * 75)
for cell_name, ds_runs in all_runs.items():
    train_l = np.mean([ds_runs[i][0]["train_loss_final"] for i in DATASETS])
    val_l   = np.mean([ds_runs[i][0]["val_loss_final"]   for i in DATASETS])
    gap = (val_l - train_l) / (train_l + 1e-9)
    flag = "⚠️ yes" if gap > 0.3 else "ok"
    print(f"{cell_name:<36}  {train_l:.6f}   {val_l:.6f}  {gap:+.2f}  {flag}")

# ── Loss curves (seed=0) ──────────────────────────────────────────────────────

fig, axes = plt.subplots(len(CELLS), len(DATASETS),
                          figsize=(18, 3.5 * len(CELLS)))
for row_idx, cell_name in enumerate(CELLS):
    for col_idx, i in enumerate(DATASETS):
        h  = all_histories[cell_name][i]
        r2 = all_runs[cell_name][i][0]["r2"]
        ax = axes[row_idx, col_idx]
        ax.plot(h["loss"],     lw=0.8, label="Train")
        ax.plot(h["val_loss"], lw=0.8, label="Val")
        ax.set_title(f"{cell_name[:20]}\nds{i} R²={r2:.3f}", fontsize=7)
        ax.legend(fontsize=6)
        ax.set_xlabel("Epoch", fontsize=7)

fig.suptitle("Phase 4 Ablation — Train vs Val Loss (seed=0)", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
plt.close(fig)

# ── Ablation bar chart ────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 5), sharey=True)
x = np.arange(len(CELLS))
labels = [c.split(":")[0] for c in CELLS]

for j, i in enumerate(DATASETS):
    ax = axes[j]
    means = [np.mean([r["r2"] for r in all_runs[c][i]]) for c in CELLS]
    stds  = [np.std ([r["r2"] for r in all_runs[c][i]]) for c in CELLS]
    colors = ["C0", "C1", "C2", "C3"]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"ds{i} ({TEMP_LABEL[i]})", fontsize=9)
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("R²")
    ax.axhline(0, color="black", lw=0.5, linestyle="--")

fig.suptitle("Phase 4 — 2×2 Ablation R² (mean±std, 3 seeds)", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "ablation_r2.png", dpi=150)
plt.close(fig)

with open(RESULTS_DIR / "ablation_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[Phase 4 complete] → {RESULTS_DIR}")
