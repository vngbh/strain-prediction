"""
Phase 4 — Deep ANN with Dropout + BatchNorm
Goal: show that upgrading architecture (not just features) is what unlocks
      consistent performance across all temperatures.

Feature set: E1, E2, Temperature, delta_R, R_ratio (all 5)
Compare 3 architectures of increasing depth against baseline.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

from utils.data_loader import load_dataset, DATASETS, INPUT_FEATURES, OUTPUT_FEATURE
from utils.metrics import evaluate, print_metrics

SEED = 0
RESULTS_DIR = Path("results/phase4_deep_ann")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

BASELINE_R2 = {0: 0.9790, 1: 0.2896, 2: 0.6170, 3: 0.9086, 4: 0.7554}
ALL_FEATURES = INPUT_FEATURES + ["Temperature", "delta_R", "R_ratio"]


def add_features(df):
    df = df.copy()
    df["delta_R"] = df["Electrode1_resistance"] - df["Electrode2_resistance"]
    df["R_ratio"] = df["Electrode1_resistance"] / df["Electrode2_resistance"]
    return df


def build_shallow(n_features):
    """Baseline architecture with all features — same depth as mentor."""
    return tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="shallow")


def build_medium(n_features):
    """Medium depth — 3 hidden layers."""
    return tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="medium")


def build_deep(n_features):
    """Deep ANN — BN for stability, light Dropout to prevent overfitting."""
    return tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="deep_bn_dropout")


ARCHITECTURES = {
    "shallow (baseline arch)": (build_shallow, 1e-4),
    "medium (3 layers)":       (build_medium,  1e-3),
    "deep (BN + Dropout)":     (build_deep,    5e-4),
}

CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
]

# Store results AND histories in one pass
all_results   = {name: {} for name in ARCHITECTURES}
all_histories = {name: {} for name in ARCHITECTURES}

print("=" * 65)
print("PHASE 4 — Deep ANN  |  Features: E1, E2, Temp, ΔR, R_ratio")
print("=" * 65)

for dataset_idx in DATASETS:
    df = add_features(load_dataset(dataset_idx))
    print(f"\n--- Dataset {dataset_idx} ---")

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[ALL_FEATURES].values)
    y = df[OUTPUT_FEATURE].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, shuffle=False
    )

    for arch_name, (build_fn, lr) in ARCHITECTURES.items():
        tf.random.set_seed(SEED)
        model = build_fn(len(ALL_FEATURES))
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])

        history = model.fit(
            X_train, y_train,
            epochs=300,
            batch_size=64,
            validation_split=0.2,
            callbacks=CALLBACKS,
            verbose=0,
        )

        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = evaluate(y_test, y_pred)
        metrics["epochs_trained"] = len(history.history["loss"])
        all_results[arch_name][f"dataset{dataset_idx}"]   = metrics
        all_histories[arch_name][f"dataset{dataset_idx}"] = history.history

        delta = metrics["r2"] - BASELINE_R2[dataset_idx]
        sign = "+" if delta >= 0 else ""
        print(f"  [{arch_name:<30}] R²={metrics['r2']:.4f}  ({sign}{delta:.4f})  "
              f"epochs={metrics['epochs_trained']}")

# ── Summary table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("R² SUMMARY")
print("=" * 65)
print(f"{'Architecture':<32}" + "".join(f"  ds{i}" for i in DATASETS) + "  mean")
print("-" * 65)

base_vals = list(BASELINE_R2.values())
print(f"{'mentor baseline':<32}" +
      "".join(f"  {v:.3f}" for v in base_vals) +
      f"  {np.mean(base_vals):.3f}")

for arch_name, results in all_results.items():
    vals = [results[f"dataset{i}"]["r2"] for i in DATASETS]
    print(f"{arch_name:<32}" +
          "".join(f"  {v:.3f}" for v in vals) +
          f"  {np.mean(vals):.3f}")

# ── Loss curves (from saved histories) ───────────────────────────────────────

fig, axes = plt.subplots(len(ARCHITECTURES), len(DATASETS),
                          figsize=(22, 4 * len(ARCHITECTURES)))
for i, arch_name in enumerate(ARCHITECTURES):
    for j, dataset_idx in enumerate(DATASETS):
        h   = all_histories[arch_name][f"dataset{dataset_idx}"]
        r2  = all_results[arch_name][f"dataset{dataset_idx}"]["r2"]
        ax  = axes[i, j]
        ax.plot(h["loss"],     label="Train", lw=0.8)
        ax.plot(h["val_loss"], label="Val",   lw=0.8)
        ax.set_title(f"{arch_name}\nds{dataset_idx} | R²={r2:.3f}", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_xlabel("Epoch", fontsize=7)

fig.suptitle("Phase 4 — Loss Curves", fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
plt.close(fig)

# ── Bar chart ─────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(DATASETS))
width = 0.18

ax.bar(x - width, base_vals, width, label="mentor baseline", alpha=0.85)
for k, (arch_name, results) in enumerate(all_results.items()):
    vals = [results[f"dataset{i}"]["r2"] for i in DATASETS]
    ax.bar(x + (k * width), vals, width, label=arch_name, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([f"ds{i}" for i in DATASETS])
ax.set_ylabel("R²")
ax.set_ylim(0, 1.05)
ax.axhline(0.95, color="green", linestyle="--", lw=1, alpha=0.5, label="R²=0.95 target")
ax.set_title("Phase 4 — R² by Architecture")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "r2_comparison.png", dpi=150)
plt.close(fig)

with open(RESULTS_DIR / "deep_ann_metrics.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n[Phase 4 complete] Results saved to {RESULTS_DIR}")
