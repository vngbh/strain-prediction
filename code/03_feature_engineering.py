"""
Phase 3 — Feature Engineering
Goal: Measure the isolated impact of adding Temperature + derived features,
      using the same model architecture as baseline (fair comparison).

New features added:
  - Temperature
  - delta_R  = Electrode1_resistance - Electrode2_resistance
  - R_ratio  = Electrode1_resistance / Electrode2_resistance
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
RESULTS_DIR = Path("results/phase3_features")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

BASELINE_R2 = {0: 0.9790, 1: 0.2896, 2: 0.6170, 3: 0.9086, 4: 0.7554}


def add_features(df):
    df = df.copy()
    df["delta_R"] = df["Electrode1_resistance"] - df["Electrode2_resistance"]
    df["R_ratio"] = df["Electrode1_resistance"] / df["Electrode2_resistance"]
    return df


# Test 3 feature sets to understand contribution of each
FEATURE_SETS = {
    "baseline (E1+E2)":         INPUT_FEATURES,
    "+Temperature":              INPUT_FEATURES + ["Temperature"],
    "+delta_R +R_ratio":        INPUT_FEATURES + ["delta_R", "R_ratio"],
    "all features":              INPUT_FEATURES + ["Temperature", "delta_R", "R_ratio"],
}


def build_model(n_features: int):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse",
        metrics=["mae"],
    )
    return model


all_results = {name: {} for name in FEATURE_SETS}

print("=" * 70)
print("FEATURE ENGINEERING — Same architecture, different feature sets")
print("=" * 70)

for dataset_idx in DATASETS:
    df = load_dataset(dataset_idx)
    df = add_features(df)
    print(f"\n--- Dataset {dataset_idx} ---")

    for feat_name, features in FEATURE_SETS.items():
        scaler = MinMaxScaler()
        X = scaler.fit_transform(df[features].values)
        y = df[OUTPUT_FEATURE].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=SEED, shuffle=False
        )

        model = build_model(len(features))
        model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_split=0.2, verbose=0)

        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = evaluate(y_test, y_pred)
        all_results[feat_name][f"dataset{dataset_idx}"] = metrics

        delta = metrics["r2"] - BASELINE_R2[dataset_idx]
        sign = "+" if delta >= 0 else ""
        print(f"  [{feat_name:<30}] R²={metrics['r2']:.4f}  ({sign}{delta:.4f} vs baseline)")


# ── Summary table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("R² SUMMARY TABLE")
print("=" * 70)
header = f"{'Feature Set':<32}" + "".join(f"  ds{i}" for i in DATASETS)
print(header)
print("-" * 70)
for feat_name, results in all_results.items():
    row = f"{feat_name:<32}"
    for i in DATASETS:
        r2 = results.get(f"dataset{i}", {}).get("r2", 0)
        row += f"  {r2:.3f}"
    print(row)

# Save metrics
with open(RESULTS_DIR / "feature_metrics.json", "w") as f:
    json.dump(all_results, f, indent=2)

# ── Bar chart comparison ───────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(DATASETS), figsize=(20, 5), sharey=True)
x = range(len(FEATURE_SETS))
feat_labels = [name.replace(" ", "\n") for name in FEATURE_SETS]

for i, ax in zip(DATASETS, axes):
    r2_values = [all_results[f][f"dataset{i}"]["r2"] for f in FEATURE_SETS]
    bars = ax.bar(x, r2_values, color=[f"C{j}" for j in range(len(FEATURE_SETS))], alpha=0.8)
    ax.axhline(BASELINE_R2[i], color="red", linestyle="--", lw=1, label="baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels, fontsize=7)
    ax.set_title(f"Dataset {i}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²")
    ax.legend(fontsize=7)

fig.suptitle("R² by Feature Set per Dataset", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "feature_comparison.png", dpi=150)
plt.close(fig)

print(f"\n[Phase 3 Feature Engineering complete] Results saved to {RESULTS_DIR}")
