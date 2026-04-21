"""
Phase 3 — Feature Engineering (revised)
Same architecture as mentor (Dense 4). Only features change.
Fixes: scaler fit on X_train only, ds1–ds4, 3 seeds.
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
RESULTS_DIR = Path("results/phase3_features")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def add_features(df):
    df = df.copy()
    df["delta_R"] = df["Electrode1_resistance"] - df["Electrode2_resistance"]
    df["R_ratio"]  = df["Electrode1_resistance"] / df["Electrode2_resistance"]
    return df


FEATURE_SETS = {
    "baseline (E1+E2)":   INPUT_FEATURES,
    "+Temperature":        INPUT_FEATURES + ["Temperature"],
    "+delta_R +R_ratio":  INPUT_FEATURES + ["delta_R", "R_ratio"],
    "all 5 features":     INPUT_FEATURES + ["Temperature", "delta_R", "R_ratio"],
}


def build_mentor_arch(n_features):
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
    return m


all_runs = {feat: {i: [] for i in DATASETS} for feat in FEATURE_SETS}

print("=" * 65)
print("PHASE 3 — Feature Engineering (mentor arch, ds1–ds4, 3 seeds)")
print("=" * 65)

for seed in SEEDS:
    for i in DATASETS:
        df = add_features(load_dataset(i))
        y  = df[OUTPUT_FEATURE].values

        for feat_name, features in FEATURE_SETS.items():
            X = df[features].values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed, shuffle=False)

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            tf.random.set_seed(seed)
            model = build_mentor_arch(len(features))
            model.fit(X_train, y_train, epochs=50, batch_size=32,
                      validation_split=0.2, verbose=0)

            y_pred = model.predict(X_test, verbose=0).flatten()
            m = evaluate(y_test, y_pred)
            m["seed"] = seed
            all_runs[feat_name][i].append(m)

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'Feature Set':<24}" + "".join(f"  ds{i}±std" for i in DATASETS) + "   mean")
print("-" * 85)

summary = {}
for feat_name, ds_runs in all_runs.items():
    row = f"{feat_name:<24}"
    means = []
    per_ds = {}
    for i in DATASETS:
        r2s = [r["r2"] for r in ds_runs[i]]
        m, s = np.mean(r2s), np.std(r2s)
        per_ds[i] = {"r2_mean": m, "r2_std": s}
        means.append(m)
        row += f"  {m:.3f}±{s:.3f}"
    row += f"   {np.mean(means):.3f}"
    print(row)
    summary[feat_name] = {"per_dataset": per_ds, "overall_mean": np.mean(means)}

# ── Bar chart ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 5), sharey=True)
x = np.arange(len(FEATURE_SETS))
labels = [f.replace(" ", "\n") for f in FEATURE_SETS]

for j, i in enumerate(DATASETS):
    ax = axes[j]
    means  = [np.mean([r["r2"] for r in all_runs[f][i]]) for f in FEATURE_SETS]
    stds   = [np.std ([r["r2"] for r in all_runs[f][i]]) for f in FEATURE_SETS]
    bars   = ax.bar(x, means, yerr=stds, capsize=4,
                    color=[f"C{k}" for k in range(len(FEATURE_SETS))], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(f"ds{i} ({TEMP_LABEL[i]})", fontsize=9)
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("R²")
    ax.axhline(0, color="black", lw=0.5, linestyle="--")

fig.suptitle("Phase 3 — Feature Sets (mentor arch, mean±std, 3 seeds)", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "feature_comparison.png", dpi=150)
plt.close(fig)

with open(RESULTS_DIR / "feature_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[Phase 3 complete] → {RESULTS_DIR}")
