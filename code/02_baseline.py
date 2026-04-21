"""
Phase 2 — Baseline ANN (revised)
Reproduce mentor model exactly on ds1–ds4.
Fixes vs original: scaler fit on X_train only, 3 seeds for robustness.
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
from utils.metrics import evaluate, print_metrics

SEEDS = [0, 42, 123]
RESULTS_DIR = Path("results/phase2_baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
    return model


all_runs = {i: [] for i in DATASETS}

print("=" * 60)
print("BASELINE — ds1–ds4, 3 seeds")
print("=" * 60)

for seed in SEEDS:
    tf.random.set_seed(seed)
    np.random.seed(seed)
    for i in DATASETS:
        df = load_dataset(i)
        X = df[INPUT_FEATURES].values
        y = df[OUTPUT_FEATURE].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, shuffle=False)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        tf.random.set_seed(seed)
        model = build_model()
        history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                            validation_split=0.2, verbose=0)

        y_pred = model.predict(X_test, verbose=0).flatten()
        m = evaluate(y_test, y_pred)
        m["train_loss_final"] = float(history.history["loss"][-1])
        m["val_loss_final"]   = float(history.history["val_loss"][-1])
        m["seed"] = seed
        m["history"] = {"loss": history.history["loss"],
                         "val_loss": history.history["val_loss"]}
        all_runs[i].append(m)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'Dataset':<14} {'R² mean':>9} {'±std':>7} {'RMSE mean':>11} {'±std':>7} {'overfit?':>10}")
print("-" * 65)
summary = {}
for i in DATASETS:
    r2s   = [r["r2"]   for r in all_runs[i]]
    rmses = [r["rmse"] for r in all_runs[i]]
    train_l = np.mean([r["train_loss_final"] for r in all_runs[i]])
    val_l   = np.mean([r["val_loss_final"]   for r in all_runs[i]])
    gap     = (val_l - train_l) / (train_l + 1e-9)
    overfit = "⚠️ yes" if gap > 0.2 else "no"
    summary[i] = {"r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                  "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses)}
    print(f"ds{i} ({TEMP_LABEL[i]})   {np.mean(r2s):>9.4f} {np.std(r2s):>7.4f} "
          f"{np.mean(rmses):>11.6f} {np.std(rmses):>7.6f}  {overfit}")

print(f"\n{'Mean':<14} {np.mean([v['r2_mean'] for v in summary.values()]):>9.4f}")

# ── Loss curves (seed=0) ──────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 4))
for j, i in enumerate(DATASETS):
    h = all_runs[i][0]["history"]
    ax = axes[j]
    ax.plot(h["loss"],     label="Train", lw=1)
    ax.plot(h["val_loss"], label="Val",   lw=1)
    r2 = all_runs[i][0]["r2"]
    ax.set_title(f"ds{i} ({TEMP_LABEL[i]}) | R²={r2:.4f}", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlabel("Epoch")
fig.suptitle("Baseline — Train vs Val Loss (seed=0)", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
plt.close(fig)

save = {str(i): {"runs": all_runs[i], "summary": summary[i]} for i in DATASETS}
with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
    json.dump(save, f, indent=2)

print(f"\n[Phase 2 complete] → {RESULTS_DIR}")
