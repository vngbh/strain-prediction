"""
Phase C — Reproduce the original paper-style baseline.

Design decisions (documented, since the brief leaves some details open):
- Train pool = ds1 + ds3 + ds4 (33C, 52C, 62C). ds2 (43C) is excluded from
  training entirely and used as the unseen-temperature interpolation test set,
  matching "exclude ~43C from training and test on it".
- The requested 70/15/15 split is applied to the TRAIN POOL only (train/val/
  "within-pool test"); ds2 is a separate, always-held-out evaluation set so
  its result is directly comparable to Phase 8's LOO(ds2) number.
- C1 (reproduction-style): 70/15/15 split via pure random shuffling of pool
  rows (matches a naive/original-paper-style split, ignores experiment
  boundaries -> can leak near-duplicate adjacent points between train/test).
- C2 (strict): 70/15/15 split by whole experiment block (no adjacent-sample
  leakage), assigned so each split gets whole experiments only.
- Both inputs (E1, E2) and outputs (Temperature, Strain) are scaled to
  [-1, 1] using a scaler fit on the TRAIN split only, then applied to
  val/test/ds2. Architecture: 3 hidden Dense(tanh) layers, output Dense(2)
  with tanh (matches the paper's stated activation and output range).
- Metrics reported after inverse-transforming predictions back to original
  units (RMSE/MAE in raw Strain / raw Celsius; R2 dimensionless either way).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_FIG = Path(__file__).resolve().parents[1] / "figures"
OUT_TAB = Path(__file__).resolve().parents[1] / "tables"

SEEDS = [0, 42, 123]
TRAIN_DS = [1, 3, 4]     # 33C, 52C, 62C
TEST_DS  = 2             # 43C, held out entirely
TEMP_LABEL = {1: "33C", 2: "43C", 3: "52C", 4: "62C"}
INPUT_FEATURES = ["Electrode1_resistance", "Electrode2_resistance"]
OUTPUT_FEATURES = ["Temperature", "Strain"]


def load_raw(i):
    df = pd.read_csv(DATA_DIR / f"dataset{i}.csv")
    df["dataset_id"] = i
    return df


def build_model():
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(16, activation="tanh"),
        tf.keras.layers.Dense(8, activation="tanh"),
        tf.keras.layers.Dense(2, activation="tanh"),
    ], name="paper_repro")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def metrics_orig_units(y_true, y_pred):
    out = {}
    for k, name in enumerate(OUTPUT_FEATURES):
        yt, yp = y_true[:, k], y_pred[:, k]
        out[name] = {
            "r2":   float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae":  float(mean_absolute_error(yt, yp)),
        }
    return out


def experiment_block_split(df, frac_train=0.7, frac_val=0.15, seed=0):
    """Assign whole (dataset_id, Experiment) blocks to train/val/test so fractions
    of TOTAL ROWS approximate the requested 70/15/15 (no adjacent-sample leakage).
    With only a handful of blocks available, counts are rounded but at least one
    block is always guaranteed for val and test."""
    blocks = (df.groupby(["dataset_id", "Experiment"]).size()
              .reset_index(name="n").sample(frac=1.0, random_state=seed).reset_index(drop=True))
    n_blocks = len(blocks)
    n_val  = max(1, round(n_blocks * frac_val))
    n_test = max(1, round(n_blocks * (1 - frac_train - frac_val)))
    n_train = n_blocks - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Not enough experiment blocks ({n_blocks}) for a 70/15/15 split")
    split_labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    blocks["split"] = split_labels
    merged = df.merge(blocks[["dataset_id", "Experiment", "split"]], on=["dataset_id", "Experiment"])
    return (merged[merged.split == "train"].drop(columns="split"),
            merged[merged.split == "val"].drop(columns="split"),
            merged[merged.split == "test"].drop(columns="split"))


def random_row_split(df, frac_train=0.7, frac_val=0.15, seed=0):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_tr, n_val = int(n * frac_train), int(n * (frac_train + frac_val))
    return shuffled.iloc[:n_tr], shuffled.iloc[n_tr:n_val], shuffled.iloc[n_val:]


def run_variant(label, split_fn, subsample_1_20=False, seeds=SEEDS):
    pool = pd.concat([load_raw(i) for i in TRAIN_DS], ignore_index=True)
    ds2  = load_raw(TEST_DS)

    if subsample_1_20:
        pool = pool.iloc[::20].reset_index(drop=True)

    results = []
    example_pred = None
    for seed in seeds:
        train_df, val_df, test_df = split_fn(pool, seed=seed)

        x_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_df[INPUT_FEATURES].values)
        y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_df[OUTPUT_FEATURES].values)

        X_tr = x_scaler.transform(train_df[INPUT_FEATURES].values)
        y_tr = y_scaler.transform(train_df[OUTPUT_FEATURES].values)
        X_val = x_scaler.transform(val_df[INPUT_FEATURES].values)
        y_val = y_scaler.transform(val_df[OUTPUT_FEATURES].values)
        X_test = x_scaler.transform(test_df[INPUT_FEATURES].values)
        X_ds2  = x_scaler.transform(ds2[INPUT_FEATURES].values)

        tf.random.set_seed(seed); np.random.seed(seed)
        model = build_model()
        cb = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
              tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0)]
        model.fit(X_tr, y_tr, epochs=300, batch_size=64, validation_data=(X_val, y_val),
                  callbacks=cb, verbose=0)

        y_pred_test = y_scaler.inverse_transform(model.predict(X_test, verbose=0))
        y_pred_ds2  = y_scaler.inverse_transform(model.predict(X_ds2, verbose=0))

        m_test = metrics_orig_units(test_df[OUTPUT_FEATURES].values, y_pred_test)
        m_ds2  = metrics_orig_units(ds2[OUTPUT_FEATURES].values, y_pred_ds2)

        results.append({"seed": seed, "within_pool_test": m_test, "ds2_43C_unseen": m_ds2,
                         "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

        if seed == seeds[0]:
            example_pred = {"y_true": ds2[OUTPUT_FEATURES].values, "y_pred": y_pred_ds2}

    summary = {"label": label, "subsample_1_20": subsample_1_20, "runs": results}
    for split_name in ["within_pool_test", "ds2_43C_unseen"]:
        for out_name in OUTPUT_FEATURES:
            r2s = [r[split_name][out_name]["r2"] for r in results]
            summary[f"{split_name}_{out_name}_r2_mean"] = float(np.mean(r2s))
            summary[f"{split_name}_{out_name}_r2_std"] = float(np.std(r2s))
            rmses = [r[split_name][out_name]["rmse"] for r in results]
            summary[f"{split_name}_{out_name}_rmse_mean"] = float(np.mean(rmses))
    return summary, example_pred


print("=" * 70)
print("PHASE C — Paper-style reproduction (train on 33/52/62C, test on 43C)")
print("=" * 70)

all_summaries = []
examples = {}

for label, split_fn, subsample in [
    ("C1_random_split", random_row_split, False),
    ("C1_random_split_1of20", random_row_split, True),
    ("C2_strict_block_split", experiment_block_split, False),
]:
    print(f"\n--- {label} ---")
    summary, example = run_variant(label, split_fn, subsample_1_20=subsample)
    all_summaries.append(summary)
    examples[label] = example
    print(f"  within-pool test: Temp R2={summary['within_pool_test_Temperature_r2_mean']:.4f}  "
          f"Strain R2={summary['within_pool_test_Strain_r2_mean']:.4f}")
    print(f"  ds2 (43C) unseen: Temp R2={summary['ds2_43C_unseen_Temperature_r2_mean']:.4f}  "
          f"Strain R2={summary['ds2_43C_unseen_Strain_r2_mean']:.4f}")

with open(OUT_TAB / "paper_reproduction_results.json", "w") as f:
    json.dump(all_summaries, f, indent=2)

flat_rows = []
for s in all_summaries:
    row = {"label": s["label"], "subsample_1_20": s["subsample_1_20"]}
    for k, v in s.items():
        if k.endswith(("_r2_mean", "_r2_std", "_rmse_mean")):
            row[k] = v
    flat_rows.append(row)
pd.DataFrame(flat_rows).to_csv(OUT_TAB / "paper_reproduction_results.csv", index=False)
print("\nSaved -> tables/paper_reproduction_results.csv")

# ── Plots: predicted vs measured strain over "time" (row index) for ds2 ───────

for label, ex in examples.items():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for k, name in enumerate(OUTPUT_FEATURES):
        ax = axes[k]
        ax.plot(ex["y_true"][:, k], lw=0.8, label="measured", alpha=0.8)
        ax.plot(ex["y_pred"][:, k], lw=0.8, label="predicted", alpha=0.8)
        ax.set_title(f"{label} — ds2 (43C) unseen — {name}")
        ax.set_xlabel("Row index (time order)"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"paper_reproduction_{label}_ds2_43C.png", dpi=150)
    plt.close(fig)

print("[Phase C complete]")
