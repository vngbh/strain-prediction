"""
Phase D-fix, part 4-6 — corrected, fair, nowcasting-only comparison.

Note: the ORIGINAL Phase D script (phase_d_simple_baselines.py) already used
flat per-row features X[t] -> y[t] (Task A / nowcasting) under a strict
experiment-block split, and NEVER built sliding windows — so it was never
affected by the window-boundary-crossing bug (that bug only exists in the
repo's own LSTM/CNN phases 5-8,10). This script re-confirms that result under
an explicit "Task A: nowcasting" label, restricted to exactly the 4 models
D-fix asks for, and saves the officially "corrected/authoritative" artifacts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_FIG = Path(__file__).resolve().parents[1] / "figures"
OUT_TAB = Path(__file__).resolve().parents[1] / "tables"

SEEDS = [0, 42, 123]
DATASETS = [1, 2, 3, 4]
TEMP_LABEL = {1: "33C", 2: "43C", 3: "52C", 4: "62C"}
KIND = {1: "extrapolation", 2: "interpolation", 3: "interpolation", 4: "extrapolation"}
E_COLS = ["Electrode1_resistance", "Electrode2_resistance"]


def load_raw(i):
    df = pd.read_csv(DATA_DIR / f"dataset{i}.csv")
    df["dataset_id"] = i
    return df


def add_interactions(df):
    df = df.copy()
    df["R1_T"] = df["Electrode1_resistance"] * df["Temperature"]
    df["R2_T"] = df["Electrode2_resistance"] * df["Temperature"]
    df["T2"] = df["Temperature"] ** 2
    return df


MODEL_FEATURES = {
    "linear_R1R2": E_COLS,
    "linear_R1R2T": E_COLS + ["Temperature"],
    "ridge_interactions": E_COLS + ["Temperature", "R1_T", "R2_T", "T2"],
    "deep_ann": E_COLS,
}


def build_ann():
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def fit_predict(model_name, X_tr, y_tr, X_te, seed):
    if model_name in ("linear_R1R2", "linear_R1R2T"):
        m = LinearRegression(); m.fit(X_tr, y_tr); return m.predict(X_te)
    if model_name == "ridge_interactions":
        m = Ridge(alpha=1.0, random_state=seed); m.fit(X_tr, y_tr); return m.predict(X_te)
    if model_name == "deep_ann":
        tf.random.set_seed(seed); np.random.seed(seed)
        m = build_ann()
        cb = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
              tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0)]
        m.fit(X_tr, y_tr, epochs=300, batch_size=32, validation_split=0.2, callbacks=cb, verbose=0)
        return m.predict(X_te, verbose=0).flatten()
    raise ValueError(model_name)


def metrics(y_true, y_pred):
    return {"r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred))}


rows = []
loo_predictions_43C = {}

print("=" * 70)
print("D-FIX — corrected, nowcasting-only, strict-split comparison")
print("=" * 70)

# D1: same-temperature, strict experiment-block holdout
for i in DATASETS:
    df = add_interactions(load_raw(i))
    exps = sorted(df["Experiment"].unique())
    train_df, test_df = df[df.Experiment == exps[0]], df[df.Experiment == exps[1]]
    y_tr_raw, y_te_raw = train_df["Strain"].values, test_df["Strain"].values
    for model_name, feats in MODEL_FEATURES.items():
        r2s = []
        for seed in SEEDS:
            scaler = MinMaxScaler().fit(train_df[feats].values)
            X_tr, X_te = scaler.transform(train_df[feats].values), scaler.transform(test_df[feats].values)
            y_pred = fit_predict(model_name, X_tr, y_tr_raw, X_te, seed)
            r2s.append(metrics(y_te_raw, y_pred)["r2"])
        m_final = metrics(y_te_raw, y_pred); m_final["r2_mean"] = float(np.mean(r2s)); m_final["r2_std"] = float(np.std(r2s))
        rows.append({"phase": "D1_same_temperature", "task": "nowcasting", "model": model_name,
                     "dataset": i, "temperature": TEMP_LABEL[i], "kind": "same-temperature",
                     "r2_mean": m_final["r2_mean"], "r2_std": m_final["r2_std"], "rmse": m_final["rmse"], "mae": m_final["mae"]})

# D2: leave-one-temperature-out
for test_id in DATASETS:
    train_ids = [i for i in DATASETS if i != test_id]
    train_df = add_interactions(pd.concat([load_raw(i) for i in train_ids], ignore_index=True))
    test_df = add_interactions(load_raw(test_id))
    y_tr_raw, y_te_raw = train_df["Strain"].values, test_df["Strain"].values
    for model_name, feats in MODEL_FEATURES.items():
        r2s, last_pred = [], None
        for seed in SEEDS:
            scaler = MinMaxScaler().fit(train_df[feats].values)
            X_tr, X_te = scaler.transform(train_df[feats].values), scaler.transform(test_df[feats].values)
            y_pred = fit_predict(model_name, X_tr, y_tr_raw, X_te, seed)
            r2s.append(metrics(y_te_raw, y_pred)["r2"]); last_pred = y_pred
        m_final = metrics(y_te_raw, last_pred); m_final["r2_mean"] = float(np.mean(r2s)); m_final["r2_std"] = float(np.std(r2s))
        rows.append({"phase": "D2_leave_one_temperature_out", "task": "nowcasting", "model": model_name,
                     "dataset": test_id, "temperature": TEMP_LABEL[test_id], "kind": KIND[test_id],
                     "r2_mean": m_final["r2_mean"], "r2_std": m_final["r2_std"], "rmse": m_final["rmse"], "mae": m_final["mae"]})
        print(f"LOO test=ds{test_id}({TEMP_LABEL[test_id]},{KIND[test_id]}) {model_name:<20} R2={m_final['r2_mean']:.4f}")
        if test_id == 2:
            loo_predictions_43C[model_name] = {"y_true": y_te_raw, "y_pred": last_pred}

res_df = pd.DataFrame(rows)
res_df.to_csv(OUT_TAB / "simple_baseline_results_corrected.csv", index=False)
print("\nSaved -> tables/simple_baseline_results_corrected.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, (model_name, pred) in zip(axes.flat, loo_predictions_43C.items()):
    ax.plot(pred["y_true"], lw=0.8, label="measured", alpha=0.8)
    ax.plot(pred["y_pred"], lw=0.8, label="predicted", alpha=0.8)
    r2 = [r for r in rows if r["model"] == model_name and r["dataset"] == 2 and r["phase"].startswith("D2")][0]["r2_mean"]
    ax.set_title(f"{model_name} | LOO 43C (nowcasting, corrected) | R2={r2:.3f}", fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle("Corrected LOO predictions at 43C (Task A: nowcasting)")
fig.tight_layout()
fig.savefig(OUT_FIG / "loo_predictions_43C_corrected.png", dpi=150)
plt.close(fig)

r2_linT_43 = [r for r in rows if r["model"] == "linear_R1R2T" and r["dataset"] == 2 and r["phase"].startswith("D2")][0]["r2_mean"]
print(f"\nConfirm: Linear+T at 43C after fixes -> R2 = {r2_linT_43:.4f} ({'POSITIVE' if r2_linT_43 > 0 else 'still negative'})")
print("[D-fix corrected baselines complete]")
