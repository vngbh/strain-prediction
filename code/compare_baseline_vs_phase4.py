"""
Fair comparison: mentor baseline vs Phase 4 medium model.
Same train/test split, same data, run in one script.
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
from utils.metrics import evaluate

SEED = 0
RESULTS_DIR = Path("results/comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

TEMP_LABEL = {0: "21°C", 1: "33°C", 2: "43°C", 3: "52°C", 4: "62°C"}


def add_features(df):
    df = df.copy()
    df["delta_R"] = df["Electrode1_resistance"] - df["Electrode2_resistance"]
    df["R_ratio"]  = df["Electrode1_resistance"] / df["Electrode2_resistance"]
    return df


def build_mentor():
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="mentor_baseline")
    m.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
    return m


def build_medium(n_features):
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="medium_ann")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
]

ALL_FEATURES = INPUT_FEATURES + ["Temperature", "delta_R", "R_ratio"]

results = {}

print("=" * 70)
print(f"{'Dataset':<12} {'Model':<20} {'R²':>7} {'RMSE':>10} {'MAE':>10} {'MSE':>12}")
print("=" * 70)

for i in DATASETS:
    df_raw = load_dataset(i)
    df_aug = add_features(df_raw)

    # ── Mentor model (2 raw features, no shuffle, 50 epochs) ─────────────
    scaler_m = MinMaxScaler()
    X_m = scaler_m.fit_transform(df_raw[INPUT_FEATURES].values)
    y   = df_raw[OUTPUT_FEATURE].values
    X_tr_m, X_te_m, y_tr, y_te = train_test_split(
        X_m, y, test_size=0.3, random_state=SEED, shuffle=False)

    tf.random.set_seed(SEED)
    mentor = build_mentor()
    mentor.fit(X_tr_m, y_tr, epochs=50, batch_size=32,
               validation_split=0.2, verbose=0)
    y_pred_m = mentor.predict(X_te_m, verbose=0).flatten()
    m_metrics = evaluate(y_te, y_pred_m)

    # ── Phase 4 medium model (5 features, callbacks) ──────────────────────
    scaler_p = MinMaxScaler()
    X_p = scaler_p.fit_transform(df_aug[ALL_FEATURES].values)
    X_tr_p, X_te_p, _, _ = train_test_split(
        X_p, y, test_size=0.3, random_state=SEED, shuffle=False)

    tf.random.set_seed(SEED)
    phase4 = build_medium(len(ALL_FEATURES))
    phase4.fit(X_tr_p, y_tr, epochs=300, batch_size=64,
               validation_split=0.2, callbacks=CALLBACKS, verbose=0)
    y_pred_p = phase4.predict(X_te_p, verbose=0).flatten()
    p_metrics = evaluate(y_te, y_pred_p)

    results[i] = {"mentor": m_metrics, "phase4": p_metrics,
                  "y_te": y_te.tolist(),
                  "y_pred_m": y_pred_m.tolist(),
                  "y_pred_p": y_pred_p.tolist()}

    label = f"ds{i} ({TEMP_LABEL[i]})"
    print(f"{label:<12} {'mentor baseline':<20} "
          f"{m_metrics['r2']:>7.4f} {m_metrics['rmse']:>10.6f} "
          f"{m_metrics['mae']:>10.6f} {m_metrics['mse']:>12.8f}")
    print(f"{'':12} {'phase4 medium':<20} "
          f"{p_metrics['r2']:>7.4f} {p_metrics['rmse']:>10.6f} "
          f"{p_metrics['mae']:>10.6f} {p_metrics['mse']:>12.8f}")

    r2_delta   = p_metrics['r2']   - m_metrics['r2']
    rmse_delta = p_metrics['rmse'] - m_metrics['rmse']
    sign_r2    = "▲" if r2_delta   >= 0 else "▼"
    sign_rmse  = "▼" if rmse_delta <= 0 else "▲"
    print(f"{'':12} {'Δ (phase4 - mentor)':<20} "
          f"{sign_r2} {r2_delta:+.4f}{'':>4} "
          f"{sign_rmse} {rmse_delta:+.6f}")
    print("-" * 70)


# ── Summary ───────────────────────────────────────────────────────────────────

print("\nSUMMARY — R²")
print(f"{'Dataset':<15} {'Mentor':>8} {'Phase4':>8} {'Δ R²':>8} {'Winner'}")
print("-" * 50)

mentor_r2s = []
phase4_r2s = []
for i in DATASETS:
    mr2 = results[i]["mentor"]["r2"]
    pr2 = results[i]["phase4"]["r2"]
    mentor_r2s.append(mr2)
    phase4_r2s.append(pr2)
    winner = "phase4 ✅" if pr2 > mr2 else "mentor"
    print(f"ds{i} ({TEMP_LABEL[i]})    {mr2:>8.4f} {pr2:>8.4f} {pr2-mr2:>+8.4f}  {winner}")

print("-" * 50)
print(f"{'Mean':<15} {np.mean(mentor_r2s):>8.4f} {np.mean(phase4_r2s):>8.4f} "
      f"{np.mean(phase4_r2s)-np.mean(mentor_r2s):>+8.4f}  "
      f"{'phase4 ✅' if np.mean(phase4_r2s) > np.mean(mentor_r2s) else 'mentor'}")

print("\nSUMMARY — RMSE")
print(f"{'Dataset':<15} {'Mentor':>10} {'Phase4':>10} {'Δ RMSE':>10} {'Winner'}")
print("-" * 55)
mentor_rmses, phase4_rmses = [], []
for i in DATASETS:
    mr = results[i]["mentor"]["rmse"]
    pr = results[i]["phase4"]["rmse"]
    mentor_rmses.append(mr)
    phase4_rmses.append(pr)
    winner = "phase4 ✅" if pr < mr else "mentor"
    print(f"ds{i} ({TEMP_LABEL[i]})    {mr:>10.6f} {pr:>10.6f} {pr-mr:>+10.6f}  {winner}")
print("-" * 55)
print(f"{'Mean':<15} {np.mean(mentor_rmses):>10.6f} {np.mean(phase4_rmses):>10.6f} "
      f"{np.mean(phase4_rmses)-np.mean(mentor_rmses):>+10.6f}")


# ── Plot: ground truth vs predicted (both models, all datasets) ───────────────

fig, axes = plt.subplots(len(DATASETS), 2, figsize=(16, 4 * len(DATASETS)))

for i in DATASETS:
    y_te    = np.array(results[i]["y_te"])
    y_pred_m = np.array(results[i]["y_pred_m"])
    y_pred_p = np.array(results[i]["y_pred_p"])
    x_ax    = np.arange(len(y_te))

    for col, (y_pred, label, color) in enumerate([
        (y_pred_m, "Mentor baseline", "C0"),
        (y_pred_p, "Phase4 medium",   "C2"),
    ]):
        ax = axes[i, col]
        ax.plot(x_ax, y_te,     lw=0.6, color="black",  alpha=0.7, label="Ground Truth")
        ax.plot(x_ax, y_pred,   lw=0.6, color=color,    alpha=0.8, label=label)
        r2 = results[i]["mentor" if col == 0 else "phase4"]["r2"]
        ax.set_title(f"ds{i} ({TEMP_LABEL[i]}) | {label} | R²={r2:.4f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Strain")

fig.suptitle("Ground Truth vs Predicted — Mentor Baseline vs Phase 4 Medium", fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "prediction_comparison.png", dpi=150)
plt.close(fig)


# ── Scatter plot: predicted vs actual ────────────────────────────────────────

fig, axes = plt.subplots(len(DATASETS), 2, figsize=(10, 4 * len(DATASETS)))

for i in DATASETS:
    y_te     = np.array(results[i]["y_te"])
    y_pred_m = np.array(results[i]["y_pred_m"])
    y_pred_p = np.array(results[i]["y_pred_p"])

    for col, (y_pred, label) in enumerate([
        (y_pred_m, "Mentor"),
        (y_pred_p, "Phase4"),
    ]):
        ax   = axes[i, col]
        r2   = results[i]["mentor" if col == 0 else "phase4"]["r2"]
        lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
        ax.scatter(y_te, y_pred, s=1, alpha=0.3, color=f"C{col}")
        ax.plot(lims, lims, "r--", lw=0.8, label="Perfect")
        ax.set_xlabel("Actual Strain")
        ax.set_ylabel("Predicted Strain")
        ax.set_title(f"ds{i} ({TEMP_LABEL[i]}) | {label} | R²={r2:.4f}", fontsize=9)
        ax.legend(fontsize=7)

fig.suptitle("Scatter: Predicted vs Actual Strain", fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "scatter_comparison.png", dpi=150)
plt.close(fig)

# Save JSON
save_results = {
    str(i): {
        "mentor": results[i]["mentor"],
        "phase4": results[i]["phase4"],
    }
    for i in DATASETS
}
with open(RESULTS_DIR / "comparison_metrics.json", "w") as f:
    json.dump(save_results, f, indent=2)

print(f"\n[Comparison complete] Saved to {RESULTS_DIR}")
