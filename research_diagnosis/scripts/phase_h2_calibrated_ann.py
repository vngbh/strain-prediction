"""
Phase H2 — Calibrated-feature ANN (final, tightly-scoped combination test).

Idea (from review): instead of feeding raw (R1, R2, T) to an ANN (which is
what the original repo's Deep ANN did, and what made it collapse out of
same-temperature domain), first physically normalize the signal using the
SAME Phase G calibration (zero-balance + 3-point), then let a small ANN
learn the residual nonlinearity on top of an already-domain-reduced
representation:

    z_i(t) = (R_i(t) - B_i) / S_i          [Phase G G3 calibration]
    [z1, z2, dz1, dz2, T] -> ANN -> eps_ANN
    eps_final = alpha*eps_ANN + (1-alpha)*eps_physics(G3)   [alpha from
                                                              calibration data only]

Architecture: Dense(8) -> Dense(4) -> Dense(1), strong L2 regularization,
early stopping, no GRU/TCN (per the reviewer's explicit scope). Same
calibration/test session split as Phase G, both directions run separately.

Decision rule (stricter than Phase H v2, per the reviewer):
  keep only if -- better than G3 on EVERY dataset; no fold degrades more
  than 5-10%; 43C improves; mean RMSE reduction >= 10%.

Do not confuse this with the original repo's same-temperature Deep ANN
R2=0.85 -- that number was measured in-domain (train/test same temperature);
this experiment is evaluated cross-session (Phase G's harder, actual
deployment protocol), so it is not a directly comparable target.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_FIG = Path(__file__).resolve().parents[1] / "figures"
OUT_TAB = Path(__file__).resolve().parents[1] / "tables"

DATASETS = [1, 2, 3, 4]
TEMP_VAL = {1: 33, 2: 43, 3: 52, 4: 62}
KIND = {1: "extrapolation", 2: "interpolation", 3: "interpolation", 4: "extrapolation"}
ELECTRODES = ["Electrode1_resistance", "Electrode2_resistance"]
PROMINENCE = 0.05


def load_raw(i):
    df = pd.read_csv(DATA_DIR / f"dataset{i}.csv")
    return df.sort_values(["Experiment", "Idx"]).reset_index(drop=True)


def detect_turning_points(strain):
    win = min(51, (len(strain) // 2) * 2 - 1); win = max(win, 5)
    smooth = savgol_filter(strain, win, 3)
    peaks, _ = find_peaks(smooth, prominence=PROMINENCE)
    troughs, _ = find_peaks(-smooth, prominence=PROMINENCE)
    return peaks, troughs


def cycles_from_troughs(troughs, n):
    bounds = [0] + list(troughs) + [n]
    return list(zip(bounds[:-1], bounds[1:]))


def calibrate_multi_point(df_calib, ref_levels):
    """G3 (3-point) calibration, identical protocol to Phase G."""
    strain = df_calib["Strain"].values
    out = {}
    for elec in ELECTRODES:
        pts_x, pts_y = [0.0], [None]
        zero_mask = strain <= (strain.min() + 0.02)
        pts_y[0] = float(np.mean(df_calib[elec].values[zero_mask])) if zero_mask.sum() else np.nan
        for lvl in ref_levels:
            mask = np.abs(strain - lvl) <= 0.03
            if mask.sum() >= 5:
                pts_x.append(lvl); pts_y.append(float(np.mean(df_calib[elec].values[mask])))
        pts_x, pts_y = np.array(pts_x), np.array(pts_y)
        valid = ~np.isnan(pts_y)
        S, B = np.polyfit(pts_x[valid], pts_y[valid], 1)
        out[elec] = (S, B)
    return out


def select_fusion_weight(eps1_val, eps2_val, eps_true_val):
    best_w, best_rmse = 0.5, np.inf
    for w1 in np.linspace(0, 1, 21):
        rmse = np.sqrt(mean_squared_error(eps_true_val, w1 * eps1_val + (1 - w1) * eps2_val))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w1
    return best_w


def smoothed_deriv(x):
    win = min(21, (len(x) // 2) * 2 - 1); win = max(win, 5)
    return np.gradient(savgol_filter(x, win, 2))


def build_ann():
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(5,)),
        tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-2)),
        tf.keras.layers.Dense(4, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-2)),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    return {"r2": float(r2_score(y_true, y_pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)), "n": int(mask.sum())}


raw = {i: load_raw(i) for i in DATASETS}
rows = []
example_predictions = {}

for held_out in DATASETS:
    df_ds = raw[held_out]
    exps = sorted(df_ds.Experiment.unique())
    directions = [("A_to_B", exps[0], exps[1]), ("B_to_A", exps[1], exps[0])]

    for direction_name, calib_exp, test_exp in directions:
        df_calib = df_ds[df_ds.Experiment == calib_exp].reset_index(drop=True)
        df_test = df_ds[df_ds.Experiment == test_exp].reset_index(drop=True)

        strain_range = df_calib["Strain"].max() - df_calib["Strain"].min()
        ref_levels = list(np.linspace(df_calib["Strain"].min() + 0.05 * strain_range,
                                       df_calib["Strain"].max() - 0.05 * strain_range, 3))
        fit = calibrate_multi_point(df_calib, ref_levels)
        S1, B1 = fit[ELECTRODES[0]]; S2, B2 = fit[ELECTRODES[1]]

        def calibrated_features(df):
            z1 = (df[ELECTRODES[0]].values - B1) / S1
            z2 = (df[ELECTRODES[1]].values - B2) / S2
            dz1, dz2 = smoothed_deriv(z1), smoothed_deriv(z2)
            T = df["Temperature"].values
            return z1, z2, dz1, dz2, T

        z1_c, z2_c, dz1_c, dz2_c, T_c = calibrated_features(df_calib)
        z1_t, z2_t, dz1_t, dz2_t, T_t = calibrated_features(df_test)
        X_calib = np.column_stack([z1_c, z2_c, dz1_c, dz2_c, T_c])
        X_test = np.column_stack([z1_t, z2_t, dz1_t, dz2_t, T_t])

        # calibration-internal train/val split (first half cycles vs second half), same protocol as Phase G/H
        peaks_c, troughs_c = detect_turning_points(df_calib["Strain"].values)
        cyc_bounds_c = cycles_from_troughs(troughs_c, len(df_calib))
        n_cyc_c = len(cyc_bounds_c)
        internal_val_mask = np.zeros(len(df_calib), dtype=bool)
        for lo, hi in cyc_bounds_c[max(1, n_cyc_c // 2):]:
            internal_val_mask[lo:hi] = True
        internal_train_mask = ~internal_val_mask

        eps_true_calib = df_calib["Strain"].values
        eps_true_test = df_test["Strain"].values

        # physics-only baseline (G3, electrode-fusion weight selected from calib-internal split)
        eps1_calib, eps2_calib = z1_c, z2_c  # z_i IS the physical per-electrode strain estimate
        w1 = select_fusion_weight(eps1_calib[internal_val_mask], eps2_calib[internal_val_mask],
                                   eps_true_calib[internal_val_mask])
        eps_phy_test = w1 * z1_t + (1 - w1) * z2_t

        # ANN: fit on internal-train, use internal-val for early stopping
        tf.random.set_seed(0); np.random.seed(0)
        ann = build_ann()
        cb = [tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0)]
        ann.fit(X_calib[internal_train_mask], eps_true_calib[internal_train_mask], epochs=300, batch_size=32,
                validation_data=(X_calib[internal_val_mask], eps_true_calib[internal_val_mask]),
                callbacks=cb, verbose=0)
        eps_ann_test = ann.predict(X_test, verbose=0).flatten()
        eps_ann_val = ann.predict(X_calib[internal_val_mask], verbose=0).flatten()

        # blend weight alpha chosen from calibration-internal validation only
        eps_phy_val = w1 * z1_c[internal_val_mask] + (1 - w1) * z2_c[internal_val_mask]
        best_alpha, best_rmse = 0.0, np.inf
        for alpha in np.linspace(0, 1, 21):
            blend = alpha * eps_ann_val + (1 - alpha) * eps_phy_val
            rmse = np.sqrt(mean_squared_error(eps_true_calib[internal_val_mask], blend))
            if rmse < best_rmse:
                best_rmse, best_alpha = rmse, alpha

        eps_final_test = best_alpha * eps_ann_test + (1 - best_alpha) * eps_phy_test

        m_g3 = metrics(eps_true_test, eps_phy_test)
        m_ann = metrics(eps_true_test, eps_ann_test)
        m_final = metrics(eps_true_test, eps_final_test)

        rows.append({"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                     "direction": direction_name, "model": "0_G3_physics_only", "alpha": 0.0, **m_g3})
        rows.append({"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                     "direction": direction_name, "model": "1_calibrated_ann_only", "alpha": 1.0, **m_ann})
        rows.append({"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                     "direction": direction_name, "model": "2_blended_final", "alpha": best_alpha, **m_final})

        if direction_name == "A_to_B":
            example_predictions[held_out] = {"y_true": eps_true_test, "G3": eps_phy_test,
                                              "ANN": eps_ann_test, "blended": eps_final_test}

    print(f"held_out=ds{held_out} ({TEMP_VAL[held_out]}C) done")

results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_TAB / "calibrated_ann_results.csv", index=False)
print(f"\nSaved {len(results_df)} rows -> tables/calibrated_ann_results.csv")

pivot_rmse = results_df.pivot_table(index=["held_out", "temperature", "kind", "direction"],
                                     columns="model", values="rmse")
pivot_rmse["pct_improvement_ann"] = 100 * (pivot_rmse["0_G3_physics_only"] - pivot_rmse["1_calibrated_ann_only"]) / pivot_rmse["0_G3_physics_only"]
pivot_rmse["pct_improvement_blended"] = 100 * (pivot_rmse["0_G3_physics_only"] - pivot_rmse["2_blended_final"]) / pivot_rmse["0_G3_physics_only"]
pivot_rmse.to_csv(OUT_TAB / "calibrated_ann_per_direction_rmse.csv")
print("\nPer-dataset x direction RMSE and %% improvement vs G3:")
print(pivot_rmse.round(4).to_string())

# ── Decision rule: better than G3 on EVERY dataset, no fold degrades >5-10%, 43C improves, mean RMSE reduction >=10% ──

decision_rows = []
for m, col in [("1_calibrated_ann_only", "pct_improvement_ann"), ("2_blended_final", "pct_improvement_blended")]:
    mean_improve = pivot_rmse[col].mean()
    worst = pivot_rmse[col].min()
    every_dataset_improves = all((pivot_rmse.xs(ds, level="held_out")[col] > 0).all() for ds in DATASETS)
    no_major_degradation = worst >= -10  # tolerance band requested: 5-10%
    ds2_improve = pivot_rmse.xs(2, level="held_out")[col].mean()
    keep = every_dataset_improves and no_major_degradation and (ds2_improve > 0) and (mean_improve >= 10)
    decision_rows.append({"model": m, "mean_pct_improvement": mean_improve, "worst_case_pct_change": worst,
                           "every_dataset_improves_both_directions": every_dataset_improves,
                           "no_fold_degrades_over_10pct": no_major_degradation,
                           "ds2_43C_pct_improvement": ds2_improve, "KEEP": keep})
decision_df = pd.DataFrame(decision_rows)
decision_df.to_csv(OUT_TAB / "calibrated_ann_decision_rule.csv", index=False)
print("\nDecision rule (keep only if: improves every dataset in both directions, no fold degrades >10%, "
      "43C improves, mean RMSE reduction >=10%):")
print(decision_df.to_string(index=False))

final_keep = decision_df[decision_df.KEEP]["model"].tolist()
print(f"\n=> Final decision: {'KEEP ' + ', '.join(final_keep) if final_keep else 'DROP -- use G3 physics-only'}")

# ── Figure ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, ds in zip(axes.flat, DATASETS):
    ex = example_predictions[ds]
    ax.plot(ex["y_true"], lw=0.8, label="measured", alpha=0.85)
    ax.plot(ex["G3"], lw=0.8, label="G3 (physics only)", alpha=0.7)
    ax.plot(ex["blended"], lw=0.8, label="blended (physics+calibrated ANN)", alpha=0.85)
    ax.set_title(f"ds{ds} ({TEMP_VAL[ds]}C, {KIND[ds]})", fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle("Phase H2 — G3 vs calibrated-feature-ANN blend, A_to_B direction")
fig.tight_layout()
fig.savefig(OUT_FIG / "calibrated_ann_vs_g3.png", dpi=150)
plt.close(fig)

print("\n[Phase H2 complete]")
