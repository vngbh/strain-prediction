"""
Phase H — Interpretable residual correction on top of the Phase G calibrated
physical inverse. Time-boxed: escalates Linear -> Ridge -> small MLP only,
per the reviewer's own decision rule (do not jump straight to GRU/TCN/
Transformer; only escalate if the simpler level shows a real gain).

Baseline (no residual): the Phase G "G3, 3-point" calibrated inverse — the
best deployable level found in Phase G — with a calibration-internal-
validated electrode fusion weight (same protocol as Phase G: never use the
real test Experiment to pick anything).

Residual features (all deployable -- computed from resistance and the
physical prediction only, NEVER from true strain):
  - dR1/dt, dR2/dt (smoothed)
  - branch (loading/unloading), inferred from resistance derivative sign
  - running (causal, expanding-window) max/min of R1, R2
  - time since last turning point (samples)
  - cycle index (causal)
  - previous physical prediction eps_phy (lag-0, itself resistance-derived)
  - rolling mean/std of R1, R2 (short window)

The residual model is fit on the CALIBRATION Experiment's own residual
(true strain is known there) and applied to the TEST Experiment (cross-
session, matching Phase G's main deployment scenario) -- both calibration
directions (A->B, B->A) are run and reported separately before averaging.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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


def infer_branch_from_resistance(df):
    r_avg = df[ELECTRODES].mean(axis=1).values
    win = min(31, (len(r_avg) // 2) * 2 - 1); win = max(win, 5)
    smooth = savgol_filter(r_avg, win, 2)
    d = np.gradient(smooth)
    thresh = np.std(d) * 0.15
    branch = np.where(d > thresh, 1, np.where(d < -thresh, -1, 0))
    last = 1
    out = np.empty_like(branch)
    for k in range(len(branch)):
        if branch[k] != 0:
            last = branch[k]
        out[k] = last
    return out, d


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


def build_features(df, S1, B1, S2, B2, eps_phy):
    """All features are deployable (resistance- or physical-prediction-derived only)."""
    R1, R2 = df[ELECTRODES[0]].values, df[ELECTRODES[1]].values
    n = len(df)
    branch, dR_avg = infer_branch_from_resistance(df)

    def smoothed_deriv(x):
        win = min(21, (len(x) // 2) * 2 - 1); win = max(win, 5)
        return np.gradient(savgol_filter(x, win, 2))

    dR1dt, dR2dt = smoothed_deriv(R1), smoothed_deriv(R2)
    run_max1 = pd.Series(R1).expanding().max().values
    run_min1 = pd.Series(R1).expanding().min().values
    run_max2 = pd.Series(R2).expanding().max().values
    run_min2 = pd.Series(R2).expanding().min().values

    branch_change = np.diff(branch, prepend=branch[0]) != 0
    turn_idx = np.where(branch_change)[0]
    time_since_turn = np.zeros(n)
    last_turn = 0
    for i in range(n):
        if i in turn_idx:
            last_turn = i
        time_since_turn[i] = i - last_turn

    peaks, troughs = detect_turning_points(df["Strain"].values)  # only used for cycle_index construction here for feature purposes
    cycles = cycles_from_troughs(troughs, n)
    cycle_idx = np.zeros(n, dtype=int)
    for ci, (lo, hi) in enumerate(cycles):
        cycle_idx[lo:hi] = ci

    roll_mean1 = pd.Series(R1).rolling(20, min_periods=1).mean().values
    roll_std1 = pd.Series(R1).rolling(20, min_periods=1).std().fillna(0).values
    roll_mean2 = pd.Series(R2).rolling(20, min_periods=1).mean().values
    roll_std2 = pd.Series(R2).rolling(20, min_periods=1).std().fillna(0).values

    feats = np.column_stack([
        dR1dt, dR2dt, branch.astype(float), run_max1 - R1, R1 - run_min1,
        run_max2 - R2, R2 - run_min2, time_since_turn, cycle_idx.astype(float),
        eps_phy, roll_mean1, roll_std1, roll_mean2, roll_std2,
    ])
    return feats


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

        eps1_calib = (df_calib[ELECTRODES[0]].values - B1) / S1
        eps2_calib = (df_calib[ELECTRODES[1]].values - B2) / S2
        eps1_test = (df_test[ELECTRODES[0]].values - B1) / S1
        eps2_test = (df_test[ELECTRODES[1]].values - B2) / S2

        peaks, troughs = detect_turning_points(df_calib["Strain"].values)
        cyc_bounds = cycles_from_troughs(troughs, len(df_calib))
        n_cyc = len(cyc_bounds)
        internal_val_mask = np.zeros(len(df_calib), dtype=bool)
        for lo, hi in cyc_bounds[max(1, n_cyc // 2):]:
            internal_val_mask[lo:hi] = True
        w1 = select_fusion_weight(eps1_calib[internal_val_mask], eps2_calib[internal_val_mask],
                                   df_calib["Strain"].values[internal_val_mask])

        eps_phy_calib = w1 * eps1_calib + (1 - w1) * eps2_calib
        eps_phy_test = w1 * eps1_test + (1 - w1) * eps2_test
        eps_true_calib = df_calib["Strain"].values
        eps_true_test = df_test["Strain"].values

        m_baseline = metrics(eps_true_test, eps_phy_test)
        rows.append({"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                     "direction": direction_name, "model": "0_calibration_only_no_residual", **m_baseline})

        feats_calib = build_features(df_calib, S1, B1, S2, B2, eps_phy_calib)
        feats_test = build_features(df_test, S1, B1, S2, B2, eps_phy_test)
        residual_calib = eps_true_calib - eps_phy_calib

        scaler = StandardScaler().fit(feats_calib)
        X_calib, X_test = scaler.transform(feats_calib), scaler.transform(feats_test)

        models = {
            "1_linear_residual": LinearRegression(),
            "2_ridge_residual": Ridge(alpha=1.0),
            "3_small_mlp_residual": MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=2000,
                                                  early_stopping=True, random_state=0),
        }
        preds = {}
        for name, model in models.items():
            model.fit(X_calib, residual_calib)
            residual_hat_test = model.predict(X_test)
            eps_final = eps_phy_test + residual_hat_test
            m = metrics(eps_true_test, eps_final)
            rows.append({"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                         "direction": direction_name, "model": name, **m})
            preds[name] = eps_final

        if direction_name == "A_to_B":
            example_predictions[held_out] = {"y_true": eps_true_test, "y_phy": eps_phy_test, **preds}

    print(f"held_out=ds{held_out} ({TEMP_VAL[held_out]}C) done")

results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_TAB / "residual_ablation_results.csv", index=False)
print(f"\nSaved {len(results_df)} rows -> tables/residual_ablation_results.csv")

summary = results_df.groupby("model")[["r2", "rmse"]].mean()
model_order = ["0_calibration_only_no_residual", "1_linear_residual", "2_ridge_residual", "3_small_mlp_residual"]
print("\nMean R2/RMSE across all datasets/directions:")
print(summary.reindex(model_order))

per_ds = results_df.groupby(["held_out", "temperature", "kind", "model"])[["r2", "rmse"]].mean().reset_index()
per_ds.to_csv(OUT_TAB / "residual_ablation_per_dataset.csv", index=False)

# ── Figures ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, ds in zip(axes.flat, DATASETS):
    ex = example_predictions[ds]
    ax.plot(ex["y_true"], lw=0.8, label="measured", alpha=0.85)
    ax.plot(ex["y_phy"], lw=0.8, label="physical (G3, no residual)", alpha=0.7)
    ax.plot(ex["3_small_mlp_residual"], lw=0.8, label="+ MLP residual", alpha=0.85)
    ax.set_title(f"ds{ds} ({TEMP_VAL[ds]}C, {KIND[ds]})", fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle("Phase H — physical-only vs +residual (MLP), A_to_B direction")
fig.tight_layout()
fig.savefig(OUT_FIG / "residual_vs_loading_state.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, ds in zip(axes.flat, DATASETS):
    ex = example_predictions[ds]
    r2 = results_df[(results_df.held_out == ds) & (results_df.direction == "A_to_B") &
                     (results_df.model == "3_small_mlp_residual")]["r2"].values[0]
    ax.plot(ex["y_true"], lw=0.8, label="measured", alpha=0.85)
    ax.plot(ex["3_small_mlp_residual"], lw=0.8, label="final (physical+residual)", alpha=0.85)
    ax.set_title(f"ds{ds} ({TEMP_VAL[ds]}C) | R2={r2:.3f}", fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle("Phase H — final predictions, all temperatures (A_to_B)")
fig.tight_layout()
fig.savefig(OUT_FIG / "final_prediction_all_temperatures.png", dpi=150)
plt.close(fig)

print("\n[Phase H complete]")
