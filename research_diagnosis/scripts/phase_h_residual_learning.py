"""
Phase H (v2, final decision run) — Interpretable residual correction on top
of the Phase G calibrated physical inverse (G3: zero-balance + 3 reference
strain points + calibration-validated electrode fusion). This version fixes
a real leakage bug found on review (see below) and applies an explicit,
pre-registered keep/drop decision rule for whether residual learning stays
in the final thesis method.

Column semantics (verified empirically before this run):
  Temperature, Electrode1_resistance, Electrode2_resistance -> inputs
  Strain -> target
  Experiment -> measurement SESSION id (exactly 2 distinct values per
    dataset, e.g. ds1 = {3,4}); each session's Idx restarts at 0 and is
    strictly increasing. Sessions are NOT the same thing as loading cycles:
    each session internally contains ~10 loading/unloading cycles, found by
    turning-point detection on the strain signal (Phase B), not given as a
    raw column.
  Idx -> within-session time order.
  Experiment and Idx are used ONLY to sort/group rows (choose calibration
  vs. test session); neither is ever passed as a model input feature.

Leakage fix: the prior version's `cycle_idx` feature was computed via
`detect_turning_points(df["Strain"].values)` on the TEST dataframe too --
i.e. it used the true strain signal (through its turning points) as an
input feature at inference time. Removed entirely (also matches the
explicit instruction to never use cycle ID as a feature).

Residual features (all deployable -- resistance- and physical-prediction-
derived only, never true strain or its derivative, never Experiment/cycle
ID):
  R1, R2, Temperature; dR1/dt, dR2/dt (smoothed); rolling mean/std of R1,R2;
  running (causal) max/min of R1, R2; branch (loading/unloading) inferred
  from resistance derivative sign; time since last turning point (inferred
  from resistance, not strain); previous physical (G3) prediction.

Models compared: G3 alone (baseline) -> Ridge -> small MLP -> small GRU
(window of the same feature set). Both calibration directions (A->B, B->A)
are run and reported separately; a pre-registered decision rule decides
whether residual learning is kept in the final thesis method.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from sklearn.linear_model import Ridge
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
GRU_WINDOW = 10
N_BOOT = 300
RNG = np.random.RandomState(0)


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
    """Deployable branch proxy -- resistance derivative sign only, no strain used."""
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
    return out


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


def build_features(df, eps_phy):
    """All features are deployable: resistance/temperature/physical-prediction
    only. No true strain, no strain derivative, no Experiment/cycle ID."""
    R1, R2, T = df[ELECTRODES[0]].values, df[ELECTRODES[1]].values, df["Temperature"].values
    n = len(df)
    branch = infer_branch_from_resistance(df)

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

    roll_mean1 = pd.Series(R1).rolling(20, min_periods=1).mean().values
    roll_std1 = pd.Series(R1).rolling(20, min_periods=1).std().fillna(0).values
    roll_mean2 = pd.Series(R2).rolling(20, min_periods=1).mean().values
    roll_std2 = pd.Series(R2).rolling(20, min_periods=1).std().fillna(0).values

    feats = np.column_stack([
        R1, R2, T, dR1dt, dR2dt, branch.astype(float),
        run_max1 - R1, R1 - run_min1, run_max2 - R2, R2 - run_min2,
        time_since_turn, eps_phy, roll_mean1, roll_std1, roll_mean2, roll_std2,
    ])
    return feats


def make_gru_windows(X, window):
    n = len(X)
    Xs = np.zeros((n, window, X.shape[1]), dtype=np.float32)
    for t in range(n):
        lo = max(0, t - window + 1)
        seg = X[lo:t + 1]
        Xs[t, window - len(seg):, :] = seg
    return Xs


def build_gru(window, n_features):
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(window, n_features)),
        tf.keras.layers.GRU(8),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    return {"r2": float(r2_score(y_true, y_pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)), "bias": float(np.mean(y_pred - y_true)),
            "n": int(mask.sum())}


def bootstrap_rmse_ci(eps_true, eps_pred, cycle_id, n_boot=N_BOOT, rng=RNG):
    cycles = np.unique(cycle_id)
    if len(cycles) < 2:
        return np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(cycles, size=len(cycles), replace=True)
        mask = np.isin(cycle_id, chosen)
        if mask.sum() < 5:
            continue
        boots.append(np.sqrt(mean_squared_error(eps_true[mask], eps_pred[mask])))
    if not boots:
        return np.nan, np.nan
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


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

        peaks_c, troughs_c = detect_turning_points(df_calib["Strain"].values)
        cyc_bounds_c = cycles_from_troughs(troughs_c, len(df_calib))
        n_cyc_c = len(cyc_bounds_c)
        internal_val_mask = np.zeros(len(df_calib), dtype=bool)
        for lo, hi in cyc_bounds_c[max(1, n_cyc_c // 2):]:
            internal_val_mask[lo:hi] = True
        w1 = select_fusion_weight(eps1_calib[internal_val_mask], eps2_calib[internal_val_mask],
                                   df_calib["Strain"].values[internal_val_mask])

        eps_phy_calib = w1 * eps1_calib + (1 - w1) * eps2_calib
        eps_phy_test = w1 * eps1_test + (1 - w1) * eps2_test
        eps_true_calib = df_calib["Strain"].values
        eps_true_test = df_test["Strain"].values

        # cycle id for TEST bootstrap -- from resistance-inferred branch changes only (not strain)
        branch_test = infer_branch_from_resistance(df_test)
        branch_change_test = np.diff(branch_test, prepend=branch_test[0]) != 0
        cycle_id_test = np.cumsum(branch_change_test & (branch_test == 1))

        m_baseline = metrics(eps_true_test, eps_phy_test)
        ci_lo, ci_hi = bootstrap_rmse_ci(eps_true_test, eps_phy_test, cycle_id_test)
        rows.append({"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                     "direction": direction_name, "model": "0_G3_no_residual",
                     "rmse_ci_lo": ci_lo, "rmse_ci_hi": ci_hi, **m_baseline})

        feats_calib = build_features(df_calib, eps_phy_calib)
        feats_test = build_features(df_test, eps_phy_test)
        residual_calib = eps_true_calib - eps_phy_calib

        scaler = StandardScaler().fit(feats_calib)
        X_calib, X_test = scaler.transform(feats_calib), scaler.transform(feats_test)

        preds = {"0_G3_no_residual": eps_phy_test}

        for name, model in [("1_ridge_residual", Ridge(alpha=1.0)),
                             ("2_small_mlp_residual", MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=2000,
                                                                    early_stopping=True, random_state=0))]:
            model.fit(X_calib, residual_calib)
            residual_hat_test = model.predict(X_test)
            eps_final = eps_phy_test + residual_hat_test
            m = metrics(eps_true_test, eps_final)
            ci_lo, ci_hi = bootstrap_rmse_ci(eps_true_test, eps_final, cycle_id_test)
            rows.append({"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                         "direction": direction_name, "model": name,
                         "rmse_ci_lo": ci_lo, "rmse_ci_hi": ci_hi, **m})
            preds[name] = eps_final

        # small GRU: window of the same feature set, predicting current residual
        Xg_calib = make_gru_windows(X_calib, GRU_WINDOW)
        Xg_test = make_gru_windows(X_test, GRU_WINDOW)
        tf.random.set_seed(0); np.random.seed(0)
        gru = build_gru(GRU_WINDOW, X_calib.shape[1])
        cb = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=0)]
        gru.fit(Xg_calib, residual_calib, epochs=100, batch_size=64, validation_split=0.15,
                callbacks=cb, verbose=0)
        residual_hat_gru = gru.predict(Xg_test, verbose=0).flatten()
        eps_final_gru = eps_phy_test + residual_hat_gru
        m_gru = metrics(eps_true_test, eps_final_gru)
        ci_lo, ci_hi = bootstrap_rmse_ci(eps_true_test, eps_final_gru, cycle_id_test)
        rows.append({"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                     "direction": direction_name, "model": "3_small_gru_residual",
                     "rmse_ci_lo": ci_lo, "rmse_ci_hi": ci_hi, **m_gru})
        preds["3_small_gru_residual"] = eps_final_gru

        if direction_name == "A_to_B":
            example_predictions[held_out] = {"y_true": eps_true_test, **preds}

    print(f"held_out=ds{held_out} ({TEMP_VAL[held_out]}C) done")

results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_TAB / "residual_ablation_results.csv", index=False)
print(f"\nSaved {len(results_df)} rows -> tables/residual_ablation_results.csv")

MODEL_ORDER = ["0_G3_no_residual", "1_ridge_residual", "2_small_mlp_residual", "3_small_gru_residual"]

# ── Per-dataset x direction (NOT averaged) + % RMSE improvement vs G3 ─────────

pivot_rmse = results_df.pivot_table(index=["held_out", "temperature", "kind", "direction"],
                                     columns="model", values="rmse").reindex(columns=MODEL_ORDER)
for m in MODEL_ORDER[1:]:
    pivot_rmse[f"pct_improvement_{m}"] = 100 * (pivot_rmse["0_G3_no_residual"] - pivot_rmse[m]) / pivot_rmse["0_G3_no_residual"]
pivot_rmse.to_csv(OUT_TAB / "residual_per_direction_rmse.csv")
print("\nPer-dataset x direction RMSE and %% improvement vs G3:")
print(pivot_rmse.round(4).to_string())

# ── Decision rule ──────────────────────────────────────────────────────────────
# Keep a residual model only if: mean RMSE improves >=10% vs G3, improvement in
# BOTH calibration directions, no dataset suffers a major degradation (>20%
# worse), and ds2 (43C) improves meaningfully.

decision_rows = []
for m in MODEL_ORDER[1:]:
    col = f"pct_improvement_{m}"
    both_dir_improve = all(
        (pivot_rmse.xs(ds, level="held_out")[col] > 0).all() for ds in DATASETS
    )
    mean_improve = pivot_rmse[col].mean()
    worst_degradation = pivot_rmse[col].min()
    major_degradation = worst_degradation < -20
    ds2_improve = pivot_rmse.xs(2, level="held_out")[col].mean()
    keep = (mean_improve >= 10) and both_dir_improve and (not major_degradation) and (ds2_improve > 0)
    decision_rows.append({"model": m, "mean_pct_improvement": mean_improve,
                           "improves_in_both_directions_every_dataset": both_dir_improve,
                           "worst_case_pct_change": worst_degradation, "major_degradation": major_degradation,
                           "ds2_43C_pct_improvement": ds2_improve, "KEEP": keep})
decision_df = pd.DataFrame(decision_rows)
decision_df.to_csv(OUT_TAB / "residual_decision_rule.csv", index=False)
print("\nDecision rule (keep model only if mean RMSE improvement >=10%, both directions improve every "
      "dataset, no dataset degrades >20%%, and 43C improves):")
print(decision_df.to_string(index=False))

best_model = None
for m in MODEL_ORDER[1:]:
    if decision_df[decision_df.model == m]["KEEP"].values[0]:
        best_model = m
print(f"\n=> Final decision: {'KEEP ' + best_model if best_model else 'DROP all residual models -- use G3 alone'}")

# ── Figures ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, ds in zip(axes.flat, DATASETS):
    ex = example_predictions[ds]
    ax.plot(ex["y_true"], lw=0.8, label="measured", alpha=0.85)
    ax.plot(ex["0_G3_no_residual"], lw=0.8, label="G3 (no residual)", alpha=0.7)
    ax.plot(ex["2_small_mlp_residual"], lw=0.8, label="+ MLP residual", alpha=0.85)
    ax.set_title(f"ds{ds} ({TEMP_VAL[ds]}C, {KIND[ds]})", fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle("Phase H — G3-only vs +residual (MLP), A_to_B direction")
fig.tight_layout()
fig.savefig(OUT_FIG / "residual_vs_loading_state.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, ds in zip(axes.flat, DATASETS):
    ex = example_predictions[ds]
    label = best_model if best_model else "2_small_mlp_residual"
    r2 = results_df[(results_df.held_out == ds) & (results_df.direction == "A_to_B") &
                     (results_df.model == label)]["r2"].values[0]
    ax.plot(ex["y_true"], lw=0.8, label="measured", alpha=0.85)
    ax.plot(ex[label], lw=0.8, label=f"final ({label})", alpha=0.85)
    ax.set_title(f"ds{ds} ({TEMP_VAL[ds]}C) | R2={r2:.3f}", fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle("Phase H — final predictions, all temperatures (A_to_B)")
fig.tight_layout()
fig.savefig(OUT_FIG / "final_prediction_all_temperatures.png", dpi=150)
plt.close(fig)

print("\n[Phase H v2 complete]")
