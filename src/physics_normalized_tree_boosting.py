"""Physics-normalized tree boosting for resistive fiber strain prediction.

Purpose
-------
Estimate mechanical strain from two resistive fiber electrodes under changing
temperature/session conditions. The method first calibrates a physically
interpretable inverse model on the target session:

    z_i = (R_i - B_i) / S_i

where B_i and S_i are obtained from zero-balance plus three reference strain
points. A small-data tree model then learns the remaining nonlinear correction
from normalized, causal time-series features.

Models:
1. HistGradientBoostingRegressor (main method): deterministic in this setup,
   low-capacity, nonlinear, strong on small tabular/time-series features.
2. ExtraTreesRegressor (stability check): randomized tree ensemble evaluated
   across seeds.

The raw resistance signal is never fed directly to a black-box model. Learning
happens after physical normalization.

Outputs
-------
results/tables/tree_boosting_results.csv
results/tables/tree_boosting_per_direction_rmse.csv
results/tables/tree_boosting_decision_rule.csv
results/figures/tree_boosting_vs_physics.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "to_know_how" / "files"
OUT_FIG = ROOT / "results" / "figures"
OUT_TAB = ROOT / "results" / "tables"

DATASETS = [1, 2, 3, 4]
TEMP_VAL = {1: 33, 2: 43, 3: 52, 4: 62}
KIND = {1: "extrapolation", 2: "interpolation", 3: "interpolation", 4: "extrapolation"}
ELECTRODES = ["Electrode1_resistance", "Electrode2_resistance"]
FULL_SCALE = 0.5  # percent strain
SEEDS = [0, 1, 42, 123]


def load_raw(i):
    return pd.read_csv(DATA_DIR / f"dataset{i}.csv").sort_values(["Experiment", "Idx"]).reset_index(drop=True)


def detect_turning_points(strain):
    win = min(51, (len(strain) // 2) * 2 - 1)
    win = max(win, 5)
    smooth = savgol_filter(strain, win, 3)
    peaks, _ = find_peaks(smooth, prominence=0.05)
    troughs, _ = find_peaks(-smooth, prominence=0.05)
    return peaks, troughs


def cycles_from_troughs(troughs, n):
    bounds = [0] + list(troughs) + [n]
    return list(zip(bounds[:-1], bounds[1:]))


def calibrate_multi_point(df_calib, ref_levels):
    """G3 calibration, identical to H2."""
    strain = df_calib["Strain"].values
    out = {}
    for elec in ELECTRODES:
        pts_x, pts_y = [0.0], [None]
        zero_mask = strain <= (strain.min() + 0.02)
        pts_y[0] = float(np.mean(df_calib[elec].values[zero_mask])) if zero_mask.sum() else np.nan
        for lvl in ref_levels:
            mask = np.abs(strain - lvl) <= 0.03
            if mask.sum() >= 5:
                pts_x.append(lvl)
                pts_y.append(float(np.mean(df_calib[elec].values[mask])))
        pts_x, pts_y = np.array(pts_x), np.array(pts_y)
        valid = ~np.isnan(pts_y)
        S, B = np.polyfit(pts_x[valid], pts_y[valid], 1)
        out[elec] = (S, B)
    return out


def select_fusion_weight(eps1_val, eps2_val, eps_true_val):
    best_w, best_rmse = 0.5, np.inf
    for w1 in np.linspace(0, 1, 21):
        pred = w1 * eps1_val + (1 - w1) * eps2_val
        rmse = np.sqrt(mean_squared_error(eps_true_val, pred))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w1
    return best_w


def smoothed_deriv(x):
    win = min(21, (len(x) // 2) * 2 - 1)
    win = max(win, 5)
    return np.gradient(savgol_filter(x, win, 2))


def rolling(series, window, op):
    r = pd.Series(series).rolling(window, min_periods=1)
    out = getattr(r, op)().values
    return np.nan_to_num(out)


def build_features(z1, z2, zf, temperature):
    dz1 = smoothed_deriv(z1)
    dz2 = smoothed_deriv(z2)
    dzf = smoothed_deriv(zf)
    branch = (dzf > 0).astype(float)
    cols = [z1, z2, zf, dz1, dz2, dzf, branch, temperature]
    for x in [z1, z2, zf, dzf]:
        for window in [21, 51, 101, 201]:
            cols.append(rolling(x, window, "mean"))
            cols.append(rolling(x, window, "std"))
            cols.append(rolling(x, window, "min"))
            cols.append(rolling(x, window, "max"))
    return np.column_stack(cols)


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pct_fs": float(100 * rmse / FULL_SCALE),
        "n": int(len(y_true)),
    }


def make_model(model_name, seed):
    if model_name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.03,
            l2_regularization=1.0,
            max_leaf_nodes=8,
            min_samples_leaf=40,
            random_state=seed,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(model_name)


def main():
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    raw = {i: load_raw(i) for i in DATASETS}
    rows = []
    examples = {}

    for seed in SEEDS:
        for model_name in ["hgb", "extra_trees"]:
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
                    S1, B1 = fit[ELECTRODES[0]]
                    S2, B2 = fit[ELECTRODES[1]]

                    z1_c = (df_calib[ELECTRODES[0]].values - B1) / S1
                    z2_c = (df_calib[ELECTRODES[1]].values - B2) / S2
                    z1_t = (df_test[ELECTRODES[0]].values - B1) / S1
                    z2_t = (df_test[ELECTRODES[1]].values - B2) / S2
                    y_calib = df_calib["Strain"].values
                    y_test = df_test["Strain"].values

                    _, troughs_c = detect_turning_points(y_calib)
                    cyc_bounds_c = cycles_from_troughs(troughs_c, len(df_calib))
                    val_mask = np.zeros(len(df_calib), dtype=bool)
                    for lo, hi in cyc_bounds_c[max(1, len(cyc_bounds_c) // 2):]:
                        val_mask[lo:hi] = True
                    train_mask = ~val_mask

                    w1 = select_fusion_weight(z1_c[val_mask], z2_c[val_mask], y_calib[val_mask])
                    zf_c = w1 * z1_c + (1 - w1) * z2_c
                    zf_t = w1 * z1_t + (1 - w1) * z2_t
                    pred_physics = zf_t
                    pred_physics_val = zf_c[val_mask]

                    X_calib = build_features(z1_c, z2_c, zf_c, df_calib["Temperature"].values)
                    X_test = build_features(z1_t, z2_t, zf_t, df_test["Temperature"].values)

                    model = make_model(model_name, seed)
                    model.fit(X_calib[train_mask], y_calib[train_mask])
                    pred_train = model.predict(X_calib[train_mask])
                    pred_val = model.predict(X_calib[val_mask])
                    pred_model = model.predict(X_test)

                    best_alpha, best_rmse = 0.0, np.inf
                    for alpha in np.linspace(0, 1, 21):
                        blend = alpha * pred_val + (1 - alpha) * pred_physics_val
                        rmse = np.sqrt(mean_squared_error(y_calib[val_mask], blend))
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_alpha = alpha
                    pred_blend = best_alpha * pred_model + (1 - best_alpha) * pred_physics

                    train_r2 = r2_score(y_calib[train_mask], pred_train)
                    val_r2 = r2_score(y_calib[val_mask], pred_val)

                    for out_model, pred, alpha in [
                        ("0_G3_physics_only", pred_physics, 0.0),
                        (f"1_{model_name}_only", pred_model, 1.0),
                        (f"2_{model_name}_blended", pred_blend, best_alpha),
                    ]:
                        rows.append({
                            "seed": seed,
                            "held_out": held_out,
                            "temperature": TEMP_VAL[held_out],
                            "kind": KIND[held_out],
                            "direction": direction_name,
                            "model": out_model,
                            "alpha": alpha,
                            "w1": w1,
                            "train_r2": train_r2 if out_model != "0_G3_physics_only" else np.nan,
                            "val_r2": val_r2 if out_model != "0_G3_physics_only" else np.nan,
                            **metrics(y_test, pred),
                        })

                    if seed == 0 and model_name == "hgb" and direction_name == "A_to_B":
                        examples[held_out] = {"y": y_test, "g3": pred_physics, "hgb": pred_blend}

                print(f"seed={seed} {model_name} ds{held_out} done")

    results = pd.DataFrame(rows)
    results.to_csv(OUT_TAB / "tree_boosting_results.csv", index=False)

    pivot = results.pivot_table(index=["seed", "held_out", "temperature", "kind", "direction"],
                                columns="model", values="rmse")
    base = pivot["0_G3_physics_only"]
    for model in ["hgb", "extra_trees"]:
        pivot[f"pct_improvement_{model}_only"] = 100 * (base - pivot[f"1_{model}_only"]) / base
        pivot[f"pct_improvement_{model}_blended"] = 100 * (base - pivot[f"2_{model}_blended"]) / base
    pivot.to_csv(OUT_TAB / "tree_boosting_per_direction_rmse.csv")

    decision_rows = []
    for seed in SEEDS:
        sub = pivot.xs(seed, level="seed")
        for model in ["hgb", "extra_trees"]:
            for variant in ["only", "blended"]:
                col = f"pct_improvement_{model}_{variant}"
                mean_improve = sub[col].mean()
                worst = sub[col].min()
                every_dataset = all((sub.xs(ds, level="held_out")[col] > 0).all() for ds in DATASETS)
                ds2_improve = sub.xs(2, level="held_out")[col].mean()
                keep = every_dataset and worst >= -10 and ds2_improve > 0 and mean_improve >= 10
                decision_rows.append({
                    "seed": seed,
                    "model": f"{model}_{variant}",
                    "mean_pct_improvement": mean_improve,
                    "worst_case_pct_change": worst,
                    "every_dataset_improves_both_directions": every_dataset,
                    "no_fold_degrades_over_10pct": worst >= -10,
                    "ds2_43C_pct_improvement": ds2_improve,
                    "KEEP": keep,
                })
    decisions = pd.DataFrame(decision_rows)

    summary = []
    for model in decisions["model"].unique():
        sub = decisions[decisions["model"] == model]
        summary.append({
            "seed": "ALL",
            "model": model,
            "mean_pct_improvement": sub["mean_pct_improvement"].mean(),
            "worst_case_pct_change": sub["worst_case_pct_change"].min(),
            "every_dataset_improves_both_directions": bool(sub["every_dataset_improves_both_directions"].all()),
            "no_fold_degrades_over_10pct": bool(sub["no_fold_degrades_over_10pct"].all()),
            "ds2_43C_pct_improvement": sub["ds2_43C_pct_improvement"].mean(),
            "KEEP": bool(sub["KEEP"].all()),
            "std_pct_improvement": sub["mean_pct_improvement"].std(ddof=1),
            "KEEP_count": int(sub["KEEP"].sum()),
            "n_seed": len(sub),
        })
    decisions = pd.concat([decisions, pd.DataFrame(summary)], ignore_index=True)
    decisions.to_csv(OUT_TAB / "tree_boosting_decision_rule.csv", index=False)

    print("\nDecision:")
    print(decisions.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, ds in zip(axes.flat, DATASETS):
        ex = examples[ds]
        ax.plot(ex["y"], lw=0.8, label="measured", alpha=0.85)
        ax.plot(ex["g3"], lw=0.8, label="G3 physics", alpha=0.7)
        ax.plot(ex["hgb"], lw=0.8, label="HGB blend", alpha=0.85)
        ax.set_title(f"ds{ds} ({TEMP_VAL[ds]}C, A_to_B)")
        ax.legend(fontsize=7)
    fig.suptitle("Physics-normalized HGB readout vs calibrated physics baseline")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "tree_boosting_vs_physics.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
