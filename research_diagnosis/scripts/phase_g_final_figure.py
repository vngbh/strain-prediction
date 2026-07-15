"""Phase G addendum: calibrated_predictions_all_temperatures.png
Recomputes the G3 (5-point) calibration for the A_to_B direction of every
dataset (best overall deployable level from calibration_summary.csv) and
plots true-vs-predicted strain over time for all 4 held-out temperatures.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from sklearn.linear_model import HuberRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_FIG = Path(__file__).resolve().parents[1] / "figures"
DATASETS = [1, 2, 3, 4]
TEMP_VAL = {1: 33, 2: 43, 3: 52, 4: 62}
ELECTRODES = ["Electrode1_resistance", "Electrode2_resistance"]


def load_raw(i):
    df = pd.read_csv(DATA_DIR / f"dataset{i}.csv")
    return df.sort_values(["Experiment", "Idx"]).reset_index(drop=True)


def calibrate_multi_point(df_calib, ref_levels):
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


fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, ds in zip(axes.flat, DATASETS):
    df = load_raw(ds)
    exps = sorted(df.Experiment.unique())
    df_calib = df[df.Experiment == exps[0]].reset_index(drop=True)
    df_test = df[df.Experiment == exps[1]].reset_index(drop=True)
    strain_range = df_calib["Strain"].max() - df_calib["Strain"].min()
    ref_levels = list(np.linspace(df_calib["Strain"].min() + 0.05 * strain_range,
                                   df_calib["Strain"].max() - 0.05 * strain_range, 5))
    fit = calibrate_multi_point(df_calib, ref_levels)
    S1, B1 = fit[ELECTRODES[0]]; S2, B2 = fit[ELECTRODES[1]]
    eps1 = (df_test[ELECTRODES[0]].values - B1) / S1
    eps2 = (df_test[ELECTRODES[1]].values - B2) / S2
    # weight ~ E2 dominant per Phase G finding; use internal-calib-selected simple average of the two single-electrode estimates' inverse in-sample residual variance
    r1 = np.var(fit[ELECTRODES[0]][0] * df_calib["Strain"].values + fit[ELECTRODES[0]][1] - df_calib[ELECTRODES[0]].values)
    r2v = np.var(fit[ELECTRODES[1]][0] * df_calib["Strain"].values + fit[ELECTRODES[1]][1] - df_calib[ELECTRODES[1]].values)
    w1 = (1 / r1) / (1 / r1 + 1 / r2v)
    eps_hat = w1 * eps1 + (1 - w1) * eps2
    ax.plot(df_test["Strain"].values, lw=0.8, label="measured", alpha=0.85)
    ax.plot(eps_hat, lw=0.8, label="predicted (G3 5-point calib)", alpha=0.85)
    ax.set_title(f"ds{ds} ({TEMP_VAL[ds]}C) -- calibrate exp{exps[0]}, test exp{exps[1]}", fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle("Phase G — G3 (5-point) calibrated predictions, all temperatures")
fig.tight_layout()
fig.savefig(OUT_FIG / "calibrated_predictions_all_temperatures.png", dpi=150)
plt.close(fig)
print("Saved figures/calibrated_predictions_all_temperatures.png")
