"""
Phase B — Data Mechanism Diagnosis
Standalone diagnostic script. Does not modify code/ or code/results/.

Cycle/loading-unloading detection note (documented deviation from the brief):
Raw sample-to-sample strain derivative is dominated by measurement noise
(~3000 sign changes per ~11000 samples), so a literal dε/dt-sign threshold
would not recover real loading/unloading branches. Instead we Savitzky-Golay
smooth the strain signal per (dataset, experiment) and find turning points
(peaks/troughs) via scipy.signal.find_peaks with a prominence threshold. This
recovers ~10 clean cycles per experiment (visually confirmed against raw
strain range 0 -> 0.5). loading_flag is then assigned per-sample from the
segment between consecutive turning points; samples within a small margin of
a turning point are flagged as dwell/turning (0), matching the spirit (not
literal derivative-threshold form) of the requested loading_flag(t) rule.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
OUT_FIG    = Path(__file__).resolve().parents[1] / "figures"
OUT_TAB    = Path(__file__).resolve().parents[1] / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

DATASETS   = [1, 2, 3, 4]
TEMP_LABEL = {1: "33C", 2: "43C", 3: "52C", 4: "62C"}
TEMP_VAL   = {1: 33, 2: 43, 3: 52, 4: 62}
ELECTRODES = ["Electrode1_resistance", "Electrode2_resistance"]
DWELL_FRAC = 0.03   # fraction of a cycle's length around each turning point flagged as dwell (0)
PROMINENCE = 0.05   # strain-units prominence for turning-point detection


def load_raw(i):
    return pd.read_csv(DATA_DIR / f"dataset{i}.csv")


def detect_turning_points(strain):
    win = min(51, (len(strain) // 2) * 2 - 1)
    win = max(win, 5)
    smooth = savgol_filter(strain, win, 3)
    peaks, _   = find_peaks(smooth, prominence=PROMINENCE)
    troughs, _ = find_peaks(-smooth, prominence=PROMINENCE)
    return smooth, peaks, troughs


def build_loading_flag(n, peaks, troughs):
    """+1 loading (trough->peak), -1 unloading (peak->trough), 0 near turning points."""
    turns = sorted([(idx, "peak") for idx in peaks] + [(idx, "trough") for idx in troughs])
    flag = np.zeros(n, dtype=int)
    for (idx_a, kind_a), (idx_b, kind_b) in zip(turns[:-1], turns[1:]):
        seg_len = idx_b - idx_a
        margin = max(1, int(seg_len * DWELL_FRAC))
        lo, hi = idx_a + margin, idx_b - margin
        if lo >= hi:
            continue
        direction = +1 if (kind_a == "trough" and kind_b == "peak") else -1
        flag[lo:hi] = direction
    return flag


def cycles_from_troughs(troughs, n):
    """Cycle i = [trough_i, trough_{i+1}) ; also include lead-in [0, trough_0) and tail as partial cycles."""
    bounds = [0] + list(troughs) + [n]
    return list(zip(bounds[:-1], bounds[1:]))


def robust_slope(x, y):
    if len(x) < 5 or np.ptp(x) < 1e-9:
        return np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def hysteresis_area(strain, resistance, load_flag):
    """Shoelace-style area between loading and unloading branch of one cycle (R vs strain)."""
    load_mask, unload_mask = load_flag == 1, load_flag == -1
    if load_mask.sum() < 5 or unload_mask.sum() < 5:
        return np.nan
    s_l, r_l = strain[load_mask], resistance[load_mask]
    s_u, r_u = strain[unload_mask], resistance[unload_mask]
    order_l = np.argsort(s_l); s_l, r_l = s_l[order_l], r_l[order_l]
    order_u = np.argsort(s_u); s_u, r_u = s_u[order_u], r_u[order_u]
    lo, hi = max(s_l.min(), s_u.min()), min(s_l.max(), s_u.max())
    if hi <= lo:
        return np.nan
    grid = np.linspace(lo, hi, 200)
    r_l_interp = np.interp(grid, s_l, r_l)
    r_u_interp = np.interp(grid, s_u, r_u)
    return float(np.trapezoid(np.abs(r_l_interp - r_u_interp), grid))


# ── Main per-dataset / per-experiment / per-cycle / per-electrode extraction ──

rows = []
all_data = {}   # ds -> dataframe with loading_flag column, for plotting

for i in DATASETS:
    df = load_raw(i)
    df = df.sort_values(["Experiment", "Idx"]).reset_index(drop=True)
    flags = np.zeros(len(df), dtype=int)

    for exp, g in df.groupby("Experiment"):
        idx0 = g.index[0]
        strain = g["Strain"].values
        smooth, peaks, troughs = detect_turning_points(strain)
        flag = build_loading_flag(len(strain), peaks, troughs)
        flags[g.index] = flag

        cycles = cycles_from_troughs(troughs, len(strain))
        for cyc_idx, (lo, hi) in enumerate(cycles):
            seg_strain = strain[lo:hi]
            seg_flag   = flag[lo:hi]
            for elec in ELECTRODES:
                seg_R = g[elec].values[lo:hi]

                # baseline B: resistance near local strain minimum within this cycle
                near0_mask = seg_strain <= (seg_strain.min() + 0.02)
                B = float(np.mean(seg_R[near0_mask])) if near0_mask.sum() > 0 else np.nan

                load_mask, unload_mask = seg_flag == 1, seg_flag == -1
                s_load  = robust_slope(seg_strain[load_mask],  seg_R[load_mask])  if load_mask.sum()  >= 5 else np.nan
                s_unload = robust_slope(seg_strain[unload_mask], seg_R[unload_mask]) if unload_mask.sum() >= 5 else np.nan
                h_area  = hysteresis_area(seg_strain, seg_R, seg_flag)

                rows.append({
                    "dataset": i, "temperature_C": TEMP_VAL[i], "experiment": exp,
                    "cycle": cyc_idx, "electrode": elec,
                    "n_samples": hi - lo,
                    "B_baseline_resistance": B,
                    "S_loading_slope": s_load,
                    "S_unloading_slope": s_unload,
                    "hysteresis_area": h_area,
                    "strain_min": float(seg_strain.min()), "strain_max": float(seg_strain.max()),
                })

    df["loading_flag"] = flags
    all_data[i] = df

mech_df = pd.DataFrame(rows)
mech_df.to_csv(OUT_TAB / "mechanism_statistics.csv", index=False)
print(f"Saved {len(mech_df)} rows -> tables/mechanism_statistics.csv")

# ── Aggregate B(T), S(T), hysteresis(T) per electrode ─────────────────────────

agg = (mech_df.groupby(["dataset", "temperature_C", "electrode"])
       .agg(B_mean=("B_baseline_resistance", "mean"), B_std=("B_baseline_resistance", "std"),
            S_load_mean=("S_loading_slope", "mean"), S_load_std=("S_loading_slope", "std"),
            S_unload_mean=("S_unloading_slope", "mean"), S_unload_std=("S_unloading_slope", "std"),
            H_mean=("hysteresis_area", "mean"), H_std=("hysteresis_area", "std"),
            n_cycles=("cycle", "count"))
       .reset_index())
agg.to_csv(OUT_TAB / "mechanism_aggregated_by_temperature.csv", index=False)
print(agg.to_string(index=False))

# variability between experiments (per dataset/electrode)
exp_var = (mech_df.groupby(["dataset", "electrode", "experiment"])
           .agg(B_mean=("B_baseline_resistance", "mean"), S_load_mean=("S_loading_slope", "mean"))
           .reset_index())
exp_var_summary = (exp_var.groupby(["dataset", "electrode"])
                   .agg(B_between_exp_std=("B_mean", "std"), S_load_between_exp_std=("S_load_mean", "std"))
                   .reset_index())
exp_var_summary.to_csv(OUT_TAB / "mechanism_between_experiment_variability.csv", index=False)

# drift: first cycle vs last cycle B, per dataset/experiment/electrode
drift_rows = []
for (ds, exp, elec), g in mech_df.groupby(["dataset", "experiment", "electrode"]):
    g = g.sort_values("cycle")
    if len(g) < 2:
        continue
    b_first, b_last = g["B_baseline_resistance"].iloc[0], g["B_baseline_resistance"].iloc[-1]
    drift_rows.append({"dataset": ds, "experiment": exp, "electrode": elec,
                        "B_first_cycle": b_first, "B_last_cycle": b_last,
                        "B_drift_abs": b_last - b_first,
                        "B_drift_pct": 100 * (b_last - b_first) / b_first if b_first else np.nan})
pd.DataFrame(drift_rows).to_csv(OUT_TAB / "mechanism_drift_first_vs_last_cycle.csv", index=False)

# ── 43C-between-33-and-52 check (multiple granularities) ──────────────────────

def between_check(vals_by_ds):
    """vals_by_ds: dict {1: val_ds1, 2: val_ds2, 3: val_ds3}. Returns True if ds2 between ds1,ds3."""
    if any(pd.isna(v) for v in vals_by_ds.values()):
        return None
    lo, hi = sorted([vals_by_ds[1], vals_by_ds[3]])
    return lo <= vals_by_ds[2] <= hi

between_rows = []
for elec in ELECTRODES:
    # per dataset mean (coarsest)
    means = {ds: mech_df[(mech_df.dataset == ds) & (mech_df.electrode == elec)]["B_baseline_resistance"].mean()
             for ds in [1, 2, 3]}
    between_rows.append({"granularity": "dataset_mean", "metric": "B", "electrode": elec,
                          "ds1_33C": means[1], "ds2_43C": means[2], "ds3_52C": means[3],
                          "ds2_between_ds1_ds3": between_check(means)})
    means_s = {ds: mech_df[(mech_df.dataset == ds) & (mech_df.electrode == elec)]["S_loading_slope"].mean()
               for ds in [1, 2, 3]}
    between_rows.append({"granularity": "dataset_mean", "metric": "S_loading", "electrode": elec,
                          "ds1_33C": means_s[1], "ds2_43C": means_s[2], "ds3_52C": means_s[3],
                          "ds2_between_ds1_ds3": between_check(means_s)})

    # per experiment (finer)
    for exp_pos in [0, 1]:  # first / second experiment of each dataset
        exps = {ds: sorted(mech_df[mech_df.dataset == ds]["experiment"].unique())[exp_pos]
                for ds in [1, 2, 3]}
        vals = {ds: mech_df[(mech_df.dataset == ds) & (mech_df.electrode == elec) & (mech_df.experiment == exps[ds])]
                ["B_baseline_resistance"].mean() for ds in [1, 2, 3]}
        between_rows.append({"granularity": f"experiment_pos{exp_pos}", "metric": "B", "electrode": elec,
                              "ds1_33C": vals[1], "ds2_43C": vals[2], "ds3_52C": vals[3],
                              "ds2_between_ds1_ds3": between_check(vals)})

between_df = pd.DataFrame(between_rows)
between_df.to_csv(OUT_TAB / "mechanism_43C_between_check.csv", index=False)
print("\n43C-between-33-52C check:\n", between_df.to_string(index=False))

# ── Figures ────────────────────────────────────────────────────────────────────

colors = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}

# R_vs_strain_all_temperatures_{E1,E2}
for elec, fname in zip(ELECTRODES, ["E1", "E2"]):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in DATASETS:
        df = all_data[i]
        ax.scatter(df["Strain"], df[elec], s=1, alpha=0.15, color=colors[i], label=f"ds{i} ({TEMP_LABEL[i]})")
    ax.set_xlabel("Strain"); ax.set_ylabel(f"{fname} resistance")
    ax.set_title(f"Resistance vs Strain — all temperatures ({fname})")
    leg = ax.legend(markerscale=10, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"R_vs_strain_all_temperatures_{fname}.png", dpi=150)
    plt.close(fig)

# R_vs_strain_loading_unloading_{E1,E2}  (ds1-4, one experiment each, colored by branch)
for elec, fname in zip(ELECTRODES, ["E1", "E2"]):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(20, 4), sharey=False)
    for j, i in enumerate(DATASETS):
        df = all_data[i]
        exp0 = sorted(df["Experiment"].unique())[0]
        g = df[df["Experiment"] == exp0]
        ax = axes[j]
        ax.scatter(g.loc[g.loading_flag == 1, "Strain"], g.loc[g.loading_flag == 1, elec],
                   s=2, alpha=0.4, color="C0", label="loading")
        ax.scatter(g.loc[g.loading_flag == -1, "Strain"], g.loc[g.loading_flag == -1, elec],
                   s=2, alpha=0.4, color="C3", label="unloading")
        ax.set_title(f"ds{i} ({TEMP_LABEL[i]}) exp{exp0}", fontsize=9)
        ax.set_xlabel("Strain")
        if j == 0:
            ax.set_ylabel(f"{fname} resistance"); ax.legend(fontsize=7, markerscale=4)
    fig.suptitle(f"Loading vs Unloading branches ({fname}, first experiment per dataset)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"R_vs_strain_loading_unloading_{fname}.png", dpi=150)
    plt.close(fig)

# baseline_B_vs_temperature
fig, ax = plt.subplots(figsize=(7, 5))
for elec, fname, mk in zip(ELECTRODES, ["E1", "E2"], ["o", "s"]):
    sub = agg[agg.electrode == elec].sort_values("temperature_C")
    ax.errorbar(sub.temperature_C, sub.B_mean, yerr=sub.B_std, marker=mk, capsize=4, label=fname)
ax.set_xlabel("Temperature (C)"); ax.set_ylabel("Baseline resistance B (near strain=0)")
ax.set_title("Baseline B(T)"); ax.legend()
fig.tight_layout()
fig.savefig(OUT_FIG / "baseline_B_vs_temperature.png", dpi=150)
plt.close(fig)

# sensitivity_S_vs_temperature
fig, ax = plt.subplots(figsize=(7, 5))
for elec, fname, mk in zip(ELECTRODES, ["E1", "E2"], ["o", "s"]):
    sub = agg[agg.electrode == elec].sort_values("temperature_C")
    ax.errorbar(sub.temperature_C, sub.S_load_mean, yerr=sub.S_load_std, marker=mk, capsize=4,
                label=f"{fname} loading", linestyle="-")
    ax.errorbar(sub.temperature_C, sub.S_unload_mean, yerr=sub.S_unload_std, marker=mk, capsize=4,
                label=f"{fname} unloading", linestyle="--")
ax.set_xlabel("Temperature (C)"); ax.set_ylabel("Sensitivity S = dR/dStrain")
ax.set_title("Sensitivity S(T)"); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_FIG / "sensitivity_S_vs_temperature.png", dpi=150)
plt.close(fig)

# hysteresis_area_vs_temperature
fig, ax = plt.subplots(figsize=(7, 5))
for elec, fname, mk in zip(ELECTRODES, ["E1", "E2"], ["o", "s"]):
    sub = agg[agg.electrode == elec].sort_values("temperature_C")
    ax.errorbar(sub.temperature_C, sub.H_mean, yerr=sub.H_std, marker=mk, capsize=4, label=fname)
ax.set_xlabel("Temperature (C)"); ax.set_ylabel("Hysteresis loop area")
ax.set_title("Hysteresis area vs Temperature"); ax.legend()
fig.tight_layout()
fig.savefig(OUT_FIG / "hysteresis_area_vs_temperature.png", dpi=150)
plt.close(fig)

print("\n[Phase B complete] figures + tables saved.")
