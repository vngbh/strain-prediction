"""
Phase E — Identify B(T) and S(T): R_i = B_i(T) + S_i(T, branch) * epsilon + error

Reuses the same turning-point/cycle detection as Phase B (see phase_b_mechanism.py
docstring for the documented deviation: smoothed-strain peak/trough detection
instead of a literal noisy derivative-sign threshold).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import HuberRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_FIG = Path(__file__).resolve().parents[1] / "figures"
OUT_TAB = Path(__file__).resolve().parents[1] / "tables"

DATASETS = [1, 2, 3, 4]
TEMP_VAL = {1: 33, 2: 43, 3: 52, 4: 62}
ELECTRODES = ["Electrode1_resistance", "Electrode2_resistance"]
THRESHOLDS = [0.01, 0.02, 0.05, 0.10]
PROMINENCE = 0.05
N_BOOT = 500
RNG = np.random.RandomState(0)


def load_raw(i):
    return pd.read_csv(DATA_DIR / f"dataset{i}.csv")


def detect_turning_points(strain):
    win = min(51, (len(strain) // 2) * 2 - 1); win = max(win, 5)
    smooth = savgol_filter(strain, win, 3)
    peaks, _ = find_peaks(smooth, prominence=PROMINENCE)
    troughs, _ = find_peaks(-smooth, prominence=PROMINENCE)
    return peaks, troughs


def build_loading_flag(n, peaks, troughs, dwell_frac=0.03):
    turns = sorted([(idx, "peak") for idx in peaks] + [(idx, "trough") for idx in troughs])
    flag = np.zeros(n, dtype=int)
    for (idx_a, kind_a), (idx_b, kind_b) in zip(turns[:-1], turns[1:]):
        seg_len = idx_b - idx_a
        margin = max(1, int(seg_len * dwell_frac))
        lo, hi = idx_a + margin, idx_b - margin
        if lo >= hi:
            continue
        flag[lo:hi] = +1 if (kind_a == "trough" and kind_b == "peak") else -1
    return flag


def cycles_from_troughs(troughs, n):
    bounds = [0] + list(troughs) + [n]
    return list(zip(bounds[:-1], bounds[1:]))


def robust_slope_intercept(x, y):
    """Huber-robust regression. Returns slope, intercept, r2 (from plain OLS fit
    for interpretability), n."""
    if len(x) < 8 or np.ptp(x) < 1e-9:
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan, n=len(x))
    huber = HuberRegressor().fit(x.reshape(-1, 1), y)
    y_ols_pred = huber.predict(x.reshape(-1, 1))
    ss_res = np.sum((y - y_ols_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(slope=float(huber.coef_[0]), intercept=float(huber.intercept_), r2=float(r2), n=len(x))


# ── Per-cycle extraction: B at multiple thresholds + robust S per branch ──────

rows = []
for i in DATASETS:
    df = load_raw(i).sort_values(["Experiment", "Idx"]).reset_index(drop=True)
    for exp, g in df.groupby("Experiment"):
        strain = g["Strain"].values
        peaks, troughs = detect_turning_points(strain)
        flag = build_loading_flag(len(strain), peaks, troughs)
        cycles = cycles_from_troughs(troughs, len(strain))
        for cyc_idx, (lo, hi) in enumerate(cycles):
            seg_strain = strain[lo:hi]
            seg_flag = flag[lo:hi]
            if hi - lo < 20:
                continue
            for elec in ELECTRODES:
                seg_R = g[elec].values[lo:hi]
                row = {"dataset": i, "temperature_C": TEMP_VAL[i], "experiment": exp,
                       "cycle": cyc_idx, "electrode": elec, "n_samples": hi - lo}
                for thr in THRESHOLDS:
                    mask = seg_strain <= (seg_strain.min() + thr)
                    row[f"B_thr{thr}"] = float(np.mean(seg_R[mask])) if mask.sum() > 0 else np.nan
                for branch, bflag in [("loading", 1), ("unloading", -1)]:
                    m = seg_flag == bflag
                    fit = robust_slope_intercept(seg_strain[m], seg_R[m]) if m.sum() >= 8 else \
                          dict(slope=np.nan, intercept=np.nan, r2=np.nan, n=int(m.sum()))
                    row[f"S_{branch}_slope"] = fit["slope"]
                    row[f"S_{branch}_intercept"] = fit["intercept"]
                    row[f"S_{branch}_r2"] = fit["r2"]
                    row[f"S_{branch}_n"] = fit["n"]
                rows.append(row)

bs_df = pd.DataFrame(rows)
bs_df.to_csv(OUT_TAB / "B_S_by_cycle.csv", index=False)
print(f"Saved {len(bs_df)} rows -> tables/B_S_by_cycle.csv")

# ── Threshold sensitivity for B ────────────────────────────────────────────────

thr_sensitivity = (bs_df.groupby(["dataset", "electrode"])[[f"B_thr{t}" for t in THRESHOLDS]]
                   .mean().reset_index())
thr_sensitivity.to_csv(OUT_TAB / "B_threshold_sensitivity.csv", index=False)
print("\nB(T) sensitivity to zero-strain threshold (mean over cycles):")
print(thr_sensitivity.to_string(index=False))

# use the middle threshold (0.02) as the primary B estimate going forward
B_COL = "B_thr0.02"

# ── Bootstrap-by-cycle aggregation per (dataset, electrode) ───────────────────

def bootstrap_mean(values, n_boot=N_BOOT, rng=RNG):
    values = np.asarray(values); values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.mean(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


agg_rows = []
for i in DATASETS:
    for elec in ELECTRODES:
        sub = bs_df[(bs_df.dataset == i) & (bs_df.electrode == elec)]
        b_mean, b_lo, b_hi = bootstrap_mean(sub[B_COL].values)
        sl_mean, sl_lo, sl_hi = bootstrap_mean(sub["S_loading_slope"].values)
        su_mean, su_lo, su_hi = bootstrap_mean(sub["S_unloading_slope"].values)
        agg_rows.append({"dataset": i, "temperature_C": TEMP_VAL[i], "electrode": elec,
                          "B_mean": b_mean, "B_ci_lo": b_lo, "B_ci_hi": b_hi,
                          "S_loading_mean": sl_mean, "S_loading_ci_lo": sl_lo, "S_loading_ci_hi": sl_hi,
                          "S_unloading_mean": su_mean, "S_unloading_ci_lo": su_lo, "S_unloading_ci_hi": su_hi})
agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv(OUT_TAB / "B_S_aggregated_bootstrap.csv", index=False)
print("\nBootstrap-CI aggregated B/S per temperature:\n", agg_df.to_string(index=False))

# loading vs unloading separation check (CIs overlap or not)
sep_rows = []
for _, r in agg_df.iterrows():
    overlap = not (r.S_loading_ci_hi < r.S_unloading_ci_lo or r.S_unloading_ci_hi < r.S_loading_ci_lo)
    sep_rows.append({"dataset": r.dataset, "electrode": r.electrode,
                      "S_loading_mean": r.S_loading_mean, "S_unloading_mean": r.S_unloading_mean,
                      "cis_overlap": overlap, "needs_separate_S": not overlap})
sep_df = pd.DataFrame(sep_rows)
sep_df.to_csv(OUT_TAB / "loading_unloading_separation_check.csv", index=False)

# ── Candidate temperature functions ────────────────────────────────────────────

def fit_constant(T, y):
    return lambda t: np.full_like(np.atleast_1d(t), np.mean(y), dtype=float)

def fit_linear(T, y):
    a, b = np.polyfit(T, y, 1)
    return lambda t: a * np.atleast_1d(t) + b

def fit_quadratic(T, y):
    if len(T) < 3:
        return fit_linear(T, y)
    a, b, c = np.polyfit(T, y, 2)
    return lambda t: a * np.atleast_1d(t) ** 2 + b * np.atleast_1d(t) + c

def fit_reduced_exp(T, y):
    def f(t, a, b):
        return a * np.exp(b * t)
    try:
        p0 = [y[0], 0.01]
        popt, _ = curve_fit(f, T, y, p0=p0, maxfev=20000)
        return lambda t: f(np.atleast_1d(t), *popt)
    except Exception:
        return fit_linear(T, y)

def fit_full_exp(T, y):
    def f(t, a, b, c):
        return a * np.exp(b * t) + c
    try:
        p0 = [1.0, 0.01, y[0]]
        popt, _ = curve_fit(f, T, y, p0=p0, maxfev=20000)
        return lambda t: f(np.atleast_1d(t), *popt)
    except Exception:
        return fit_linear(T, y)

def fit_pchip(T, y):
    order = np.argsort(T)
    interp = PchipInterpolator(np.array(T)[order], np.array(y)[order], extrapolate=True)
    return lambda t: interp(np.atleast_1d(t))


CANDIDATES = {"constant": fit_constant, "linear": fit_linear, "quadratic": fit_quadratic,
              "reduced_exponential": fit_reduced_exp, "full_exponential": fit_full_exp,
              "monotonic_pchip": fit_pchip}


def loo_evaluate(metric_col, label):
    """LOO across the 4 temperatures: fit on other 3, predict held-out, compare to measured."""
    out_rows = []
    for held_out in DATASETS:
        train_ids = [i for i in DATASETS if i != held_out]
        for elec in ELECTRODES:
            T_train = np.array([TEMP_VAL[i] for i in train_ids])
            y_train = np.array([agg_df[(agg_df.dataset == i) & (agg_df.electrode == elec)][metric_col].values[0]
                                 for i in train_ids])
            y_true = agg_df[(agg_df.dataset == held_out) & (agg_df.electrode == elec)][metric_col].values[0]
            T_held = TEMP_VAL[held_out]
            for cand_name, fit_fn in CANDIDATES.items():
                try:
                    f = fit_fn(T_train, y_train)
                    y_pred = float(f(T_held)[0]) if hasattr(f(T_held), "__len__") else float(f(T_held))
                except Exception:
                    y_pred = np.nan
                abs_err = abs(y_pred - y_true)
                rel_err = abs_err / abs(y_true) if y_true else np.nan
                out_rows.append({"metric": label, "held_out_dataset": held_out, "held_out_temp_C": T_held,
                                  "electrode": elec, "candidate": cand_name,
                                  "y_true": y_true, "y_pred": y_pred,
                                  "abs_error": abs_err, "rel_error_pct": 100 * rel_err})
    return pd.DataFrame(out_rows)


B_loo = loo_evaluate("B_mean", "B")
S_loo = loo_evaluate("S_loading_mean", "S_loading")
B_loo.to_csv(OUT_TAB / "B_temperature_model_LOO.csv", index=False)
S_loo.to_csv(OUT_TAB / "S_temperature_model_LOO.csv", index=False)

print("\nB(T) LOO mean abs relative error by candidate function:")
print(B_loo.groupby("candidate")["rel_error_pct"].mean().sort_values().to_string())
print("\nS(T) LOO mean abs relative error by candidate function:")
print(S_loo.groupby("candidate")["rel_error_pct"].mean().sort_values().to_string())

# ── Monotonicity checks ────────────────────────────────────────────────────────

def is_monotonic(vals):
    d = np.diff(vals)
    return bool(np.all(d > 0) or np.all(d < 0))

mono_rows = []
for elec in ELECTRODES:
    b_by_T = [agg_df[(agg_df.dataset == i) & (agg_df.electrode == elec)]["B_mean"].values[0] for i in DATASETS]
    s_by_T = [agg_df[(agg_df.dataset == i) & (agg_df.electrode == elec)]["S_loading_mean"].values[0] for i in DATASETS]
    mono_rows.append({"electrode": elec, "B_values_by_T": b_by_T, "B_monotonic": is_monotonic(b_by_T),
                       "S_values_by_T": s_by_T, "S_monotonic": is_monotonic(s_by_T)})
mono_df = pd.DataFrame(mono_rows)
mono_df.to_csv(OUT_TAB / "monotonicity_check.csv", index=False)
print("\nMonotonicity check:\n", mono_df.to_string(index=False))

# 43C between 33C/52C, using this phase's bootstrap-CI B/S values
between_rows = []
for elec in ELECTRODES:
    for metric in ["B_mean", "S_loading_mean"]:
        v = {i: agg_df[(agg_df.dataset == i) & (agg_df.electrode == elec)][metric].values[0] for i in [1, 2, 3]}
        lo, hi = sorted([v[1], v[3]])
        between_rows.append({"electrode": elec, "metric": metric, "ds1_33C": v[1], "ds2_43C": v[2], "ds3_52C": v[3],
                              "ds2_between": lo <= v[2] <= hi})
between_df = pd.DataFrame(between_rows)
between_df.to_csv(OUT_TAB / "phaseE_43C_between_check.csv", index=False)
print("\n43C-between check (Phase E, bootstrap-CI values):\n", between_df.to_string(index=False))

# ── Figures ────────────────────────────────────────────────────────────────────

for metric_col, loo_df, fname in [("B_mean", B_loo, "B_temperature_fit_LOO.png"),
                                   ("S_loading_mean", S_loo, "S_temperature_fit_LOO.png")]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, elec in zip(axes, ELECTRODES):
        sub = agg_df[agg_df.electrode == elec].sort_values("temperature_C")
        ax.plot(sub.temperature_C, sub[metric_col], "ko-", label="measured", zorder=5)
        T_grid = np.linspace(30, 65, 100)
        for cand_name, fit_fn in CANDIDATES.items():
            T_all = np.array([TEMP_VAL[i] for i in DATASETS])
            y_all = np.array([agg_df[(agg_df.dataset == i) & (agg_df.electrode == elec)][metric_col].values[0]
                               for i in DATASETS])
            try:
                f = fit_fn(T_all, y_all)
                ax.plot(T_grid, f(T_grid), lw=1, alpha=0.6, label=cand_name)
            except Exception:
                pass
        ax.set_title(f"{metric_col} vs T — {elec}", fontsize=9)
        ax.set_xlabel("Temperature (C)")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT_FIG / fname, dpi=150)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
for elec, mk in zip(ELECTRODES, ["o", "s"]):
    sub = agg_df[agg_df.electrode == elec].sort_values("temperature_C")
    ax.errorbar(sub.temperature_C - 0.3, sub.S_loading_mean,
                yerr=[sub.S_loading_mean - sub.S_loading_ci_lo, sub.S_loading_ci_hi - sub.S_loading_mean],
                marker=mk, capsize=4, label=f"{elec.split('_')[0]} loading", linestyle="-")
    ax.errorbar(sub.temperature_C + 0.3, sub.S_unloading_mean,
                yerr=[sub.S_unloading_mean - sub.S_unloading_ci_lo, sub.S_unloading_ci_hi - sub.S_unloading_mean],
                marker=mk, capsize=4, label=f"{elec.split('_')[0]} unloading", linestyle="--")
ax.set_xlabel("Temperature (C)"); ax.set_ylabel("Sensitivity S (robust slope)")
ax.set_title("Loading vs Unloading sensitivity (bootstrap 95% CI)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT_FIG / "loading_unloading_sensitivity.png", dpi=150)
plt.close(fig)

print("\n[Phase E complete]")
