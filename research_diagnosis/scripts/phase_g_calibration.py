"""
Phase G — Calibration-Level Study.

Question: what is the minimum target-temperature/target-session calibration
needed to make the physical forward model R_i = B_i + S_i*epsilon deployable?
Temperature is assumed known (no joint T-epsilon estimation here, per brief).

Protocol (mandatory, per brief):
  - For each of the 4 datasets, its 2 Experiments serve as calib/test in BOTH
    directions (A->B and B->A), reported separately before any averaging.
  - Nothing about the real test Experiment is ever used to pick an electrode,
    a calibration level, epsilon_ref, fusion weights, or any hyperparameter --
    all such choices are made from calibration-internal data only (an
    internal calib/val split carved out of the calibration Experiment itself).

Calibration levels:
  G0  zero target-domain data: B_i(T*), S_i(T*) predicted from the OTHER 3
      temperatures (Phase E full-exponential LOO fit) -- identical setup to
      Phase F's known-T inverse, reused here as the zero-shot reference.
  G1  zero-balance: B_i measured directly from near-zero-strain samples in
      the calibration Experiment (budgets: first stable segment, 10, 50, 100,
      all available); S_i(T*) still from the Phase E LOO fit.
  G2  zero-balance + one reference strain point: S_i(T*) estimated from a
      single calibration strain level; the level itself (low/mid/high) is
      chosen via an internal calib-only train/val split, never the real test.
  G3  multi-point static calibration: zero + {2,3,5} reference strain levels,
      linear (and robust) fit of B_i, S_i from the calibration Experiment.
  G4  one full loading-unloading cycle: B_i, common S_i, branch-split S_i,
      hysteresis area, residual variance, and fusion weights all estimated
      from exactly one cycle; evaluated on LATER cycles of the SAME
      Experiment (same-session) and on the OTHER Experiment (cross-session).

Electrode fusion variants at G1-G4 (fit only on calibration data):
  E1-only, E2-only, equal-weight, calibration-internal-validation-weighted
  (used for both "residual-variance" and "calibration-error" weighting --
  collapsed into one internal-holdout-RMSE-based weight, a documented
  simplification), and a constrained linear fusion (w1+w2=1, w1,w2>=0)
  whose weight is grid-searched using the SAME internal calib/val split.

Session-shift: same-session RMSE (calibrate on part of an Experiment, test on
a held-back part of the SAME Experiment) vs cross-session RMSE (calibrate on
one Experiment, test on the OTHER) at every calibration level.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_FIG = Path(__file__).resolve().parents[1] / "figures"
OUT_TAB = Path(__file__).resolve().parents[1] / "tables"

DATASETS = [1, 2, 3, 4]
TEMP_VAL = {1: 33, 2: 43, 3: 52, 4: 62}
KIND = {1: "extrapolation", 2: "interpolation", 3: "interpolation", 4: "extrapolation"}
ELECTRODES = ["Electrode1_resistance", "Electrode2_resistance"]
PROMINENCE = 0.05
N_BOOT = 300
RNG = np.random.RandomState(0)


# ── Shared helpers (duplicated from phase_f_analytical_inverse.py; that
#    script executes its full pipeline at import time, so it is not imported
#    here to avoid re-running it) ────────────────────────────────────────────

def load_raw(i):
    df = pd.read_csv(DATA_DIR / f"dataset{i}.csv")
    df["dataset_id"] = i
    return df.sort_values(["Experiment", "Idx"]).reset_index(drop=True)


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
    return out


def fit_full_exp(T, y):
    def f(t, a, b, c):
        return a * np.exp(b * t) + c
    try:
        popt, _ = curve_fit(f, T, y, p0=[1.0, 0.01, y[0]], maxfev=20000)
        return lambda t: f(np.asarray(t, dtype=float), *popt)
    except Exception:
        a, b = np.polyfit(T, y, 1)
        return lambda t: a * np.asarray(t, dtype=float) + b


def extract_cycle_BS(df_exp, strain_col="Strain"):
    strain = df_exp[strain_col].values
    peaks, troughs = detect_turning_points(strain)
    flag = build_loading_flag(len(strain), peaks, troughs)
    cycles = cycles_from_troughs(troughs, len(strain))
    out = []
    for cyc_idx, (lo, hi) in enumerate(cycles):
        if hi - lo < 20:
            continue
        seg_strain = strain[lo:hi]; seg_flag = flag[lo:hi]
        for elec in ELECTRODES:
            seg_R = df_exp[elec].values[lo:hi]
            mask_b = seg_strain <= (seg_strain.min() + 0.02)
            B = float(np.mean(seg_R[mask_b])) if mask_b.sum() > 0 else np.nan
            row = {"electrode": elec, "cycle": cyc_idx, "B": B, "lo": lo, "hi": hi}
            for branch, bflag in [("loading", 1), ("unloading", -1)]:
                m = seg_flag == bflag
                if m.sum() >= 8 and np.ptp(seg_strain[m]) > 1e-9:
                    huber = HuberRegressor().fit(seg_strain[m].reshape(-1, 1), seg_R[m])
                    resid = seg_R[m] - huber.predict(seg_strain[m].reshape(-1, 1))
                    row[f"S_{branch}"] = float(huber.coef_[0])
                    row[f"resid_var_{branch}"] = float(np.var(resid))
                else:
                    row[f"S_{branch}"] = np.nan; row[f"resid_var_{branch}"] = np.nan
            out.append(row)
    return pd.DataFrame(out)


def get_train_BS(train_ids, all_cycle_bs):
    fits = {}
    for elec in ELECTRODES:
        for branch in ["loading", "unloading"]:
            T_arr, B_arr, S_arr = [], [], []
            for ds_id in train_ids:
                sub = all_cycle_bs[ds_id]; sub = sub[sub.electrode == elec]
                T_arr.append(TEMP_VAL[ds_id]); B_arr.append(sub["B"].mean()); S_arr.append(sub[f"S_{branch}"].mean())
            fits[(elec, branch)] = {"B_val": fit_full_exp(np.array(T_arr), np.array(B_arr)),
                                     "S_val": fit_full_exp(np.array(T_arr), np.array(S_arr))}
    return fits


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if mask.sum() < 2:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "n": int(mask.sum())}
    return {"r2": float(r2_score(y_true, y_pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)), "bias": float(np.mean(y_pred - y_true)),
            "n": int(mask.sum())}


def bootstrap_ci_by_cycle(df, eps_true_col, eps_pred_col, cycle_col, n_boot=N_BOOT, rng=RNG):
    """95% CI on RMSE, resampling whole cycles with replacement."""
    cycles = df[cycle_col].unique()
    if len(cycles) < 2:
        return np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(cycles, size=len(cycles), replace=True)
        sub = pd.concat([df[df[cycle_col] == c] for c in chosen], ignore_index=True)
        if len(sub) < 5:
            continue
        boots.append(np.sqrt(mean_squared_error(sub[eps_true_col], sub[eps_pred_col])))
    if not boots:
        return np.nan, np.nan
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def eps_from_BS(R, B, S):
    return (R - B) / S


def fuse(eps1, eps2, w1):
    return w1 * eps1 + (1 - w1) * eps2


def select_fusion_weight(eps1_val, eps2_val, eps_true_val):
    """Grid-search w1 in [0,1] on an INTERNAL calibration validation split only."""
    best_w, best_rmse = 0.5, np.inf
    for w1 in np.linspace(0, 1, 21):
        rmse = np.sqrt(mean_squared_error(eps_true_val, fuse(eps1_val, eps2_val, w1)))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w1
    return best_w


def electrode_fusion_variants(R1, R2, B1, S1, B2, S2, eps1_val_internal=None, eps2_val_internal=None,
                               eps_true_internal=None):
    """Returns dict of variant_name -> eps_hat array (evaluated on R1/R2, the
    real test rows). The fusion weight is selected using ONLY the separately
    supplied calibration-internal validation predictions/targets, never R1/R2
    itself, since R1/R2 belongs to the real held-out test Experiment."""
    eps1, eps2 = eps_from_BS(R1, B1, S1), eps_from_BS(R2, B2, S2)
    variants = {"E1_only": eps1, "E2_only": eps2, "equal_weight": fuse(eps1, eps2, 0.5)}
    if eps1_val_internal is not None and len(eps1_val_internal) >= 5:
        w1 = select_fusion_weight(eps1_val_internal, eps2_val_internal, eps_true_internal)
        variants["constrained_linear_fusion"] = fuse(eps1, eps2, w1)
        variants["_selected_w1"] = w1
    else:
        variants["constrained_linear_fusion"] = fuse(eps1, eps2, 0.5)
        variants["_selected_w1"] = 0.5
    return variants


# ── Load data, precompute per-cycle B/S (for G0's train-pool LOO fits) ───────

print("Loading data and extracting per-cycle B/S...")
raw = {i: load_raw(i) for i in DATASETS}
cycle_bs = {}
for i in DATASETS:
    parts = []
    for exp, g in raw[i].groupby("Experiment"):
        p = extract_cycle_BS(g); p["experiment"] = exp; parts.append(p)
    cycle_bs[i] = pd.concat(parts, ignore_index=True)


def calibrate_zero_balance(df_calib, budget):
    """Returns {electrode: (B, n_used)} using near-zero-strain samples."""
    strain = df_calib["Strain"].values
    zero_mask = strain <= (strain.min() + 0.02)
    zero_idx = np.where(zero_mask)[0]
    if budget == "first_segment":
        # first contiguous run of zero_mask from the start of the Experiment
        k = 0
        while k < len(strain) and zero_mask[k]:
            k += 1
        chosen = np.arange(0, max(k, 1))
        chosen = chosen[zero_mask[chosen]] if len(chosen) else zero_idx[:1]
        if len(chosen) == 0:
            chosen = zero_idx[:10]
    elif budget == "all":
        chosen = zero_idx
    else:
        chosen = zero_idx[:budget]
    out = {}
    for elec in ELECTRODES:
        vals = df_calib[elec].values[chosen] if len(chosen) else df_calib[elec].values[zero_idx[:1]]
        out[elec] = (float(np.mean(vals)), len(chosen))
    return out


def calibrate_one_ref(df_calib, B_by_elec, eps_ref, window=0.03):
    strain = df_calib["Strain"].values
    mask = np.abs(strain - eps_ref) <= window
    out = {}
    for elec in ELECTRODES:
        if mask.sum() < 5:
            out[elec] = np.nan
            continue
        R_mean = float(np.mean(df_calib[elec].values[mask]))
        out[elec] = (R_mean - B_by_elec[elec]) / eps_ref
    return out


def calibrate_multi_point(df_calib, ref_levels, robust=False):
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
        pts_x, pts_y = pts_x[valid], pts_y[valid]
        if len(pts_x) < 2:
            out[elec] = (np.nan, np.nan)
            continue
        if robust and len(pts_x) >= 3:
            m = HuberRegressor().fit(pts_x.reshape(-1, 1), pts_y)
            out[elec] = (float(m.coef_[0]), float(m.intercept_))
        else:
            S, B = np.polyfit(pts_x, pts_y, 1)
            out[elec] = (S, B)
    return out


all_results = []          # long table: every (dataset, direction, level, variant) row
fusion_rows = []          # electrode fusion detail
session_rows = []         # same-session vs cross-session
bootstrap_rows = []

for held_out in DATASETS:
    train_ids = [i for i in DATASETS if i != held_out]
    loo_fits = get_train_BS(train_ids, cycle_bs)

    df_ds = raw[held_out]
    exps = sorted(df_ds.Experiment.unique())
    directions = [("A_to_B", exps[0], exps[1]), ("B_to_A", exps[1], exps[0])]

    for direction_name, calib_exp, test_exp in directions:
        df_calib = df_ds[df_ds.Experiment == calib_exp].reset_index(drop=True)
        df_test = df_ds[df_ds.Experiment == test_exp].reset_index(drop=True)
        R1_test, R2_test = df_test[ELECTRODES[0]].values, df_test[ELECTRODES[1]].values
        eps_true_test = df_test["Strain"].values
        T_test = df_test["Temperature"].values

        # internal calib/val split (within the CALIBRATION experiment only,
        # by cycle, never touching the real test experiment)
        calib_cycles_bounds = cycles_from_troughs(
            detect_turning_points(df_calib["Strain"].values)[1], len(df_calib))
        n_cyc = len(calib_cycles_bounds)
        internal_calib_cycles = calib_cycles_bounds[:max(1, n_cyc // 2)]
        internal_val_cycles = calib_cycles_bounds[max(1, n_cyc // 2):]
        internal_val_mask = np.zeros(len(df_calib), dtype=bool)
        for lo, hi in internal_val_cycles:
            internal_val_mask[lo:hi] = True
        eps_true_internal_val = df_calib["Strain"].values[internal_val_mask]
        R1_internal_val = df_calib[ELECTRODES[0]].values[internal_val_mask]
        R2_internal_val = df_calib[ELECTRODES[1]].values[internal_val_mask]

        def record(level, variant, m, extra=None):
            row = {"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                   "direction": direction_name, "level": level, "variant": variant,
                   "r2": m["r2"], "rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"], "n": m["n"]}
            if extra:
                row.update(extra)
            all_results.append(row)

        # ---- G0: zero target-domain calibration (reuses Phase E/F LOO fit) ----
        B1_g0 = loo_fits[(ELECTRODES[0], "loading")]["B_val"](T_test)
        S1_g0 = loo_fits[(ELECTRODES[0], "loading")]["S_val"](T_test)
        B2_g0 = loo_fits[(ELECTRODES[1], "loading")]["B_val"](T_test)
        S2_g0 = loo_fits[(ELECTRODES[1], "loading")]["S_val"](T_test)
        eps1_g0, eps2_g0 = eps_from_BS(R1_test, B1_g0, S1_g0), eps_from_BS(R2_test, B2_g0, S2_g0)
        for vname, eps_hat in [("E1_only", eps1_g0), ("E2_only", eps2_g0),
                               ("equal_weight", fuse(eps1_g0, eps2_g0, 0.5))]:
            record("G0_zero_calibration", vname, metrics(eps_true_test, eps_hat))

        # ---- G1: zero-balance (multiple budgets) ----
        g1_budgets = ["first_segment", 10, 50, 100, "all"]
        g1_best_variant_per_budget = {}
        for budget in g1_budgets:
            calib_B = calibrate_zero_balance(df_calib, budget)
            B1 = calib_B[ELECTRODES[0]][0]; B2 = calib_B[ELECTRODES[1]][0]
            n_used = calib_B[ELECTRODES[0]][1]
            S1_used = loo_fits[(ELECTRODES[0], "loading")]["S_val"](T_test)
            S2_used = loo_fits[(ELECTRODES[1], "loading")]["S_val"](T_test)
            eps1 = eps_from_BS(R1_test, B1, S1_used); eps2 = eps_from_BS(R2_test, B2, S2_used)
            eps1_val = eps_from_BS(R1_internal_val, B1, loo_fits[(ELECTRODES[0], "loading")]["S_val"](TEMP_VAL[held_out]))
            eps2_val = eps_from_BS(R2_internal_val, B2, loo_fits[(ELECTRODES[1], "loading")]["S_val"](TEMP_VAL[held_out]))
            fusion = electrode_fusion_variants(R1_test, R2_test, B1, S1_used, B2, S2_used,
                                                eps1_val_internal=eps1_val, eps2_val_internal=eps2_val,
                                                eps_true_internal=eps_true_internal_val)
            for vname, eps_hat in fusion.items():
                if vname.startswith("_"):
                    continue
                m = metrics(eps_true_test, eps_hat)
                record(f"G1_zero_balance_n{budget}", vname, m, extra={"n_calib_samples": n_used})

        # ---- G2: zero-balance + one reference strain (level chosen internally) ----
        B_zb = calibrate_zero_balance(df_calib, "all")
        B1_zb, B2_zb = B_zb[ELECTRODES[0]][0], B_zb[ELECTRODES[1]][0]
        strain_range = df_calib["Strain"].max() - df_calib["Strain"].min()
        ref_candidates = [df_calib["Strain"].min() + f * strain_range for f in [0.15, 0.5, 0.85]]
        ref_labels = ["low", "mid", "high"]
        best_ref, best_ref_rmse = None, np.inf
        for lvl, lbl in zip(ref_candidates, ref_labels):
            S_internal = calibrate_one_ref(df_calib[~internal_val_mask].reset_index(drop=True),
                                            {ELECTRODES[0]: B1_zb, ELECTRODES[1]: B2_zb}, lvl)
            if np.isnan(S_internal[ELECTRODES[0]]) or np.isnan(S_internal[ELECTRODES[1]]):
                continue
            eps1_v = eps_from_BS(R1_internal_val, B1_zb, S_internal[ELECTRODES[0]])
            eps2_v = eps_from_BS(R2_internal_val, B2_zb, S_internal[ELECTRODES[1]])
            rmse_v = np.sqrt(mean_squared_error(eps_true_internal_val, fuse(eps1_v, eps2_v, 0.5)))
            record("G2_internal_selection_diagnostic", f"ref_{lbl}", metrics(eps_true_internal_val, fuse(eps1_v, eps2_v, 0.5)))
            if rmse_v < best_ref_rmse:
                best_ref_rmse, best_ref, best_ref_label = rmse_v, lvl, lbl
        if best_ref is not None:
            S_g2 = calibrate_one_ref(df_calib, {ELECTRODES[0]: B1_zb, ELECTRODES[1]: B2_zb}, best_ref)
            eps1_val_g2 = eps_from_BS(R1_internal_val, B1_zb, S_g2[ELECTRODES[0]])
            eps2_val_g2 = eps_from_BS(R2_internal_val, B2_zb, S_g2[ELECTRODES[1]])
            fusion = electrode_fusion_variants(R1_test, R2_test, B1_zb, S_g2[ELECTRODES[0]], B2_zb, S_g2[ELECTRODES[1]],
                                                eps1_val_internal=eps1_val_g2, eps2_val_internal=eps2_val_g2,
                                                eps_true_internal=eps_true_internal_val)
            for vname, eps_hat in fusion.items():
                if vname.startswith("_"):
                    continue
                record("G2_zero_balance_plus_one_ref", vname, metrics(eps_true_test, eps_hat),
                       extra={"selected_ref_level": best_ref_label})

        # ---- G3: multi-point static calibration (2, 3, 5 ref points) ----
        for n_pts in [2, 3, 5]:
            ref_levels = list(np.linspace(df_calib["Strain"].min() + 0.05 * strain_range,
                                           df_calib["Strain"].max() - 0.05 * strain_range, n_pts))
            for robust in [False, True]:
                fit = calibrate_multi_point(df_calib, ref_levels, robust=robust)
                S1_g3, B1_g3 = fit[ELECTRODES[0]]; S2_g3, B2_g3 = fit[ELECTRODES[1]]
                if np.isnan(S1_g3) or np.isnan(S2_g3):
                    continue
                eps1_val_g3 = eps_from_BS(R1_internal_val, B1_g3, S1_g3)
                eps2_val_g3 = eps_from_BS(R2_internal_val, B2_g3, S2_g3)
                fusion = electrode_fusion_variants(R1_test, R2_test, B1_g3, S1_g3, B2_g3, S2_g3,
                                                    eps1_val_internal=eps1_val_g3, eps2_val_internal=eps2_val_g3,
                                                    eps_true_internal=eps_true_internal_val)
                fit_label = f"G3_{n_pts}pt_{'robust' if robust else 'linear'}"
                for vname, eps_hat in fusion.items():
                    if vname.startswith("_"):
                        continue
                    record(fit_label, vname, metrics(eps_true_test, eps_hat))

        # ---- G4: one complete loading-unloading cycle ----
        # Cycle 0 (the lead-in segment from the Experiment's start to the first
        # trough) typically has NO samples flagged "loading" (the rise from the
        # very start to the first peak precedes any trough->peak turn pair, so
        # build_loading_flag leaves it unflagged/0) -- S_loading is NaN for it.
        # Use the first cycle index with BOTH branches valid for both electrodes.
        calib_cycle_bs = extract_cycle_BS(df_calib)
        valid_cycles = (calib_cycle_bs.groupby("cycle")
                         .apply(lambda g: g["S_loading"].notna().all() and g["S_unloading"].notna().all(),
                                include_groups=False))
        valid_cycle_ids = sorted(valid_cycles[valid_cycles].index.tolist())
        chosen_cycle = valid_cycle_ids[0] if valid_cycle_ids else None
        first_cycle_lo, first_cycle_hi = (calib_cycles_bounds[chosen_cycle] if chosen_cycle is not None
                                           else calib_cycles_bounds[0])
        cyc0 = calib_cycle_bs[calib_cycle_bs.cycle == chosen_cycle] if chosen_cycle is not None else calib_cycle_bs.iloc[:0]
        if len(cyc0) == len(ELECTRODES):
            B_by_elec = dict(zip(cyc0.electrode, cyc0.B))
            S_common = {e: 0.5 * (cyc0[cyc0.electrode == e]["S_loading"].values[0] +
                                   cyc0[cyc0.electrode == e]["S_unloading"].values[0]) for e in ELECTRODES}
            branch_test = infer_branch_from_resistance(df_test)
            S1_load = cyc0[cyc0.electrode == ELECTRODES[0]]["S_loading"].values[0]
            S1_unl = cyc0[cyc0.electrode == ELECTRODES[0]]["S_unloading"].values[0]
            S2_load = cyc0[cyc0.electrode == ELECTRODES[1]]["S_loading"].values[0]
            S2_unl = cyc0[cyc0.electrode == ELECTRODES[1]]["S_unloading"].values[0]
            S1_branch = np.where(branch_test >= 0, S1_load, S1_unl)
            S2_branch = np.where(branch_test >= 0, S2_load, S2_unl)

            # same-session eval: later cycles of the SAME calib Experiment (excluding cycle 0)
            later_mask = np.arange(len(df_calib)) >= first_cycle_hi
            if later_mask.sum() > 20:
                R1_later, R2_later = df_calib[ELECTRODES[0]].values[later_mask], df_calib[ELECTRODES[1]].values[later_mask]
                eps_true_later = df_calib["Strain"].values[later_mask]
                eps_common_later = fuse(eps_from_BS(R1_later, B_by_elec[ELECTRODES[0]], S_common[ELECTRODES[0]]),
                                         eps_from_BS(R2_later, B_by_elec[ELECTRODES[1]], S_common[ELECTRODES[1]]), 0.5)
                m_same = metrics(eps_true_later, eps_common_later)
                record("G4_one_cycle_common_S", "equal_weight_SAME_SESSION", m_same)

            eps_common_test = fuse(eps_from_BS(R1_test, B_by_elec[ELECTRODES[0]], S_common[ELECTRODES[0]]),
                                    eps_from_BS(R2_test, B_by_elec[ELECTRODES[1]], S_common[ELECTRODES[1]]), 0.5)
            eps_branch_test = fuse(eps_from_BS(R1_test, B_by_elec[ELECTRODES[0]], S1_branch),
                                    eps_from_BS(R2_test, B_by_elec[ELECTRODES[1]], S2_branch), 0.5)
            m_cross_common = metrics(eps_true_test, eps_common_test)
            m_cross_branch = metrics(eps_true_test, eps_branch_test)
            record("G4_one_cycle_common_S", "equal_weight_CROSS_SESSION", m_cross_common)
            record("G4_one_cycle_branch_S", "equal_weight_CROSS_SESSION", m_cross_branch)

            if later_mask.sum() > 20:
                session_rows.append({"held_out": held_out, "direction": direction_name, "level": "G4_one_cycle",
                                      "same_session_rmse": m_same["rmse"], "cross_session_rmse": m_cross_common["rmse"],
                                      "session_shift_penalty": m_cross_common["rmse"] - m_same["rmse"]})

    print(f"held_out=ds{held_out} ({TEMP_VAL[held_out]}C, {KIND[held_out]}) done")

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_TAB / "calibration_results_all.csv", index=False)
print(f"\nSaved {len(results_df)} rows -> tables/calibration_results_all.csv")

# ── Summary table: best variant per (dataset, direction, level) + level progression ──

summary_rows = []
for (ds, direction, level), g in results_df.groupby(["held_out", "direction", "level"]):
    g = g[g["rmse"].notna()]
    if len(g) == 0:
        continue
    best = g.loc[g["rmse"].idxmin()]
    summary_rows.append({"held_out": ds, "temperature": TEMP_VAL[ds], "kind": KIND[ds],
                          "direction": direction, "level": level, "best_variant": best["variant"],
                          "r2": best["r2"], "rmse": best["rmse"], "mae": best["mae"], "n": best["n"]})
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_TAB / "calibration_summary.csv", index=False)

pd.DataFrame(session_rows).to_csv(OUT_TAB / "session_shift_penalty.csv", index=False)

# ── Bootstrap CI (by cycle) for the best variant at each level, ds2 as example-heavy case ──

boot_rows = []
for ds in DATASETS:
    df_ds = raw[ds]
    exps = sorted(df_ds.Experiment.unique())
    df_test_for_boot = df_ds[df_ds.Experiment == exps[1]].reset_index(drop=True)
    peaks, troughs = detect_turning_points(df_test_for_boot["Strain"].values)
    cycles = cycles_from_troughs(troughs, len(df_test_for_boot))
    cyc_id = np.zeros(len(df_test_for_boot), dtype=int)
    for ci, (lo, hi) in enumerate(cycles):
        cyc_id[lo:hi] = ci
    df_test_for_boot = df_test_for_boot.assign(cycle_id=cyc_id)
    train_ids = [i for i in DATASETS if i != ds]
    loo_fits = get_train_BS(train_ids, cycle_bs)
    T_t = df_test_for_boot["Temperature"].values
    B1 = loo_fits[(ELECTRODES[0], "loading")]["B_val"](T_t); S1 = loo_fits[(ELECTRODES[0], "loading")]["S_val"](T_t)
    B2 = loo_fits[(ELECTRODES[1], "loading")]["B_val"](T_t); S2 = loo_fits[(ELECTRODES[1], "loading")]["S_val"](T_t)
    eps_hat = fuse(eps_from_BS(df_test_for_boot[ELECTRODES[0]].values, B1, S1),
                    eps_from_BS(df_test_for_boot[ELECTRODES[1]].values, B2, S2), 0.5)
    tmp = pd.DataFrame({"eps_true": df_test_for_boot["Strain"].values, "eps_pred": eps_hat, "cycle_id": cyc_id})
    lo_ci, hi_ci = bootstrap_ci_by_cycle(tmp, "eps_true", "eps_pred", "cycle_id")
    boot_rows.append({"held_out": ds, "level": "G0_zero_calibration", "variant": "equal_weight",
                       "rmse_ci_lo": lo_ci, "rmse_ci_hi": hi_ci})
pd.DataFrame(boot_rows).to_csv(OUT_TAB / "calibration_bootstrap_ci.csv", index=False)

# reference-strain selection diagnostic table (already partly in all_results; also save standalone)
ref_sel = results_df[results_df.level == "G2_internal_selection_diagnostic"]
ref_sel.to_csv(OUT_TAB / "reference_strain_selection.csv", index=False)

# electrode fusion detail (subset of all_results for readability)
fusion_view = results_df[results_df.variant.isin(
    ["E1_only", "E2_only", "equal_weight", "constrained_linear_fusion"])]
fusion_view.to_csv(OUT_TAB / "electrode_fusion_calibration.csv", index=False)

print("\n[Phase G tables saved]")

# ── Figures ────────────────────────────────────────────────────────────────────

LEVEL_ORDER = ["G0_zero_calibration", "G1_zero_balance_nfirst_segment", "G1_zero_balance_n10",
               "G1_zero_balance_n50", "G1_zero_balance_n100", "G1_zero_balance_nall",
               "G2_zero_balance_plus_one_ref", "G3_2pt_linear", "G3_3pt_linear", "G3_5pt_linear",
               "G4_one_cycle_common_S"]
LEVEL_SHORT = {l: l.replace("G1_zero_balance_n", "G1_n=").replace("_zero_calibration", "")
               .replace("_zero_balance_plus_one_ref", "").replace("_one_cycle_common_S", "")
               .replace("pt_linear", "pt") for l in LEVEL_ORDER}

best_by_level = (results_df[results_df.level.isin(LEVEL_ORDER)]
                  .groupby(["held_out", "level"])[["r2", "rmse"]].min().reset_index())

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ds in DATASETS:
    sub = best_by_level[best_by_level.held_out == ds].set_index("level").reindex(LEVEL_ORDER)
    axes[0].plot(range(len(LEVEL_ORDER)), sub["r2"], marker="o", label=f"ds{ds} ({TEMP_VAL[ds]}C)")
    axes[1].plot(range(len(LEVEL_ORDER)), sub["rmse"], marker="o", label=f"ds{ds} ({TEMP_VAL[ds]}C)")
for ax, ylabel in zip(axes, ["R2 (best variant)", "RMSE (best variant)"]):
    ax.set_xticks(range(len(LEVEL_ORDER))); ax.set_xticklabels([LEVEL_SHORT[l] for l in LEVEL_ORDER], rotation=45, fontsize=7)
    ax.set_ylabel(ylabel); ax.legend(fontsize=7)
fig.suptitle("Calibration curve: G0 -> G4 (best electrode-fusion variant per level)")
fig.tight_layout(); fig.savefig(OUT_FIG / "calibration_curve_R2.png", dpi=150); plt.close(fig)
# separate RMSE-only figure too (brief asks for both explicitly)
fig, ax = plt.subplots(figsize=(8, 5))
for ds in DATASETS:
    sub = best_by_level[best_by_level.held_out == ds].set_index("level").reindex(LEVEL_ORDER)
    ax.plot(range(len(LEVEL_ORDER)), sub["rmse"], marker="o", label=f"ds{ds} ({TEMP_VAL[ds]}C)")
ax.set_xticks(range(len(LEVEL_ORDER))); ax.set_xticklabels([LEVEL_SHORT[l] for l in LEVEL_ORDER], rotation=45, fontsize=7)
ax.set_ylabel("RMSE"); ax.set_title("Calibration curve: RMSE vs calibration level"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT_FIG / "calibration_curve_RMSE.png", dpi=150); plt.close(fig)

if session_rows:
    sdf = pd.DataFrame(session_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sdf))
    ax.bar(x - 0.15, sdf.same_session_rmse, width=0.3, label="Same-session")
    ax.bar(x + 0.15, sdf.cross_session_rmse, width=0.3, label="Cross-session")
    ax.set_xticks(x); ax.set_xticklabels([f"ds{r.held_out}\n{r.direction}" for r in sdf.itertuples()], fontsize=7)
    ax.set_ylabel("RMSE"); ax.set_title("Same-session vs cross-session (G4, one-cycle calibration)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT_FIG / "same_vs_cross_session_calibration.png", dpi=150); plt.close(fig)

fusion_summary = (fusion_view.groupby(["level", "variant"])["rmse"].mean().reset_index())
fig, ax = plt.subplots(figsize=(10, 5))
for variant in ["E1_only", "E2_only", "equal_weight", "constrained_linear_fusion"]:
    sub = fusion_summary[fusion_summary.variant == variant]
    sub = sub.set_index("level").reindex([l for l in LEVEL_ORDER if l in sub.level.values])
    ax.plot(range(len(sub)), sub["rmse"], marker="o", label=variant)
ax.set_xticks(range(len(sub))); ax.set_xticklabels([LEVEL_SHORT.get(l, l) for l in sub.index], rotation=45, fontsize=7)
ax.set_ylabel("Mean RMSE across datasets/directions"); ax.set_title("Electrode fusion strategy by calibration level")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT_FIG / "electrode_fusion_by_calibration_level.png", dpi=150); plt.close(fig)

ref_diag = results_df[results_df.level == "G2_internal_selection_diagnostic"]
if len(ref_diag):
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds in DATASETS:
        sub = ref_diag[ref_diag.held_out == ds]
        ax.plot(sub.variant, sub.rmse, marker="o", label=f"ds{ds}")
    ax.set_ylabel("Internal-validation RMSE"); ax.set_title("Reference-strain-level selection (calib-internal only)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT_FIG / "reference_strain_comparison.png", dpi=150); plt.close(fig)

print("[Phase G complete]")
