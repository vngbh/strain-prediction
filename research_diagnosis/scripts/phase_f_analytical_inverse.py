"""
Phase F (v2, post-review fixes) — Analytical inverse baseline (F1 known-T,
F2 unknown-T joint) + Jacobian/identifiability analysis + mandatory
comparison.

Three corrections applied after review of the v1 results:

1. F2 RMSE=29.9 bug: the [-0.05, 0.55] epsilon constraint only excluded bad
   candidates from winning the argmin -- rows where EVERY grid candidate was
   invalid (possible near a near-singular S(T)) still returned an unclipped,
   unbounded epsilon from whatever candidate happened to sit at the grid's
   first index. Fixed by (a) hard-clipping the final returned epsilon to the
   physical range, and (b) explicitly counting/reporting rows where the
   search found no valid candidate ("search_failed" rate).

2. "Oracle" renamed / split into two distinct concepts:
   - `session_calibrated_inverse` (was "oracle" in v1): B/S measured on the
     SAME held-out dataset's FIRST Experiment, evaluated on its SECOND
     Experiment. Still subject to session-to-session (Experiment-to-Experiment)
     baseline/sensitivity shift -- NOT a pure forward-model-error measurement.
   - `true_oracle_within_experiment`: B/S measured on HALF the cycles of the
     SAME test Experiment (even-indexed cycles), evaluated on the OTHER half
     (odd-indexed cycles) of that same Experiment. This isolates pure
     forward-model-form error (hysteresis/nonlinearity not captured by the
     static R = B(T) + S(T)*epsilon form) with no cross-experiment session
     shift and no cross-temperature B/S-prediction error at all.

3. Electrode/variant selection leakage: v1 picked "E1_only" as the preferred
   F1 variant by looking at its performance on the real test set -- a form of
   model-selection leakage. Fixed by a nested, train-pool-only selection: for
   each real LOO fold, each of the 3 training datasets is used once as a
   pseudo-held-out set (fit B/S on the other 2 training datasets only,
   evaluate each variant's RMSE on it), the resulting 3-fold nested RMSE is
   averaged per variant, and the variant with the lowest nested (train-only)
   RMSE is the one applied to the real test set. The full per-variant table
   on the real test set is still reported, but explicitly labeled diagnostic/
   post-hoc, not used for selection.

Branch (loading/unloading) is inferred from the sign of a smoothed dR/dt
(both electrodes averaged) -- never from the true strain derivative, so the
inverse models stay deployable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_FIG = Path(__file__).resolve().parents[1] / "figures"
OUT_TAB = Path(__file__).resolve().parents[1] / "tables"

DATASETS = [1, 2, 3, 4]
TEMP_VAL = {1: 33, 2: 43, 3: 52, 4: 62}
KIND = {1: "extrapolation", 2: "interpolation", 3: "interpolation", 4: "extrapolation"}
ELECTRODES = ["Electrode1_resistance", "Electrode2_resistance"]
PROMINENCE = 0.05
SEEDS = [0, 42, 123]


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
    """Deployable branch proxy: sign of smoothed d(mean(E1,E2))/dt. No true strain used."""
    r_avg = df[ELECTRODES].mean(axis=1).values
    win = min(31, (len(r_avg) // 2) * 2 - 1); win = max(win, 5)
    smooth = savgol_filter(r_avg, win, 2)
    d = np.gradient(smooth)
    thresh = np.std(d) * 0.15
    branch = np.where(d > thresh, 1, np.where(d < -thresh, -1, 0))
    # fill dwell (0) with the previous non-zero branch (forward-fill), default to loading
    last = 1
    out = np.empty_like(branch)
    for k in range(len(branch)):
        if branch[k] != 0:
            last = branch[k]
        out[k] = last
    return out


def extract_cycle_BS(df_exp, strain_col="Strain"):
    """Returns list of dicts: per-electrode per-branch (B via thr=0.02, S via Huber slope+residuals)."""
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
            row = {"electrode": elec, "cycle": cyc_idx, "B": B}
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


def fit_full_exp(T, y):
    def f(t, a, b, c):
        return a * np.exp(b * t) + c
    try:
        popt, _ = curve_fit(f, T, y, p0=[1.0, 0.01, y[0]], maxfev=20000)
        def val(t): return f(np.asarray(t, dtype=float), *popt)
        def deriv(t): return popt[0] * popt[1] * np.exp(popt[1] * np.asarray(t, dtype=float))
        return val, deriv
    except Exception:
        a, b = np.polyfit(T, y, 1)
        return (lambda t: a * np.asarray(t, dtype=float) + b), (lambda t: np.full_like(np.asarray(t, dtype=float), a))


def get_train_BS(train_ids, all_cycle_bs):
    """Aggregate mean B/S per electrode/branch across the given train datasets, fit full_exponential(T)."""
    fits = {}
    residual_vars = {}
    for elec in ELECTRODES:
        for branch in ["loading", "unloading"]:
            T_arr, B_arr, S_arr, rv_list = [], [], [], []
            for ds_id in train_ids:
                sub = all_cycle_bs[ds_id]
                sub = sub[sub.electrode == elec]
                T_arr.append(TEMP_VAL[ds_id])
                B_arr.append(sub["B"].mean())
                S_arr.append(sub[f"S_{branch}"].mean())
                rv_list.extend(sub[f"resid_var_{branch}"].dropna().tolist())
            B_val, B_deriv = fit_full_exp(np.array(T_arr), np.array(B_arr))
            S_val, S_deriv = fit_full_exp(np.array(T_arr), np.array(S_arr))
            fits[(elec, branch)] = {"B_val": B_val, "B_deriv": B_deriv, "S_val": S_val, "S_deriv": S_deriv}
            residual_vars[(elec, branch)] = np.mean(rv_list) if rv_list else 1.0
    return fits, residual_vars


def known_T_inverse(R1, R2, T, branch, fits, residual_vars, mode="both_weighted"):
    """epsilon_hat = sum_i w_i S_i(T)(R_i - B_i(T)) / sum_i w_i S_i(T)^2 ; branch-aware S."""
    def per_electrode(elec, R):
        S_load = fits[(elec, "loading")]["S_val"](T)
        S_unload = fits[(elec, "unloading")]["S_val"](T)
        B_load = fits[(elec, "loading")]["B_val"](T)
        if mode == "common_S":
            S = 0.5 * (S_load + S_unload); B = B_load
        else:
            S = np.where(branch >= 0, S_load, S_unload)
            B = B_load  # B is branch-independent (near-zero-strain baseline)
        w = 1.0 / residual_vars[(elec, "loading" if mode != "unloading_only" else "unloading")]
        return S, B, w

    S1, B1, w1 = per_electrode(ELECTRODES[0], R1)
    S2, B2, w2 = per_electrode(ELECTRODES[1], R2)

    if mode == "E1_only":
        w2 = 0 * w2 if np.isscalar(w2) else np.zeros_like(S2)
        w1 = 1.0
    elif mode == "E2_only":
        w1 = 0.0
        w2 = 1.0

    num = w1 * S1 * (R1 - B1) + w2 * S2 * (R2 - B2)
    den = w1 * S1 ** 2 + w2 * S2 ** 2
    return num / np.where(den == 0, np.nan, den)


def unknown_T_inverse(R1, R2, branch, fits, residual_vars, T_grid_coarse, T_refine_halfwidth=2.0, n_refine=21):
    """Vectorized coarse-to-fine grid search over T; epsilon solved analytically at each T."""
    n = len(R1)

    def total_residual_and_eps(T_candidates):
        # T_candidates: (n, k) array -- one row per sample, k candidate T values
        S1_load = fits[(ELECTRODES[0], "loading")]["S_val"](T_candidates)
        S1_unl = fits[(ELECTRODES[0], "unloading")]["S_val"](T_candidates)
        S2_load = fits[(ELECTRODES[1], "loading")]["S_val"](T_candidates)
        S2_unl = fits[(ELECTRODES[1], "unloading")]["S_val"](T_candidates)
        B1 = fits[(ELECTRODES[0], "loading")]["B_val"](T_candidates)
        B2 = fits[(ELECTRODES[1], "loading")]["B_val"](T_candidates)
        br = branch.reshape(-1, 1)
        S1 = np.where(br >= 0, S1_load, S1_unl)
        S2 = np.where(br >= 0, S2_load, S2_unl)
        w1 = 1.0 / residual_vars[(ELECTRODES[0], "loading")]
        w2 = 1.0 / residual_vars[(ELECTRODES[1], "loading")]
        R1c, R2c = R1.reshape(-1, 1), R2.reshape(-1, 1)
        num = w1 * S1 * (R1c - B1) + w2 * S2 * (R2c - B2)
        den = w1 * S1 ** 2 + w2 * S2 ** 2
        eps_hat = num / np.where(den == 0, np.nan, den)
        resid = w1 * (R1c - B1 - S1 * eps_hat) ** 2 + w2 * (R2c - B2 - S2 * eps_hat) ** 2
        # F2 explicit constraint (brief item): epsilon must stay within the observed
        # mechanical range. Candidate (T, epsilon) pairs outside this range are
        # physically inadmissible and are excluded from the grid search rather than
        # silently allowed to win via a near-singular S(T) blow-up.
        out_of_range = (eps_hat < -0.05) | (eps_hat > 0.55)
        resid = np.where(out_of_range, np.inf, resid)
        return eps_hat, resid

    Tg = np.tile(T_grid_coarse, (n, 1))
    eps_hat, resid = total_residual_and_eps(Tg)
    all_invalid_coarse = ~np.isfinite(resid).any(axis=1)
    best_idx = np.nanargmin(resid, axis=1)
    T_best_coarse = T_grid_coarse[best_idx]

    fine_offsets = np.linspace(-T_refine_halfwidth, T_refine_halfwidth, n_refine)
    T_fine = T_best_coarse.reshape(-1, 1) + fine_offsets.reshape(1, -1)
    eps_fine, resid_fine = total_residual_and_eps(T_fine)
    all_invalid_fine = ~np.isfinite(resid_fine).any(axis=1)
    best_fine_idx = np.nanargmin(resid_fine, axis=1)
    T_final = T_fine[np.arange(n), best_fine_idx]
    eps_final = eps_fine[np.arange(n), best_fine_idx]

    # Fix #1: rows where every candidate (coarse AND fine) was physically
    # inadmissible have no valid solution -- their raw eps_final/T_final are
    # meaningless (whatever sat at the first grid index) and must not be
    # allowed to silently distort RMSE. Hard-clip the output to the physical
    # range and flag these rows explicitly instead.
    search_failed = all_invalid_coarse | all_invalid_fine
    eps_final = np.clip(eps_final, -0.05, 0.55)
    T_final = np.clip(T_final, T_grid_coarse.min(), T_grid_coarse.max())
    return T_final, eps_final, search_failed


def true_oracle_within_experiment(df_test, T_true, branch, residual_vars):
    """Fix #2: calibrate B/S on HALF the cycles of the SAME test Experiment
    (even-indexed), evaluate on the OTHER half (odd-indexed) of that same
    Experiment. No cross-Experiment session shift, no cross-temperature
    B/S-prediction error -- isolates pure forward-model-form error."""
    strain = df_test["Strain"].values
    peaks, troughs = detect_turning_points(strain)
    cycles = cycles_from_troughs(troughs, len(strain))
    calib_mask = np.zeros(len(strain), dtype=bool)
    test_mask = np.zeros(len(strain), dtype=bool)
    for cyc_idx, (lo, hi) in enumerate(cycles):
        (calib_mask if cyc_idx % 2 == 0 else test_mask)[lo:hi] = True

    calib_bs = extract_cycle_BS(df_test[calib_mask].reset_index(drop=True))
    oracle_fits = {}
    for elec in ELECTRODES:
        sub = calib_bs[calib_bs.electrode == elec]
        B_o = sub["B"].mean()
        for branch_name in ["loading", "unloading"]:
            S_o = sub[f"S_{branch_name}"].mean()
            oracle_fits[(elec, branch_name)] = {
                "B_val": (lambda t, B_o=B_o: np.full_like(np.asarray(t, dtype=float), B_o)),
                "S_val": (lambda t, S_o=S_o: np.full_like(np.asarray(t, dtype=float), S_o))}

    R1_t, R2_t = df_test.loc[test_mask, ELECTRODES[0]].values, df_test.loc[test_mask, ELECTRODES[1]].values
    eps_hat = known_T_inverse(R1_t, R2_t, T_true[test_mask], branch[test_mask], oracle_fits, residual_vars, mode="both_weighted")
    return df_test.loc[test_mask, "Strain"].values, eps_hat


def nested_select_variant(train_ids, cycle_bs, raw):
    """Fix #3: select the F1 variant (E1_only/E2_only/both_weighted/common_S)
    using ONLY the training pool -- no peeking at the real held-out test set.
    Each of the 3 training datasets is used once as a pseudo-held-out set;
    B/S are fit on the other 2 training datasets and evaluated (known-T) on
    the pseudo-held-out set's second Experiment (mirrors the real protocol)."""
    variants = ["both_weighted", "E1_only", "E2_only", "common_S"]
    nested_rmses = {v: [] for v in variants}
    for pseudo_held_out in train_ids:
        nested_train_ids = [i for i in train_ids if i != pseudo_held_out]
        nested_fits, nested_resvar = get_train_BS(nested_train_ids, cycle_bs)
        df_p = raw[pseudo_held_out]
        exps_p = sorted(df_p.Experiment.unique())
        df_p_test = df_p[df_p.Experiment == exps_p[1]].reset_index(drop=True)
        R1_p, R2_p = df_p_test[ELECTRODES[0]].values, df_p_test[ELECTRODES[1]].values
        T_p, eps_p_true = df_p_test["Temperature"].values, df_p_test["Strain"].values
        branch_p = infer_branch_from_resistance(df_p_test)
        for v in variants:
            eps_hat_p = known_T_inverse(R1_p, R2_p, T_p, branch_p, nested_fits, nested_resvar, mode=v)
            nested_rmses[v].append(metrics(eps_p_true, eps_hat_p)["rmse"])
    avg_rmse = {v: float(np.mean(nested_rmses[v])) for v in variants}
    selected = min(avg_rmse, key=avg_rmse.get)
    return selected, avg_rmse


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    return {"r2": float(r2_score(y_true, y_pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)), "n": int(mask.sum())}


def build_ann():
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(3,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


# ── Precompute per-cycle B/S for every dataset (needed for train-pool fits & oracle) ──

print("Extracting per-cycle B/S for all datasets (for Phase F fits)...")
raw = {i: load_raw(i) for i in DATASETS}
cycle_bs = {}
for i in DATASETS:
    df = raw[i]
    parts = []
    for exp, g in df.groupby("Experiment"):
        p = extract_cycle_BS(g)
        p["experiment"] = exp
        parts.append(p)
    cycle_bs[i] = pd.concat(parts, ignore_index=True)

comparison_rows = []
f1_variant_rows = []
nested_selection_rows = []
known_examples, unknown_examples = {}, {}

for held_out in DATASETS:
    train_ids = [i for i in DATASETS if i != held_out]
    fits, residual_vars = get_train_BS(train_ids, cycle_bs)

    df_held = raw[held_out]
    exps = sorted(df_held.Experiment.unique())
    exp_calib, exp_test = exps[0], exps[1]
    df_calib = df_held[df_held.Experiment == exp_calib].reset_index(drop=True)
    df_test = df_held[df_held.Experiment == exp_test].reset_index(drop=True)

    R1, R2 = df_test["Electrode1_resistance"].values, df_test["Electrode2_resistance"].values
    T_true = df_test["Temperature"].values
    eps_true = df_test["Strain"].values
    branch = infer_branch_from_resistance(df_test)

    # F1 variants (known T) -- reported for diagnostics; NOT used to pick the
    # headline result (see nested_select_variant below, Fix #3)
    for mode in ["both_weighted", "E1_only", "E2_only", "common_S"]:
        eps_hat = known_T_inverse(R1, R2, T_true, branch, fits, residual_vars, mode=mode)
        m = metrics(eps_true, eps_hat)
        f1_variant_rows.append({"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                                 "variant": mode, "selection": "post_hoc_diagnostic_only", **m})

    # Fix #3: pick the variant using ONLY the training pool (nested LOO), never the real test set
    selected_variant, nested_avg_rmse = nested_select_variant(train_ids, cycle_bs, raw)
    eps_hat_f1 = known_T_inverse(R1, R2, T_true, branch, fits, residual_vars, mode=selected_variant)
    m_f1 = metrics(eps_true, eps_hat_f1)
    known_examples[held_out] = {"y_true": eps_true, "y_pred": eps_hat_f1, "variant": selected_variant}

    # F2 (unknown T): coarse grid over the full observed T range, then refine
    T_grid_coarse = np.linspace(25, 70, 46)
    T_hat, eps_hat_f2, search_failed = unknown_T_inverse(R1, R2, branch, fits, residual_vars, T_grid_coarse)
    m_f2_eps = metrics(eps_true, eps_hat_f2)
    m_f2_T = metrics(T_true, T_hat)
    f2_fail_rate = float(np.mean(search_failed))
    unknown_examples[held_out] = {"y_true": eps_true, "y_pred": eps_hat_f2}

    # Fix #2a: session-calibrated inverse (was "oracle" in v1) -- B/S measured on
    # the SAME held-out dataset's FIRST Experiment, evaluated on its SECOND
    # Experiment. Still subject to Experiment-to-Experiment session shift.
    calib_bs = extract_cycle_BS(df_calib)
    session_fits = {}
    for elec in ELECTRODES:
        sub = calib_bs[calib_bs.electrode == elec]
        B_o = sub["B"].mean()
        for branch_name in ["loading", "unloading"]:
            S_o = sub[f"S_{branch_name}"].mean()
            session_fits[(elec, branch_name)] = {"B_val": (lambda t, B_o=B_o: np.full_like(np.asarray(t, dtype=float), B_o)),
                                                   "S_val": (lambda t, S_o=S_o: np.full_like(np.asarray(t, dtype=float), S_o))}
    eps_hat_session = known_T_inverse(R1, R2, T_true, branch, session_fits, residual_vars, mode="both_weighted")
    m_session = metrics(eps_true, eps_hat_session)

    # Fix #2b: TRUE oracle -- calibrate and test within the SAME Experiment
    # (even/odd cycle split), isolating pure forward-model-form error.
    eps_true_true_oracle, eps_hat_true_oracle = true_oracle_within_experiment(df_test, T_true, branch, residual_vars)
    m_true_oracle = metrics(eps_true_true_oracle, eps_hat_true_oracle)

    # Linear+T and Deep ANN, retrained on the SAME train pool, evaluated on the SAME test rows
    train_pool = pd.concat([raw[i] for i in train_ids], ignore_index=True)
    feats = ["Electrode1_resistance", "Electrode2_resistance", "Temperature"]
    scaler = MinMaxScaler().fit(train_pool[feats].values)
    X_tr = scaler.transform(train_pool[feats].values)
    X_te = scaler.transform(df_test[feats].values)
    y_tr = train_pool["Strain"].values

    lin = LinearRegression().fit(X_tr, y_tr)
    m_linear = metrics(eps_true, lin.predict(X_te))

    ann_r2s, ann_pred = [], None
    for seed in SEEDS:
        tf.random.set_seed(seed); np.random.seed(seed)
        m_ann = build_ann()
        cb = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
              tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0)]
        m_ann.fit(X_tr, y_tr, epochs=300, batch_size=32, validation_split=0.2, callbacks=cb, verbose=0)
        ann_pred = m_ann.predict(X_te, verbose=0).flatten()
        ann_r2s.append(metrics(eps_true, ann_pred)["r2"])
    m_deep_ann = metrics(eps_true, ann_pred); m_deep_ann["r2_seeds"] = ann_r2s; m_deep_ann["r2"] = float(np.mean(ann_r2s))

    for label, m in [("1_linear_R1R2T", m_linear), ("2_deep_ann", m_deep_ann),
                     (f"3_analytical_inverse_knownT_selected_{selected_variant}", m_f1),
                     ("4_analytical_inverse_unknownT", m_f2_eps),
                     ("5a_session_calibrated_inverse", m_session),
                     ("5b_true_oracle_within_experiment", m_true_oracle)]:
        comparison_rows.append({"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                                 "method": label, "r2": m["r2"], "rmse": m["rmse"], "mae": m["mae"], "n": m["n"]})
    comparison_rows.append({"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                             "method": "4b_unknownT_temperature_recovery", "r2": m_f2_T["r2"], "rmse": m_f2_T["rmse"],
                             "mae": m_f2_T["mae"], "n": m_f2_T["n"], "search_failed_rate": f2_fail_rate})
    nested_selection_rows.append({"held_out": held_out, "selected_variant": selected_variant,
                                   **{f"nested_train_rmse_{v}": r for v, r in nested_avg_rmse.items()}})

    print(f"held_out=ds{held_out}({TEMP_VAL[held_out]}C,{KIND[held_out]}): "
          f"linear+T R2={m_linear['r2']:.3f} | deep_ann R2={m_deep_ann['r2']:.3f} | "
          f"inv_knownT(sel={selected_variant}) R2={m_f1['r2']:.3f} | "
          f"inv_unknownT(eps) R2={m_f2_eps['r2']:.3f} (fail_rate={f2_fail_rate:.1%}, T_recovery R2={m_f2_T['r2']:.3f}) | "
          f"session_calib R2={m_session['r2']:.3f} | true_oracle R2={m_true_oracle['r2']:.3f}")

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUT_TAB / "analytical_inverse_results.csv", index=False)
pd.DataFrame(f1_variant_rows).to_csv(OUT_TAB / "analytical_inverse_F1_variants.csv", index=False)
pd.DataFrame(nested_selection_rows).to_csv(OUT_TAB / "analytical_inverse_variant_selection_nested.csv", index=False)
print("\nSaved -> tables/analytical_inverse_results.csv, tables/analytical_inverse_F1_variants.csv, "
      "tables/analytical_inverse_variant_selection_nested.csv")

# ── Jacobian / identifiability map (uses the ds1+ds2+ds3 train fit, held_out=4, as a representative case) ──

fits_rep, _ = get_train_BS([1, 2, 3], cycle_bs)
T_range = np.linspace(30, 65, 60)
eps_range = np.linspace(0.0, 0.5, 60)
Tg, Eg = np.meshgrid(T_range, eps_range)

dR1_dT = fits_rep[(ELECTRODES[0], "loading")]["B_deriv"](Tg) + fits_rep[(ELECTRODES[0], "loading")]["S_deriv"](Tg) * Eg
dR2_dT = fits_rep[(ELECTRODES[1], "loading")]["B_deriv"](Tg) + fits_rep[(ELECTRODES[1], "loading")]["S_deriv"](Tg) * Eg
dR1_de = fits_rep[(ELECTRODES[0], "loading")]["S_val"](Tg)
dR2_de = fits_rep[(ELECTRODES[1], "loading")]["S_val"](Tg)

cond_map = np.zeros_like(Tg)
det_map = np.zeros_like(Tg)
for a in range(Tg.shape[0]):
    for b in range(Tg.shape[1]):
        J = np.array([[dR1_dT[a, b], dR1_de[a, b]], [dR2_dT[a, b], dR2_de[a, b]]])
        det_map[a, b] = np.linalg.det(J)
        sv = np.linalg.svd(J, compute_uv=False)
        cond_map[a, b] = sv[0] / sv[1] if sv[1] > 0 else np.inf

jac_df = pd.DataFrame({"T": Tg.flatten(), "epsilon": Eg.flatten(),
                        "det_J": det_map.flatten(), "condition_number": cond_map.flatten()})
jac_df.to_csv(OUT_TAB / "jacobian_condition_map.csv", index=False)
print(f"\nJacobian condition number: median={np.median(cond_map):.2f}, "
      f"max={np.nanmax(cond_map[np.isfinite(cond_map)]):.2f}, "
      f"min={np.nanmin(cond_map):.2f}")
print(f"det(J) sign uniform (never crosses zero -> globally invertible over this grid)? "
      f"{bool(np.all(det_map > 0) or np.all(det_map < 0))}")

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.pcolormesh(T_range, eps_range, np.log10(cond_map), shading="auto", cmap="viridis")
plt.colorbar(im, ax=ax, label="log10(condition number)")
ax.set_xlabel("Temperature (C)"); ax.set_ylabel("Strain")
ax.set_title("Jacobian condition number over (T, strain) space\n(lower = more stable inversion)")
fig.tight_layout()
fig.savefig(OUT_FIG / "jacobian_condition_map.png", dpi=150)
plt.close(fig)

# ── Plots ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ex = known_examples[2]
axes[0].plot(ex["y_true"], lw=0.8, label="measured"); axes[0].plot(ex["y_pred"], lw=0.8, label="predicted")
axes[0].set_title("Analytical inverse (known T) - ds2 (43C) test experiment"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT_FIG / "analytical_inverse_43C.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, i in zip(axes.flat, DATASETS):
    ex = known_examples[i]
    ax.plot(ex["y_true"], lw=0.8, label="measured"); ax.plot(ex["y_pred"], lw=0.8, label="predicted (known T)")
    ax.set_title(f"ds{i} ({TEMP_VAL[i]}C, {KIND[i]})", fontsize=9); ax.legend(fontsize=7)
fig.suptitle("Analytical inverse (known-T), all held-out temperatures")
fig.tight_layout(); fig.savefig(OUT_FIG / "analytical_inverse_all_temperatures.png", dpi=150); plt.close(fig)

# Three-way error decomposition:
#   true_oracle_rmse           = pure forward-model-form error (same Experiment, split by cycle)
#   session_calibrated_rmse    = + Experiment-to-Experiment session shift (still same temperature)
#   deployable_rmse            = + B(T)/S(T) LOO-prediction error at a genuinely unseen temperature
decomp_rows = []
for held_out in DATASETS:
    f1_row = comparison_df[(comparison_df.held_out == held_out) &
                            (comparison_df.method.str.startswith("3_analytical_inverse_knownT"))]
    f1_rmse = f1_row["rmse"].values[0]
    selected_variant_label = f1_row["method"].values[0]
    session_rmse = comparison_df[(comparison_df.held_out == held_out) &
                                  (comparison_df.method == "5a_session_calibrated_inverse")]["rmse"].values[0]
    true_oracle_rmse = comparison_df[(comparison_df.held_out == held_out) &
                                      (comparison_df.method == "5b_true_oracle_within_experiment")]["rmse"].values[0]
    decomp_rows.append({"held_out": held_out, "temperature": TEMP_VAL[held_out], "kind": KIND[held_out],
                         "selected_variant": selected_variant_label,
                         "true_oracle_rmse_forward_model_only": true_oracle_rmse,
                         "session_calibrated_rmse": session_rmse,
                         "deployable_knownT_rmse": f1_rmse,
                         "extra_rmse_from_session_shift": session_rmse - true_oracle_rmse,
                         "extra_rmse_from_BS_prediction_error": f1_rmse - session_rmse})
decomp_df = pd.DataFrame(decomp_rows)
decomp_df.to_csv(OUT_TAB / "inverse_error_decomposition.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(DATASETS))
w = 0.25
ax.bar(x - w, decomp_df.true_oracle_rmse_forward_model_only, width=w, label="True oracle (forward-model form only)")
ax.bar(x, decomp_df.session_calibrated_rmse, width=w, label="Session-calibrated (+ session shift)")
ax.bar(x + w, decomp_df.deployable_knownT_rmse, width=w, label="Deployable (+ B/S LOO-prediction error)")
ax.set_xticks(x); ax.set_xticklabels([f"ds{i}\n{TEMP_VAL[i]}C\n{KIND[i]}" for i in DATASETS], fontsize=8)
ax.set_ylabel("RMSE (strain)"); ax.set_title("Error decomposition: forward-model vs session-shift vs B(T)/S(T)-prediction error")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT_FIG / "inverse_error_decomposition.png", dpi=150); plt.close(fig)

print("\n[Phase F complete]")
