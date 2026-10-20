"""Pre-registered test: apparent alignment between resistance and the reference strain.

Implements research/directions/2026-10-21-preregistration-reference-lag-v2.md (wider-grid,
data-informed sensitivity analysis of the frozen b82b769 test); do not change
without a new dated pre-registration. dataset0 is sealed and never loaded. No mechanism is
inferred from any output (see the pre-registration's identifiability limit).

Usage: .venv/bin/python -I research/probes/probe_reference_lag_v2.py <repo root> <out prefix>
"""
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, sys.argv[1] + "/src")
from strain.spec import DATASETS, ELECTRODES, TEMP_VAL, load_raw  # noqa: E402

assert 0 not in DATASETS, "dataset0 is sealed"
DT = 0.1
SHIFTS = np.round(np.arange(-15.0, 15.0 + 1e-9, DT), 1)
MARGIN = 15.0
BINS = np.round(np.arange(0.05, 0.45 + 1e-9, 0.01), 4)
N_BOOT = 100
MIN_VALID = 14


def cycles_of(eps):
    troughs, _ = find_peaks(-eps, prominence=0.25, distance=600)
    return list(zip(troughs[:-1], troughs[1:]))


def support_mask(t, cycles, which=None):
    """Samples inside the selected complete cycles whose shifted reference exists for every shift."""
    mask = np.zeros(len(t), bool)
    for i, (a, b) in enumerate(cycles):
        if which is None or i in which:
            mask[a:b] = True
    return mask & (t >= t[0] + MARGIN) & (t <= t[-1] - MARGIN)


def design(t, eps, tau, mask):
    shifted = np.interp(t[mask] - tau, t, eps)
    return np.column_stack([np.ones(mask.sum()), shifted, t[mask]])


def rms_curve(t, eps, r, mask):
    out = []
    for tau in SHIFTS:
        X = design(t, eps, tau, mask)
        coef, *_ = np.linalg.lstsq(X, r[mask], rcond=None)
        out.append(float(np.sqrt(np.mean((r[mask] - X @ coef) ** 2))))
    return np.array(out)


def refine(curve):
    i = int(np.argmin(curve))
    if i in (0, len(curve) - 1) or np.sum(curve == curve[i]) > 1:
        return np.nan
    y0, y1, y2 = curve[i - 1], curve[i], curve[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom <= 0:
        return np.nan
    return float(SHIFTS[i] + 0.5 * (y0 - y2) / denom * DT)


def nuisance(t, eps, r, mask, tau):
    coef, *_ = np.linalg.lstsq(design(t, eps, tau, mask), r[mask], rcond=None)
    return coef


def loop_area(t, eps, r, cycles, which, tau, coef_eval, tau_compare):
    """Sum over valid cycles; a cycle is valid only if all bins are filled at tau and tau_compare."""
    a0, slope, c = coef_eval
    total, valid = 0.0, 0
    for i, (a, b) in enumerate(cycles):
        if i not in which:
            continue
        idx = np.arange(a, b)
        if t[idx[0]] < t[0] + MARGIN or t[idx[-1]] > t[-1] - MARGIN:
            continue
        gaps_by_shift = []
        for shift in (tau, tau_compare):
            e = np.interp(t[idx] - shift, t, eps)
            z = (r[idx] - a0 - c * t[idx]) / slope
            peak = int(np.argmax(e))
            up, down = np.arange(len(idx)) <= peak, np.arange(len(idx)) > peak
            gaps = []
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                in_bin = (e >= lo) & (e < hi)
                if not (in_bin & up).any() or not (in_bin & down).any():
                    gaps = None
                    break
                gaps.append(abs(z[in_bin & down].mean() - z[in_bin & up].mean()))
            gaps_by_shift.append(gaps)
        if any(g is None for g in gaps_by_shift):
            continue
        centers = (BINS[:-1] + BINS[1:]) / 2
        total += float(np.trapezoid(gaps_by_shift[0], centers))
        valid += 1
    return (total if valid else np.nan), valid


def cycle_phases(t, eps, r, cycles):
    out = {1: [], 3: [], 5: []}
    for a, b in cycles:
        tt, e, rr = t[a:b], eps[a:b], r[a:b]
        rr = rr - np.polyval(np.polyfit(tt, rr, 1), tt)
        e = e - np.polyval(np.polyfit(tt, e, 1), tt)  # same linear detrend for both signals
        period = tt[-1] - tt[0] + DT
        for k in out:
            basis = np.exp(-2j * np.pi * k * (tt - tt[0]) / period)
            h = (rr @ basis) / (e @ basis)
            out[k].append(-np.angle(h) / (2 * np.pi * k / period))
    return {k: float(np.median(v)) for k, v in out.items()}


def interval(draws):
    """95 % percentile interval over resolved draws; invalid if more than 10 % are unresolved."""
    ok = np.isfinite(draws)
    if ok.mean() < 0.9:
        return np.nan, np.nan
    return float(np.percentile(draws[ok], 2.5)), float(np.percentile(draws[ok], 97.5))


rows = []
rng = np.random.default_rng(0)
for dataset in DATASETS:
    frame = load_raw(dataset)
    for experiment in sorted(frame.Experiment.unique()):
        s = frame[frame.Experiment == experiment].reset_index(drop=True)
        eps = s.Strain.to_numpy()
        t = s.Idx.to_numpy() - float(s.Idx.iloc[0])
        cycles = cycles_of(eps)
        odd, even = set(range(0, len(cycles), 2)), set(range(1, len(cycles), 2))
        full = support_mask(t, cycles)
        picks = [rng.integers(0, len(cycles), len(cycles)) for _ in range(N_BOOT)]  # shared by both electrodes
        boot_by_electrode = {}
        for k, electrode in enumerate(ELECTRODES, start=1):
            r = s[electrode].to_numpy()
            boot = []
            for pick in picks:
                weights = np.zeros(len(t))
                for i in pick:
                    a, b = cycles[i]
                    weights[a:b] += 1
                keep = (weights > 0) & full
                rep = np.repeat(np.flatnonzero(keep), weights[keep].astype(int))
                curve = []
                for shift in SHIFTS:
                    X = np.column_stack([np.ones(len(rep)), np.interp(t[rep] - shift, t, eps), t[rep]])
                    coef, *_ = np.linalg.lstsq(X, r[rep], rcond=None)
                    curve.append(float(np.sqrt(np.mean((r[rep] - X @ coef) ** 2))))
                boot.append(refine(np.array(curve)))
            boot_by_electrode[k] = np.array(boot)
        diff = boot_by_electrode[1] - boot_by_electrode[2]
        diff_ci = interval(diff)
        for k, electrode in enumerate(ELECTRODES, start=1):
            r = s[electrode].to_numpy()
            tau = refine(rms_curve(t, eps, r, full))
            boot = boot_by_electrode[k]
            ci = interval(boot)
            unresolved_fraction = float(np.mean(~np.isfinite(boot)))
            tau_odd = refine(rms_curve(t, eps, r, support_mask(t, cycles, odd)))
            tau_even = refine(rms_curve(t, eps, r, support_mask(t, cycles, even)))
            coef0 = nuisance(t, eps, r, full, 0.0)
            collapse = {}
            for name, fit_tau, which in (("even_by_odd", tau_odd, even), ("odd_by_even", tau_even, odd)):
                if not np.isfinite(fit_tau):
                    collapse[name], collapse[name + "_refit"] = np.nan, np.nan
                    continue
                a_shift, n1 = loop_area(t, eps, r, cycles, which, fit_tau, coef0, 0.0)
                a_zero, _ = loop_area(t, eps, r, cycles, which, 0.0, coef0, fit_tau)
                coef_s = nuisance(t, eps, r, full, fit_tau)
                a_shift_r, _ = loop_area(t, eps, r, cycles, which, fit_tau, coef_s, 0.0)
                collapse[name] = a_shift / a_zero if a_zero and np.isfinite(a_zero) else np.nan
                collapse[name + "_refit"] = a_shift_r / a_zero if a_zero and np.isfinite(a_zero) else np.nan
            lag_rms = (float(np.sqrt(np.mean((eps[full] - np.interp(t[full] - tau, t, eps)) ** 2))) if np.isfinite(tau)
                       else np.nan)
            phases = cycle_phases(t, eps, r, cycles)
            rows.append({"dataset": dataset, "temperature": TEMP_VAL[dataset], "experiment": experiment,
                         "electrode": k, "cycles": len(cycles), "tau_s": tau, "ci_low": ci[0], "ci_high": ci[1],
                         "boot_unresolved_fraction": unresolved_fraction,
                         "session_diff_ci_low": diff_ci[0], "session_diff_ci_high": diff_ci[1],
                         "tau_odd": tau_odd, "tau_even": tau_even, **{f"collapse_{k2}": v for k2, v in collapse.items()},
                         "shift_only_rms_pct": lag_rms, "harm_tau1": phases[1], "harm_tau3": phases[3],
                         "harm_tau5": phases[5]})
        print(f"done dataset {dataset} experiment {experiment}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(sys.argv[2] + "_per_electrode_session.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(3).to_string(index=False))

valid = out[out.tau_s.notna() & out.ci_low.notna()]
print(f"\nvalid electrode-sessions: {len(valid)}/16 (minimum {MIN_VALID})")
if len(valid) < MIN_VALID:
    print("PRIMARY: insufficient coverage")
else:
    pos = valid[(valid.tau_s >= 0.3) & (valid.ci_low > 0)]
    neg = valid[(valid.tau_s <= -0.3) & (valid.ci_high < 0)]
    within = valid[(valid.ci_low >= -0.2) & (valid.ci_high <= 0.2)]
    print(f"tau >= 0.3 s with CI > 0: {len(pos)}; tau <= -0.3 s with CI < 0: {len(neg)}; "
          f"CI within +/-0.2 s: {len(within)}")
    if max(len(pos), len(neg)) >= 14:
        print("PRIMARY: consistent apparent alignment (no mechanism inferred)")
    elif len(within) >= 14:
        print("PRIMARY: no material apparent alignment")
    else:
        print("PRIMARY: inconclusive")
session = out.pivot_table(index=["temperature", "experiment"], columns="electrode", values="tau_s")
diff_ci = out[out.electrode == 1].set_index(["temperature", "experiment"])[["session_diff_ci_low", "session_diff_ci_high"]]
print("\nper-session tau (electrode 1, 2), difference, and paired-bootstrap difference interval:\n",
      session.assign(diff=session[1] - session[2]).join(diff_ci).round(3).to_string())
print("\nsession-mean tau versus temperature (descriptive, no test):\n",
      session.mean(axis=1).rename("session_mean_tau_s").round(3).to_string())
