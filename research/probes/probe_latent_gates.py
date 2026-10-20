"""Pre-registered Gates A and B for the latent-state direction.

Implements research/directions/2026-10-21-preregistration-latent-gates.md; do not change
without a new dated pre-registration. dataset0 is sealed and never loaded here.

Usage: .venv/bin/python -I research/probes/probe_latent_gates.py <repo root> <out prefix>
"""
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, sys.argv[1] + "/src")
import strain.calibration as cal  # noqa: E402
from strain.spec import DATASETS, ELECTRODES, TEMP_VAL, load_raw  # noqa: E402

assert 0 not in DATASETS, "dataset0 is sealed"
BIN_EDGES = np.arange(0.20, 0.45 + 1e-9, 0.05)
HORIZON = 1000
CRB_LIMIT = 0.01
MIN_CYCLES = 6
COEFS = ["dB1", "dS1", "dB2", "dS2"]


def stop(reason):
    raise SystemExit(f"RUN STOPPED: {reason}")


def cycles_of(strain):
    troughs, _ = find_peaks(-strain, prominence=0.25, distance=600)
    starts = troughs.tolist()
    if strain[0] <= strain.min() + 0.02:
        starts = [0, *starts]
    return list(zip(starts[:-1], starts[1:]))


def branch_bins(strain, resistance):
    loading = np.gradient(strain) > 0
    xs, ys = [], []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        for branch in (loading, ~loading):
            mask = branch & (strain >= lo) & (strain < hi)
            if mask.sum() == 0:
                return None
            xs.append(strain[mask].mean())
            ys.append(resistance[mask].mean())
    slope, intercept = np.polyfit(xs, ys, 1)
    return intercept, slope


# ---------------------------------------------------------------- coefficients
rows, dropped, sessions = [], 0, {}
for dataset in DATASETS:
    frame = load_raw(dataset)
    for experiment in sorted(frame.Experiment.unique()):
        s = frame[frame.Experiment == experiment].reset_index(drop=True)
        sessions[(dataset, experiment)] = s
        strain = s.Strain.to_numpy()
        t = s.Idx.to_numpy() - float(s.Idx.iloc[0])
        for index, (a, b) in enumerate(cycles_of(strain), start=1):
            fits = [branch_bins(strain[a:b], s[e].to_numpy()[a:b]) for e in ELECTRODES]
            if any(f is None for f in fits):
                dropped += 1
                continue
            rows.append({"dataset": dataset, "temperature": TEMP_VAL[dataset], "experiment": experiment,
                         "cycle": index, "mid_time_s": float(t[(a + b) // 2]),
                         "B1": fits[0][0], "S1": fits[0][1], "B2": fits[1][0], "S2": fits[1][1]})
coef = pd.DataFrame(rows)
for c in ("B1", "S1", "B2", "S2"):
    coef["d" + c] = coef[c] / coef.groupby(["dataset", "experiment"])[c].transform("median") - 1
counts = coef.groupby(["dataset", "experiment"]).size().reindex(list(sessions), fill_value=0)
if counts.min() < MIN_CYCLES:
    stop(f"insufficient coverage: {counts.to_dict()}")
if not np.isfinite(coef[COEFS].to_numpy()).all():
    stop("non-finite coefficients")
coef.to_csv(sys.argv[2] + "_cycle_coefficients.csv", index=False)
print(f"cycles kept {len(coef)} (per session {counts.min()}-{counts.max()}), dropped {dropped}")


def rank_one(source):
    sigma = source[COEFS].std().to_numpy()
    if (sigma <= 0).any():
        stop("zero source variance")
    _, _, vt = np.linalg.svd(source[COEFS].to_numpy() / sigma, full_matrices=False)
    v = vt[0] * np.sign(vt[0][np.argmax(np.abs(vt[0]))])
    return v, sigma


def predict_e2(v, sigma, target):
    x = target[["dB1", "dS1"]].to_numpy() / sigma[:2]
    state = x @ v[:2] / (v[:2] @ v[:2])
    return np.outer(state, v[2:]) * sigma[2:]


def loss(pred, target, sigma):
    """Mean over sessions of per-session RMSE of standardized [dB2, dS2]."""
    err = (np.asarray(pred) - target[["dB2", "dS2"]].to_numpy()) / sigma[2:]
    frame = pd.DataFrame(err, index=target.index).assign(key=target.experiment.to_numpy())
    per_session = frame.groupby("key").apply(lambda g: np.sqrt(np.mean(g[[0, 1]].to_numpy() ** 2)))
    comp = np.sqrt(np.mean(err ** 2, axis=0))
    return float(per_session.mean()), comp


def controls(source, target):
    t1 = {c: np.polyfit(source.mid_time_s, source[c], 1) for c in COEFS}
    t2 = source.groupby("cycle")[COEFS].mean()
    pred_t1 = pd.DataFrame({c: np.polyval(t1[c], target.mid_time_s) for c in COEFS}, index=target.index)
    pred_t2 = t2.loc[target.cycle].set_index(target.index)
    return pred_t1, pred_t2


# ---------------------------------------------------------------- Gate A
gate_a, fold_models = [], {}
for held in DATASETS:
    source, target = coef[coef.dataset != held], coef[coef.dataset == held]
    support = set(source.cycle)
    excluded = int((~target.cycle.isin(support)).sum())
    target = target[target.cycle.isin(support)]
    v, sigma = rank_one(source)
    fold_models[held] = (v, sigma)
    pred_t1, pred_t2 = controls(source, target)
    zero = np.zeros((len(target), 2))
    scores = {"L": loss(predict_e2(v, sigma, target), target, sigma),
              "T0": loss(zero, target, sigma),
              "T1": loss(pred_t1[["dB2", "dS2"]].to_numpy(), target, sigma),
              "T2": loss(pred_t2[["dB2", "dS2"]].to_numpy(), target, sigma)}
    best = min(scores[k][0] for k in ("T0", "T1", "T2"))
    a1 = best > 0 and scores["L"][0] <= 0.9 * best
    a1_status = "inconclusive (zero control loss)" if best <= 0 else ("pass" if a1 else "fail")
    # A2: residuals after the better (on sources) of T1/T2
    src_t1, src_t2 = controls(source, source)
    src_err = {k: float(np.mean((p[COEFS].to_numpy() - source[COEFS].to_numpy()) ** 2))
               for k, p in (("T1", src_t1), ("T2", src_t2))}
    chosen = min(src_err, key=src_err.get)
    src_resid = source.copy()
    src_resid[COEFS] = source[COEFS].to_numpy() - (src_t1 if chosen == "T1" else src_t2)[COEFS].to_numpy()
    tgt_resid = target.copy()
    tgt_resid[COEFS] = target[COEFS].to_numpy() - (pred_t1 if chosen == "T1" else pred_t2)[COEFS].to_numpy()
    vr, sr = rank_one(src_resid)
    resid_l = loss(predict_e2(vr, sr, tgt_resid), tgt_resid, sr)
    resid_zero = loss(zero, tgt_resid, sr)
    a2 = resid_zero[0] > 0 and resid_l[0] <= 0.9 * resid_zero[0]
    a2_status = "inconclusive (zero control loss)" if resid_zero[0] <= 0 else ("pass" if a2 else "fail")
    gate_a.append({"temperature": TEMP_VAL[held], "excluded_cycles": excluded,
                   **{f"loss_{k}": s[0] for k, s in scores.items()},
                   "L_comp_dB2_dS2": np.round(scores["L"][1], 3).tolist(),
                   "A1_pass": a1, "A1_status": a1_status, "resid_control": chosen, "loss_resid_L": resid_l[0],
                   "loss_resid_zero": resid_zero[0], "A2_pass": a2, "A2_status": a2_status})
gate_a = pd.DataFrame(gate_a)
pd.set_option("display.width", 240)
print("\nGate A:\n", gate_a.round(4).to_string(index=False))
pass_a = bool(gate_a.A1_pass.all() and gate_a.A2_pass.all())
print("GATE A:", "PASS" if pass_a else "FAIL")


# ---------------------------------------------------------------- Gate B
def noise_covariance(s):
    strain = s.Strain.to_numpy()
    loading = np.gradient(strain) > 0
    resid = {e: np.full(len(strain), np.nan) for e in ELECTRODES}  # keeps original sample positions
    for a, b in cycles_of(strain):
        for branch in (loading[a:b], ~loading[a:b]):
            idx = np.arange(a, b)[branch]
            if len(idx) < 10:
                continue
            for e in ELECTRODES:
                y = s[e].to_numpy()[idx]
                resid[e][idx] = y - np.polyval(np.polyfit(strain[idx], y, 1), strain[idx])
    r1, r2 = resid[ELECTRODES[0]], resid[ELECTRODES[1]]
    both = np.isfinite(r1) & np.isfinite(r2)
    scale = [1.4826 * np.median(np.abs(r[both] - np.median(r[both]))) for r in (r1, r2)]
    if min(scale) <= 0 or not np.isfinite(scale).all():
        stop("zero or non-finite noise")
    rho = float(np.corrcoef(r1[both], r2[both])[0, 1])
    pair = np.isfinite(r1[:-1]) & np.isfinite(r1[1:])  # neighbours one sampling interval apart
    lag1 = [float(np.corrcoef(r[:-1][pair], r[1:][pair])[0, 1]) for r in (r1, r2)]
    cov = np.array([[scale[0] ** 2, rho * scale[0] * scale[1]], [rho * scale[0] * scale[1], scale[1] ** 2]])
    if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min() <= 0:
        stop("noise covariance not positive definite")
    return cov, rho, lag1, scale


gate_b = []
for held in DATASETS:
    v, sigma = fold_models[held]
    b1, g1, b2, g2 = v * sigma  # relative loadings
    bounds, diag = [], []
    for (dataset, experiment), s in sessions.items():
        if dataset != held:
            continue
        fit, _, _ = cal.four_point_calibration(cal.calibration_prefix(s))
        (S10, B10), (S20, B20) = fit[ELECTRODES[0]], fit[ELECTRODES[1]]
        if not np.isfinite([S10, B10, S20, B20]).all() or min(abs(S10), abs(S20)) <= 0:
            stop("invalid prefix calibration")
        cov, rho, lag1, scale = noise_covariance(s)
        diag.append({"session": experiment, "rho": round(rho, 3), "lag1": np.round(lag1, 3).tolist(),
                     "noise_ohm": np.round(scale, 2).tolist()})
        inv = np.linalg.inv(cov)
        a = np.array([S10, S20])
        d = float(a @ inv @ a)
        strain = s.Strain.to_numpy()
        for start in range(0, len(strain) - HORIZON + 1, HORIZON):
            eps = strain[start:start + HORIZON]
            h = np.column_stack([B10 * b1 + eps * S10 * g1, B20 * b2 + eps * S20 * g2])
            c = h @ inv @ a
            q = float(np.einsum("ki,ij,kj->", h, inv, h))
            m = q - float(np.sum(c ** 2)) / d
            if m <= 1e-12 * q:
                bounds.append(np.inf)
                continue
            var = 1.0 / d + (c / d) ** 2 / m
            bounds.append(float(np.sqrt(np.mean(var))))
    rms = float(np.sqrt(np.mean(np.square(bounds)))) if np.isfinite(bounds).all() else np.inf
    gate_b.append({"temperature": TEMP_VAL[held], "smoothing_crb_strain_rms_pct": rms,
                   "relative_loadings_b1_g1_b2_g2": np.round([b1, g1, b2, g2], 4).tolist(),
                   "noise_diagnostics": diag, "pass": rms < CRB_LIMIT})
gate_b = pd.DataFrame(gate_b)
print("\nGate B:\n", gate_b.to_string(index=False))
pass_b = bool(gate_b["pass"].all())
print("GATE B:", "PASS" if pass_b else "FAIL")
pd.concat([gate_a.assign(gate="A"), gate_b.assign(gate="B")], ignore_index=True).to_csv(
    sys.argv[2] + "_gates.csv", index=False)
print("\nDECISION:", "build the latent-state estimator" if pass_a and pass_b
      else "stop the shared-state method direction on this corpus")
