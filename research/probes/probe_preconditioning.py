"""Pre-registered test: one preconditioning cycle before calibration.

Implements research/directions/2026-10-21-preregistration-preconditioning.md exactly; do not
change protocols, metric, or decision rule here without a new dated pre-registration.

Usage: .venv/bin/python -I research/probes/probe_preconditioning.py <repo root> <out prefix>
Writes <out prefix>_rmse.csv and <out prefix>_calibration_content.csv.
"""
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, sys.argv[1] + "/src")
import strain.calibration as cal  # noqa: E402
from strain.features import build_features  # noqa: E402
from strain.protocol import make_hgb  # noqa: E402
from strain.spec import DATASETS, ELECTRODES, ROLLING_WINDOWS, TEMP_VAL, load_raw  # noqa: E402

CAL_SECONDS = 125.0
EVENT_SECONDS = 250.0
raw = {d: load_raw(d) for d in DATASETS}
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


class ProtocolFailure(Exception):
    pass


def sessions(dataset):
    frame = raw[dataset]
    return [frame[frame.Experiment == e].reset_index(drop=True) for e in sorted(frame.Experiment.unique())]


def elapsed(session):
    return session["Idx"].to_numpy() - float(session["Idx"].iloc[0])


def first_cycle_end(session):
    """First strain trough after the first peak, searched only inside the calibration event."""
    t = elapsed(session)
    strain = session["Strain"].to_numpy()[t <= EVENT_SECONDS]
    peaks, _ = find_peaks(strain, prominence=0.25, distance=600)
    troughs, _ = find_peaks(-strain, prominence=0.25, distance=600)
    if len(peaks) == 0:
        raise ProtocolFailure("no peak inside the event")
    troughs = troughs[troughs > peaks[0]]
    if len(troughs) == 0:
        raise ProtocolFailure("no trough after the first peak inside the event")
    c1 = float(t[troughs[0]])
    if c1 + CAL_SECONDS > EVENT_SECONDS:
        raise ProtocolFailure(f"c1 + 125 = {c1 + CAL_SECONDS:.1f} s exceeds the event")
    return c1


def rule(name, session):
    if name == "P0":  # never invokes cycle detection
        return 0.0, CAL_SECONDS
    if name == "P1":
        c1 = first_cycle_end(session)
        return c1, c1 + CAL_SECONDS
    if name == "P2":
        return 0.0, EVENT_SECONDS
    if name == "P3":
        return CAL_SECONDS, EVENT_SECONDS
    raise ValueError(name)


def window(session, lo, hi):
    t = elapsed(session)
    return session[(t >= lo) & (t <= hi)].reset_index(drop=True)


def calibrate(calibration_window):
    fit, points, _ = cal.four_point_calibration(calibration_window)
    return fit, points, cal.fusion_weight_from_points(fit, points)


def features(session, calibration_window):
    fit, _, weight = calibrate(calibration_window)
    z = cal.normalize(session, fit, weight)
    return build_features(z, session["Temperature"].to_numpy(), "causal", False, ROLLING_WINDOWS), z[2]


def content(session, lo, hi):
    w = window(session, lo, hi)
    strain = w["Strain"].to_numpy()
    rng = strain.max() - strain.min()
    levels = np.linspace(strain.min() + 0.05 * rng, strain.max() - 0.05 * rng, 3)
    loading = np.gradient(strain) > 0
    row = {"cal_start_s": lo, "cal_end_s": hi, "strain_min": strain.min(), "strain_max": strain.max()}
    masks = [strain <= strain.min() + 0.02] + [np.abs(strain - level) <= 0.03 for level in levels]
    for k, (level, mask) in enumerate(zip([0.0, *levels], masks)):
        row[f"level{k}"] = level
        row[f"band{k}_rows"] = int(mask.sum())
        row[f"band{k}_loading_rows"] = int((mask & loading).sum())
        row[f"band{k}_unloading_rows"] = int((mask & ~loading).sum())
    return row


PROTOCOLS = {"P0": ("P0", "P0"), "P1": ("P1", "P1"), "P2": ("P2", "P2"), "P3": ("P3", "P3"), "P1t": ("P0", "P1")}

# ---------------------------------------------------------------- pre-scoring leakage check
for dataset in DATASETS:
    for s in sessions(dataset):
        corrupted = s.copy()
        late = elapsed(corrupted) > EVENT_SECONDS
        corrupted.loc[late, "Strain"] = np.random.default_rng(0).uniform(-5, 5, late.sum())
        for name in ("P1", "P2", "P3"):
            lo, hi = rule(name, s)
            lo_c, hi_c = rule(name, corrupted)
            fa, _, wa = calibrate(window(s, lo, hi))
            fb, _, wb = calibrate(window(corrupted, lo_c, hi_c))
            same = (lo, hi) == (lo_c, hi_c) and wa == wb and all(
                np.allclose(fa[e], fb[e], rtol=0, atol=0) for e in ELECTRODES)
            if not same:
                raise SystemExit(f"LEAKAGE CHECK FAILED: dataset {dataset} {name}")
print("pre-scoring leakage check passed (post-event labels do not affect c1 or calibration)")

# ---------------------------------------------------------------- scoring
rows, contents = [], []
for held in DATASETS:
    models = {}
    for source_rule in sorted({s for s, _ in PROTOCOLS.values()}):
        x, y = [], []
        for d in DATASETS:
            if d == held:
                continue
            for s in sessions(d):
                x.append(features(s, window(s, *rule(source_rule, s)))[0])
                y.append(s["Strain"].to_numpy())
        models[source_rule] = make_hgb().fit(np.vstack(x), np.concatenate(y))
    a, b = sessions(held)
    for direction, calib, test in (("A_to_B", a, b), ("B_to_A", b, a)):
        truth = test["Strain"].to_numpy()
        for protocol, (source_rule, target_rule) in PROTOCOLS.items():
            lo, hi = rule(target_rule, calib)
            f, physics = features(test, window(calib, lo, hi))
            key = {"protocol": protocol, "held_out_dataset": held, "temperature": TEMP_VAL[held], "direction": direction}
            rows.append({**key, "physics_rmse": rmse(physics, truth),
                         "hgb_rmse": rmse(models[source_rule].predict(f), truth)})
            contents.append({**key, **content(calib, lo, hi)})
frame = pd.DataFrame(rows)
frame.to_csv(sys.argv[2] + "_rmse.csv", index=False)
pd.DataFrame(contents).to_csv(sys.argv[2] + "_calibration_content.csv", index=False)

# ---------------------------------------------------------------- pre-registered decision
hgb = frame.pivot_table(index=["temperature", "direction"], columns="protocol", values="hgb_rmse")
phys = frame.pivot_table(index=["temperature", "direction"], columns="protocol", values="physics_rmse")
print("HGB RMSE per direction:\n", hgb.round(4).to_string())
print("\nphysics RMSE per direction:\n", phys.round(4).to_string())
print("\nmean HGB RMSE:", hgb.mean().round(4).to_dict())
print("mean physics RMSE:", phys.mean().round(4).to_dict())

verdict = {}
for q in ("P0", "P2"):
    m_ok = hgb["P1"].mean() <= 0.95 * hgb[q].mean()
    improvement = 100 * (1 - hgb["P1"] / hgb[q]).mean()
    wins = int((hgb["P1"] < hgb[q]).sum())
    worst = 100 * (hgb["P1"] / hgb[q] - 1).max()
    verdict[q] = m_ok and improvement >= 5 and wins >= 6
    print(f"\nP1 vs {q}: M_P1={hgb['P1'].mean():.4f} vs 0.95*M_{q}={0.95 * hgb[q].mean():.4f} -> {m_ok}; "
          f"I_{q}={improvement:.1f}% (need >= 5) ; wins {wins}/8 (need >= 6); worst relative deterioration {worst:+.1f}%")
if verdict["P0"] and verdict["P2"]:
    print("\nDECISION: support this candidate calibration convention within the observed corpus")
elif verdict["P0"]:
    print("\nDECISION: No demonstrated advantage from exclusion over pooling")
else:
    print("\nDECISION: Decision criterion unmet")
