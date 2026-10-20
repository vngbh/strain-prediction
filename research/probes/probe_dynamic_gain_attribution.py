"""Pre-registered test: does re-aligning labels contract the static-dynamic RMSE gap?

Implements research/directions/2026-10-21-preregistration-dynamic-gain-attribution.md; do not
change without a new dated pre-registration. dataset0 is sealed and never loaded.

Usage: .venv/bin/python -I research/probes/probe_dynamic_gain_attribution.py <repo root> <out prefix>
"""
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

root = sys.argv[1]
sys.path.insert(0, root + "/src")
import strain.calibration as cal  # noqa: E402
from strain.features import build_features  # noqa: E402
from strain.protocol import make_hgb  # noqa: E402
from strain.spec import CALIBRATION_PREFIX_SECONDS, DATASETS, TEMP_VAL, VARIANTS, load_raw  # noqa: E402

assert 0 not in DATASETS, "dataset0 is sealed"
VARIANT = {v["name"]: v for v in VARIANTS}
STATIC, DYNAMIC = VARIANT["physics_static_no_T_direct_HGB"], VARIANT["final_physics_causal_no_T_direct_HGB"]
FLOOR = 0.005


def stop(reason):
    raise SystemExit(f"RUN STOPPED: {reason}")


# ---------------------------------------------------------------- inputs and guards
raw = {d: load_raw(d) for d in DATASETS}
sessions = {}
for d in DATASETS:
    for e in sorted(raw[d].Experiment.unique()):
        s = raw[d][raw[d].Experiment == e].reset_index(drop=True)
        t = s.Idx.to_numpy() - float(s.Idx.iloc[0])
        if not (np.isfinite(t).all() and (np.diff(t) > 0).all()):
            stop(f"timestamps not finite/strictly increasing: {d}/{e}")
        troughs, _ = find_peaks(-s.Strain.to_numpy(), prominence=0.25, distance=600)
        cycle_id = np.full(len(t), -1)
        for i, (a, b) in enumerate(zip(troughs[:-1], troughs[1:])):
            cycle_id[a:b] = i
        sessions[(d, e)] = {"frame": s, "t": t, "cycle": cycle_id}

lag = pd.read_csv(f"{root}/research/probes/out/reference_lag_v2_per_electrode_session.csv")
offsets = lag[["dataset", "experiment", "electrode", "tau_s", "tau_odd", "tau_even"]]
for key, g in offsets.groupby(["dataset", "experiment"]):
    if len(g) != 2 or set(g.electrode) != {1, 2} or not np.isfinite(g[["tau_s", "tau_odd", "tau_even"]].to_numpy()).all():
        stop(f"session {key}: need exactly electrodes {{1, 2}} with finite tau_s, tau_odd, tau_even")
if set(map(tuple, offsets[["dataset", "experiment"]].drop_duplicates().to_numpy())) != set(sessions):
    stop("offset sessions do not match the evaluated sessions")
taus = {c: offsets.groupby(["dataset", "experiment"])[c].mean() for c in ("tau_s", "tau_odd", "tau_even")}


def shifted(key, tau):
    """eps(t - tau); NaN where t - tau falls outside the session's time range."""
    s, t = sessions[key]["frame"], sessions[key]["t"]
    q = t - tau
    out = np.interp(q, t, s.Strain.to_numpy())
    out[(q < t[0]) | (q > t[-1])] = np.nan
    return out


# ---------------------------------------------------------------- re-implemented frozen protocol
def calibrate(key, cal_labels):
    s, t = sessions[key]["frame"], sessions[key]["t"]
    mask = (t <= CALIBRATION_PREFIX_SECONDS) & np.isfinite(cal_labels)  # clock never restarted
    window = s[mask].copy()
    window["Strain"] = cal_labels[mask]
    fit, points, _ = cal.four_point_calibration(window.reset_index(drop=True))
    return fit, cal.fusion_weight_from_points(fit, points)


def features(key, fit, weight, variant):
    s = sessions[key]["frame"]
    z = cal.normalize(s, fit, weight)
    x = build_features(z, s.Temperature.to_numpy(), variant["derivative"], variant["include_temperature"],
                       variant["windows"], variant.get("include_dynamics", True))
    return x, z[2]


def run_arm(label_of, cal_label_of, score_mask_of, held_tau=None):
    """label_of(held, key) -> labels; cal_label_of(held, key) -> calibration labels;
    score_mask_of(key) -> extra scoring mask. Returns one row per direction and variant."""
    rows = []
    for held in DATASETS:
        for variant in (STATIC, DYNAMIC):
            xs, ys = [], []
            for d in DATASETS:
                if d == held:
                    continue
                for key in [k for k in sessions if k[0] == d]:
                    labels = label_of(held, key)
                    fit, weight = calibrate(key, cal_label_of(held, key))
                    x, _ = features(key, fit, weight, variant)
                    valid = np.isfinite(labels)
                    xs.append(x[valid])
                    ys.append(labels[valid])
            model = make_hgb().fit(np.vstack(xs), np.concatenate(ys))
            a, b = [k for k in sessions if k[0] == held]
            for direction, c, t in (("A_to_B", a, b), ("B_to_A", b, a)):
                fit, weight = calibrate(c, cal_label_of(held, c))
                x, physics = features(t, fit, weight, variant)
                labels = label_of(held, t)
                mask = np.isfinite(labels) & score_mask_of(t)
                pred = model.predict(x)
                rmse = lambda p: float(np.sqrt(np.mean((p[mask] - labels[mask]) ** 2)))
                rows.append({"held_out_dataset": held, "temperature": TEMP_VAL[held], "direction": direction,
                             "variant": variant["name"], "rmse": rmse(pred), "physics_rmse": rmse(physics),
                             "scored_rows": int(mask.sum())})
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame[["rmse", "physics_rmse"]].to_numpy()).all() or len(frame) != 16:
        stop("incomplete or non-finite fold rows")
    phys = frame.pivot_table(index=["held_out_dataset", "direction"], columns="variant", values="physics_rmse")
    if (phys.max(axis=1) - phys.min(axis=1)).max() > 1e-12:
        stop("physics RMSE differs between variants within a fold")
    return frame


everything = lambda key: np.ones(len(sessions[key]["t"]), bool)
original = lambda held, key: sessions[key]["frame"].Strain.to_numpy()
oracle = lambda held, key: shifted(key, float(taus["tau_s"].loc[key]))


def source_constant(held, key):
    tau = float(taus["tau_s"].drop(index=held, level="dataset").median())
    return shifted(key, tau)


arms = {}
arms["A0_original"] = run_arm(original, original, everything)

# sanity gate: reproduce the repository per-direction RMSE
repo = pd.read_csv(f"{root}/results/tables/frozen_protocol_ablations_per_fold.csv")
check = arms["A0_original"].merge(repo[["variant", "held_out_dataset", "direction", "rmse"]],
                                  on=["variant", "held_out_dataset", "direction"], suffixes=("", "_repo"))
if len(check) != 16 or (check.rmse - check.rmse_repo).abs().max() > 1e-9:
    stop(f"A0 does not reproduce the repository (max diff {(check.rmse - check.rmse_repo).abs().max():.2e})")
print("sanity gate passed: A0 reproduces repository static and dynamic RMSE")

arms["A1_oracle"] = run_arm(oracle, oracle, everything)
arms["A2_source_constant"] = run_arm(source_constant, source_constant, everything)
arms["A3_oracle_fixed_normalization"] = run_arm(oracle, original, everything)
odd = lambda key: (sessions[key]["cycle"] >= 0) & (sessions[key]["cycle"] % 2 == 0)
even = lambda key: (sessions[key]["cycle"] >= 0) & (sessions[key]["cycle"] % 2 == 1)
tau_odd = lambda held, key: shifted(key, float(taus["tau_odd"].loc[key]))
tau_even = lambda held, key: shifted(key, float(taus["tau_even"].loc[key]))
arms["A4_run1_tauodd_score_even"] = run_arm(tau_odd, tau_odd, even)
arms["A4_run2_taueven_score_odd"] = run_arm(tau_even, tau_even, odd)
arms["A0_score_even"] = run_arm(original, original, even)
arms["A0_score_odd"] = run_arm(original, original, odd)

frame = pd.concat([f.assign(arm=k) for k, f in arms.items()], ignore_index=True)
frame.to_csv(sys.argv[2] + "_per_fold.csv", index=False)


def gaps(arm):
    w = arms[arm].pivot_table(index=["held_out_dataset", "direction"], columns="variant", values="rmse")
    return w[STATIC["name"]] - w[DYNAMIC["name"]]


def verdict(g_arm, g_ref):
    if g_ref.mean() < FLOOR:
        return np.nan, "ratio uninterpretable"
    F = 1 - g_arm.mean() / g_ref.mean()
    smaller = int((g_arm < g_ref).sum())
    if F < 0:
        label = "gap expansion"
    elif F >= 0.5:
        label = "substantial contraction" if smaller >= 6 else "heterogeneous"
    elif F <= 0.2:
        label = "minor contraction"
    else:
        label = "partial contraction"
    return F, f"{label} (smaller gap in {smaller}/8)"


g0 = gaps("A0_original")
table = pd.DataFrame({"G_A0": g0})
for arm in ("A1_oracle", "A2_source_constant", "A3_oracle_fixed_normalization"):
    table[f"G_{arm}"] = gaps(arm)
g_cf = (gaps("A4_run1_tauodd_score_even") + gaps("A4_run2_taueven_score_odd")) / 2
g0_cf = (gaps("A0_score_even") + gaps("A0_score_odd")) / 2
table["G_A4_crossfit"], table["G_A0_crossfit_subsets"] = g_cf, g0_cf
for arm in ("A1_oracle", "A2_source_constant", "A3_oracle_fixed_normalization"):
    table[f"dG_{arm}"] = table[f"G_{arm}"] - table["G_A0"]
table["dG_A4_crossfit"] = table["G_A4_crossfit"] - table["G_A0_crossfit_subsets"]
table.to_csv(sys.argv[2] + "_gaps.csv")
pd.set_option("display.width", 240)
print("\nper-direction static-dynamic RMSE gaps (% strain):\n", table.round(5).to_string())

results = {}
for arm in ("A1_oracle", "A2_source_constant", "A3_oracle_fixed_normalization"):
    results[arm] = verdict(table[f"G_{arm}"], g0)
results["A4_crossfit"] = verdict(g_cf, g0_cf)
for arm, (F, label) in results.items():
    print(f"{arm}: F = {F:.3f} -> {label}")

a1, a2 = results["A1_oracle"][1], results["A2_source_constant"][1]
print("\nPRIMARY (diagnostic, A1):", a1)
print("PRIMARY (transfer, A2):", a2)
if a1.startswith("substantial") and not a2.startswith("substantial"):
    print("COMBINED: oracle-specific contraction; transfer unsupported")

print("\nleave-one-temperature-group-out F:")
for held in DATASETS:
    keep = g0.index.get_level_values(0) != held
    row = {arm: verdict(table[f"G_{arm}"][keep], g0[keep])[0] for arm in ("A1_oracle", "A2_source_constant")}
    print(f"  without {TEMP_VAL[held]} degC:", {k: round(v, 3) for k, v in row.items()})
print("\nmean RMSE by arm (static, dynamic, physics):")
print(frame.groupby(["arm", "variant"])[["rmse", "physics_rmse"]].mean().round(4).to_string())
