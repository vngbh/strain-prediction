"""Claude's evidence for the research debate (scratch, not project code). Uses the frozen protocol.

A. Verify Codex: calibration points at measured band means instead of nominal levels.
B. Mechanism: does HGB's gain need z1/z2 separately (electrode-difference information)?
   B1 fused-only: features from zf only.  B2 final + explicit d = z1 - z2 features.
C. Physics drift observer, frozen: zf - k * trailing_mean(z1 - z2, W) with k fitted on SOURCE
   sessions only (target labels untouched). Compare to physics and HGB.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1] + "/src")
import strain.calibration as cal  # noqa: E402
import strain.protocol as proto  # noqa: E402
from strain.features import build_features, rolling  # noqa: E402
from strain.metrics import summarize  # noqa: E402
from strain.spec import DATASETS, ELECTRODES, ROLLING_WINDOWS, VARIANTS, load_raw  # noqa: E402

raw = {d: load_raw(d) for d in DATASETS}
final = VARIANTS[0]


def summary(frame, label):
    s = summarize(frame).iloc[0]
    print(f"{label:<44} meanRMSE={s.mean_rmse:.4f} physRMSE={s.mean_physics_rmse:.4f} "
          f"meanImp={s.mean_pct_improvement:5.1f}% worst={s.worst_fold_pct_improvement:6.1f}% "
          f"dirs={s.directions_improved}/8")
    return frame


# ---------------- baseline
base, _ = proto.run_variants(raw, [final])
summary(base, "final (repository)")

# ---------------- A: measured band means
orig_fpc = cal.four_point_calibration


def measured_mean_calibration(df):
    strain = df["Strain"].to_numpy()
    rng = strain.max() - strain.min()
    levels = np.linspace(strain.min() + 0.05 * rng, strain.max() - 0.05 * rng, 3)
    points, used = [], []
    for i, level in enumerate([0.0, *levels]):
        mask = strain <= strain.min() + 0.02 if i == 0 else np.abs(strain - level) <= 0.03
        used.extend(np.flatnonzero(mask).tolist())
        points.append((float(strain[mask].mean()), *(df[e].to_numpy()[mask].mean() for e in ELECTRODES)))
    points = np.asarray(points)
    fit = {e: tuple(np.polyfit(points[:, 0], points[:, k], 1)) for k, e in enumerate(ELECTRODES, start=1)}
    info = {"calibration_mode": "measured_means", "calibration_rows_used": len(set(used)),
            "calibration_duration_seconds": float(df["Idx"].iloc[-1] - df["Idx"].iloc[0])}
    return fit, points, info


cal.four_point_calibration = measured_mean_calibration
a, _ = proto.run_variants(raw, [final])
summary(a, "A: measured band means (Codex claim)")
print("   per-fold improvement:", a.pct_improvement.round(1).tolist())
cal.four_point_calibration = orig_fpc

# ---------------- B: mechanism ablations via a feature hook
orig_build = proto.build_features


def fused_only(signals, temperature, derivative_mode, include_temperature, windows, include_dynamics=True):
    _, _, xf = signals
    return orig_build((xf, xf, xf), temperature, derivative_mode, include_temperature, windows, include_dynamics)


def with_difference(signals, temperature, derivative_mode, include_temperature, windows, include_dynamics=True):
    x1, x2, xf = signals
    feats = orig_build(signals, temperature, derivative_mode, include_temperature, windows, include_dynamics)
    d = x1 - x2
    extra = [d] + [rolling(d, w, "mean") for w in (201, 1001)]
    return np.column_stack([feats, *extra])


for label, hook in [("B1: fused zf only (no electrode difference)", fused_only),
                    ("B2: final + explicit z1-z2 trailing means", with_difference)]:
    proto.build_features = hook
    f, _ = proto.run_variants(raw, [final])
    summary(f, label)
proto.build_features = orig_build

# ---------------- C: frozen 1-parameter physics drift observer
W = 1001


def observer_inputs(session, calibration_source=None):
    (z1, z2, zf), _ = cal.session_signals(session, "physics", calibration_source)
    return zf, rolling(z1 - z2, W, "mean")


rows = []
for held in DATASETS:
    X, y = [], []
    for d in DATASETS:
        if d == held:
            continue
        for e in sorted(raw[d].Experiment.unique()):
            s = raw[d][raw[d].Experiment == e].reset_index(drop=True)
            zf, dd = observer_inputs(s)
            X.append(dd); y.append(zf - s.Strain.to_numpy())
    X, y = np.concatenate(X), np.concatenate(y)
    k = float(X @ y / (X @ X))
    exps = sorted(raw[held].Experiment.unique())
    for direction, c, t in (("A_to_B", exps[0], exps[1]), ("B_to_A", exps[1], exps[0])):
        calib = raw[held][raw[held].Experiment == c].reset_index(drop=True)
        test = raw[held][raw[held].Experiment == t].reset_index(drop=True)
        zf, dd = observer_inputs(test, calib)
        truth = test.Strain.to_numpy()
        phys = np.sqrt(np.mean((zf - truth) ** 2)); obs = np.sqrt(np.mean((zf - k * dd - truth) ** 2))
        hgb = base[(base.held_out_dataset == held) & (base.direction == direction)].rmse.iloc[0]
        rows.append(dict(held=held, direction=direction, k=round(k, 2), physics=phys, observer=obs, hgb=hgb,
                         obs_imp=100 * (phys - obs) / phys))
c = pd.DataFrame(rows)
print("\nC: frozen physics drift observer (k fitted on source sessions only)")
print(c.round(4).to_string(index=False))
print(f"   mean RMSE physics={c.physics.mean():.4f} observer={c.observer.mean():.4f} hgb={c.hgb.mean():.4f} "
      f"observer meanImp={c.obs_imp.mean():.1f}% dirs improved={(c.obs_imp > 0).sum()}/8")
