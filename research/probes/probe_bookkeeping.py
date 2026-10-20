"""Calibration bookkeeping sensitivity: nominal band levels (repository) vs measured band means.

Same frozen LOTO and HGB; only the strain coordinate assigned to each of the four calibration
aggregates changes (nominal level, with exactly 0 for the low band, vs the mean measured strain of
the averaged rows). Writes per-fold rows for both conventions.

Usage: .venv/bin/python -I research/probes/probe_bookkeeping.py <repo root> <out csv>
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1] + "/src")
import strain.calibration as cal  # noqa: E402
from strain.protocol import run_variants  # noqa: E402
from strain.spec import DATASETS, ELECTRODES, VARIANTS, load_raw  # noqa: E402

nominal = cal.four_point_calibration


def measured_means(df):
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
    info = {"calibration_mode": "measured_band_means", "calibration_rows_used": len(set(used)),
            "calibration_duration_seconds": float(df["Idx"].iloc[-1] - df["Idx"].iloc[0])}
    return fit, points, info


raw = {d: load_raw(d) for d in DATASETS}
frames = []
for convention, function in [("nominal_levels", nominal), ("measured_band_means", measured_means)]:
    cal.four_point_calibration = function
    frame, _ = run_variants(raw, [VARIANTS[0]])
    frames.append(frame.assign(convention=convention))
cal.four_point_calibration = nominal
out = pd.concat(frames, ignore_index=True)
out.to_csv(sys.argv[2], index=False)
print(out.groupby("convention").agg(physics=("physics_rmse", "mean"), hgb=("rmse", "mean"),
                                    mean_imp=("pct_improvement", "mean"), worst=("pct_improvement", "min"),
                                    improved=("pct_improvement", lambda s: int((s > 0).sum()))).round(4))
