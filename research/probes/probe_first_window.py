"""Is the first calibration window of a session atypical (first-cycle effect)?

For every session in dataset1-4, estimate the four-level calibration (B_i, S_i) and its
within-window self-fit RMSE at window starts every 10 s. Report how far the 0 s window lies
from the distribution of later windows (robust z = (x0 - median_later) / (1.4826 * MAD_later)).
Descriptive only: windows overlap, and drift alone also makes early windows differ.

Usage: .venv/bin/python -I research/probes/probe_first_window.py <repo root> <out csv>
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1] + "/src")
import strain.calibration as cal  # noqa: E402
from strain.spec import CALIBRATION_PREFIX_SECONDS, DATASETS, ELECTRODES, TEMP_VAL, load_raw  # noqa: E402

STEP = 10.0
rows = []
for dataset in DATASETS:
    frame = load_raw(dataset)
    for experiment in sorted(frame.Experiment.unique()):
        session = frame[frame.Experiment == experiment].reset_index(drop=True)
        t0, t_end = float(session.Idx.iloc[0]), float(session.Idx.iloc[-1])
        for start in np.arange(0.0, t_end - t0 - CALIBRATION_PREFIX_SECONDS + 1e-9, STEP):
            w = session[(session.Idx >= t0 + start) & (session.Idx <= t0 + start + CALIBRATION_PREFIX_SECONDS)]
            w = w.reset_index(drop=True)
            fit, points, _ = cal.four_point_calibration(w)
            z1, z2, _ = cal.normalize(w, fit, 0.5)
            strain = w.Strain.to_numpy()
            row = {"dataset": dataset, "temperature": TEMP_VAL[dataset], "experiment": experiment,
                   "window_start_s": start}
            for k, electrode in enumerate(ELECTRODES, start=1):
                slope, baseline = fit[electrode]
                row[f"B{k}"], row[f"S{k}"] = baseline, slope
            row["selffit1"] = float(np.sqrt(np.mean((z1 - strain) ** 2)))
            row["selffit2"] = float(np.sqrt(np.mean((z2 - strain) ** 2)))
            rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(sys.argv[2], index=False)


def robust_z(group, column):
    first = group.loc[group.window_start_s == 0, column].iloc[0]
    later = group.loc[group.window_start_s >= CALIBRATION_PREFIX_SECONDS, column]
    mad = 1.4826 * np.median(np.abs(later - later.median()))
    return (first - later.median()) / mad if mad > 0 else np.nan


columns = ["B1", "S1", "B2", "S2", "selffit1", "selffit2"]
z = out.groupby(["temperature", "experiment"]).apply(lambda g: pd.Series({c: robust_z(g, c) for c in columns}))
print("Robust z of the 0 s window vs non-overlapping later windows (|z| > 3 = atypical):")
print(z.round(1).to_string())
print("\nsessions with |z| > 3:", (z.abs() > 3).sum().to_dict())
pct = out.groupby(["temperature", "experiment"]).apply(
    lambda g: pd.Series({c: 100 * (g.loc[g.window_start_s == 0, c].iloc[0] / g.loc[g.window_start_s >= 125, c].median() - 1)
                         for c in ["S1", "S2"]}))
print("\nS at 0 s relative to the later-window median (%):")
print(pct.round(1).to_string())
