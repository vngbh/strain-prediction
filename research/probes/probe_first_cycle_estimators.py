"""First-cycle apparent sensitivity under several estimators (reproduces Codex round-6 checks).

Descriptive calibration physics: uses reference strain labels of every session (offline analysis,
not a deployment path). For each session and electrode, compare the first-cycle slope dR/d(strain)
with later cycles.

Estimators on fixed 125 s windows at 0/250/500/750 s (first vs median of the later three):
  ols           R ~ 1 + strain
  ols_time      R ~ 1 + strain + t   (sensitivity check for baseline drift; time can also absorb hysteresis)
Cycle-aligned estimator (cycles = trough-to-trough of the reference strain):
  cycle_bins    equal-weight 0.05 % strain bins over 0.20-0.45 %, loading and unloading branch means
                pooled with equal weight; cycle 1 vs median of cycles 3-7.
"Faster initial decline": first-to-second drop larger than the mean later per-step drop.

Usage: .venv/bin/python -I research/probes/probe_first_cycle_estimators.py <repo root> <out csv>
"""
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, sys.argv[1] + "/src")
from strain.spec import DATASETS, ELECTRODES, TEMP_VAL, load_raw  # noqa: E402

WINDOW_STARTS = (0.0, 250.0, 500.0, 750.0)
BIN_EDGES = np.arange(0.20, 0.45 + 1e-9, 0.05)


def ols_slope(strain, resistance, t=None):
    columns = [np.ones_like(strain), strain] + ([t] if t is not None else [])
    return float(np.linalg.lstsq(np.column_stack(columns), resistance, rcond=None)[0][1])


def cycle_bins_slope(strain, resistance):
    loading = np.gradient(strain) > 0
    xs, ys = [], []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        for branch in (loading, ~loading):
            mask = branch & (strain >= lo) & (strain < hi)
            if mask.sum() < 5:
                return np.nan
            xs.append(strain[mask].mean())
            ys.append(resistance[mask].mean())
    return float(np.polyfit(xs, ys, 1)[0])


rows = []
for dataset in DATASETS:
    frame = load_raw(dataset)
    for experiment in sorted(frame.Experiment.unique()):
        s = frame[frame.Experiment == experiment].reset_index(drop=True)
        strain = s.Strain.to_numpy()
        t = s.Idx.to_numpy() - float(s.Idx.iloc[0])
        troughs, _ = find_peaks(-strain, prominence=0.25, distance=600)
        bounds = [0, *troughs.tolist(), len(strain)]
        cycles = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 300]
        for k, electrode in enumerate(ELECTRODES, start=1):
            resistance = s[electrode].to_numpy()
            row = {"temperature": TEMP_VAL[dataset], "experiment": experiment, "electrode": k}
            for name, extra_time in (("ols", False), ("ols_time", True)):
                slopes = []
                for start in WINDOW_STARTS:
                    m = (t >= start) & (t <= start + 125.0)
                    slopes.append(ols_slope(strain[m], resistance[m], t[m] if extra_time else None))
                row[f"{name}_first_vs_later_pct"] = 100 * (slopes[0] / np.median(slopes[1:]) - 1)
                steps = -np.diff(slopes)
                row[f"{name}_faster_initial_decline"] = bool(steps[0] > steps[1:].mean())
            per_cycle = [cycle_bins_slope(strain[a:b], resistance[a:b]) for a, b in cycles[:7]]
            later = [v for v in per_cycle[2:7] if np.isfinite(v)]
            row["cycles_found"] = len(cycles)
            row["cycle_bins_first_vs_later_pct"] = (100 * (per_cycle[0] / np.median(later) - 1)
                                                   if np.isfinite(per_cycle[0]) and later else np.nan)
            steps = -np.diff([v for v in per_cycle if np.isfinite(v)])
            row["cycle_bins_faster_initial_decline"] = bool(steps[0] > steps[1:].mean()) if len(steps) > 1 else np.nan
            rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(sys.argv[2], index=False)
pd.set_option("display.width", 220)
print(out.round(2).to_string(index=False))
for name in ("ols", "ols_time", "cycle_bins"):
    col = out[f"{name}_first_vs_later_pct"]
    print(f"{name:<11} first higher: {int((col > 0).sum())}/{col.notna().sum()}  range {col.min():+.1f}..{col.max():+.1f}%  "
          f"faster initial decline: {int(out[f'{name}_faster_initial_decline'].astype(bool).sum())}/{len(out)}")
