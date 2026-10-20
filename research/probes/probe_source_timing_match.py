"""Why does the source calibration offset matter? Matching vs mid-session (prespecified).

Train one dynamic HGB per single source calibration offset in OFFSETS (row count identical:
one copy each). Score every model on every target calibration window (10 s step), then bin
target windows by start time into BINS whose centres are the OFFSETS.

Prespecified reading (written before running):
  H_match: in each bin, the best single offset is the one whose value is the bin centre.
           Supported if this holds in >= 3 of 4 bins.
  H_mid:   one offset is best in all 4 bins (mid-session calibration simply trains better).
Metric per bin and model: mean over directions of the per-direction median RMSE in that bin.

Usage: .venv/bin/python -I research/probes/probe_source_timing_match.py <repo root> <out csv>
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1] + "/src")
import strain.calibration as cal  # noqa: E402
from strain.features import build_features  # noqa: E402
from strain.protocol import make_hgb  # noqa: E402
from strain.spec import CALIBRATION_PREFIX_SECONDS, DATASETS, ROLLING_WINDOWS, TEMP_VAL, load_raw  # noqa: E402

STEP = 10.0
OFFSETS = (0.0, 240.0, 480.0, 720.0)
BINS = [(0.0, 120.0), (120.0, 360.0), (360.0, 600.0), (600.0, np.inf)]  # centred on OFFSETS
raw = {d: load_raw(d) for d in DATASETS}
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


def sessions(dataset):
    frame = raw[dataset]
    return [frame[frame.Experiment == e].reset_index(drop=True) for e in sorted(frame.Experiment.unique())]


def window(session, offset):
    t0 = float(session["Idx"].iloc[0]) + offset
    w = session[(session["Idx"] >= t0) & (session["Idx"] <= t0 + CALIBRATION_PREFIX_SECONDS)]
    return w.reset_index(drop=True)


def features(session, calibration_window):
    signals, _ = cal.session_signals(session, "physics", calibration_window)
    return build_features(signals, session["Temperature"].to_numpy(), "causal", False, ROLLING_WINDOWS)


rows = []
for held in DATASETS:
    models = {}
    for offset in OFFSETS:
        x, y = [], []
        for d in DATASETS:
            if d == held:
                continue
            for s in sessions(d):
                x.append(features(s, window(s, offset)))
                y.append(s["Strain"].to_numpy())
        models[offset] = make_hgb().fit(np.vstack(x), np.concatenate(y))
    a, b = sessions(held)
    for direction, calib, test in (("A_to_B", a, b), ("B_to_A", b, a)):
        truth = test["Strain"].to_numpy()
        span = float(calib["Idx"].iloc[-1] - calib["Idx"].iloc[0]) - CALIBRATION_PREFIX_SECONDS
        for start in np.arange(0.0, span + 1e-9, STEP):
            f = features(test, window(calib, start))
            row = {"held_out_dataset": held, "temperature": TEMP_VAL[held], "direction": direction,
                   "window_start_s": start}
            for offset, model in models.items():
                row[f"rmse_src{int(offset)}"] = rmse(model.predict(f), truth)
            rows.append(row)
frame = pd.DataFrame(rows)
frame["bin_centre"] = np.select([(frame.window_start_s >= lo) & (frame.window_start_s < hi) for lo, hi in BINS],
                                [int(o) for o in OFFSETS])
frame.to_csv(sys.argv[2], index=False)

columns = [f"rmse_src{int(o)}" for o in OFFSETS]
table = frame.groupby(["bin_centre", "temperature", "direction"])[columns].median().groupby("bin_centre").mean()
print("Mean over directions of per-direction median RMSE, rows = target window bin, cols = source offset:")
print(table.round(4).to_string())
best = table.idxmin(axis=1).str.replace("rmse_src", "").astype(int)
matches = int((best.index == best.to_numpy()).sum())
print("\nbest source offset per bin:", best.to_dict())
print(f"H_match holds in {matches}/4 bins (supported if >= 3)")
print("H_mid (one offset best in all bins):", best.nunique() == 1)
per_direction = frame.groupby(["bin_centre", "temperature", "direction"])[columns].median().idxmin(axis=1)
diag = (per_direction.str.replace("rmse_src", "").astype(int) == per_direction.index.get_level_values(0)).mean()
print(f"per (bin, direction) cells where the matched offset is best: {diag:.2f} (chance 0.25)")
