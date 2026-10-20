"""Control for analysis 4C: is the calibration-diversity gain just 4x training rows?

Trains the dynamic HGB on the 0 s-calibrated source sessions replicated len(OFFSETS) times
(same row count as the diversity model, no calibration diversity) and scores it on the same
target windows as probe_calibration_shift.py, then compares with that script's CSV.

Usage: .venv/bin/python -I research/probes/probe_diversity_volume_control.py <repo root> <shift csv>
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1] + "/src")
import strain.calibration as cal  # noqa: E402
from strain.features import build_features  # noqa: E402
from strain.protocol import make_hgb  # noqa: E402
from strain.spec import CALIBRATION_PREFIX_SECONDS, DATASETS, ROLLING_WINDOWS, load_raw  # noqa: E402

REPLICAS = 4
raw = {d: load_raw(d) for d in DATASETS}
shift = pd.read_csv(sys.argv[2])
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


values = []
for held in DATASETS:
    x, y = [], []
    for d in DATASETS:
        if d == held:
            continue
        for s in sessions(d):
            f = features(s, window(s, 0.0))
            x += [f] * REPLICAS
            y += [s["Strain"].to_numpy()] * REPLICAS
    model = make_hgb()
    model.fit(np.vstack(x), np.concatenate(y))
    a, b = sessions(held)
    for direction, calib, test in (("A_to_B", a, b), ("B_to_A", b, a)):
        rows = shift[(shift.held_out_dataset == held) & (shift.direction == direction)]
        for start in rows.window_start_s:
            values.append(rmse(model.predict(features(test, window(calib, start))), test["Strain"].to_numpy()))
shift["replicated_rmse"] = values
g = shift.groupby(["temperature", "direction"])
print(g[["dynamic_rmse", "replicated_rmse", "diverse_rmse"]].median().round(4).to_string())
print("\nfold-mean of medians:", g[["dynamic_rmse", "replicated_rmse", "diverse_rmse"]].median().mean().round(4).to_dict())
print("share of windows diverse < replicated, per fold:",
      g.apply(lambda d: round((d.diverse_rmse < d.replicated_rmse).mean(), 2)).tolist())
