"""Direction 1, analysis 4: dynamic vs static across windows, matched source shifts, and
calibration-diversity training. All settings fixed before running (not tuned on any fold).

A. Static vs dynamic HGB over every target calibration window (10 s step), both trained with the
   repository source convention (source self-calibration at 0 s).
B. Matched shift: source sessions self-calibrate at offset s (same s as the target window),
   s in OFFSETS; compare with the 0 s-trained model on the same target windows.
C. Calibration-diversity training: every source session normalized with each window in OFFSETS,
   stacked into one training set; evaluated over all target windows.
Windows overlap and share sessions; report per fold, never as independent events.

Usage: .venv/bin/python -I research/probes/probe_calibration_shift.py <repo root> <out csv>
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
raw = {d: load_raw(d) for d in DATASETS}
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


def sessions(dataset):
    frame = raw[dataset]
    return [frame[frame.Experiment == e].reset_index(drop=True) for e in sorted(frame.Experiment.unique())]


def window(session, offset):
    t0 = float(session["Idx"].iloc[0]) + offset
    w = session[(session["Idx"] >= t0) & (session["Idx"] <= t0 + CALIBRATION_PREFIX_SECONDS)]
    return w.reset_index(drop=True)


def features(session, calibration_window, dynamic):
    signals, _ = cal.session_signals(session, "physics", calibration_window)
    return build_features(signals, session["Temperature"].to_numpy(), "causal", False,
                          ROLLING_WINDOWS if dynamic else (), include_dynamics=dynamic), signals[2]


def fit(held, offsets, dynamic):
    x, y = [], []
    for d in DATASETS:
        if d == held:
            continue
        for s in sessions(d):
            for offset in offsets:
                x.append(features(s, window(s, offset), dynamic)[0])
                y.append(s["Strain"].to_numpy())
    model = make_hgb()
    model.fit(np.vstack(x), np.concatenate(y))
    return model


rows = []
for held in DATASETS:
    static0, dynamic0 = fit(held, (0.0,), False), fit(held, (0.0,), True)
    diverse = fit(held, OFFSETS, True)
    matched = {o: (dynamic0 if o == 0.0 else fit(held, (o,), True)) for o in OFFSETS}
    a, b = sessions(held)
    for direction, calib, test in (("A_to_B", a, b), ("B_to_A", b, a)):
        truth = test["Strain"].to_numpy()
        span = float(calib["Idx"].iloc[-1] - calib["Idx"].iloc[0]) - CALIBRATION_PREFIX_SECONDS
        for start in np.arange(0.0, span + 1e-9, STEP):
            w = window(calib, start)
            x_dyn, physics = features(test, w, True)
            x_sta, _ = features(test, w, False)
            row = {"held_out_dataset": held, "temperature": TEMP_VAL[held], "direction": direction,
                   "window_start_s": start, "physics_rmse": rmse(physics, truth),
                   "static_rmse": rmse(static0.predict(x_sta), truth),
                   "dynamic_rmse": rmse(dynamic0.predict(x_dyn), truth),
                   "diverse_rmse": rmse(diverse.predict(x_dyn), truth), "matched_rmse": np.nan}
            if start in matched:
                row["matched_rmse"] = rmse(matched[start].predict(x_dyn), truth)
            rows.append(row)
frame = pd.DataFrame(rows)
frame.to_csv(sys.argv[2], index=False)

g = frame.groupby(["temperature", "direction"])
print("A/C per fold over all target windows (median RMSE, p90 RMSE, shares):")
print(pd.DataFrame({
    "n": g.size(),
    "physics_med": g.physics_rmse.median(),
    "static_med": g.static_rmse.median(),
    "dynamic_med": g.dynamic_rmse.median(),
    "dynamic_p90": g.dynamic_rmse.quantile(0.9),
    "diverse_med": g.diverse_rmse.median(),
    "diverse_p90": g.diverse_rmse.quantile(0.9),
    "share_dyn<static": g.apply(lambda d: (d.dynamic_rmse < d.static_rmse).mean()),
    "share_diverse<dyn": g.apply(lambda d: (d.diverse_rmse < d.dynamic_rmse).mean()),
}).round(4).to_string())
print("\nfold-mean of per-fold medians:",
      g[["physics_rmse", "static_rmse", "dynamic_rmse", "diverse_rmse"]].median().mean().round(4).to_dict())
print("fold-mean of per-fold p90:",
      g[["dynamic_rmse", "diverse_rmse"]].quantile(0.9).mean().round(4).to_dict())
print("fold-mean of per-fold max:", g[["dynamic_rmse", "diverse_rmse"]].max().mean().round(4).to_dict())

m = frame.dropna(subset=["matched_rmse"])
print("\nB matched source/target offset (dynamic, 0 s-trained vs matched-trained vs diverse):")
print(m.pivot_table(index=["temperature", "direction"], columns="window_start_s",
                    values=["dynamic_rmse", "matched_rmse", "diverse_rmse"]).round(4).to_string())
print("mean over folds and offsets:", m[["dynamic_rmse", "matched_rmse", "diverse_rmse"]].mean().round(4).to_dict())
