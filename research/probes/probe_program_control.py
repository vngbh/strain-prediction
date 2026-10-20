"""Direction 1, analysis 3: loading-program control (prespecified with Codex before running).

Same frozen LOTO, same HGB, same repository calibration. Feature set replaced by:
  static [z1, z2, zf] + causal program state estimated from resistance only:
  branch (+1 loading / -1 unloading, deadband on the trailing-21 derivative of zf),
  seconds since last branch switch to loading (trough) and to unloading (peak),
  completed cycle count, duration of the last completed cycle.
No rolling amplitude statistics. Nothing uses Strain or future samples. DEADBAND fixed a priori
(~20% of the typical loading rate, 0.009 %/s at 10 Hz), not tuned on any fold.

Prespecified kill rule for "detailed history demonstrates transferable sensor memory":
  program-control mean RMSE within 5% of dynamic HGB AND per-direction RMSE within 10% in >= 7/8.

Usage: .venv/bin/python -I research/probes/probe_program_control.py <repo root> [out csv]
"""
import sys

import numpy as np

sys.path.insert(0, sys.argv[1] + "/src")
import strain.protocol as proto  # noqa: E402
from strain.features import causal_derivative  # noqa: E402
from strain.metrics import summarize  # noqa: E402
from strain.spec import DATASETS, VARIANTS, load_raw  # noqa: E402

DEADBAND = 0.0002  # z units per sample
DT = 0.1


def program_state(zf):
    derivative = causal_derivative(zf)
    n = len(zf)
    branch = np.empty(n)
    since_trough = np.empty(n)
    since_peak = np.empty(n)
    cycles = np.empty(n)
    last_cycle = np.empty(n)
    state, t_trough, t_peak, count, previous_trough, cycle_length = -1.0, 0.0, 0.0, 0.0, None, 0.0
    for i in range(n):
        t = i * DT
        if state < 0 and derivative[i] > DEADBAND:  # unloading -> loading: a trough just passed
            state, t_trough = 1.0, t
            if previous_trough is not None:
                count, cycle_length = count + 1, t - previous_trough
            previous_trough = t
        elif state > 0 and derivative[i] < -DEADBAND:  # loading -> unloading: a peak just passed
            state, t_peak = -1.0, t
        branch[i], since_trough[i], since_peak[i] = state, t - t_trough, t - t_peak
        cycles[i], last_cycle[i] = count, cycle_length
    return np.column_stack([branch, since_trough, since_peak, cycles, last_cycle])


original_build = proto.build_features


def program_features(signals, temperature, derivative_mode, include_temperature, windows, include_dynamics=True):
    x1, x2, xf = signals
    return np.column_stack([x1, x2, xf, program_state(xf)])


raw = {d: load_raw(d) for d in DATASETS}
results = {}
for label, build in [("dynamic (final)", original_build), ("program control", program_features)]:
    proto.build_features = build
    frame, _ = proto.run_variants(raw, [VARIANTS[0]])
    results[label] = frame
    s = summarize(frame).iloc[0]
    print(f"{label:<18} meanRMSE={s.mean_rmse:.4f} meanImp={s.mean_pct_improvement:5.1f}% "
          f"worst={s.worst_fold_pct_improvement:6.1f}% dirs={s.directions_improved}/8")
proto.build_features = original_build
if len(sys.argv) > 2:
    import pandas as pd

    pd.concat([f.assign(feature_set=k) for k, f in results.items()], ignore_index=True).to_csv(sys.argv[2], index=False)

dyn, prog = results["dynamic (final)"].rmse.to_numpy(), results["program control"].rmse.to_numpy()
mean_ratio = prog.mean() / dyn.mean()
per_dir = prog / dyn
within = int((per_dir <= 1.10).sum())
print(f"\nprogram/dynamic mean RMSE ratio = {mean_ratio:.3f} (kill if <= 1.05)")
print(f"per-direction ratios = {np.round(per_dir, 2).tolist()}; within 10%: {within}/8 (kill if >= 7)")
killed = mean_ratio <= 1.05 and within >= 7
print("VERDICT:", "sensor-memory claim KILLED (program control matches)" if killed
      else "program control does NOT match; memory claim survives this control (not proven)")
