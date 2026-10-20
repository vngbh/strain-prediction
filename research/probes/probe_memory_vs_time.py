"""Direction 1, analysis 2: is the dynamic gain sensor memory, or elapsed session time / program?

Frozen LOTO with the repository calibration (fixed 125 s prefix), feature sets swapped via a hook:
  static            [z1, z2, zf]
  static + t        [z1, z2, zf, seconds since session start]
  dynamic (final)   repository features
  dynamic + t       repository features + seconds since session start
If static + t approaches dynamic, much of the "history" gain is explainable by elapsed time
(drift trajectory or loading program), not by sensor memory. Elapsed time is available online.

Usage: .venv/bin/python -I research/probes/probe_memory_vs_time.py <repo root>
"""
import sys

import numpy as np

sys.path.insert(0, sys.argv[1] + "/src")
import strain.protocol as proto  # noqa: E402
from strain.metrics import summarize  # noqa: E402
from strain.spec import DATASETS, VARIANTS, load_raw  # noqa: E402

raw = {d: load_raw(d) for d in DATASETS}
original_build = proto.build_features
original_signals = proto.session_signals
elapsed = {}


def signals_with_time(df, representation, calibration_source=None):
    elapsed["t"] = df["Idx"].to_numpy() - float(df["Idx"].iloc[0])
    return original_signals(df, representation, calibration_source)


def hook(include_dynamics, add_time):
    def build(signals, temperature, derivative_mode, include_temperature, windows, _include_dynamics=True):
        feats = original_build(signals, temperature, derivative_mode, include_temperature, windows, include_dynamics)
        return np.column_stack([feats, elapsed["t"]]) if add_time else feats
    return build


proto.session_signals = signals_with_time
for label, dyn, add_t in [("static", False, False), ("static + elapsed time", False, True),
                          ("dynamic (final)", True, False), ("dynamic + elapsed time", True, True)]:
    proto.build_features = hook(dyn, add_t)
    frame, _ = proto.run_variants(raw, [VARIANTS[0]])
    s = summarize(frame).iloc[0]
    print(f"{label:<24} meanRMSE={s.mean_rmse:.4f} meanImp={s.mean_pct_improvement:5.1f}% "
          f"worst={s.worst_fold_pct_improvement:6.1f}% dirs={s.directions_improved}/8  "
          f"per-fold={frame.pct_improvement.round(1).tolist()}")
