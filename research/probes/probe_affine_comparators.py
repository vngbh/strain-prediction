"""Two-electrode affine comparators: full-label oracle per session vs 125 s opposite-session fit.

Usage: .venv/bin/python -I research/probes/probe_affine_comparators.py <repo root>
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1] + "/src")
from strain.calibration import calibration_prefix  # noqa: E402
from strain.spec import DATASETS, FINAL_VARIANT, RESULTS_DIR, load_raw  # noqa: E402


def design(df):
    return np.column_stack([np.ones(len(df)), df.Electrode1_resistance, df.Electrode2_resistance])


hgb = pd.read_csv(RESULTS_DIR / "tables" / "frozen_protocol_ablations_per_fold.csv")
hgb = hgb[hgb.variant == FINAL_VARIANT]
rows = []
for held in DATASETS:
    raw = load_raw(held)
    exps = sorted(raw.Experiment.unique())
    for direction, c, t in (("A_to_B", exps[0], exps[1]), ("B_to_A", exps[1], exps[0])):
        cal = raw[raw.Experiment == c].reset_index(drop=True)
        test = raw[raw.Experiment == t].reset_index(drop=True)
        prefix = calibration_prefix(cal)
        short = np.linalg.lstsq(design(prefix), prefix.Strain.values, rcond=None)[0]
        oracle = np.linalg.lstsq(design(test), test.Strain.values, rcond=None)[0]
        rmse = lambda coef: float(np.sqrt(np.mean((design(test) @ coef - test.Strain.values) ** 2)))
        fold = hgb[(hgb.held_out_dataset == held) & (hgb.direction == direction)].iloc[0]
        rows.append(dict(T=fold.temperature, direction=direction, physics=fold.physics_rmse,
                         affine2_125s=rmse(short), affine2_oracle_full_labels=rmse(oracle), hgb=fold.rmse))
frame = pd.DataFrame(rows)
print(frame.round(4).to_string(index=False))
print(frame.drop(columns=["T", "direction"]).mean().round(4).to_string())
