"""Fixed protocol specification: paths, constants, ordered variant registries, raw loader.

Every other module reads its configuration from here. Registry order is part of
the output contract: VARIANTS order sets CSV row order and figure label order.
This module touches the filesystem only through load_raw (read-only).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # src/strain/spec.py -> repository root
DATA_DIR = ROOT / "to_know_how" / "files"
RESULTS_DIR = ROOT / "results"
DATASETS = (1, 2, 3, 4)
TEMP_VAL = {1: 33, 2: 43, 3: 52, 4: 62}
ELECTRODES = ("Electrode1_resistance", "Electrode2_resistance")
FULL_SCALE = 0.5  # percent strain
ROLLING_WINDOWS = (21, 51, 101, 201)
CALIBRATION_PREFIX_SECONDS = 125.0
FINAL_VARIANT = "final_physics_causal_no_T_direct_HGB"


VARIANTS = [
    {
        "name": FINAL_VARIANT,
        "representation": "physics",
        "derivative": "causal",
        "include_temperature": False,
        "target": "direct",
        "model": "hgb",
        "windows": ROLLING_WINDOWS,
    },
    {
        "name": "physics_static_no_T_direct_HGB",
        "representation": "physics",
        "derivative": "causal",
        "include_temperature": False,
        "include_dynamics": False,
        "target": "direct",
        "model": "hgb",
        "windows": (),
    },
    {
        "name": "physics_causal_no_T_residual_HGB",
        "representation": "physics",
        "derivative": "causal",
        "include_temperature": False,
        "target": "residual",
        "model": "hgb",
        "windows": ROLLING_WINDOWS,
    },
    {
        "name": "physics_centered_no_T_direct_HGB",
        "representation": "physics",
        "derivative": "centered",
        "include_temperature": False,
        "target": "direct",
        "model": "hgb",
        "windows": ROLLING_WINDOWS,
    },
    {
        "name": "physics_causal_with_T_direct_HGB",
        "representation": "physics",
        "derivative": "causal",
        "include_temperature": True,
        "target": "direct",
        "model": "hgb",
        "windows": ROLLING_WINDOWS,
    },
    {
        "name": "physics_causal_no_T_Ridge",
        "representation": "physics",
        "derivative": "causal",
        "include_temperature": False,
        "target": "direct",
        "model": "ridge",
        "windows": ROLLING_WINDOWS,
    },
    {
        "name": "physics_causal_smoothing_only",
        "representation": "physics",
        "derivative": "causal",
        "include_temperature": False,
        "target": "direct",
        "model": "smoothing",
        "windows": ROLLING_WINDOWS,
    },
    {
        "name": "raw_with_T_direct_HGB",
        "representation": "raw",
        "derivative": "causal",
        "include_temperature": True,
        "target": "direct",
        "model": "hgb",
        "windows": ROLLING_WINDOWS,
    },
    {
        "name": "raw_no_T_direct_HGB",
        "representation": "raw",
        "derivative": "causal",
        "include_temperature": False,
        "target": "direct",
        "model": "hgb",
        "windows": ROLLING_WINDOWS,
    },
    {
        "name": "raw_low_strain_offset_with_T_direct_HGB",
        "representation": "zero_offset",
        "derivative": "causal",
        "include_temperature": True,
        "target": "direct",
        "model": "hgb",
        "windows": ROLLING_WINDOWS,
    },
    {
        "name": "raw_low_strain_offset_no_T_direct_HGB",
        "representation": "zero_offset",
        "derivative": "causal",
        "include_temperature": False,
        "target": "direct",
        "model": "hgb",
        "windows": ROLLING_WINDOWS,
    },
]

# Figure labels, aligned with VARIANTS by position.
VARIANT_LABELS = [
    "Final normalized HGB",
    "Static normalized HGB",
    "Residual HGB",
    "Centered HGB",
    "HGB with T",
    "Ridge",
    "Smoothing only",
    "Raw HGB + T",
    "Raw HGB, no T",
    "Raw + low-strain offset + T",
    "Raw + low-strain offset, no T",
]
assert len(VARIANT_LABELS) == len(VARIANTS)


WINDOW_VARIANTS = [
    ("history_local_only", ()),
    ("history_to_21", (21,)),
    ("history_to_51", (21, 51)),
    ("history_to_101", (21, 51, 101)),
    ("history_to_201", ROLLING_WINDOWS),
]


def load_raw(dataset):
    path = DATA_DIR / f"dataset{dataset}.csv"
    return pd.read_csv(path).sort_values(["Experiment", "Idx"]).reset_index(drop=True)
