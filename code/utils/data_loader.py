"""
Shared data-loading utilities for all phases.

Input:  data/dataset{0-4}.csv — columns: Idx, Experiment, Temperature,
        Electrode1_resistance, Electrode2_resistance, Strain
Output: pd.DataFrame (single dataset) or concatenated DataFrame (multi-dataset)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"

ALL_DATASETS  = [0, 1, 2, 3, 4]
DATASETS      = [1, 2, 3, 4]        # ds0 excluded — single experiment, likely sample data
INPUT_FEATURES = ["Electrode1_resistance", "Electrode2_resistance"]
OUTPUT_FEATURE = "Strain"
TEMP_LABEL    = {0: "21°C", 1: "33°C", 2: "43°C", 3: "52°C", 4: "62°C"}


def load_dataset(idx: int) -> pd.DataFrame:
    """Load one dataset CSV; adds dataset_id column for traceability."""
    path = DATA_DIR / f"dataset{idx}.csv"
    df = pd.read_csv(path)
    df["dataset_id"] = idx
    return df


def load_all_datasets() -> pd.DataFrame:
    """Concatenate ds1–ds4 into one DataFrame (excludes ds0)."""
    return pd.concat([load_dataset(i) for i in DATASETS], ignore_index=True)


def load_datasets(indices: list[int]) -> pd.DataFrame:
    """Concatenate an arbitrary subset of datasets by index list."""
    return pd.concat([load_dataset(i) for i in indices], ignore_index=True)
