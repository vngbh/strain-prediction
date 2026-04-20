import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"

DATASETS = [0, 1, 2, 3, 4]
INPUT_FEATURES = ["Electrode1_resistance", "Electrode2_resistance"]
OUTPUT_FEATURE = "Strain"
ALL_FEATURES = INPUT_FEATURES + ["Temperature"]


def load_dataset(idx: int) -> pd.DataFrame:
    path = DATA_DIR / f"dataset{idx}.csv"
    df = pd.read_csv(path)
    df["dataset_id"] = idx
    return df


def load_all_datasets() -> pd.DataFrame:
    return pd.concat([load_dataset(i) for i in DATASETS], ignore_index=True)


def load_datasets(indices: list[int]) -> pd.DataFrame:
    return pd.concat([load_dataset(i) for i in indices], ignore_index=True)
