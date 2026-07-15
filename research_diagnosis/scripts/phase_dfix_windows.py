"""
Phase D-fix, part 2 — build corrected windows with provenance, and quantify how
many windows the OLD repo approach (code/utils/sequence.py make_sequences,
applied after a temporal 70/30 split that ignores Experiment boundaries) would
have silently corrupted by splicing across an Experiment boundary.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from windowing import build_windows_provenance

DATA_DIR = ROOT / "data"
OUT_TAB = Path(__file__).resolve().parents[1] / "tables"
DATASETS = [1, 2, 3, 4]
FEATURES = ["Electrode1_resistance", "Electrode2_resistance"]
WINDOW = 20


def load_raw(i):
    df = pd.read_csv(DATA_DIR / f"dataset{i}.csv")
    df["dataset_id"] = i
    return df


raw = {i: load_raw(i) for i in DATASETS}

X, y, prov = build_windows_provenance(raw, FEATURES, "Strain", WINDOW, task="nowcasting")
prov.to_csv(OUT_TAB / "window_provenance.csv", index=False)
print(f"Corrected windows built: {len(prov)} (task=nowcasting, window={WINDOW})")
print(prov.groupby(["dataset", "experiment"]).size())

# ── Quantify how many windows the OLD repo approach corrupted ────────────────
# Old approach: concat whole dataset (both experiments) in raw row order,
# take first 70% as "train", then slide a window of 20 across that whole
# 70% block irrespective of the internal experiment boundary.

corrupted_counts = []
for i in DATASETS:
    df = raw[i].sort_values(["Experiment", "Idx"]).reset_index(drop=True)
    n = len(df)
    split = int(n * 0.7)
    train_block = df.iloc[:split]
    boundary_positions = np.where(train_block["Experiment"].values[1:] != train_block["Experiment"].values[:-1])[0] + 1
    # a window [t-WINDOW+1, t] is corrupted if any boundary falls strictly inside it
    n_windows_old = max(0, len(train_block) - WINDOW)
    corrupted = 0
    for b in boundary_positions:
        lo, hi = max(0, b - WINDOW), b  # windows whose target t in [b, b+WINDOW-1] and start before b
        for t in range(b, min(b + WINDOW, len(train_block))):
            if t >= WINDOW:
                corrupted += 1
    corrupted_counts.append({"dataset": i, "n_train_rows_old": len(train_block),
                              "n_windows_old": n_windows_old,
                              "n_boundaries_in_train_region": len(boundary_positions),
                              "n_corrupted_windows_old": corrupted,
                              "pct_corrupted": round(100 * corrupted / n_windows_old, 4) if n_windows_old else None})

corrupted_df = pd.DataFrame(corrupted_counts)
corrupted_df.to_csv(OUT_TAB / "window_corruption_old_approach.csv", index=False)
print("\nOld-approach window corruption (repo make_sequences on temporal 70/30 split):")
print(corrupted_df.to_string(index=False))
print("\n[Phase D-fix windowing complete]")
