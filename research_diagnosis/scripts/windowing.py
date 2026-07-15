"""
Corrected, provenance-tracked sequence builder (Phase D-fix item 2).

Rules enforced:
  - A window never crosses an Experiment boundary.
  - A window never crosses a dataset/temperature boundary
    (each dataset is windowed independently by construction: build per-source
    then concatenate, never window a pre-concatenated multi-source array).
  - Every window records its provenance: source dataset, Experiment,
    start index (within that Experiment's own row order) and target index.

Two task definitions kept explicitly distinct (Phase A finding #4 /
D-fix item 3):
  Task A - nowcasting:      target y[t] uses X up to and including t.
  Task B - forecasting:     target y[t] uses X only up to t-1 (window excludes t).
"""

import numpy as np
import pandas as pd


def build_windows_single_experiment(df_exp, feature_cols, target_col, window, task="nowcasting"):
    """df_exp: rows of ONE (dataset, Experiment) block, already sorted by Idx."""
    X = df_exp[feature_cols].values
    y = df_exp[target_col].values
    n = len(df_exp)
    Xs, ys, starts, targets = [], [], [], []
    lo = window if task == "nowcasting" else window  # window length identical; slice differs
    for t in range(window, n):
        if task == "nowcasting":
            seg = X[t - window + 1:t + 1]     # includes t
        elif task == "forecasting":
            seg = X[t - window:t]             # excludes t
        else:
            raise ValueError(task)
        Xs.append(seg)
        ys.append(y[t])
        starts.append(t - window + (1 if task == "nowcasting" else 0))
        targets.append(t)
    if not Xs:
        return np.empty((0, window, len(feature_cols))), np.empty((0,)), [], []
    return np.array(Xs), np.array(ys), starts, targets


def build_windows_provenance(datasets_dict, feature_cols, target_col, window, task="nowcasting"):
    """datasets_dict: {dataset_id: raw_df (with 'Experiment' column, sorted by Idx)}.
    Returns X (n, window, features), y (n,), provenance DataFrame with one row per window."""
    all_X, all_y, prov_rows = [], [], []
    for ds_id, df in datasets_dict.items():
        for exp_id, g in df.groupby("Experiment"):
            g = g.sort_values("Idx").reset_index(drop=True)
            Xs, ys, starts, targets = build_windows_single_experiment(g, feature_cols, target_col, window, task)
            if len(ys) == 0:
                continue
            all_X.append(Xs); all_y.append(ys)
            for s, t in zip(starts, targets):
                prov_rows.append({"dataset": ds_id, "experiment": exp_id,
                                   "start_idx": s, "target_idx": t, "task": task})
    if not all_X:
        return np.empty((0, window, len(feature_cols))), np.empty((0,)), pd.DataFrame(prov_rows)
    return np.concatenate(all_X), np.concatenate(all_y), pd.DataFrame(prov_rows)
