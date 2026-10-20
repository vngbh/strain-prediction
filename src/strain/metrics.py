"""Per-fold scores and per-variant summaries.

Two aggregate improvement summaries answer different questions and must not be
swapped in prose: mean_pct_improvement (mean of per-direction improvements, the
primary metric) and ratio_of_mean_rmse_reduction (1 - mean RMSE / mean physics RMSE).
summarize keeps VARIANTS order (groupby sort=False).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .spec import FULL_SCALE


def score(y_true, prediction):
    rmse = float(np.sqrt(mean_squared_error(y_true, prediction)))
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, prediction)),
        "pct_fs": 100 * rmse / FULL_SCALE,
    }


def summarize(frame):
    rows = []
    for variant, group in frame.groupby("variant", sort=False):
        rows.append(
            {
                "variant": variant,
                "mean_physics_rmse": group["physics_rmse"].mean(),
                "mean_rmse": group["rmse"].mean(),
                "mean_r2": group["r2"].mean(),
                "mean_pct_improvement": group["pct_improvement"].mean(),
                "ratio_of_mean_rmse_reduction": 100
                * (group["physics_rmse"].mean() - group["rmse"].mean())
                / group["physics_rmse"].mean(),
                "worst_fold_pct_improvement": group["pct_improvement"].min(),
                "temperature_groups_improved_both_directions": sum(
                    (part["pct_improvement"] > 0).all() for _, part in group.groupby("held_out_dataset")
                ),
                "directions_improved": int((group["pct_improvement"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)
