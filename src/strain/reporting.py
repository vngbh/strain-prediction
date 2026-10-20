"""Derived result tables and figures for the final variant.

Pure consumers of protocol outputs: they never fit models or read raw files.
Figure functions write only into the directory they are given.
error_regions labels turning points from the TRUTH signal with a centered
derivative; that is offline analysis of errors, not a model input.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .features import centered_derivative
from .metrics import summarize
from .spec import FINAL_VARIANT, TEMP_VAL, VARIANT_LABELS, VARIANTS


def final_rows(per_fold):
    return per_fold[per_fold["variant"] == FINAL_VARIANT]


def temperature_summary(per_fold):
    return final_rows(per_fold).groupby(
        ["held_out_dataset", "temperature", "temperature_position"], as_index=False
    ).agg(
        physics_rmse=("physics_rmse", "mean"),
        model_rmse=("rmse", "mean"),
        model_r2=("r2", "mean"),
        pct_improvement=("pct_improvement", "mean"),
    )


def position_summary(per_fold):
    return final_rows(per_fold).groupby("temperature_position", as_index=False).agg(
        physics_rmse=("physics_rmse", "mean"),
        model_rmse=("rmse", "mean"),
        model_r2=("r2", "mean"),
        pct_improvement=("pct_improvement", "mean"),
    )


def error_regions(final_artifacts):
    rows = []
    for (dataset, direction), artifact in final_artifacts.items():
        test = artifact["test"]
        y = test["Strain"].to_numpy()
        prediction = artifact["prediction"]
        physics = artifact["physics_prediction"]
        truth_loading = centered_derivative(y) > 0
        changes = np.flatnonzero(np.diff(truth_loading.astype(int)))
        distance = np.min(np.abs(np.arange(len(y))[:, None] - changes[None, :]), axis=1)
        near = distance <= 50
        thirds = np.array_split(np.arange(len(y)), 3)
        regions = {
            "near_turning": near,
            "stable_loading": truth_loading & ~near,
            "stable_unloading": ~truth_loading & ~near,
            "early_third": np.isin(np.arange(len(y)), thirds[0]),
            "middle_third": np.isin(np.arange(len(y)), thirds[1]),
            "late_third": np.isin(np.arange(len(y)), thirds[2]),
        }
        for region, mask in regions.items():
            model_rmse = np.sqrt(mean_squared_error(y[mask], prediction[mask]))
            physics_rmse = np.sqrt(mean_squared_error(y[mask], physics[mask]))
            rows.append(
                {
                    "dataset": dataset,
                    "temperature": TEMP_VAL[dataset],
                    "direction": direction,
                    "region": region,
                    "n": int(mask.sum()),
                    "model_rmse": model_rmse,
                    "physics_rmse": physics_rmse,
                    "pct_improvement": 100 * (physics_rmse - model_rmse) / physics_rmse,
                }
            )
    return pd.DataFrame(rows)


def plot_protocol_results(per_fold, final_artifacts, fig_dir):
    summary = summarize(per_fold).set_index("variant")
    order = [variant["name"] for variant in VARIANTS]
    labels = VARIANT_LABELS
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    values = summary.loc[order, "mean_pct_improvement"]
    colors = ["#167D73" if value > 0 else "#C44E52" for value in values]
    axes[0].barh(labels, values, color=colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Mean RMSE improvement vs matched physics (%)")
    axes[0].set_title("All ablations under frozen LOTO")

    colors_by_dataset = {1: "#3377AA", 2: "#CC6677", 3: "#228833", 4: "#AA4499"}
    for (dataset, direction), artifact in final_artifacts.items():
        axes[1].scatter(
            artifact["test"]["Strain"].to_numpy()[::20],
            artifact["prediction"][::20],
            s=5,
            alpha=0.25,
            color=colors_by_dataset[dataset],
            marker="o" if direction == "A_to_B" else "x",
            label=f"{TEMP_VAL[dataset]} degC" if direction == "A_to_B" else None,
        )
    axes[1].plot([0, 0.5], [0, 0.5], "--", color="black", linewidth=0.8)
    axes[1].set_xlabel("Measured strain (%)")
    axes[1].set_ylabel("Predicted strain (%)")
    axes[1].set_title("Final frozen model parity")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(fig_dir / "frozen_protocol_audit.png", dpi=160)
    plt.close(figure)

    rows = final_rows(per_fold)
    best = rows.loc[rows["r2"].idxmax()]
    worst = rows.loc[rows["r2"].idxmin()]
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    for axis, row, label in [(axes[0], best, "Best R2 fold"), (axes[1], worst, "Lowest R2 fold")]:
        artifact = final_artifacts[(int(row["held_out_dataset"]), row["direction"])]
        measured = artifact["test"]["Strain"].to_numpy()
        axis.plot(measured, color="#222222", linewidth=0.8, label="Measured")
        axis.plot(artifact["physics_prediction"], color="#D98C28", linewidth=0.7, label="Physics only")
        axis.plot(artifact["prediction"], color="#167D73", linewidth=0.8, label="Frozen causal HGB")
        axis.set_ylabel("Strain (%)")
        axis.set_title(
            f"{label}: {int(row['temperature'])} degC {row['direction']}, "
            f"R2={row['r2']:.3f}, RMSE={row['rmse']:.3f}%"
        )
        axis.legend(fontsize=8, ncol=3)
    axes[1].set_xlabel("Sample")
    figure.tight_layout()
    figure.savefig(fig_dir / "frozen_best_worst_predictions.png", dpi=160)
    plt.close(figure)
