"""Audit every major design choice under the final frozen LOTO protocol.

All analyses use the existing four-temperature corpus. They are retrospective:
the model family and fixed hyperparameters were developed after inspecting this
corpus, so these checks strengthen internal validity but are not an untouched
external confirmation.
"""

import hashlib
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "to_know_how" / "files"
OUT_TAB = ROOT / "results" / "tables"
OUT_FIG = ROOT / "results" / "figures"
DATASETS = (1, 2, 3, 4)
TEMP_VAL = {1: 33, 2: 43, 3: 52, 4: 62}
ELECTRODES = ("Electrode1_resistance", "Electrode2_resistance")
FULL_SCALE = 0.5  # percent strain
ROLLING_WINDOWS = (21, 51, 101, 201)
CALIBRATION_PREFIX_SECONDS = 125.0


VARIANTS = [
    {
        "name": "final_physics_causal_no_T_direct_HGB",
        "representation": "physics",
        "derivative": "causal",
        "include_temperature": False,
        "target": "direct",
        "model": "hgb",
        "windows": ROLLING_WINDOWS,
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


def rolling(signal, window, operation):
    values = getattr(pd.Series(signal).rolling(window, min_periods=1), operation)().to_numpy()
    return np.nan_to_num(values)


def centered_derivative(signal):
    """Offline reference derivative requiring 10 future samples."""
    return np.gradient(savgol_filter(signal, 21, 2))


def causal_derivative(signal):
    """Online-compatible derivative of a trailing 21-sample mean."""
    smooth = rolling(signal, 21, "mean")
    derivative = np.empty_like(smooth, dtype=float)
    derivative[0] = 0.0
    derivative[1:] = np.diff(smooth)
    return derivative


def make_hgb():
    return HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.03,
        l2_regularization=1.0,
        max_leaf_nodes=8,
        min_samples_leaf=40,
        random_state=0,
    )


def four_point_calibration(df):
    """Estimate B/S from zero and three strain-level windows."""
    strain = df["Strain"].to_numpy()
    strain_range = strain.max() - strain.min()
    levels = np.linspace(strain.min() + 0.05 * strain_range, strain.max() - 0.05 * strain_range, 3)
    points, used_indices = [], []
    for point_index, level in enumerate([0.0, *levels]):
        mask = strain <= strain.min() + 0.02 if point_index == 0 else np.abs(strain - level) <= 0.03
        used_indices.extend(np.flatnonzero(mask).tolist())
        points.append((float(level), *(df[electrode].to_numpy()[mask].mean() for electrode in ELECTRODES)))
    points = np.asarray(points)
    fit = {}
    for electrode_index, electrode in enumerate(ELECTRODES, start=1):
        slope, baseline = np.polyfit(points[:, 0], points[:, electrode_index], 1)
        fit[electrode] = (slope, baseline)
    info = {
        "calibration_mode": "fixed_125s_four_level_windows",
        "calibration_rows_used": int(len(np.unique(used_indices))),
        "calibration_duration_seconds": float(df["Idx"].iloc[-1] - df["Idx"].iloc[0]),
    }
    return fit, points, info


def fusion_weight_from_points(fit, points):
    strain, resistance1, resistance2 = points.T
    slope1, baseline1 = fit[ELECTRODES[0]]
    slope2, baseline2 = fit[ELECTRODES[1]]
    z1 = (resistance1 - baseline1) / slope1
    z2 = (resistance2 - baseline2) / slope2
    best_weight, best_rmse = 0.5, np.inf
    for weight1 in np.linspace(0, 1, 21):
        prediction = weight1 * z1 + (1 - weight1) * z2
        rmse = np.sqrt(mean_squared_error(strain, prediction))
        if rmse < best_rmse:
            best_weight, best_rmse = weight1, rmse
    return best_weight


def normalize(df, fit, weight1):
    slope1, baseline1 = fit[ELECTRODES[0]]
    slope2, baseline2 = fit[ELECTRODES[1]]
    z1 = (df[ELECTRODES[0]].to_numpy() - baseline1) / slope1
    z2 = (df[ELECTRODES[1]].to_numpy() - baseline2) / slope2
    return z1, z2, weight1 * z1 + (1 - weight1) * z2


def score(y_true, prediction):
    rmse = float(np.sqrt(mean_squared_error(y_true, prediction)))
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, prediction)),
        "pct_fs": 100 * rmse / FULL_SCALE,
    }


def build_features(signals, temperature, derivative_mode, include_temperature, windows):
    derivative = causal_derivative if derivative_mode == "causal" else centered_derivative
    x1, x2, xf = signals
    dx1, dx2, dxf = derivative(x1), derivative(x2), derivative(xf)
    columns = [x1, x2, xf, dx1, dx2, dxf, (dxf > 0).astype(float)]
    if include_temperature:
        columns.append(temperature)
    for signal in [x1, x2, xf, dxf]:
        for window in windows:
            columns.extend(
                [
                    rolling(signal, window, "mean"),
                    rolling(signal, window, "std"),
                    rolling(signal, window, "min"),
                    rolling(signal, window, "max"),
                ]
            )
    return np.column_stack(columns)


def low_strain_offset_calibration(df):
    prefix = calibration_prefix(df)
    strain = prefix["Strain"].to_numpy()
    mask = strain <= strain.min() + 0.02
    return {electrode: float(prefix[electrode].to_numpy()[mask].mean()) for electrode in ELECTRODES}


def calibration_prefix(df):
    """Select calibration by a fixed time budget, never by full-series labels."""
    cutoff = float(df["Idx"].iloc[0]) + CALIBRATION_PREFIX_SECONDS
    return df[df["Idx"] <= cutoff].reset_index(drop=True)


def session_signals(df, representation, calibration_source=None):
    """Transform a session using calibration from calibration_source."""
    source = df if calibration_source is None else calibration_source
    if representation == "physics":
        prefix = calibration_prefix(source)
        fit, points, info = four_point_calibration(prefix)
        weight = fusion_weight_from_points(fit, points)
        return normalize(df, fit, weight), info
    if representation == "zero_offset":
        baselines = low_strain_offset_calibration(source)
        x1 = df[ELECTRODES[0]].to_numpy() - baselines[ELECTRODES[0]]
        x2 = df[ELECTRODES[1]].to_numpy() - baselines[ELECTRODES[1]]
        info = {
            "calibration_mode": "fixed_125s_low_strain_window",
            "calibration_rows_used": int(
                (calibration_prefix(source)["Strain"] <= calibration_prefix(source)["Strain"].min() + 0.02).sum()
            ),
            "calibration_duration_seconds": float(
                calibration_prefix(source)["Idx"].iloc[-1] - calibration_prefix(source)["Idx"].iloc[0]
            ),
        }
        return (x1, x2, 0.5 * (x1 + x2)), info
    x1, x2 = (df[electrode].to_numpy() for electrode in ELECTRODES)
    return (x1, x2, 0.5 * (x1 + x2)), {
        "calibration_mode": "none",
        "calibration_rows_used": 0,
        "calibration_duration_seconds": 0.0,
    }


def fit_source_model(raw, held_out, variant):
    x_train, y_train, physics_train = [], [], []
    for dataset in DATASETS:
        if dataset == held_out:
            continue
        for experiment in sorted(raw[dataset].Experiment.unique()):
            session = raw[dataset][raw[dataset].Experiment == experiment].reset_index(drop=True)
            signals, _ = session_signals(session, variant["representation"])
            x_train.append(
                build_features(
                    signals,
                    session["Temperature"].to_numpy(),
                    variant["derivative"],
                    variant["include_temperature"],
                    variant["windows"],
                )
            )
            strain = session["Strain"].to_numpy()
            y_train.append(strain)
            if variant["representation"] == "physics":
                physics_train.append(signals[2])
            else:
                physics_train.append(np.zeros_like(strain))

    if variant["model"] == "smoothing":
        return None
    model = make_hgb() if variant["model"] == "hgb" else make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    target = np.concatenate(y_train)
    if variant["target"] == "residual":
        target = target - np.concatenate(physics_train)
    model.fit(np.vstack(x_train), target)
    return model


def model_fingerprint(model):
    if model is None:
        return "not_applicable"
    return hashlib.sha256(pickle.dumps(model)).hexdigest()


def evaluate_fold(raw, held_out, direction, calibration_experiment, test_experiment, variant, model):
    calibration = raw[held_out][raw[held_out].Experiment == calibration_experiment].reset_index(drop=True)
    test = raw[held_out][raw[held_out].Experiment == test_experiment].reset_index(drop=True)

    physics_signals, physics_info = session_signals(test, "physics", calibration)
    physics_prediction = physics_signals[2]
    physics_score = score(test["Strain"].to_numpy(), physics_prediction)

    signals, calibration_info = session_signals(test, variant["representation"], calibration)
    features = build_features(
        signals,
        test["Temperature"].to_numpy(),
        variant["derivative"],
        variant["include_temperature"],
        variant["windows"],
    )
    fingerprint_before = model_fingerprint(model)
    if variant["model"] == "smoothing":
        prediction = rolling(signals[2], 21, "mean")
    else:
        prediction = model.predict(features)
        if variant["target"] == "residual":
            prediction = physics_prediction + prediction
    fingerprint_after = model_fingerprint(model)

    result = score(test["Strain"].to_numpy(), prediction)
    row = {
        "variant": variant["name"],
        "held_out_dataset": held_out,
        "temperature": TEMP_VAL[held_out],
        "temperature_position": "edge" if held_out in (1, 4) else "interior",
        "direction": direction,
        "target_temperature_hgb_training_rows": 0,
        "calibration_mode": calibration_info["calibration_mode"],
        "calibration_rows_used": calibration_info["calibration_rows_used"],
        "calibration_duration_seconds": calibration_info["calibration_duration_seconds"],
        "source_model_fingerprint": fingerprint_before,
        "source_model_unchanged": fingerprint_before == fingerprint_after,
        "physics_rmse": physics_score["rmse"],
        "physics_r2": physics_score["r2"],
        **result,
        "pct_improvement": 100 * (physics_score["rmse"] - result["rmse"]) / physics_score["rmse"],
    }
    artifacts = {
        "calibration": calibration,
        "test": test,
        "model": model,
        "signals": signals,
        "physics_signals": physics_signals,
        "features": features,
        "prediction": prediction,
        "physics_prediction": physics_prediction,
        "physics_info": physics_info,
        "model_fingerprint_before": fingerprint_before,
        "model_fingerprint_after": fingerprint_after,
    }
    return row, artifacts


def run_variants(raw, variants):
    rows, final_artifacts = [], {}
    for variant in variants:
        for held_out in DATASETS:
            model = fit_source_model(raw, held_out, variant)
            experiments = sorted(raw[held_out].Experiment.unique())
            directions = [
                ("A_to_B", experiments[0], experiments[1]),
                ("B_to_A", experiments[1], experiments[0]),
            ]
            for direction, calibration_experiment, test_experiment in directions:
                row, artifacts = evaluate_fold(
                    raw,
                    held_out,
                    direction,
                    calibration_experiment,
                    test_experiment,
                    variant,
                    model,
                )
                rows.append(row)
                if variant["name"] == "final_physics_causal_no_T_direct_HGB":
                    final_artifacts[(held_out, direction)] = artifacts
    return pd.DataFrame(rows), final_artifacts


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


def causality_and_label_audit(final_artifacts):
    rows = []
    shared_model_pass = {}
    for dataset in DATASETS:
        artifacts = [artifact for (key_dataset, _), artifact in final_artifacts.items() if key_dataset == dataset]
        shared_model_pass[dataset] = (
            len({id(artifact["model"]) for artifact in artifacts}) == 1
            and len({artifact["model_fingerprint_before"] for artifact in artifacts}) == 1
        )

    for (dataset, direction), artifact in final_artifacts.items():
        test = artifact["test"]
        full_features = artifact["features"]
        model = artifact["model"]
        feature_differences, prediction_differences = [], []
        for stop in [250, len(test) // 3, 2 * len(test) // 3, len(test) - 1]:
            test_prefix = test.iloc[: stop + 1].reset_index(drop=True)
            prefix_signals, _ = session_signals(test_prefix, "physics", artifact["calibration"])
            prefix_features = build_features(
                prefix_signals,
                test_prefix["Temperature"].to_numpy(),
                "causal",
                False,
                ROLLING_WINDOWS,
            )
            feature_differences.append(np.max(np.abs(prefix_features[-1] - full_features[stop])))
            prediction_differences.append(
                abs(float(model.predict(prefix_features[-1:])[0]) - float(model.predict(full_features[stop : stop + 1])[0]))
            )

        # Inference must run when the forbidden test label column is absent.
        test_without_labels = test.drop(columns=["Strain"])
        modified_signals, _ = session_signals(test_without_labels, "physics", artifact["calibration"])
        rebuilt_features = build_features(
            modified_signals,
            test_without_labels["Temperature"].to_numpy(),
            "causal",
            False,
            ROLLING_WINDOWS,
        )
        label_prediction_difference = np.max(
            np.abs(model.predict(rebuilt_features) - artifact["prediction"])
        )

        # Removing the forbidden calibration tail entirely must change nothing.
        truncated_calibration = calibration_prefix(artifact["calibration"])
        modified_signals, _ = session_signals(test, "physics", truncated_calibration)
        modified_features = build_features(
            modified_signals,
            test["Temperature"].to_numpy(),
            "causal",
            False,
            ROLLING_WINDOWS,
        )
        calibration_tail_difference = np.max(
            np.abs(model.predict(modified_features) - artifact["prediction"])
        )

        # Strongly alter all observations after t. Predictions through t must
        # remain unchanged when every feature is truly causal.
        future_feature_differences, future_prediction_differences = [], []
        for stop in [250, len(test) // 3, 2 * len(test) // 3]:
            altered = test.copy()
            future = np.arange(stop + 1, len(altered))
            for electrode in ELECTRODES:
                scale = max(float(altered[electrode].std()), 1.0)
                altered.loc[future, electrode] += scale * np.linspace(10.0, 20.0, len(future))
            altered.loc[future, "Temperature"] += 100.0
            altered_signals, _ = session_signals(altered, "physics", artifact["calibration"])
            altered_features = build_features(
                altered_signals,
                altered["Temperature"].to_numpy(),
                "causal",
                False,
                ROLLING_WINDOWS,
            )
            future_feature_differences.append(
                np.max(np.abs(altered_features[: stop + 1] - full_features[: stop + 1]))
            )
            future_prediction_differences.append(
                np.max(
                    np.abs(
                        model.predict(altered_features[: stop + 1])
                        - artifact["prediction"][: stop + 1]
                    )
                )
            )

        frozen_model_pass = (
            artifact["model_fingerprint_before"] == artifact["model_fingerprint_after"]
            and shared_model_pass[dataset]
        )
        rows.append(
            {
                "dataset": dataset,
                "temperature": TEMP_VAL[dataset],
                "direction": direction,
                "max_prefix_feature_difference": max(feature_differences),
                "max_prefix_prediction_difference": max(prediction_differences),
                "max_test_label_prediction_difference": label_prediction_difference,
                "max_post_calibration_label_prediction_difference": calibration_tail_difference,
                "max_future_observation_feature_difference": max(future_feature_differences),
                "max_future_observation_prediction_difference": max(future_prediction_differences),
                "causality_pass": max(feature_differences) < 1e-12 and max(prediction_differences) < 1e-12,
                "test_label_access_pass": label_prediction_difference < 1e-12,
                "calibration_scope_pass": calibration_tail_difference < 1e-12,
                "future_observation_pass": max(future_feature_differences) < 1e-12
                and max(future_prediction_differences) < 1e-12,
                "frozen_shared_source_model_pass": frozen_model_pass,
            }
        )
    return pd.DataFrame(rows)


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


def window_sensitivity(raw):
    variants = []
    for name, windows in WINDOW_VARIANTS:
        variants.append(
            {
                "name": name,
                "representation": "physics",
                "derivative": "causal",
                "include_temperature": False,
                "target": "direct",
                "model": "hgb",
                "windows": windows,
            }
        )
    frame, _ = run_variants(raw, variants)
    return frame, summarize(frame)


def plot_protocol_results(per_fold, final_artifacts):
    summary = summarize(per_fold).set_index("variant")
    order = [variant["name"] for variant in VARIANTS]
    labels = [
        "Final normalized HGB",
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
    figure.savefig(OUT_FIG / "frozen_protocol_audit.png", dpi=160)
    plt.close(figure)

    final_rows = per_fold[per_fold["variant"] == "final_physics_causal_no_T_direct_HGB"]
    best = final_rows.loc[final_rows["r2"].idxmax()]
    worst = final_rows.loc[final_rows["r2"].idxmin()]
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
    figure.savefig(OUT_FIG / "frozen_best_worst_predictions.png", dpi=160)
    plt.close(figure)


def main():
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    raw = {dataset: load_raw(dataset) for dataset in DATASETS}

    per_fold, final_artifacts = run_variants(raw, VARIANTS)
    summary = summarize(per_fold)
    per_fold.to_csv(OUT_TAB / "frozen_protocol_ablations_per_fold.csv", index=False)
    summary.to_csv(OUT_TAB / "frozen_protocol_ablations_summary.csv", index=False)

    final_rows = per_fold[per_fold["variant"] == "final_physics_causal_no_T_direct_HGB"]
    temperature_summary = final_rows.groupby(
        ["held_out_dataset", "temperature", "temperature_position"], as_index=False
    ).agg(
        physics_rmse=("physics_rmse", "mean"),
        model_rmse=("rmse", "mean"),
        model_r2=("r2", "mean"),
        pct_improvement=("pct_improvement", "mean"),
    )
    temperature_summary.to_csv(OUT_TAB / "frozen_protocol_by_temperature.csv", index=False)

    position_summary = final_rows.groupby("temperature_position", as_index=False).agg(
        physics_rmse=("physics_rmse", "mean"),
        model_rmse=("rmse", "mean"),
        model_r2=("r2", "mean"),
        pct_improvement=("pct_improvement", "mean"),
    )
    position_summary.to_csv(OUT_TAB / "frozen_protocol_edge_vs_interior.csv", index=False)

    audit = causality_and_label_audit(final_artifacts)
    audit.to_csv(OUT_TAB / "causality_label_access_audit.csv", index=False)
    assert audit["causality_pass"].all()
    assert audit["test_label_access_pass"].all()
    assert audit["calibration_scope_pass"].all()
    assert audit["future_observation_pass"].all()
    assert audit["frozen_shared_source_model_pass"].all()

    regions = error_regions(final_artifacts)
    regions.to_csv(OUT_TAB / "frozen_protocol_error_regions.csv", index=False)

    window_per_fold, window_summary = window_sensitivity(raw)
    window_per_fold.to_csv(OUT_TAB / "frozen_window_sensitivity_per_fold.csv", index=False)
    window_summary.to_csv(OUT_TAB / "frozen_window_sensitivity_summary.csv", index=False)

    plot_protocol_results(per_fold, final_artifacts)

    print("\nFrozen-protocol ablations:")
    print(summary.round(4).to_string(index=False))
    print("\nCausality and label-access audit:")
    print(audit.to_string(index=False))
    print("\nWindow sensitivity:")
    print(window_summary.round(4).to_string(index=False))
    print("\nEdge versus interior held-out temperatures:")
    print(position_summary.round(4).to_string(index=False))
    print("\nError regions, averaged across directions:")
    print(regions.groupby("region")[["model_rmse", "physics_rmse", "pct_improvement"]].mean().round(4))


if __name__ == "__main__":
    main()
