"""Frozen leave-one-temperature-out (LOTO) protocol.

For each held-out temperature group: fit one model on all sessions of the other
groups (each source session self-calibrated from its own 125 s prefix), freeze
it, then for each direction calibrate on one target session's prefix and predict
the opposite target session. Target labels never reach model fitting.
Ordering contract: DATASETS order, then sorted experiment ids; A = lower id.
Audited by: frozen_shared_source_model_pass (audits.py).
"""

import hashlib
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import session_signals
from .features import build_features, rolling
from .metrics import score, summarize
from .spec import DATASETS, FINAL_VARIANT, TEMP_VAL, WINDOW_VARIANTS


def make_hgb():
    return HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.03,
        l2_regularization=1.0,
        max_leaf_nodes=8,
        min_samples_leaf=40,
        random_state=0,
    )


def model_fingerprint(model):
    if model is None:
        return "not_applicable"
    return hashlib.sha256(pickle.dumps(model)).hexdigest()


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
                    variant.get("include_dynamics", True),
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
        variant.get("include_dynamics", True),
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
                if variant["name"] == FINAL_VARIANT:
                    final_artifacts[(held_out, direction)] = artifacts
    return pd.DataFrame(rows), final_artifacts


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
