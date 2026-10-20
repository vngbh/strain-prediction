"""Executable leakage audits over the final-variant artifacts.

Five checks per direction: prefix causality, absent test labels, truncated
calibration tail, altered future observations, and one shared frozen source
model per temperature group. assert_audit_passes is the failure gate run by the
entry script. Passing means no leakage in these tested paths, not proof that no
untested path exists.
"""

import numpy as np
import pandas as pd

from .calibration import calibration_prefix, session_signals
from .features import build_features
from .spec import DATASETS, ELECTRODES, ROLLING_WINDOWS, TEMP_VAL

AUDIT_PASS_COLUMNS = (
    "causality_pass",
    "test_label_access_pass",
    "calibration_scope_pass",
    "future_observation_pass",
    "frozen_shared_source_model_pass",
)


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


def assert_audit_passes(audit):
    for column in AUDIT_PASS_COLUMNS:
        assert audit[column].all(), f"audit failed: {column}"
