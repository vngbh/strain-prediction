"""LABEL-ACCESS boundary: physical calibration and normalization z_i = (R_i - B_i) / S_i.

This is the only module that reads the `Strain` column of a target-temperature
session. It may read labels only inside calibration_prefix (the fixed first
125 s). Caller obligation: pass the calibration session explicitly as
`calibration_source`; when it is omitted, the session calibrates itself from its
own prefix, which is correct only for source-session preprocessing.
Audited by: test_label_access_pass, calibration_scope_pass (audits.py).
"""

import numpy as np
from sklearn.metrics import mean_squared_error

from .spec import CALIBRATION_PREFIX_SECONDS, ELECTRODES


def calibration_prefix(df):
    """Select calibration by a fixed time budget, never by full-series labels."""
    cutoff = float(df["Idx"].iloc[0]) + CALIBRATION_PREFIX_SECONDS
    return df[df["Idx"] <= cutoff].reset_index(drop=True)


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


def low_strain_offset_calibration(df):
    prefix = calibration_prefix(df)
    strain = prefix["Strain"].to_numpy()
    mask = strain <= strain.min() + 0.02
    return {electrode: float(prefix[electrode].to_numpy()[mask].mean()) for electrode in ELECTRODES}


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
