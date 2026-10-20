"""Direction 1, analysis 1: distribution of transfer error over calibration-window positions.

For every fold (calibration session C -> test session T) the frozen source HGB is fitted once.
The 125 s calibration window is then slid over C every STEP seconds; each window calibrates
(physics, four-level) and normalizes T, and both physics and the frozen HGB are scored on T.
Window descriptors use only labels INSIDE that calibration window (allowed calibration labels).
Results are reported per fold; windows overlap and share sessions, so they are not independent.

Usage: .venv/bin/python -I research/probes/probe_window_distribution.py <repo root> <out csv>
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1] + "/src")
import strain.calibration as cal  # noqa: E402
from strain.features import build_features  # noqa: E402
from strain.protocol import fit_source_model  # noqa: E402
from strain.spec import (  # noqa: E402
    CALIBRATION_PREFIX_SECONDS, DATASETS, ELECTRODES, ROLLING_WINDOWS, TEMP_VAL, VARIANTS, load_raw,
)

STEP = 10.0
raw = {d: load_raw(d) for d in DATASETS}
final = VARIANTS[0]
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


def descriptors(window):
    strain = window["Strain"].to_numpy()
    fit, points, _ = cal.four_point_calibration(window)
    weight = cal.fusion_weight_from_points(fit, points)
    z1, z2, zf = cal.normalize(window, fit, weight)
    low = strain <= strain.min() + 0.02
    rng = strain.max() - strain.min()
    levels = np.linspace(strain.min() + 0.05 * rng, strain.max() - 0.05 * rng, 3)
    band = np.zeros_like(strain, bool)
    for level in levels:
        band |= np.abs(strain - level) <= 0.03
    loading = np.gradient(strain) > 0
    return {
        "coverage": float(rng),
        "low_band_measured_mean": float(strain[low].mean()),
        "band_loading_fraction": float(loading[band].mean()),
        "self_fit_rmse": rmse(zf, strain),
        "fusion_weight": float(weight),
        "electrode_disagreement": float(np.sqrt(np.mean((z1 - z2) ** 2))),
    }


rows = []
for held in DATASETS:
    model = fit_source_model(raw, held, final)
    exps = sorted(raw[held].Experiment.unique())
    for direction, c, t in (("A_to_B", exps[0], exps[1]), ("B_to_A", exps[1], exps[0])):
        calib = raw[held][raw[held].Experiment == c].reset_index(drop=True)
        test = raw[held][raw[held].Experiment == t].reset_index(drop=True)
        truth = test["Strain"].to_numpy()
        t0, t_end = float(calib["Idx"].iloc[0]), float(calib["Idx"].iloc[-1])
        for start in np.arange(0.0, t_end - t0 - CALIBRATION_PREFIX_SECONDS + 1e-9, STEP):
            lo = t0 + start
            window = calib[(calib["Idx"] >= lo) & (calib["Idx"] <= lo + CALIBRATION_PREFIX_SECONDS)]
            window = window.reset_index(drop=True)
            try:
                signals, _ = cal.session_signals(test, "physics", window)
                desc = descriptors(window)
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
                continue
            if not np.all(np.isfinite(signals[2])):
                continue
            features = build_features(signals, test["Temperature"].to_numpy(), "causal", False, ROLLING_WINDOWS)
            physics, hgb = rmse(signals[2], truth), rmse(model.predict(features), truth)
            rows.append({"held_out_dataset": held, "temperature": TEMP_VAL[held], "direction": direction,
                         "window_start_s": start, "physics_rmse": physics, "hgb_rmse": hgb,
                         "pct_improvement": 100 * (physics - hgb) / physics, **desc})
frame = pd.DataFrame(rows)
frame.to_csv(sys.argv[2], index=False)

print("Per fold (window position distribution):")
g = frame.groupby(["temperature", "direction"])
summary = pd.DataFrame({
    "windows": g.size(),
    "hgb_rmse_median": g.hgb_rmse.median(),
    "hgb_rmse_p10": g.hgb_rmse.quantile(0.1),
    "hgb_rmse_p90": g.hgb_rmse.quantile(0.9),
    "physics_rmse_median": g.physics_rmse.median(),
    "share_windows_hgb_better": g.pct_improvement.apply(lambda s: (s > 0).mean()),
    "improvement_at_0s": g.apply(lambda d: d.loc[d.window_start_s == 0, "pct_improvement"].iloc[0]),
    "improvement_median": g.pct_improvement.median(),
})
print(summary.round(3).to_string())

print("\nSpearman correlation of window descriptors with HGB RMSE, per fold (sign consistency matters):")
cols = ["window_start_s", "coverage", "low_band_measured_mean", "band_loading_fraction",
        "self_fit_rmse", "fusion_weight", "electrode_disagreement", "physics_rmse"]
corr = g.apply(lambda d: d[cols].corrwith(d.hgb_rmse, method="spearman"))
print(corr.round(2).to_string())
print("\nfolds with positive / negative correlation:")
print(pd.DataFrame({"positive": (corr > 0.3).sum(), "negative": (corr < -0.3).sum()}).to_string())
