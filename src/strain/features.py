"""TIME-CAUSALITY boundary: dynamic features computed within one session.

Features at sample t use samples <= t only, except centered_derivative, which
reads 10 future samples and is reserved for the labelled offline ablation and
for error-region labelling of the truth signal (reporting.py). Callers must
build features per session; never concatenate sessions before rolling.
Column order (signals, then derivatives, then signal x window x operation) is
part of the output contract. Audited by: causality_pass,
future_observation_pass (audits.py).
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


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


def build_features(signals, temperature, derivative_mode, include_temperature, windows, include_dynamics=True):
    x1, x2, xf = signals
    columns = [x1, x2, xf]
    if include_temperature:
        columns.append(temperature)
    if not include_dynamics:
        return np.column_stack(columns)

    derivative = causal_derivative if derivative_mode == "causal" else centered_derivative
    dx1, dx2, dxf = derivative(x1), derivative(x2), derivative(xf)
    columns.extend([dx1, dx2, dxf, (dxf > 0).astype(float)])
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
