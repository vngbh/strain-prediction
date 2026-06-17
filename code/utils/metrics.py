"""
Regression evaluation utilities used by all phase scripts.

Input:  y_true, y_pred — 1-D numpy arrays of actual and predicted Strain values
Output: dict with keys {mse, mae, rmse, r2} (all floats)
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate(y_true, y_pred) -> dict:
    """Return MSE, MAE, RMSE, and R² as a flat dict."""
    return {
        "mse":  float(mean_squared_error(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def print_metrics(metrics: dict, label: str = ""):
    """One-line console summary of a metrics dict from evaluate()."""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}MSE={metrics['mse']:.6f} | RMSE={metrics['rmse']:.6f} | MAE={metrics['mae']:.6f} | R²={metrics['r2']:.4f}")
