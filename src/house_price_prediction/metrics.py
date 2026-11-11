"""Evaluation metrics reported on the original dollar scale."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must have the same shape.")
    if actual.size == 0:
        raise ValueError("Cannot evaluate an empty set of predictions.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted values must be finite.")
    if (actual <= 0).any():
        raise ValueError("Actual prices must be greater than zero to calculate MAPE.")

    return {
        "r2": float(r2_score(actual, predicted)),
        "mae_usd": float(mean_absolute_error(actual, predicted)),
        "rmse_usd": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_pct": float(100 * mean_absolute_percentage_error(actual, predicted)),
    }
