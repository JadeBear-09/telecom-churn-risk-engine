from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Return weighted average gap between predicted and observed churn rates."""
    y_true = np.asarray(y_true, dtype=float)
    y_probability = np.asarray(y_probability, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0

    for idx in range(n_bins):
        lower = bins[idx]
        upper = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_probability >= lower) & (y_probability <= upper)
        else:
            mask = (y_probability >= lower) & (y_probability < upper)
        if not np.any(mask):
            continue

        bin_weight = float(mask.mean())
        observed_rate = float(y_true[mask].mean())
        predicted_rate = float(y_probability[mask].mean())
        error += bin_weight * abs(observed_rate - predicted_rate)

    return float(error)


def calibration_summary(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    n_bins: int = 10,
) -> dict[str, object]:
    """Summarize probability calibration for model monitoring."""
    y_true = np.asarray(y_true, dtype=float)
    y_probability = np.asarray(y_probability, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []

    for idx in range(n_bins):
        lower = bins[idx]
        upper = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_probability >= lower) & (y_probability <= upper)
        else:
            mask = (y_probability >= lower) & (y_probability < upper)

        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": int(mask.sum()),
                "mean_predicted_probability": float(y_probability[mask].mean()) if np.any(mask) else 0.0,
                "observed_churn_rate": float(y_true[mask].mean()) if np.any(mask) else 0.0,
            }
        )

    return {
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "expected_calibration_error": expected_calibration_error(y_true, y_probability, n_bins=n_bins),
        "bins": rows,
    }
