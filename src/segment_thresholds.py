from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.threshold_tuning import evaluate_thresholds, select_optimal_threshold


DEFAULT_SEGMENT = "Unknown"


def _threshold_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    rows = evaluate_thresholds(y_true, y_probability, thresholds=np.asarray([threshold], dtype=float))
    return rows[0]


def select_segment_thresholds(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    segments: pd.Series,
    default_threshold: float,
    min_segment_size: int = 40,
    min_recall: float = 0.65,
) -> dict[str, Any]:
    """Choose cost-aware thresholds per customer segment when support is sufficient."""
    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)
    segment_values = segments.fillna(DEFAULT_SEGMENT).astype(str).to_numpy()

    thresholds: dict[str, Any] = {
        "__default__": {
            "threshold": float(default_threshold),
            "support": int(len(y_true)),
            "source": "global",
            **_threshold_metrics(y_true, y_probability, default_threshold),
        }
    }

    for segment in sorted(np.unique(segment_values)):
        mask = segment_values == segment
        support = int(mask.sum())
        if support < min_segment_size or len(np.unique(y_true[mask])) < 2:
            thresholds[segment] = {
                "threshold": float(default_threshold),
                "support": support,
                "source": "global_fallback",
            }
            continue

        segment_threshold, _ = select_optimal_threshold(
            y_true[mask],
            y_probability[mask],
            min_recall=min_recall,
        )
        thresholds[segment] = {
            "threshold": float(segment_threshold),
            "support": support,
            "source": "segment",
            **_threshold_metrics(y_true[mask], y_probability[mask], segment_threshold),
        }

    return thresholds


def threshold_for_segment(
    segment: object,
    segment_thresholds: dict[str, Any] | None,
    default_threshold: float = 0.5,
) -> float:
    if not segment_thresholds:
        return float(default_threshold)
    segment_key = str(segment) if segment is not None else DEFAULT_SEGMENT
    row = segment_thresholds.get(segment_key) or segment_thresholds.get("__default__", {})
    return float(row.get("threshold", default_threshold))


def thresholds_for_segments(
    segments: pd.Series,
    segment_thresholds: dict[str, Any] | None,
    default_threshold: float = 0.5,
) -> np.ndarray:
    return np.asarray(
        [threshold_for_segment(segment, segment_thresholds, default_threshold) for segment in segments],
        dtype=float,
    )
