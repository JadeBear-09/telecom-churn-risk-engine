from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.business_cost import total_business_cost


def evaluate_thresholds(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """Score thresholds from model probability into business metrics."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        y_pred = (y_probability >= threshold).astype(int)
        costs = total_business_cost(y_true, y_pred)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                **costs,
            }
        )
    return rows


def select_optimal_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    min_recall: float = 0.65,
) -> tuple[float, list[dict[str, float | int]]]:
    """Minimize business cost while enforcing recall floor when possible."""
    rows = evaluate_thresholds(y_true, y_probability)
    candidates = [row for row in rows if float(row["recall"]) >= min_recall]
    if not candidates:
        candidates = rows
    best = min(candidates, key=lambda row: (int(row["total_cost"]), -float(row["recall"])))
    return float(best["threshold"]), rows

