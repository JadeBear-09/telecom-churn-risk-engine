from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix


FALSE_NEGATIVE_COST = 5000
FALSE_POSITIVE_COST = 500


@dataclass(frozen=True)
class BusinessCost:
    false_negative_cost: int = FALSE_NEGATIVE_COST
    false_positive_cost: int = FALSE_POSITIVE_COST


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def total_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_negative_cost: int = FALSE_NEGATIVE_COST,
    false_positive_cost: int = FALSE_POSITIVE_COST,
) -> dict[str, int]:
    counts = confusion_counts(y_true, y_pred)
    fn_cost = counts["fn"] * false_negative_cost
    fp_cost = counts["fp"] * false_positive_cost
    return {
        **counts,
        "false_negative_cost": int(fn_cost),
        "false_positive_cost": int(fp_cost),
        "total_cost": int(fn_cost + fp_cost),
    }

