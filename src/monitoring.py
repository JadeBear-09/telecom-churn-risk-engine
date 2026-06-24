from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd


def missing_value_rate(df: pd.DataFrame) -> dict[str, float]:
    return {col: float(rate) for col, rate in df.isna().mean().sort_values(ascending=False).items()}


def calculate_psi(expected: Iterable[float], actual: Iterable[float], buckets: int = 10) -> float:
    expected_arr = np.asarray(list(expected), dtype=float)
    actual_arr = np.asarray(list(actual), dtype=float)
    expected_arr = expected_arr[~np.isnan(expected_arr)]
    actual_arr = actual_arr[~np.isnan(actual_arr)]
    if len(expected_arr) == 0 or len(actual_arr) == 0:
        return 0.0

    breakpoints = np.percentile(expected_arr, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 2:
        return 0.0

    expected_counts, _ = np.histogram(expected_arr, bins=breakpoints)
    actual_counts, _ = np.histogram(actual_arr, bins=breakpoints)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def numeric_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list[str],
) -> dict[str, dict[str, float | str]]:
    drift = {}
    for feature in numeric_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        psi = calculate_psi(reference_df[feature], current_df[feature])
        drift[feature] = {
            "psi": psi,
            "status": "alert" if psi >= 0.25 else "watch" if psi >= 0.10 else "stable",
        }
    return drift


def prediction_distribution(probabilities: Iterable[float], bins: int = 10) -> dict[str, float]:
    probs = np.asarray(list(probabilities), dtype=float)
    counts, edges = np.histogram(probs, bins=np.linspace(0, 1, bins + 1))
    total = max(int(counts.sum()), 1)
    return {
        f"{edges[idx]:.1f}-{edges[idx + 1]:.1f}": float(count / total)
        for idx, count in enumerate(counts)
    }


def simulate_api_metrics(request_count: int = 1000, error_rate: float = 0.01) -> dict[str, float]:
    rng = np.random.default_rng(42)
    latency = np.clip(rng.normal(85, 25, request_count), 15, 350)
    return {
        "p50_latency_ms": float(np.percentile(latency, 50)),
        "p95_latency_ms": float(np.percentile(latency, 95)),
        "error_rate": float(error_rate),
    }


def monitoring_snapshot(
    current_df: pd.DataFrame,
    probabilities: Iterable[float],
    started_at: float | None = None,
    request_count: int = 1000,
    error_count: int = 10,
) -> dict[str, object]:
    probs = np.asarray(list(probabilities), dtype=float)
    latency_ms = (time.time() - started_at) * 1000 if started_at else None
    error_rate = error_count / max(request_count, 1)
    api_metrics = simulate_api_metrics(request_count=request_count, error_rate=error_rate)
    if latency_ms is not None:
        api_metrics["latest_latency_ms"] = float(latency_ms)

    return {
        "missing_value_rate": missing_value_rate(current_df),
        "prediction_distribution": prediction_distribution(probs),
        "average_churn_probability": float(np.mean(probs)) if len(probs) else 0.0,
        "high_risk_customer_pct": float(np.mean(probs >= 0.65)) if len(probs) else 0.0,
        "api_metrics": api_metrics,
    }

