from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _persuadability_prior(feature_frame: pd.DataFrame) -> np.ndarray:
    monthly = pd.to_numeric(feature_frame.get("monthly_charges", 0), errors="coerce").fillna(0)
    complaints = pd.to_numeric(feature_frame.get("complaint_count_90d", 0), errors="coerce").fillna(0)
    downtime = pd.to_numeric(feature_frame.get("network_downtime_minutes_30d", 0), errors="coerce").fillna(0)
    is_month_to_month = (feature_frame.get("contract", "") == "Month-to-month").astype(float)
    billing_issue = pd.to_numeric(feature_frame.get("billing_issue_flag", 0), errors="coerce").fillna(0)
    high_value = ((monthly >= 80) | (feature_frame.get("customer_segment", "") == "Premium")).astype(float)

    prior = (
        0.04
        + is_month_to_month * 0.05
        + billing_issue * 0.04
        + high_value * 0.03
        - (complaints >= 4).astype(float) * 0.03
        - (downtime >= 180).astype(float) * 0.03
    )
    return np.clip(np.asarray(prior, dtype=float), 0.01, 0.22)


def _simulate_retention_experiment(
    feature_frame: pd.DataFrame,
    y_churn: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create demo treatment/outcome labels until real retention experiments exist."""
    rng = np.random.default_rng(random_state)
    monthly = pd.to_numeric(feature_frame.get("monthly_charges", 0), errors="coerce").fillna(0)
    propensity = np.clip(0.18 + (monthly >= 80).astype(float) * 0.10 + rng.normal(0, 0.03, len(feature_frame)), 0.08, 0.45)
    treatment = rng.binomial(1, propensity)

    persuadability = _persuadability_prior(feature_frame)
    retained = (1 - np.asarray(y_churn, dtype=int)).copy()
    recovered = (treatment == 1) & (y_churn == 1) & (rng.random(len(feature_frame)) < persuadability)
    retained[recovered] = 1
    return treatment.astype(int), retained.astype(int)


def train_uplift_model(
    transformed_features: np.ndarray,
    feature_frame: pd.DataFrame,
    y_churn: np.ndarray,
    random_state: int = 42,
) -> dict[str, Any]:
    treatment, retained = _simulate_retention_experiment(feature_frame, y_churn, random_state)
    treated_mask = treatment == 1
    control_mask = treatment == 0

    artifact: dict[str, Any] = {
        "method": "two_model_synthetic_retention_baseline",
        "uses_synthetic_experiment": True,
        "treatment_rate": float(treatment.mean()),
        "treated_rows": int(treated_mask.sum()),
        "control_rows": int(control_mask.sum()),
    }

    if (
        treated_mask.sum() < 30
        or control_mask.sum() < 30
        or len(np.unique(retained[treated_mask])) < 2
        or len(np.unique(retained[control_mask])) < 2
    ):
        artifact["available"] = False
        return artifact

    treatment_model = RandomForestClassifier(
        n_estimators=120,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    control_model = RandomForestClassifier(
        n_estimators=120,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=random_state + 1,
        n_jobs=-1,
    )
    treatment_model.fit(transformed_features[treated_mask], retained[treated_mask])
    control_model.fit(transformed_features[control_mask], retained[control_mask])

    artifact.update(
        {
            "available": True,
            "treatment_model": treatment_model,
            "control_model": control_model,
        }
    )
    return artifact


def heuristic_persuadability_score(churn_probability: float, feature_frame: pd.DataFrame | None = None) -> float:
    probability = float(np.clip(churn_probability, 0.0, 1.0))
    score = 4.0 * probability * (1.0 - probability)

    if feature_frame is not None and not feature_frame.empty:
        row = feature_frame.iloc[0]
        if row.get("contract") == "Month-to-month":
            score += 0.08
        if float(row.get("billing_issue_flag", 0) or 0) == 1:
            score += 0.06
        if float(row.get("complaint_count_90d", 0) or 0) >= 4:
            score -= 0.08
        if float(row.get("network_downtime_minutes_30d", 0) or 0) >= 180:
            score -= 0.08

    return float(np.clip(score, 0.0, 1.0))


def score_persuadability(
    uplift_artifact: dict[str, Any] | None,
    transformed_features: np.ndarray,
    feature_frame: pd.DataFrame | None,
    churn_probability: float,
) -> float:
    if uplift_artifact and uplift_artifact.get("available"):
        treatment_model = uplift_artifact["treatment_model"]
        control_model = uplift_artifact["control_model"]
        p_retained_treated = treatment_model.predict_proba(transformed_features)[:, 1]
        p_retained_control = control_model.predict_proba(transformed_features)[:, 1]
        uplift = np.clip(p_retained_treated - p_retained_control, 0.0, 1.0)
        return float(uplift[0])
    return heuristic_persuadability_score(churn_probability, feature_frame)


def uplift_tier(score: float) -> str:
    if score < 0.15:
        return "Low"
    if score < 0.35:
        return "Medium"
    return "High"
