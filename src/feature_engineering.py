from __future__ import annotations

import numpy as np
import pandas as pd


RAW_NUMERIC_FEATURES = [
    "senior_citizen",
    "tenure",
    "monthly_charges",
    "total_charges",
    "complaint_count_90d",
    "support_ticket_count_90d",
    "network_downtime_minutes_30d",
    "late_payment_count_6m",
    "billing_dispute_count_6m",
    "avg_data_usage_gb_30d",
    "plan_change_count_12m",
]

RAW_CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "region",
    "customer_segment",
    "last_interaction_sentiment",
]

ENGINEERED_FEATURES = [
    "complaints_per_month",
    "support_calls_per_month",
    "charges_per_tenure",
    "is_month_to_month",
    "has_recent_downtime",
    "late_payment_ratio",
    "billing_issue_flag",
    "high_value_customer_flag",
    "high_usage_low_satisfaction_flag",
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "churn"


DEFAULT_VALUES = {
    "senior_citizen": 0,
    "tenure": 0,
    "monthly_charges": 0.0,
    "total_charges": 0.0,
    "complaint_count_90d": 0,
    "support_ticket_count_90d": 0,
    "network_downtime_minutes_30d": 0.0,
    "late_payment_count_6m": 0,
    "billing_dispute_count_6m": 0,
    "avg_data_usage_gb_30d": 0.0,
    "plan_change_count_12m": 0,
    "gender": "Unknown",
    "partner": "No",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "region": "Unknown",
    "customer_segment": "Growth",
    "last_interaction_sentiment": "neutral",
}


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for col, default in DEFAULT_VALUES.items():
        if col not in output.columns:
            output[col] = default
    return output


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-readable churn predictors."""
    engineered = ensure_feature_columns(df)

    for col in RAW_NUMERIC_FEATURES:
        engineered[col] = pd.to_numeric(engineered[col], errors="coerce").fillna(DEFAULT_VALUES[col])

    tenure_safe = engineered["tenure"].clip(lower=1)
    engineered["complaints_per_month"] = engineered["complaint_count_90d"] / 3.0
    engineered["support_calls_per_month"] = engineered["support_ticket_count_90d"] / 3.0
    engineered["charges_per_tenure"] = engineered["total_charges"] / tenure_safe
    engineered["is_month_to_month"] = (engineered["contract"] == "Month-to-month").astype(int)
    engineered["has_recent_downtime"] = (engineered["network_downtime_minutes_30d"] >= 60).astype(int)
    engineered["late_payment_ratio"] = engineered["late_payment_count_6m"] / 6.0
    engineered["billing_issue_flag"] = (
        (engineered["billing_dispute_count_6m"] > 0) | (engineered["late_payment_count_6m"] >= 2)
    ).astype(int)
    engineered["high_value_customer_flag"] = (
        (engineered["monthly_charges"] >= 80) | (engineered["customer_segment"] == "Premium")
    ).astype(int)
    engineered["high_usage_low_satisfaction_flag"] = (
        (engineered["avg_data_usage_gb_30d"] >= 75)
        & (engineered["last_interaction_sentiment"].str.lower() == "negative")
    ).astype(int)

    engineered[ENGINEERED_FEATURES] = engineered[ENGINEERED_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    for col in RAW_CATEGORICAL_FEATURES:
        engineered[col] = engineered[col].fillna(DEFAULT_VALUES[col]).astype(str)

    return engineered


def get_feature_columns() -> tuple[list[str], list[str]]:
    return NUMERIC_FEATURES, CATEGORICAL_FEATURES


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].astype(int)

