from __future__ import annotations

import re

import numpy as np
import pandas as pd


COLUMN_RENAMES = {
    "customerID": "customer_id",
    "CustomerID": "customer_id",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}


def _snake_case(name: str) -> str:
    name = COLUMN_RENAMES.get(name, name)
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns, types, and target labels."""
    cleaned = df.copy()
    cleaned.columns = [_snake_case(col) for col in cleaned.columns]

    if "customer_id" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)

    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})

    numeric_cols = [
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
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if {"total_charges", "monthly_charges", "tenure"}.issubset(cleaned.columns):
        estimated_total = cleaned["monthly_charges"] * cleaned["tenure"].clip(lower=1)
        cleaned["total_charges"] = cleaned["total_charges"].fillna(estimated_total)
        cleaned["total_charges"] = cleaned["total_charges"].fillna(cleaned["total_charges"].median())

    if "senior_citizen" in cleaned.columns:
        cleaned["senior_citizen"] = cleaned["senior_citizen"].fillna(0).astype(int)

    if "churn" in cleaned.columns:
        churn_text = cleaned["churn"].astype(str).str.strip().str.lower()
        cleaned["churn"] = churn_text.map(
            {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
        ).astype(int)

    categorical_cols = cleaned.select_dtypes(include=["object"]).columns
    cleaned[categorical_cols] = cleaned[categorical_cols].fillna("Unknown")

    return cleaned
