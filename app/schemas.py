from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    customer_id: str = Field(default="CUST-DEMO-001")
    gender: str = Field(default="Female")
    senior_citizen: int = Field(default=0, ge=0, le=1)
    partner: str = Field(default="Yes")
    dependents: str = Field(default="No")
    tenure: int = Field(default=8, ge=0)
    phone_service: str = Field(default="Yes")
    multiple_lines: str = Field(default="No")
    internet_service: str = Field(default="Fiber optic")
    online_security: str = Field(default="No")
    online_backup: str = Field(default="No")
    device_protection: str = Field(default="No")
    tech_support: str = Field(default="No")
    streaming_tv: str = Field(default="Yes")
    streaming_movies: str = Field(default="Yes")
    contract: str = Field(default="Month-to-month")
    paperless_billing: str = Field(default="Yes")
    payment_method: str = Field(default="Electronic check")
    monthly_charges: float = Field(default=92.0, ge=0)
    total_charges: float = Field(default=736.0, ge=0)
    complaint_count_90d: int = Field(default=3, ge=0)
    support_ticket_count_90d: int = Field(default=4, ge=0)
    network_downtime_minutes_30d: float = Field(default=115.0, ge=0)
    late_payment_count_6m: int = Field(default=2, ge=0)
    billing_dispute_count_6m: int = Field(default=1, ge=0)
    avg_data_usage_gb_30d: float = Field(default=86.0, ge=0)
    plan_change_count_12m: int = Field(default=1, ge=0)
    region: str = Field(default="South")
    customer_segment: str = Field(default="Premium")
    last_interaction_sentiment: str = Field(default="negative")


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    decision_threshold: float
    churn_prediction: int
    risk_band: str
    top_reasons: list[str]
    persuadability_score: float
    uplift_tier: str
    recommended_action: str
    expected_business_impact: dict[str, Any]


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerInput]
