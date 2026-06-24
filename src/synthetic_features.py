from __future__ import annotations

import numpy as np
import pandas as pd


SYNTHETIC_FEATURES = [
    "complaint_count_90d",
    "support_ticket_count_90d",
    "network_downtime_minutes_30d",
    "late_payment_count_6m",
    "billing_dispute_count_6m",
    "avg_data_usage_gb_30d",
    "plan_change_count_12m",
    "region",
    "customer_segment",
    "last_interaction_sentiment",
]


def _risk_proxy(df: pd.DataFrame) -> np.ndarray:
    monthly = pd.to_numeric(df.get("monthly_charges", 0), errors="coerce").fillna(0)
    tenure = pd.to_numeric(df.get("tenure", 0), errors="coerce").fillna(0)
    risk = (
        0.18
        + (df.get("contract", "") == "Month-to-month").astype(float) * 0.24
        + (df.get("tech_support", "") == "No").astype(float) * 0.14
        + (df.get("payment_method", "") == "Electronic check").astype(float) * 0.12
        + (monthly > monthly.quantile(0.7)).astype(float) * 0.12
        + (tenure < 12).astype(float) * 0.10
    )
    return np.clip(np.asarray(risk, dtype=float), 0.05, 0.95)


def add_synthetic_features(
    df: pd.DataFrame,
    random_state: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Add realistic telecom operations features when source dataset lacks them."""
    enriched = df.copy()
    rng = np.random.default_rng(random_state)
    n_rows = len(enriched)
    risk = _risk_proxy(enriched)

    generated: dict[str, np.ndarray] = {
        "complaint_count_90d": rng.poisson(np.clip(0.25 + risk * 3.0, 0.1, 5.0)).astype(int),
        "support_ticket_count_90d": rng.poisson(np.clip(0.35 + risk * 3.4, 0.1, 5.5)).astype(int),
        "network_downtime_minutes_30d": np.clip(rng.gamma(shape=1.8 + risk * 2.0, scale=18.0), 0, 420).round(1),
        "late_payment_count_6m": rng.poisson(np.clip(0.15 + risk * 1.8, 0.05, 3.0)).astype(int),
        "billing_dispute_count_6m": rng.binomial(3, np.clip(0.05 + risk * 0.22, 0.02, 0.55)).astype(int),
        "avg_data_usage_gb_30d": np.clip(
            rng.normal(38 + risk * 28 + (enriched.get("internet_service", "") == "Fiber optic") * 22, 12),
            0,
            180,
        ).round(1),
        "plan_change_count_12m": rng.poisson(np.clip(0.12 + risk * 1.1, 0.02, 2.0)).astype(int),
        "region": rng.choice(["Northeast", "South", "Midwest", "West"], size=n_rows, p=[0.23, 0.35, 0.2, 0.22]),
    }

    monthly = pd.to_numeric(enriched.get("monthly_charges", 0), errors="coerce").fillna(0)
    tenure = pd.to_numeric(enriched.get("tenure", 0), errors="coerce").fillna(0)
    generated["customer_segment"] = np.select(
        [monthly >= 85, (monthly <= 45) & (tenure >= 18)],
        ["Premium", "Value"],
        default="Growth",
    )

    sentiment = []
    for row_risk in risk:
        sentiment.append(
            rng.choice(
                ["positive", "neutral", "negative"],
                p=[
                    max(0.10, 0.58 - row_risk * 0.38),
                    0.30,
                    min(0.60, 0.12 + row_risk * 0.38),
                ],
            )
        )
    generated["last_interaction_sentiment"] = np.asarray(sentiment)

    for col, values in generated.items():
        if overwrite or col not in enriched.columns:
            enriched[col] = values
        else:
            enriched[col] = enriched[col].fillna(pd.Series(values, index=enriched.index))

    return enriched

