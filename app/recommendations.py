from __future__ import annotations

from src.business_cost import FALSE_NEGATIVE_COST


def risk_band(churn_probability: float) -> str:
    if churn_probability < 0.30:
        return "Low"
    if churn_probability < 0.65:
        return "Medium"
    return "High"


def recommend_action(churn_probability: float, top_reasons: list[str]) -> str:
    band = risk_band(churn_probability)
    reasons = " ".join(top_reasons).lower()

    if band == "Low":
        return "Normal engagement with routine value check-in."

    if "network downtime" in reasons or "support" in reasons or "complaints" in reasons:
        return (
            "Priority support outreach with service recovery message and network issue follow-up."
            if band == "Medium"
            else "Retention call plus priority network support and service credit review."
        )

    if "billing" in reasons or "late payment" in reasons or "charges" in reasons:
        return (
            "Personalized plan recommendation with billing support message."
            if band == "Medium"
            else "Retention call with targeted discount and billing support escalation."
        )

    if "month-to-month" in reasons:
        return (
            "Proactive annual-plan offer matched to current usage."
            if band == "Medium"
            else "Retention specialist call with contract upgrade incentive."
        )

    return (
        "Personalized plan recommendation or proactive support message."
        if band == "Medium"
        else "Retention call with discount, billing support, and priority support routing."
    )


def expected_business_impact(churn_probability: float, action: str) -> dict[str, float | str]:
    band = risk_band(churn_probability)
    retention_cost = {"Low": 25, "Medium": 150, "High": 400}[band]
    assumed_save_rate = {"Low": 0.04, "Medium": 0.12, "High": 0.20}[band]
    expected_loss_without_action = churn_probability * FALSE_NEGATIVE_COST
    expected_loss_reduction = expected_loss_without_action * assumed_save_rate
    net_expected_value = expected_loss_reduction - retention_cost
    return {
        "risk_band": band,
        "assumed_action": action,
        "estimated_retention_cost": float(retention_cost),
        "expected_loss_without_action": round(float(expected_loss_without_action), 2),
        "expected_loss_reduction": round(float(expected_loss_reduction), 2),
        "net_expected_value": round(float(net_expected_value), 2),
    }

