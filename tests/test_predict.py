from __future__ import annotations

from app.predict import predict_one
from app.schemas import CustomerInput


def test_predict_returns_expected_fields():
    response = predict_one(CustomerInput())
    assert response["customer_id"] == "CUST-DEMO-001"
    assert 0 <= response["churn_probability"] <= 1
    assert 0 <= response["decision_threshold"] <= 1
    assert response["churn_prediction"] in {0, 1}
    assert response["risk_band"] in {"Low", "Medium", "High"}
    assert isinstance(response["top_reasons"], list)
    assert 0 <= response["persuadability_score"] <= 1
    assert response["uplift_tier"] in {"Low", "Medium", "High"}
    assert response["recommended_action"]
    assert "net_expected_value" in response["expected_business_impact"]
