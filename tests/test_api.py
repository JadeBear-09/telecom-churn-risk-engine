from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_contract():
    response = client.post("/predict", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["customer_id"] == "CUST-DEMO-001"
    assert "churn_probability" in payload
    assert "decision_threshold" in payload
    assert "churn_prediction" in payload
    assert "persuadability_score" in payload
    assert "risk_band" in payload
    assert "recommended_action" in payload
