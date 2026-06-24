from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.explainability import top_prediction_reasons
from app.recommendations import expected_business_impact, recommend_action, risk_band
from app.schemas import CustomerInput
from src.feature_engineering import FEATURE_COLUMNS, engineer_features
from src.segment_thresholds import threshold_for_segment
from src.uplift_modeling import score_persuadability, uplift_tier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, Any]:
    model_path = MODEL_DIR / "churn_model.pkl"
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    metrics_path = MODEL_DIR / "metrics.json"
    uplift_path = MODEL_DIR / "uplift_model.pkl"
    explainer_background_path = MODEL_DIR / "explainer_background.pkl"
    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError("Model artifacts missing. Run `make train` before scoring.")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"selected_threshold": 0.5}
    uplift_model = joblib.load(uplift_path) if uplift_path.exists() else None
    explainer_background = None
    if explainer_background_path.exists():
        background_payload = joblib.load(explainer_background_path)
        explainer_background = background_payload.get("features")
    return {
        "model": joblib.load(model_path),
        "preprocessor": joblib.load(preprocessor_path),
        "metrics": metrics,
        "uplift_model": uplift_model,
        "explainer_background": explainer_background,
    }


def _customer_to_dataframe(customer: CustomerInput | dict[str, Any]) -> pd.DataFrame:
    if isinstance(customer, CustomerInput):
        record = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()
    else:
        record = dict(customer)
    return pd.DataFrame([record])


def _predict_probability(model: Any, features: Any) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0, 1])
    score = float(model.decision_function(features)[0])
    return float(1 / (1 + np.exp(-score)))


def predict_one(customer: CustomerInput) -> dict[str, Any]:
    artifacts = load_artifacts()
    raw_df = _customer_to_dataframe(customer)
    engineered = engineer_features(raw_df)
    feature_frame = engineered[FEATURE_COLUMNS]
    transformed = artifacts["preprocessor"].transform(feature_frame)
    churn_probability = _predict_probability(artifacts["model"], transformed)
    top_reasons = top_prediction_reasons(
        artifacts["model"],
        artifacts["preprocessor"],
        feature_frame,
        artifacts.get("explainer_background"),
        top_n=5,
    )
    segment = str(engineered.iloc[0].get("customer_segment", "Unknown"))
    decision_threshold = threshold_for_segment(
        segment,
        artifacts["metrics"].get("segment_thresholds"),
        default_threshold=float(artifacts["metrics"].get("selected_threshold", 0.5)),
    )
    churn_prediction = int(churn_probability >= decision_threshold)
    persuadability_score = score_persuadability(
        artifacts.get("uplift_model"),
        transformed,
        feature_frame,
        churn_probability,
    )
    action = recommend_action(churn_probability, top_reasons)
    return {
        "customer_id": customer.customer_id,
        "churn_probability": round(churn_probability, 4),
        "decision_threshold": round(decision_threshold, 4),
        "churn_prediction": churn_prediction,
        "risk_band": risk_band(churn_probability),
        "top_reasons": top_reasons,
        "persuadability_score": round(persuadability_score, 4),
        "uplift_tier": uplift_tier(persuadability_score),
        "recommended_action": action,
        "expected_business_impact": expected_business_impact(churn_probability, action),
    }


def predict_batch(customers: list[CustomerInput]) -> list[dict[str, Any]]:
    return [predict_one(customer) for customer in customers]
