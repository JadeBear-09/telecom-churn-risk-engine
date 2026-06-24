from __future__ import annotations

import numpy as np
import pandas as pd

from app.explainability import _translate_feature_name
from app.recommendations import risk_band
from src.calibration import calibration_summary
from src.feature_engineering import ENGINEERED_FEATURES, engineer_features
from src.segment_thresholds import select_segment_thresholds, threshold_for_segment
from src.threshold_tuning import select_optimal_threshold


def test_feature_engineering_creates_expected_columns():
    df = pd.DataFrame(
        [
            {
                "customer_id": "T-1",
                "tenure": 3,
                "monthly_charges": 95.0,
                "total_charges": 285.0,
                "contract": "Month-to-month",
                "complaint_count_90d": 3,
                "support_ticket_count_90d": 4,
                "network_downtime_minutes_30d": 90,
                "late_payment_count_6m": 2,
                "billing_dispute_count_6m": 1,
                "avg_data_usage_gb_30d": 88,
                "customer_segment": "Premium",
                "last_interaction_sentiment": "negative",
            }
        ]
    )
    engineered = engineer_features(df)
    for column in ENGINEERED_FEATURES:
        assert column in engineered.columns
    assert np.isfinite(engineered[ENGINEERED_FEATURES].to_numpy()).all()
    assert engineered.loc[0, "is_month_to_month"] == 1
    assert engineered.loc[0, "billing_issue_flag"] == 1


def test_risk_band_mapping():
    assert risk_band(0.10) == "Low"
    assert risk_band(0.30) == "Medium"
    assert risk_band(0.65) == "High"


def test_contract_explanation_does_not_overgeneralize():
    assert _translate_feature_name("cat__contract_Month-to-month") == "month-to-month contract"
    assert _translate_feature_name("cat__contract_One year") == "Contract One year"


def test_threshold_tuning_returns_valid_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_probability = np.array([0.05, 0.20, 0.45, 0.55, 0.80, 0.95])
    threshold, rows = select_optimal_threshold(y_true, y_probability, min_recall=0.5)
    assert 0.10 <= threshold <= 0.90
    assert rows
    assert all("total_cost" in row for row in rows)


def test_calibration_summary_returns_expected_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_probability = np.array([0.1, 0.2, 0.8, 0.9])
    summary = calibration_summary(y_true, y_probability, n_bins=2)
    assert 0 <= summary["brier_score"] <= 1
    assert 0 <= summary["expected_calibration_error"] <= 1
    assert len(summary["bins"]) == 2


def test_segment_thresholds_fall_back_for_small_segments():
    y_true = np.array([0, 0, 1, 1])
    y_probability = np.array([0.1, 0.2, 0.8, 0.9])
    segments = pd.Series(["Premium", "Premium", "Value", "Value"])
    thresholds = select_segment_thresholds(y_true, y_probability, segments, default_threshold=0.3, min_segment_size=10)
    assert threshold_for_segment("Premium", thresholds, default_threshold=0.3) == 0.3
    assert threshold_for_segment("Unknown", thresholds, default_threshold=0.3) == 0.3
