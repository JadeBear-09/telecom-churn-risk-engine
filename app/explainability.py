from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


BUSINESS_REASON_MAP = {
    "contract_Month-to-month": "month-to-month contract",
    "complaint_count_90d": "high complaints",
    "complaints_per_month": "high complaints",
    "support_ticket_count_90d": "high support ticket volume",
    "support_calls_per_month": "high support ticket volume",
    "network_downtime_minutes_30d": "recent network downtime",
    "has_recent_downtime": "recent network downtime",
    "late_payment_count_6m": "late payment history",
    "late_payment_ratio": "late payment history",
    "billing_dispute_count_6m": "billing dispute history",
    "billing_issue_flag": "billing issue history",
    "monthly_charges": "high monthly charges",
    "charges_per_tenure": "high charges relative to tenure",
    "tenure": "short tenure",
    "tech_support_No": "no active tech support",
    "online_security_No": "no online security add-on",
    "payment_method_Electronic check": "electronic check payment risk",
    "last_interaction_sentiment_negative": "negative last interaction sentiment",
    "high_usage_low_satisfaction_flag": "high usage with low satisfaction",
}


def _feature_names(preprocessor: Any) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def _model_signal(model: Any) -> np.ndarray | None:
    estimator = _unwrap_estimator(model)
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)
    if hasattr(estimator, "coef_"):
        return np.abs(np.asarray(estimator.coef_[0], dtype=float))
    return None


def _unwrap_estimator(model: Any) -> Any:
    if model.__class__.__name__ == "FrozenEstimator" and hasattr(model, "estimator"):
        return _unwrap_estimator(model.estimator)
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        estimator = getattr(calibrated[0], "estimator", None) or getattr(calibrated[0], "base_estimator", None)
        if estimator is not None:
            return _unwrap_estimator(estimator)
    return model


def _translate_feature_name(feature_name: str) -> str:
    compact = feature_name.replace("num__", "").replace("cat__", "")
    for raw_name, reason in BUSINESS_REASON_MAP.items():
        if raw_name in compact:
            return reason
    compact = compact.replace("_", " ")
    return compact[:1].upper() + compact[1:]


def rule_based_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if str(row.get("contract", "")).lower() == "month-to-month":
        reasons.append("month-to-month contract")
    if float(row.get("complaint_count_90d", 0) or 0) >= 2:
        reasons.append("high complaints")
    if float(row.get("support_ticket_count_90d", 0) or 0) >= 3:
        reasons.append("high support ticket volume")
    if float(row.get("network_downtime_minutes_30d", 0) or 0) >= 60:
        reasons.append("recent network downtime")
    if float(row.get("late_payment_count_6m", 0) or 0) >= 2:
        reasons.append("late payment history")
    if float(row.get("billing_dispute_count_6m", 0) or 0) > 0:
        reasons.append("billing dispute history")
    if float(row.get("monthly_charges", 0) or 0) >= 80:
        reasons.append("high monthly charges")
    if float(row.get("tenure", 0) or 0) <= 6:
        reasons.append("short tenure")
    if str(row.get("last_interaction_sentiment", "")).lower() == "negative":
        reasons.append("negative last interaction sentiment")
    if int(row.get("high_usage_low_satisfaction_flag", 0) or 0) == 1:
        reasons.append("high usage with low satisfaction")
    return reasons


def shap_local_reasons(
    model: Any,
    preprocessor: Any,
    engineered_df: pd.DataFrame,
    explainer_background: np.ndarray | None = None,
    top_n: int = 5,
) -> list[str]:
    """Return local SHAP reasons when shap is installed; otherwise return empty."""
    try:
        import shap

        transformed = np.asarray(preprocessor.transform(engineered_df))
        feature_names = _feature_names(preprocessor)
        if not feature_names:
            return []

        background = explainer_background
        if background is None:
            background = np.zeros((1, transformed.shape[1]), dtype=float)

        explainer = shap.Explainer(
            lambda values: model.predict_proba(values)[:, 1],
            background,
            feature_names=feature_names,
        )
        values = explainer(transformed)
        shap_values = np.asarray(values.values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        local_values = shap_values[0]

        reasons: list[str] = []
        for idx in np.argsort(np.abs(local_values))[::-1]:
            reason = _translate_feature_name(feature_names[idx])
            if reason not in reasons:
                reasons.append(reason)
            if len(reasons) >= top_n:
                break
        return reasons
    except Exception:
        return []


def top_prediction_reasons(
    model: Any,
    preprocessor: Any,
    engineered_df: pd.DataFrame,
    explainer_background: np.ndarray | None = None,
    top_n: int = 5,
) -> list[str]:
    """Return business-language drivers for one prediction."""
    row = engineered_df.iloc[0]
    reasons = rule_based_reasons(row)
    for reason in shap_local_reasons(model, preprocessor, engineered_df, explainer_background, top_n=top_n):
        if reason not in reasons:
            reasons.append(reason)

    try:
        transformed = preprocessor.transform(engineered_df)
        feature_names = _feature_names(preprocessor)
        signal = _model_signal(model)
        if signal is not None and len(feature_names) == transformed.shape[1]:
            local_values = np.abs(np.asarray(transformed[0]).ravel()) * signal
            for idx in np.argsort(local_values)[::-1]:
                reason = _translate_feature_name(feature_names[idx])
                if reason not in reasons:
                    reasons.append(reason)
                if len(reasons) >= top_n:
                    break
    except Exception:
        pass

    return reasons[:top_n] if reasons else ["model indicates elevated churn pattern"]


def global_feature_importance(model: Any, preprocessor: Any, top_n: int = 20) -> list[dict[str, float | str]]:
    names = _feature_names(preprocessor)
    signal = _model_signal(model)
    if signal is None or not names:
        return []
    ranked = np.argsort(signal)[::-1][:top_n]
    return [
        {"feature": _translate_feature_name(names[idx]), "raw_feature": names[idx], "importance": float(signal[idx])}
        for idx in ranked
    ]
