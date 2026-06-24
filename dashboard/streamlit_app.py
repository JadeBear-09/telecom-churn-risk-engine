from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.explainability import global_feature_importance  # noqa: E402
from app.batch_validation import (  # noqa: E402
    BATCH_COLUMNS,
    BATCH_REQUIRED_COLUMNS,
    format_missing_columns_error,
    validate_batch_columns,
)
from app.predict import MODEL_DIR, load_artifacts, predict_one  # noqa: E402
from app.schemas import CustomerInput  # noqa: E402
from src.monitoring import monitoring_snapshot  # noqa: E402


st.set_page_config(page_title="Telecom Churn Risk Engine", layout="wide")


BATCH_COLUMN_GROUPS = {
    "Account": [
        "customer_id",
        "tenure",
        "contract",
        "payment_method",
        "paperless_billing",
    ],
    "Services": [
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ],
    "Billing": [
        "monthly_charges",
        "total_charges",
        "late_payment_count_6m",
        "billing_dispute_count_6m",
    ],
    "Support and usage": [
        "complaint_count_90d",
        "support_ticket_count_90d",
        "network_downtime_minutes_30d",
        "avg_data_usage_gb_30d",
        "plan_change_count_12m",
        "last_interaction_sentiment",
    ],
    "Profile": [
        "gender",
        "senior_citizen",
        "partner",
        "dependents",
        "region",
        "customer_segment",
    ],
}


def _load_metrics() -> dict:
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text())


def _template_csv() -> str:
    example = CustomerInput().model_dump()
    columns = ["customer_id", *[col for col in BATCH_COLUMNS if col != "customer_id"]]
    return pd.DataFrame([example], columns=columns).to_csv(index=False)


def _render_batch_requirements() -> None:
    required_count = len(BATCH_REQUIRED_COLUMNS)
    optional_count = len(BATCH_COLUMNS) - required_count
    st.caption(f"{required_count} required columns. {optional_count} optional column: customer_id.")

    group_cols = st.columns(3)
    for idx, (group, columns) in enumerate(BATCH_COLUMN_GROUPS.items()):
        with group_cols[idx % len(group_cols)]:
            st.markdown(f"**{group}**")
            rows = []
            for column in columns:
                rows.append(
                    {
                        "Column": column,
                        "Required": "No" if column == "customer_id" else "Yes",
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.download_button(
        "Download CSV template",
        data=_template_csv(),
        file_name="telecom_churn_batch_template.csv",
        mime="text/csv",
    )


def _customer_form() -> CustomerInput:
    left, middle, right = st.columns(3)
    with left:
        customer_id = st.text_input("Customer ID", "CUST-DEMO-001")
        tenure = st.number_input("Tenure", min_value=0, max_value=100, value=8)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=92.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=736.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
    with middle:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1)
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], index=1)
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], index=1)
        complaint_count_90d = st.number_input("Complaints 90d", min_value=0, value=3)
        support_ticket_count_90d = st.number_input("Support Tickets 90d", min_value=0, value=4)
        network_downtime_minutes_30d = st.number_input("Downtime Minutes 30d", min_value=0.0, value=115.0)
    with right:
        late_payment_count_6m = st.number_input("Late Payments 6m", min_value=0, value=2)
        billing_dispute_count_6m = st.number_input("Billing Disputes 6m", min_value=0, value=1)
        avg_data_usage_gb_30d = st.number_input("Avg Data Usage GB 30d", min_value=0.0, value=86.0)
        last_interaction_sentiment = st.selectbox("Last Interaction Sentiment", ["positive", "neutral", "negative"], index=2)
        region = st.selectbox("Region", ["Northeast", "South", "Midwest", "West"], index=1)
        customer_segment = st.selectbox("Segment", ["Value", "Growth", "Premium"], index=2)

    return CustomerInput(
        customer_id=customer_id,
        gender="Female",
        senior_citizen=0,
        partner="Yes",
        dependents="No",
        tenure=int(tenure),
        phone_service="Yes",
        multiple_lines="No",
        internet_service=internet_service,
        online_security=online_security,
        online_backup="No",
        device_protection="No",
        tech_support=tech_support,
        streaming_tv="Yes",
        streaming_movies="Yes",
        contract=contract,
        paperless_billing="Yes",
        payment_method=payment_method,
        monthly_charges=float(monthly_charges),
        total_charges=float(total_charges),
        complaint_count_90d=int(complaint_count_90d),
        support_ticket_count_90d=int(support_ticket_count_90d),
        network_downtime_minutes_30d=float(network_downtime_minutes_30d),
        late_payment_count_6m=int(late_payment_count_6m),
        billing_dispute_count_6m=int(billing_dispute_count_6m),
        avg_data_usage_gb_30d=float(avg_data_usage_gb_30d),
        plan_change_count_12m=1,
        region=region,
        customer_segment=customer_segment,
        last_interaction_sentiment=last_interaction_sentiment,
    )


def _score_batch(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    missing, unexpected = validate_batch_columns(df)
    if missing:
        raise ValueError(format_missing_columns_error(missing))
    if unexpected:
        st.info(f"Ignoring extra columns: {', '.join(unexpected[:12])}")
    results = []
    for idx, row in df.iterrows():
        record = row.to_dict()
        record.setdefault("customer_id", f"UPLOAD-{idx:05d}")
        results.append(predict_one(CustomerInput(**record)))
    return pd.DataFrame(results)


st.title("Telecom Customer Churn Risk Scoring")

if not (MODEL_DIR / "churn_model.pkl").exists():
    st.warning("Model artifacts missing. Run `make train` from project root.")

tab_score, tab_batch, tab_metrics, tab_monitor = st.tabs(["Score", "Batch", "Metrics", "Monitoring"])

with tab_score:
    customer = _customer_form()
    if st.button("Score Customer", type="primary"):
        result = predict_one(customer)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Churn Probability", f"{result['churn_probability']:.1%}")
        kpi2.metric("Risk Band", result["risk_band"])
        kpi3.metric("Decision Threshold", f"{result['decision_threshold']:.1%}")
        kpi4.metric("Persuadability", f"{result['persuadability_score']:.1%}")
        st.subheader("Top Reasons")
        st.write(result["top_reasons"])
        st.subheader("Recommended Action")
        st.write(result["recommended_action"])
        st.metric("Net Expected Value", f"${result['expected_business_impact']['net_expected_value']:,.0f}")

with tab_batch:
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    with st.expander("CSV format", expanded=True):
        st.write("Upload one row per customer. Extra columns are ignored.")
        _render_batch_requirements()
    if uploaded is not None:
        try:
            batch_results = _score_batch(uploaded)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.dataframe(batch_results, use_container_width=True)
            st.subheader("Churn Risk Distribution")
            st.bar_chart(batch_results["risk_band"].value_counts())
            st.subheader("High-Risk Customers")
            st.dataframe(batch_results[batch_results["risk_band"] == "High"], use_container_width=True)
            st.subheader("Most Persuadable Customers")
            st.dataframe(batch_results.sort_values("persuadability_score", ascending=False).head(20), use_container_width=True)

with tab_metrics:
    metrics = _load_metrics()
    if metrics:
        metric_cols = st.columns(5)
        for col, key in zip(metric_cols, ["accuracy", "roc_auc", "precision", "recall", "f1"]):
            col.metric(key.replace("_", " ").title(), f"{metrics.get(key, 0):.3f}")

        st.subheader("Confusion Matrix")
        st.dataframe(pd.DataFrame(metrics.get("confusion_matrix", []), index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        st.subheader("Threshold Tuning Cost Curve")
        threshold_df = pd.DataFrame(metrics.get("threshold_tuning", []))
        if not threshold_df.empty:
            st.line_chart(threshold_df.set_index("threshold")[["total_cost", "false_negative_cost", "false_positive_cost"]])

        st.subheader("Calibration")
        calibration = metrics.get("calibration", {}).get("test", {})
        if calibration:
            st.write(
                {
                    "brier_score": calibration.get("brier_score"),
                    "expected_calibration_error": calibration.get("expected_calibration_error"),
                }
            )

        st.subheader("Segment Thresholds")
        segment_thresholds = metrics.get("segment_thresholds", {})
        if segment_thresholds:
            st.dataframe(pd.DataFrame.from_dict(segment_thresholds, orient="index"), use_container_width=True)

        st.subheader("Feature Importance")
        try:
            artifacts = load_artifacts()
            importance = pd.DataFrame(global_feature_importance(artifacts["model"], artifacts["preprocessor"]))
            if not importance.empty:
                st.bar_chart(importance.head(15).set_index("feature")["importance"])
        except Exception as exc:
            st.info(f"Feature importance unavailable: {exc}")
    else:
        st.info("Metrics unavailable. Run `make train`.")

with tab_monitor:
    sample_path = PROJECT_ROOT / "data" / "sample_inputs" / "sample_customers.csv"
    if sample_path.exists():
        sample_df = pd.read_csv(sample_path).head(100)
        probabilities = [predict_one(CustomerInput(**row.to_dict()))["churn_probability"] for _, row in sample_df.iterrows()]
        snapshot = monitoring_snapshot(sample_df, probabilities)
        mon1, mon2 = st.columns(2)
        mon1.metric("Average Churn Probability", f"{snapshot['average_churn_probability']:.1%}")
        mon2.metric("High-Risk Customer %", f"{snapshot['high_risk_customer_pct']:.1%}")
        st.subheader("Prediction Distribution")
        st.bar_chart(pd.Series(snapshot["prediction_distribution"]))
        st.subheader("API Latency/Error Snapshot")
        st.json(snapshot["api_metrics"])
        st.subheader("Missing Value Rate")
        st.dataframe(pd.Series(snapshot["missing_value_rate"], name="missing_rate"))
    else:
        st.info("Sample customers unavailable. Run `make train`.")
