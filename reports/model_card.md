# Model Card

## Model Purpose

The model predicts telecom customer churn risk and supports retention prioritization. It is designed to help identify customers who may cancel service and explain the likely reasons behind the risk.

This model is not meant to make fully automated customer decisions. It should support human review and retention workflow prioritization.

## Intended Users

Primary users:

- ML engineers evaluating the training and serving workflow
- data analysts reviewing churn drivers
- customer success or retention teams using risk bands
- technical reviewers evaluating the end-to-end workflow

Secondary users:

- product managers interested in customer experience signals
- business stakeholders reviewing retention strategy

## Input Features

Core customer and account features:

- tenure
- monthly charges
- total charges
- contract type
- payment method
- internet service
- support add-ons
- streaming services
- senior citizen flag
- partner and dependent flags

Synthetic telecom/business features:

- complaint count in last 90 days
- support ticket count in last 90 days
- network downtime minutes in last 30 days
- late payment count in last 6 months
- billing dispute count in last 6 months
- average data usage in last 30 days
- plan change count in last 12 months
- region
- customer segment
- last interaction sentiment

Engineered features:

- complaints per month
- support calls per month
- charges per tenure
- month-to-month contract flag
- recent downtime flag
- late payment ratio
- billing issue flag
- high-value customer flag
- high-usage low-satisfaction flag

## Output Fields

The prediction API returns:

- `customer_id`
- `churn_probability`
- `risk_band`
- `top_reasons`
- `recommended_action`
- `expected_business_impact`

Risk bands:

```text
Low     0.00 - 0.30
Medium  0.30 - 0.65
High    0.65 - 1.00
```

## Training Data

The project supports the public IBM Telco Customer Churn dataset. The loader now prefers `data/raw/Telco-Customer-Churn.csv`, then the older IBM filename, then downloads the public IBM CSV URL. If the real source is unavailable, it creates a compatible synthetic fallback dataset.

Important note:

The fallback dataset is useful for exercising the engineering structure and demo behavior. It should not be treated as real production evidence. A production system should train on actual CRM, billing, support, network, and cancellation data.

## Model Candidates

Training compares:

- Logistic Regression baseline
- Random Forest
- XGBoost if available
- Gradient Boosting if XGBoost is not available

Logistic Regression is included as an interpretable baseline. Tree-based models are included to capture nonlinear relationships and feature interactions.

## Evaluation Metrics

Saved metrics include:

- accuracy
- ROC-AUC
- precision
- recall
- F1 score
- confusion matrix
- classification report
- precision-recall curve data
- threshold tuning table

The project emphasizes recall and business cost, not only accuracy. Churn prediction usually penalizes missed churners more heavily than extra outreach.

## Current Artifact Snapshot

Current metrics from the real IBM CSV run:

| Field | Value |
| --- | ---: |
| Selected model | `logistic_regression` |
| Rows | `7043` |
| ROC-AUC | `0.841` |
| Accuracy | `0.649` |
| Precision | `0.425` |
| Recall | `0.922` |
| F1 | `0.582` |
| Selected threshold | `0.13` |
| Brier score | `0.139` |
| Expected calibration error | `0.023` |

Interpretation:

The current threshold favors recall because the false negative cost is much higher than the false positive cost. This is appropriate for a retention demo where missing churners is expensive, but the exact operating point should be revisited with real intervention cost, customer lifetime value, and retention-team capacity.

## Threshold Selected

Thresholds from `0.10` to `0.90` are evaluated. The selected threshold minimizes business cost while targeting reasonable recall when possible.

Cost matrix:

```text
False Negative = ₹5,000
False Positive = ₹500
```

The current selected threshold is saved in:

```text
models/metrics.json
```

Current demo threshold: `0.13`, with segment thresholds saved under `segment_thresholds`.

## Explainability Method

The model explanation layer returns business-language reasons rather than raw encoded feature names.

Explanation sources:

- rule-based mapping of high-risk feature values
- model feature importances for tree models
- absolute coefficient magnitude for Logistic Regression

Examples:

- month-to-month contract
- high complaints
- recent network downtime
- late payment history
- high monthly charges
- negative last interaction sentiment

The serving layer can use SHAP for local explanations when `shap` is installed. If SHAP is unavailable, it falls back to business rules plus model coefficient or feature-importance signals.

## Limitations

- Synthetic fallback features are simulated.
- Current system is not trained on live telecom production data.
- Recommendations are rule-based and need business validation.
- Uplift modeling currently uses a synthetic treatment/control baseline until real outreach data exists.
- The model registry is local JSON metadata, not a managed production registry.
- No real-time feature store is included.
- Label delay is not fully implemented; actual churn labels may arrive weeks later.
- Fairness and segment-level performance analysis should be added before production use.

## Ethical Considerations

Churn scoring can affect how customers receive offers and support. The model should not be used to unfairly exclude customers from service quality improvements or support.

Important checks before production:

- Monitor performance by region, segment, age-related proxy fields, and plan type.
- Avoid using sensitive attributes unless there is a justified and compliant reason.
- Keep human review for high-impact retention decisions.
- Make sure offers and support actions are fair and auditable.
- Do not use the model to penalize customers for complaints or payment hardship.

## Monitoring Requirements

Production monitoring should include:

- input schema validation
- missing value rates
- numeric feature drift
- categorical distribution changes
- prediction distribution drift
- high-risk customer percentage
- API latency
- API error rate
- delayed model performance after churn labels arrive
- business KPIs such as contacted customers, saved customers, retention cost, and net retained revenue

Retraining should be considered when drift persists, performance drops, business policies change, or enough new labeled churn outcomes are available.
