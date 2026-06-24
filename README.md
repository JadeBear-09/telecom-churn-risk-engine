---
title: Telecom Churn Risk Engine
emoji: 📉
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Telecom Customer Churn Risk Scoring & Retention Recommendation Engine

End-to-end classical machine learning project for telecom churn risk scoring. The system predicts whether a customer is likely to churn, explains the main risk drivers, maps the probability into a business risk band, recommends a retention action, exposes predictions through FastAPI, and provides a Streamlit dashboard for scoring, evaluation, and monitoring.

This project is intentionally built as a practical ML engineering portfolio project, not only a notebook. It shows how a model can move from data preparation to training, threshold selection, API serving, dashboarding, testing, Docker deployment, and production monitoring.

## Project Overview

The project answers one business question:

> Which customers are most likely to churn, why are they risky, and what should the retention team do next?

The output is not just a model prediction. Each scored customer receives:

- churn probability
- risk band: `Low`, `Medium`, or `High`
- top risk reasons in business language
- recommended retention action
- expected business impact based on a cost matrix

## Project Snapshot

| Area | Implementation |
| --- | --- |
| ML task | Binary classification for churn risk |
| Main library | scikit-learn |
| Serving layer | FastAPI |
| Dashboard | Streamlit |
| Deployment | Docker, Hugging Face Spaces-ready |
| Current demo model | Logistic Regression selected from candidate models |
| Current demo threshold | `0.13`, selected by cost-based threshold tuning on a calibration split |
| Scope note | Uses the public IBM Telco CSV from GitHub when present or downloadable; synthetic fallback remains available if the real source is unavailable |

## Business Problem

Telecom companies lose recurring revenue when customers cancel service. Retention teams cannot manually contact every customer, so they need a prioritized list of customers who are likely to churn and enough context to take action.

Model accuracy alone is not enough for this use case. A retention workflow needs high recall for likely churners, reasonable precision so teams do not waste outreach effort, and a threshold that reflects the cost of missing a churner versus contacting a customer unnecessarily.

## Why This Project Matters

This project demonstrates ML concepts that come up often in interviews and real projects:

- tabular classification
- data cleaning and feature engineering
- baseline versus stronger models
- ROC-AUC, precision, recall, F1, and confusion matrix interpretation
- threshold tuning with business costs
- explainability for non-technical stakeholders
- FastAPI model serving
- Streamlit dashboarding
- Docker and Hugging Face Spaces deployment
- monitoring for drift, API health, and business KPIs

## Architecture

```text
Raw IBM Telco CSV or synthetic fallback data
        |
        v
Data cleaning
        |
        v
Synthetic telecom/business feature generation
        |
        v
Feature engineering
        |
        v
ColumnTransformer preprocessing
        |
        v
Model training
  - Logistic Regression baseline
  - Random Forest challenger
  - XGBoost if installed, otherwise Gradient Boosting
        |
        v
Evaluation and threshold tuning
        |
        v
Saved artifacts
  - models/churn_model.pkl
  - models/preprocessor.pkl
  - models/metrics.json
        |
        +--------------------+
        |                    |
        v                    v
FastAPI scoring service    Streamlit dashboard
        |                    |
        v                    v
Prediction, reasons,       Single scoring, batch scoring,
risk band, action          metrics, monitoring snapshot
```

## Repository Structure

```text
app/                  FastAPI app, prediction logic, schemas, explanations, recommendations
src/                  Data ingestion, cleaning, feature engineering, training, evaluation, monitoring
dashboard/            Streamlit dashboard
notebooks/            EDA, feature engineering, training, and threshold-analysis notebook stubs
data/                 Raw data, processed data, and sample scoring inputs
models/               Trained model, preprocessor, metrics, evaluation output
reports/              Business case, model card, monitoring plan, interview notes, concept guide
tests/                Pytest coverage for features, API, prediction contract, threshold tuning
Dockerfile            Hugging Face Spaces-compatible Docker image
Makefile              Common commands for install, train, evaluate, API, dashboard, tests
```

## Dataset Description

The project supports the public IBM Telco Customer Churn dataset. The current real source is:

```text
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

`src/data_ingestion.py` looks for local real CSV files first:

```text
data/raw/Telco-Customer-Churn.csv
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

If neither file exists, the loader downloads the IBM CSV from the URL above. If that download is unavailable, it creates a compatible synthetic fallback dataset with the same kind of customer, contract, billing, and service fields.

Current artifacts were generated from the real IBM CSV with `7043` rows. To reproduce, run `make train`.

Core fields include:

- customer ID
- tenure
- contract type
- monthly and total charges
- internet service type
- support add-ons
- payment method
- churn label

## Synthetic Feature Explanation

The IBM dataset is useful, but real telecom churn systems usually include operational and business signals that are not present in the public CSV. To make the project closer to a production setting, `src/synthetic_features.py` adds realistic extra features:

- `complaint_count_90d`
- `support_ticket_count_90d`
- `network_downtime_minutes_30d`
- `late_payment_count_6m`
- `billing_dispute_count_6m`
- `avg_data_usage_gb_30d`
- `plan_change_count_12m`
- `region`
- `customer_segment`
- `last_interaction_sentiment`

These features are simulated when real source data is unavailable. In a production project, they would come from CRM, billing, support-ticket, network-quality, and customer-service systems.

## Optional Operational Source Files

`src/data_sources.py` can merge source-system CSV extracts before synthetic fallback fills missing fields. Place any of these files in `data/raw/`:

| File | Expected fields |
| --- | --- |
| `crm_customers.csv` | `customer_id`, `customer_segment`, `region`, `last_interaction_sentiment`, `plan_change_count_12m` |
| `support_tickets.csv` | `customer_id`, `created_at`, `category` or `ticket_type`, optional `subject` |
| `billing_events.csv` | `customer_id`, `event_date`, `event_type` |
| `network_outages.csv` | `customer_id` or `region`, `event_date`, `downtime_minutes` or `outage_minutes` |

If these files are absent, training remains fully runnable with the synthetic fallback data.

## Feature Engineering

Feature engineering turns raw fields into more useful model signals:

- `complaints_per_month`: complaint intensity normalized over 90 days
- `support_calls_per_month`: support load normalized over 90 days
- `charges_per_tenure`: customer spend relative to time with company
- `is_month_to_month`: high-flexibility contract flag
- `has_recent_downtime`: recent network experience risk flag
- `late_payment_ratio`: payment friction indicator
- `billing_issue_flag`: late payment or billing dispute signal
- `high_value_customer_flag`: premium or high-spend customer
- `high_usage_low_satisfaction_flag`: high usage combined with negative sentiment

The preprocessing pipeline uses `ColumnTransformer` as a safe baseline, not as a claim that these choices are always best:

- numeric features: median imputation and scaling
- categorical features: most-frequent imputation and one-hot encoding

These preprocessing steps are fitted only on the training split, then reused on validation, test, API, and dashboard inputs. This prevents data leakage.

In a real project, preprocessing is chosen after looking at the dataset:

- missing-value patterns decide whether median, mode, `"Unknown"`, or missing-indicator features make sense
- numeric distributions decide whether standard scaling, robust scaling, log transforms, or no scaling is better
- categorical cardinality decides whether one-hot encoding is safe or whether rare-category grouping, hashing, or target encoding should be tested
- model family matters: Logistic Regression benefits from scaling, while tree models usually do not need scaled numeric features
- final choices should be compared with cross-validation and the same business metrics used for model selection

## Model Training Approach

Training code lives in `src/train.py`.

The pipeline:

1. Load IBM data or synthetic fallback data.
2. Clean column names, missing values, and target labels.
3. Add synthetic telecom features when needed.
4. Engineer business-relevant features.
5. Split train and test data with stratification.
6. Fit preprocessing only on training data.
7. Train multiple classical ML models.
8. Select the model using validation ROC-AUC with recall awareness.
9. Tune the decision threshold using business cost.
10. Save model, preprocessor, metrics, and sample inputs.

## Baseline Model vs Final Model

The baseline model is Logistic Regression. It is useful because it is fast, stable, and easier to explain in interviews. It also gives a sanity check before using more complex models.

Challenger models include:

- Random Forest
- XGBoost if installed
- Gradient Boosting if XGBoost is unavailable

The final selected model is whichever performs best in the training run based on ROC-AUC and recall-aware selection. In the current generated fallback dataset run, Logistic Regression is selected because it performs best on the validation split while staying interpretable.

## Evaluation Metrics

Metrics are saved in `models/metrics.json`.

The project reports:

- accuracy
- ROC-AUC
- precision
- recall
- F1 score
- confusion matrix
- classification report
- precision-recall curve data

Accuracy is included but not treated as the main decision metric. For churn, recall and precision are more useful because the business cares about catching churners while keeping retention outreach manageable.

Current real IBM data run:

| Metric | Value |
| --- | ---: |
| Model | `logistic_regression` |
| Rows | `7043` |
| ROC-AUC | `0.841` |
| Accuracy | `0.649` |
| Precision | `0.425` |
| Recall | `0.922` |
| F1 | `0.582` |
| Selected threshold | `0.13` |
| Brier score | `0.139` |
| Expected calibration error | `0.023` |

These numbers are useful for demonstrating the workflow on the public IBM dataset. Operational features such as support tickets and outages are still simulated unless source-system CSVs are added.

## Threshold Tuning

The model outputs probabilities, but the business needs a decision: who should receive retention action?

Instead of using the default `0.50` threshold, `src/threshold_tuning.py` evaluates thresholds from `0.10` to `0.90`.
Training now uses a train/calibration/test split. Candidate models are selected on the calibration split, probability calibration is applied with sigmoid calibration, and the final test split is only used for final evaluation.

Cost matrix:

```text
False Negative: ₹5,000
False Positive: ₹500
```

False negatives are expensive because a likely churner is missed. False positives are cheaper because they usually mean an unnecessary retention action. The selected threshold minimizes total business cost while maintaining a recall floor when possible.

Segment-specific thresholds are saved in `models/metrics.json` under `segment_thresholds`. The current real-data run uses:

| Segment | Threshold |
| --- | ---: |
| Global default | `0.13` |
| Growth | `0.16` |
| Premium | `0.12` |
| Value | `0.13` |

## Explainability

Explainability code lives in `app/explainability.py`.

The API returns top risk reasons in business language, for example:

- month-to-month contract
- high complaints
- recent network downtime
- late payment history
- high monthly charges
- negative last interaction sentiment

The implementation uses rule-based business reason mapping, optional SHAP local explanations when `shap` is installed, and a fallback based on model feature importance or Logistic Regression coefficients. SHAP background data is saved as `models/explainer_background.pkl`.

Install optional SHAP support with:

```bash
python -m pip install -r requirements-explainability.txt
```

## FastAPI Endpoints

Run API:

```bash
make api
```

Default local URL:

```text
http://localhost:8000
```

Endpoints:

```text
GET  /health
POST /predict
POST /batch_predict
GET  /model/metrics
GET  /customer/{customer_id}/risk
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"CUST-001","contract":"Month-to-month","monthly_charges":95,"tenure":4}'
```

Example response fields:

```json
{
  "customer_id": "CUST-001",
  "churn_probability": 0.82,
  "decision_threshold": 0.14,
  "churn_prediction": 1,
  "risk_band": "High",
  "top_reasons": ["month-to-month contract", "high monthly charges"],
  "persuadability_score": 0.31,
  "uplift_tier": "Medium",
  "recommended_action": "Retention call with targeted discount and billing support escalation.",
  "expected_business_impact": {
    "expected_loss_without_action": 4100.0,
    "net_expected_value": 420.0
  }
}
```

## Streamlit Dashboard

Run dashboard:

```bash
make dashboard
```

Open:

```text
http://localhost:8501
```

Dashboard sections:

- single customer scoring form
- batch CSV upload
- model metrics
- confusion matrix
- churn risk distribution
- high-risk customer table
- most persuadable customer table
- feature importance chart
- threshold tuning and business cost chart
- calibration metrics
- segment threshold table
- monitoring snapshot

## Batch Retention Queue

Generate a ranked daily retention queue:

```bash
make batch-score
```

Default input:

```text
data/sample_inputs/sample_customers.csv
```

Default output:

```text
data/processed/retention_queue_YYYY-MM-DD.csv
```

The queue ranks customers by calibrated churn probability, segment-specific decision threshold, expected value, and persuadability score.

## Docker Deployment

Build and run locally:

```bash
docker build -t telecom-churn-risk-engine .
docker run -p 7860:7860 telecom-churn-risk-engine
```

The Docker image installs dependencies, trains the model artifacts during build, exposes port `7860`, and starts the Streamlit dashboard.

For API-first deployment, replace the Docker `CMD` with:

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

## Hugging Face Spaces Deployment

Steps:

1. Create a new Hugging Face Space.
2. Choose Docker as the SDK.
3. Push this repository to the Space.
4. Hugging Face builds the Docker image.
5. The app starts on port `7860`.

For a portfolio demo, Streamlit is the easiest default because it shows the model, metrics, and monitoring views. For an API demo, switch the Docker `CMD` to Uvicorn.

## Monitoring Plan

Production monitoring should track:

- input drift on numeric features
- missing value rates
- prediction distribution drift
- average churn probability
- percentage of high-risk customers
- API latency
- API error rate
- delayed model performance once churn labels arrive
- business KPIs such as contacted customers, saved customers, and retention cost

Detailed monitoring notes are in `reports/monitoring_plan.md`.

## Limitations

- The project is classical ML only, not a deep learning system.
- Synthetic features are simulated when real telecom operations data is absent.
- Current fallback data is useful for demonstrating architecture, not for making real business decisions.
- Recommendation logic is rules-based and should be validated with business teams.
- Uplift modeling currently uses a synthetic two-model retention experiment baseline until real treatment/control outreach data exists.
- Local experiment tracking and registry metadata are JSON-based; MLflow can replace this for team production use.
- Real production use would require data contracts, monitoring dashboards, feature store integration, and controlled A/B testing.

## Implemented Improvements

- Optional CRM, support ticket, billing, and network outage CSV merges in `src/data_sources.py`.
- Optional SHAP local explanations with fallback explanations when `shap` is unavailable.
- Probability calibration with Brier score and expected calibration error reporting.
- Segment-specific thresholds by `customer_segment`.
- Two-model uplift/persuadability baseline and API response fields.
- Local JSON experiment tracking and model registry metadata in `models/experiment_runs/` and `models/model_registry.json`.
- GitHub Actions CI plus scheduled daily retention queue workflow.
- Batch scoring job for retention queue generation in `src/batch_scoring.py`.

## Remaining Improvements

- Replace synthetic operational and uplift labels with real production CRM, billing, support, network, outreach, and churn-label data.
- Promote local JSON tracking to MLflow or a managed registry if this becomes team-operated.
- Validate intervention impact through A/B testing before using persuadability scores for budget allocation.

## Resume Bullets

- Built an end-to-end telecom churn risk engine using Python, scikit-learn, FastAPI, Streamlit, Docker, and pytest.
- Engineered business-focused churn features from customer, billing, support, network, and sentiment signals.
- Trained baseline and tree-based classical ML models with reusable preprocessing through `ColumnTransformer`.
- Tuned the decision threshold using a business cost matrix instead of relying on a default `0.50` cutoff.
- Served churn predictions with risk bands, explainable top reasons, retention recommendations, and expected business impact.
- Added monitoring logic for missing values, feature drift, prediction drift, latency, error rate, and business KPIs.

## Interview Framing

The strongest way to present this project:

> I built a churn prediction system as a decision workflow. The model predicts risk, threshold tuning decides who should receive action, explainability tells the retention team why the customer is risky, and monitoring checks whether the system remains reliable after deployment.

Honest scope statement:

> The fallback dataset makes the project runnable end to end. In production, I would retrain on real CRM, billing, network, support, and cancellation data before making business decisions.

## Quick Start

```bash
cd telecom-churn-risk-engine
make install
make train
make test
make api
make dashboard
```
