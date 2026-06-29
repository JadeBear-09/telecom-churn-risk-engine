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

# Telecom Churn Risk Engine

<p align="center">
  <a href="https://github.com/JadeBear-09/telecom-churn-risk-engine/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/JadeBear-09/telecom-churn-risk-engine?style=social"></a>
  <a href="https://github.com/JadeBear-09/telecom-churn-risk-engine/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/JadeBear-09/telecom-churn-risk-engine/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Last commit" src="https://img.shields.io/github/last-commit/JadeBear-09/telecom-churn-risk-engine">
  <img alt="Issues" src="https://img.shields.io/github/issues/JadeBear-09/telecom-churn-risk-engine">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white">
</p>

<p align="center">
  <a href="https://jade09-telecom-churn-risk-engine.hf.space">Live demo</a>
  · <a href="#quick-start">Quick start</a>
  · <a href="#api-contract">API</a>
  · <a href="#model-snapshot">Model snapshot</a>
  · <a href="docs/COMMIT_GUIDE.md">Commit guide</a>
</p>

End-to-end telecom churn workflow that turns customer records into risk bands,
plain-language drivers, and retention actions. The repo is built to be readable in
interviews: training code, serving API, dashboard, tests, model card, monitoring notes,
and CI all live in one place.

## Why This Exists

Telecom teams cannot contact every customer. This project ranks customers by churn
risk, explains why each customer is risky, and recommends the next retention action.

```text
customer data -> churn probability -> risk band -> top reasons -> retention action
```

Each scored customer receives:

- churn probability and tuned decision threshold
- `Low`, `Medium`, or `High` risk band
- top business-readable risk reasons
- recommended retention action
- estimated business impact
- uplift tier and persuadability score for outreach prioritization

## What Ships

| Area | Implementation |
| --- | --- |
| Training | Reproducible scikit-learn pipeline with feature engineering and threshold tuning |
| Serving | FastAPI service with single, batch, health, metric, and customer-risk endpoints |
| Review UI | Streamlit dashboard for single scoring, batch uploads, metrics, and monitoring |
| Business layer | Cost-aware thresholding, risk bands, retention recommendations, model card |
| Reliability | Pytest suite, GitHub Actions CI, Docker build, monitoring checks |
| Artifacts | Model metrics snapshot, sample payloads, notebook stubs, reports |

## Architecture

```text
IBM Telco CSV or synthetic fallback
        |
        v
data cleaning + telecom feature engineering
        |
        v
scikit-learn preprocessing and model comparison
        |
        v
calibration + cost-aware threshold tuning
        |
        v
saved model artifacts + metrics
        |
        +--------------------------+
        |                          |
        v                          v
FastAPI scoring service       Streamlit review dashboard
```

## Model Snapshot

| Metric | Value |
| --- | ---: |
| Dataset | IBM Telco Customer Churn, `7,043` rows |
| Selected model | Logistic Regression |
| ROC-AUC | `0.841` |
| Recall | `0.922` |
| Precision | `0.425` |
| Tuned global threshold | `0.13` |
| Threshold policy | False negatives cost `10x` false positives |

The low threshold is intentional: in this business framing, missing likely churners is
more expensive than contacting some customers who would have stayed anyway.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
make install
make train
make test
```

Run the API:

```bash
make api
```

Run the dashboard:

```bash
make dashboard
```

Default local URLs:

```text
FastAPI:   http://localhost:8000
Streamlit: http://localhost:8501
```

Docker:

```bash
docker build -t telecom-churn-risk-engine .
docker run -p 7860:7860 telecom-churn-risk-engine
```

## API Contract

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health |
| `POST` | `/predict` | Score one customer |
| `POST` | `/batch_predict` | Score a batch of telecom customers |
| `GET` | `/model/metrics` | Return model metric snapshot |
| `GET` | `/customer/{customer_id}/risk` | Return stored/sample customer risk |

Prediction responses include:

```text
customer_id
churn_probability
decision_threshold
churn_prediction
risk_band
top_reasons
persuadability_score
uplift_tier
recommended_action
expected_business_impact
```

## Dashboard

The Streamlit dashboard supports:

- single customer scoring
- telecom CSV batch scoring
- model metric review
- confusion matrix and risk distribution
- feature importance
- threshold tuning view
- calibration metrics
- segment thresholds
- monitoring snapshot

Batch upload is intentionally strict. Non-telecom CSV files are rejected instead of
being scored with meaningless defaults.

## Quality Gates

```bash
make test
python -m src.evaluate
python -m src.batch_scoring
```

CI trains model artifacts and runs the pytest suite on every push and pull request.
A scheduled workflow can generate the daily retention queue artifact.

## Project Map

```text
app/          FastAPI app, schemas, prediction, explanations, recommendations
src/          data ingestion, cleaning, features, training, evaluation, monitoring
dashboard/    Streamlit app
tests/        pytest suite
reports/      model card, business case, monitoring plan
notebooks/    lightweight EDA/training notebook stubs
data/         sample inputs and generated processed outputs
models/       generated model artifacts and metrics snapshot
```

## Deeper Review

- [Model card](reports/model_card.md)
- [Business case](reports/business_case.md)
- [Monitoring plan](reports/monitoring_plan.md)
- [Commit guide](docs/COMMIT_GUIDE.md)
- [Contributing guide](CONTRIBUTING.md)

## Roadmap

- Add real outreach experiment data for uplift validation
- Add model registry integration and artifact versioning
- Add production data contracts and drift dashboards
- Add richer SHAP-based explanations behind an optional dependency
- Add A/B testing hooks for retention campaign measurement

## Scope And Limits

- This is a classical ML project, not deep learning.
- Public IBM Telco data is used for the demo.
- Operational fields such as support tickets, complaints, outages, and sentiment are
  simulated unless real source CSVs are supplied.
- Recommendations are rule-based and need domain validation before production use.
- Uplift modeling uses a synthetic treatment/control baseline until real outreach
  experiment data exists.
- Real deployment would need production data contracts, access control, observability,
  model registry, and A/B testing.
