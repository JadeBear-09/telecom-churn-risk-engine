# Monitoring Plan

A churn model can perform well during offline evaluation and still become unreliable in production. Monitoring is needed because customer behavior, pricing, network quality, support processes, and data pipelines change over time.

## Monitoring Summary

| Area | What to monitor | Example action |
| --- | --- | --- |
| Input data | schema, missing values, feature distributions | pause scoring or investigate upstream data feed |
| Predictions | score distribution, average score, high-risk share | check for drift or business incident |
| Performance | ROC-AUC, precision, recall, calibration after labels arrive | retrain or adjust threshold |
| API health | latency, error rate, request volume | scale service or roll back deployment |
| Business KPIs | contacted customers, saved customers, offer cost, retained revenue | adjust outreach strategy |

## Input Drift

Input drift means feature distributions change after deployment.

Examples:

- monthly charges increase after pricing changes
- network downtime rises after outages
- support ticket patterns change after new support tooling
- customer segment mix changes after marketing campaigns

Planned checks:

- PSI for numeric features
- distribution comparison for categorical features
- mean, median, min, max, and percentile shifts
- schema validation for expected columns and data types

Important monitored features:

- `tenure`
- `monthly_charges`
- `total_charges`
- `complaint_count_90d`
- `support_ticket_count_90d`
- `network_downtime_minutes_30d`
- `late_payment_count_6m`
- `avg_data_usage_gb_30d`

## Prediction Drift

Prediction drift means model outputs change even if model code has not changed.

Monitor:

- average churn probability
- median churn probability
- churn probability distribution by decile
- percentage of low, medium, and high-risk customers
- top reasons frequency

Example concern:

If high-risk customers suddenly increase by 30%, the cause may be real business stress, data drift, schema bugs, or model input corruption. The system should alert before retention teams act on a potentially broken queue.

## Missing Value Monitoring

Missing values often reveal upstream data pipeline issues.

Monitor:

- missing rate by feature
- missing rate by data source
- sudden changes from previous day or week
- required fields missing from API requests

Example alert:

```text
ALERT: missing value rate for monthly_charges crossed 10%.
Action: check billing data feed and recent schema changes.
```

## Model Performance Monitoring

True churn labels arrive later, so performance monitoring is delayed. Once labels are available, track:

- ROC-AUC
- precision
- recall
- F1
- confusion matrix
- calibration
- business cost at selected threshold
- performance by region and customer segment

Offline validation should be repeated on fresh labeled data. If the model catches fewer true churners over time, retention value drops even if API uptime remains good.

## Latency Monitoring

The API should respond fast enough for support and dashboard usage.

Monitor:

- p50 latency
- p95 latency
- p99 latency
- timeout rate

Example alert:

```text
ALERT: p95 prediction API latency exceeded 500 ms for 15 minutes.
Action: check model server load, dependency health, and recent deployment.
```

## Error Rate Monitoring

Monitor:

- HTTP 4xx validation errors
- HTTP 5xx server errors
- failed batch predictions
- model artifact loading failures
- schema mismatch errors

High validation errors may indicate that upstream systems changed field names, categories, or payload structure.

Example alert:

```text
ALERT: prediction API error rate exceeded 2%.
Action: inspect request logs and validate input schema.
```

## Business KPI Monitoring

Model monitoring should connect to business outcomes.

Track:

- number of customers scored
- number of customers contacted
- percentage of high-risk customers contacted
- offer acceptance rate
- saved customers
- retention cost
- net revenue retained
- churn rate by contacted versus non-contacted groups

Offline model metrics do not prove business value. Business value should be measured through A/B testing or careful comparison groups.

## Retraining Triggers

Retrain or review the model when:

- input drift persists for multiple days
- prediction distribution shifts sharply
- high-risk customer share changes by more than 30%
- missing values cross 10% for important fields
- recall drops below the business target
- precision drops enough to overload retention teams
- new product plans or pricing changes launch
- enough new churn labels arrive
- business cost at selected threshold increases materially

## Example Alerts

```text
ALERT: high-risk customer percentage increased by 30% day over day.
Possible causes: real churn risk increase, outage, pricing change, data bug.
```

```text
ALERT: missing values crossed 10% for support_ticket_count_90d.
Possible causes: support platform feed failed or schema changed.
```

```text
ALERT: average churn score shifted from 0.34 to 0.52 in one day.
Possible causes: model input drift, changed customer mix, upstream data issue.
```

```text
ALERT: API p95 latency exceeded 500 ms.
Possible causes: server overload, slow artifact loading, dependency issue.
```

## Alert Severity Guide

| Severity | Example | Response |
| --- | --- | --- |
| Low | one feature PSI enters watch range | review in next monitoring cycle |
| Medium | average churn score shifts sharply for one day | compare input data against reference window |
| High | missing values cross 10% for critical fields | investigate source pipeline before trusting scores |
| Critical | API error rate spikes or model artifacts fail to load | roll back deployment or disable automated scoring |

## Operational Response

For each alert, the team should:

1. Check whether the issue is data, model, API, or real business movement.
2. Compare current data with a known good reference window.
3. Validate recent deployments and upstream schema changes.
4. Temporarily fall back to a previous model or threshold if needed.
5. Document the incident and update monitoring rules if needed.
