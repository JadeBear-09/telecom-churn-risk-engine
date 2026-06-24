# Concept Guide

This guide explains the main ML concepts used in the churn project in beginner-friendly language.

## How These Concepts Connect In This Project

The churn engine is a classification system. It uses a train/test split to evaluate performance, feature engineering to create better input signals, metrics like recall and precision to understand mistakes, threshold tuning to convert probabilities into business decisions, explainability to make predictions understandable, and monitoring to check whether the system stays reliable after deployment.

## Classification

Classification is a machine learning task where the model predicts a category. In this project, the categories are:

- `0`: customer does not churn
- `1`: customer churns

The model produces a churn probability, then a threshold converts that probability into a class decision.

## Confusion Matrix

A confusion matrix compares model predictions with actual outcomes.

```text
                    Predicted No Churn   Predicted Churn
Actual No Churn     True Negative        False Positive
Actual Churn        False Negative       True Positive
```

It helps explain what types of mistakes the model makes.

## True Positive

A true positive happens when the customer actually churns and the model predicts churn.

In this project:

```text
Model says high risk, and customer churns.
```

This is a correct churn warning.

## False Positive

A false positive happens when the customer does not churn, but the model predicts churn.

In this project:

```text
Model says high risk, but customer stays.
```

This may cause unnecessary retention outreach.

## True Negative

A true negative happens when the customer does not churn and the model predicts no churn.

In this project:

```text
Model says low risk, and customer stays.
```

This is a correct non-churn prediction.

## False Negative

A false negative happens when the customer churns, but the model predicts no churn.

In this project:

```text
Model says low risk, but customer leaves.
```

This is usually the most expensive mistake because the business misses a chance to intervene.

## Precision

Precision measures how many predicted churn risks were actually churners.

```text
precision = true positives / (true positives + false positives)
```

Business meaning:

Out of customers the retention team contacts, how many were truly at risk?

High precision means less wasted outreach.

## Recall

Recall measures how many actual churners the model caught.

```text
recall = true positives / (true positives + false negatives)
```

Business meaning:

Out of all customers who would churn, how many did the model identify?

High recall matters when missed churners are expensive.

## F1 Score

F1 combines precision and recall into one number.

```text
F1 = 2 * (precision * recall) / (precision + recall)
```

It is useful when both false positives and false negatives matter. It is not always the final business metric, but it helps compare models.

## ROC-AUC

ROC-AUC measures how well the model ranks churners above non-churners across thresholds.

Simple interpretation:

- `0.50`: random ranking
- `0.70`: useful separation
- `0.90`: strong separation
- `1.00`: perfect ranking

ROC-AUC is useful for model comparison before choosing a threshold.

## Threshold

A threshold converts probability into a decision.

Example:

```text
churn probability = 0.62
threshold = 0.50
prediction = churn
```

If the threshold is lower, the model catches more churners but creates more false positives. If the threshold is higher, the model becomes more selective but may miss churners.

## Feature Engineering

Feature engineering means creating useful input variables from raw data.

Example:

Raw fields:

- `complaint_count_90d`
- `tenure`
- `monthly_charges`

Engineered features:

- `complaints_per_month`
- `charges_per_tenure`
- `billing_issue_flag`

Good feature engineering helps classical ML models learn stronger business patterns.

## Train/Test Split

A train/test split separates data into two parts:

- training data: used to fit the model
- test data: used to evaluate how the model performs on unseen data

This helps estimate how the model may behave on future customers.

## Data Leakage

Data leakage happens when the model sees information during training that would not be available at prediction time.

Examples:

- using cancellation date as an input feature
- fitting preprocessing on the full dataset before train/test split
- using future support tickets to predict current churn risk

Data leakage makes offline metrics look better than real production performance.

## SHAP And Explainability

Explainability helps humans understand why a model made a prediction.

SHAP is a popular technique that estimates how much each feature contributed to a prediction. For example, it can show that month-to-month contract and recent complaints increased churn risk.

This project uses a lighter fallback approach:

- map risky feature values to business-language reasons
- use feature importances or model coefficients when available

## Model Drift

Model drift happens when model performance gets worse over time.

Reasons:

- customer behavior changes
- pricing changes
- competitors launch new offers
- network quality changes
- support process changes

Drift is detected by monitoring later churn labels and comparing fresh performance against historical performance.

## Prediction Drift

Prediction drift happens when model output distribution changes.

Example:

```text
Average churn probability was 0.32 last month.
Average churn probability is 0.55 today.
```

This may be caused by real business changes or data pipeline issues. It should trigger investigation.

## Business Cost Matrix

A business cost matrix assigns costs to model mistakes.

For churn:

```text
False Negative = ₹5,000
False Positive = ₹500
```

False negatives are missed churners. False positives are unnecessary retention actions.

Threshold tuning uses this matrix to choose a decision threshold that better matches business value.
