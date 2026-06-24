# Interview Notes

## 30-Second Explanation

I built a telecom churn risk engine that predicts which customers are likely to leave, explains the main risk reasons, and recommends retention actions. It includes a scikit-learn training pipeline, threshold tuning with a business cost matrix, FastAPI endpoints, a Streamlit dashboard, Docker deployment, tests, and monitoring logic.

## 2-Minute Explanation

This project treats churn prediction as a full ML product, not just a notebook. The pipeline loads IBM Telco-style churn data, cleans it, adds realistic telecom features like complaints, support tickets, network downtime, late payments, and sentiment, then engineers business features such as complaint rate, charges per tenure, billing issue flags, and month-to-month contract flags.

I train Logistic Regression as a baseline and compare it with tree-based models like Random Forest and XGBoost or Gradient Boosting. The model is evaluated with ROC-AUC, precision, recall, F1, confusion matrix, and classification report. I do not rely on a default `0.50` threshold. Instead, I tune the threshold using a cost matrix where false negatives are much more expensive than false positives.

The prediction API returns churn probability, risk band, top reasons, recommended action, and expected business impact. A Streamlit dashboard supports single scoring, batch scoring, model metrics, threshold analysis, feature importance, and monitoring snapshots.

## Technical Deep-Dive Explanation

The project uses a modular Python structure. Data ingestion handles either the IBM Telco CSV or a compatible synthetic fallback. Data cleaning normalizes columns, converts `Churn` into a binary target, handles missing values, and standardizes types.

Feature engineering creates both numeric and categorical features. Preprocessing uses `ColumnTransformer`: numeric columns are imputed and scaled, categorical columns are imputed and one-hot encoded. This avoids fitting preprocessing on the test set and reduces data leakage risk.

The training script splits data with stratification, trains candidate models, compares performance, selects a model, tunes thresholds from `0.10` to `0.90`, and saves artifacts with `joblib`. Metrics and threshold tuning results are saved to `models/metrics.json`.

The serving layer loads the model and preprocessor, applies the same feature engineering logic, generates a churn probability, maps it to a risk band, creates explanations, and recommends an action. Tests verify feature engineering, risk band mapping, threshold tuning, API contract, and prediction output fields.

## Business Explanation For Non-Technical Interviewer

The goal is to help a telecom company decide which customers need retention attention. Instead of giving the business a raw model score only, the system says:

- how risky the customer is
- why the customer is risky
- what action the team should take
- what business impact the action may have

For example, if a customer is high risk because of network downtime and complaints, the recommended action is priority support rather than a generic discount. This makes the model more useful for real teams.

## Likely Interview Questions

### Why did you choose churn prediction?

Churn prediction is a common real business problem with clear value. It connects ML to revenue, customer experience, and operational decision-making. It also lets me demonstrate classification, feature engineering, threshold tuning, explainability, deployment, and monitoring.

### Why not use accuracy only?

Accuracy can be misleading when classes are imbalanced or when different mistakes have different costs. In churn, missing a true churner is usually more expensive than contacting a customer who would not churn. Recall, precision, and business cost are more useful than accuracy alone.

### What is ROC-AUC?

ROC-AUC measures how well the model ranks churners above non-churners across many thresholds. A value near `0.5` means random ranking. A value closer to `1.0` means stronger separation. It is useful for comparing models before choosing an operating threshold.

### What is precision?

Precision answers: out of customers predicted as churn risks, how many actually churned?

High precision means the retention queue is cleaner and fewer outreach actions are wasted.

### What is recall?

Recall answers: out of customers who actually churned, how many did the model catch?

High recall matters when missing churners is expensive. For retention, recall is often more important than accuracy.

### Why does threshold tuning matter?

The model outputs probabilities, but the business needs a yes/no action decision. The default threshold of `0.50` may not match business costs. Threshold tuning chooses a cutoff that better balances missed churners and unnecessary outreach.

### Why use Logistic Regression as a baseline?

Logistic Regression is fast, stable, and interpretable. It gives a simple benchmark. If a complex model cannot beat it, the complex model may not be worth the extra operational cost.

### Why use XGBoost or LightGBM?

Gradient boosted trees often perform well on tabular business data. They can capture nonlinear patterns and feature interactions better than simple linear models. In this project, XGBoost is used if available; otherwise Gradient Boosting is used as a dependency-light fallback.

### How did you avoid data leakage?

I split train and test data before fitting preprocessing. Imputation, scaling, and one-hot encoding are fitted only on training data through `ColumnTransformer`. The target is not used in feature engineering. Threshold tuning is performed on held-out predictions, not on training predictions.

### How would you monitor this in production?

I would monitor input drift, missing values, prediction drift, API latency, API error rate, and delayed model performance once churn labels arrive. I would also track business KPIs such as contacted customers, saved customers, offer cost, and retained revenue.

### What would you improve next?

I added optional CRM, billing, support, and network CSV ingestion, probability calibration, segment-specific thresholds, optional SHAP explanations, local experiment tracking, and a batch retention queue. Next I would replace the synthetic fallback data and synthetic uplift baseline with real outreach experiment data, then validate retention actions with A/B testing.

## Strong Talking Points

- The project turns model output into a business workflow.
- Threshold tuning is based on cost, not default probability cutoff.
- Explanations are written in language a retention team can understand.
- Monitoring includes both ML health and API health.
- The fallback synthetic data is clearly labeled and not overclaimed.

## Follow-Up Questions To Be Ready For

### Why is the selected threshold low?

The cost matrix makes false negatives much more expensive than false positives. A lower threshold catches more churners, which improves recall. The tradeoff is more false positives, so the business must make sure the retention team can handle the larger outreach queue.

### Would you deploy this exact model to production?

No. I would treat this as a production-style project structure and demo. Before real deployment, I would retrain on real CRM, billing, network, support, outreach, and churn-label data; validate calibration on delayed labels; analyze fairness and segments; and run an A/B test for retention actions.

### What makes this more than a Kaggle-style notebook?

It includes reusable training code, saved artifacts, threshold tuning, an API, a dashboard, tests, Docker deployment, monitoring logic, documentation, and a business-facing recommendation layer.

## Honest Limitations To Mention

- The base churn dataset is the public IBM Telco CSV; operational add-on features are still simulated unless real CRM, billing, support, and network extracts are provided.
- Real production use needs actual labeled telecom data.
- Rule-based recommendations should be validated by domain experts.
- SHAP is optional because it adds dependency weight; the code falls back to stable business-rule explanations.
- The uplift baseline should be replaced with real treatment/control data before business budget allocation.
- Business value should be proven with controlled experiments.
