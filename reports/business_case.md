# Business Case

## Why Churn Prediction Matters

Customer churn is one of the most important business problems for telecom companies. Telecom revenue is recurring, so every lost customer reduces future monthly revenue. Acquiring a new customer is usually more expensive than retaining an existing one, especially in competitive markets where discounts, device subsidies, and marketing spend are high.

A churn model helps the business move from reactive cancellation handling to proactive retention.

## How Churn Affects Telecom Business

Churn affects several parts of the business:

- Revenue loss: cancelled customers stop generating monthly recurring revenue.
- Acquisition pressure: marketing and sales teams must replace lost customers.
- Margin pressure: replacement customers may require promotional offers.
- Customer experience signal: rising churn can indicate billing, support, pricing, or network problems.
- Capacity planning: churn risk can help retention teams prioritize limited outreach capacity.

For a telecom company, churn is not only a model target. It is a commercial risk signal.

## Why Risk Bands Matter More Than Raw Probabilities

A probability such as `0.73` is useful for ranking, but business teams usually need simpler operating categories:

```text
0.00 - 0.30  Low risk
0.30 - 0.65  Medium risk
0.65 - 1.00  High risk
```

Risk bands help teams decide what action to take:

- Low risk: normal engagement.
- Medium risk: proactive support message or personalized plan recommendation.
- High risk: retention call, discount review, billing support, or priority network support.

This makes the model easier to operationalize for sales, support, and customer success teams.

## Risk Band Operating Policy

| Risk band | Probability range | Business action | Why this works |
| --- | ---: | --- | --- |
| Low | `0.00 - 0.30` | Normal engagement | Avoids spending retention budget on low-risk accounts |
| Medium | `0.30 - 0.65` | Plan recommendation or proactive support | Addresses early warning signs before risk becomes severe |
| High | `0.65 - 1.00` | Retention call, discount review, billing help, priority network support | Focuses human effort on customers most likely to leave |

This policy is intentionally simple. In production, the business could tune actions by customer lifetime value, offer eligibility, plan type, and retention team capacity.

## Why Action Recommendation Matters

Prediction alone does not create business value. Value comes when a team acts on the prediction.

This project links risk reasons to actions:

- If risk is driven by complaints or network downtime, route the customer to priority support.
- If risk is driven by billing disputes or late payments, offer billing help.
- If risk is driven by high charges, suggest a better-fit plan or discount review.
- If risk is driven by month-to-month contract, offer a contract upgrade incentive.

This turns the model from a scoring tool into a retention recommendation engine.

## Cost-Based Threshold Tuning

Most classifiers use a default threshold of `0.50`. That is rarely the best business decision.

For churn, missing a real churner is usually more expensive than contacting a customer who would not have churned. The project uses a cost matrix:

```text
False Negative = ₹5,000
False Positive = ₹500
```

Interpretation:

- False Negative: model predicts low risk, but customer churns. Business loses revenue and misses chance to intervene.
- False Positive: model predicts high risk, but customer would not churn. Business spends on unnecessary outreach or offer.

Because false negatives are more expensive, the selected threshold may be lower than `0.50` to catch more churners. This usually improves recall but can reduce precision. The right threshold depends on retention capacity and intervention cost.

## Example Business Interpretation

Example model output:

```json
{
  "customer_id": "CUST-001",
  "churn_probability": 0.82,
  "risk_band": "High",
  "top_reasons": [
    "month-to-month contract",
    "high complaints",
    "recent network downtime"
  ],
  "recommended_action": "Retention call plus priority network support and service credit review.",
  "expected_business_impact": {
    "expected_loss_without_action": 4100,
    "estimated_retention_cost": 400,
    "net_expected_value": 420
  }
}
```

Business interpretation:

This customer is high risk. The main drivers are contract flexibility, recent complaints, and service quality issues. A generic discount may not be enough; the better action is a retention call combined with network support follow-up. If the customer is saved, the expected value of intervention can exceed the cost of outreach.

## Practical Business Use

A retention team could use this project to create a daily customer queue:

1. Score active customers.
2. Sort by churn probability or expected business value.
3. Filter to medium and high-risk customers.
4. Route customers by top reason.
5. Track contact outcome, saved customers, offer cost, and later churn labels.

This is the difference between a model and an operating workflow.

## What A Stakeholder Should Take Away

The model is not valuable because it predicts a label. It is valuable if it helps the business contact the right customers, with the right reason, at the right cost. That is why the project includes risk bands, action recommendations, threshold tuning, and business impact fields.
