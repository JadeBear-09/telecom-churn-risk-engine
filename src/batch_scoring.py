from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from app.predict import PROJECT_ROOT, predict_one
from app.schemas import CustomerInput


RISK_RANK = {"High": 3, "Medium": 2, "Low": 1}


def _score_row(row: pd.Series, idx: int) -> dict[str, Any]:
    record = row.to_dict()
    record.setdefault("customer_id", f"BATCH-{idx:06d}")
    result = predict_one(CustomerInput(**record))
    impact = result["expected_business_impact"]
    return {
        "customer_id": result["customer_id"],
        "churn_probability": result["churn_probability"],
        "risk_band": result["risk_band"],
        "churn_prediction": result["churn_prediction"],
        "decision_threshold": result["decision_threshold"],
        "persuadability_score": result["persuadability_score"],
        "uplift_tier": result["uplift_tier"],
        "net_expected_value": impact["net_expected_value"],
        "recommended_action": result["recommended_action"],
        "top_reasons": "; ".join(result["top_reasons"]),
    }


def generate_retention_queue(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    if input_path is None:
        input_path = PROJECT_ROOT / "data" / "sample_inputs" / "sample_customers.csv"
    input_path = Path(input_path)
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "processed" / f"retention_queue_{date.today().isoformat()}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(input_path)
    scored = pd.DataFrame([_score_row(row, idx) for idx, row in customers.iterrows()])
    scored["risk_rank"] = scored["risk_band"].map(RISK_RANK).fillna(0).astype(int)
    scored["queue_score"] = (
        scored["churn_probability"] * 0.55
        + scored["persuadability_score"] * 0.30
        + (scored["net_expected_value"].clip(lower=0) / 5000.0) * 0.15
    )

    queue = scored[(scored["churn_prediction"] == 1) | (scored["risk_band"].isin(["Medium", "High"]))].copy()
    queue = queue.sort_values(["queue_score", "net_expected_value"], ascending=[False, False])
    if top_n is not None:
        queue = queue.head(top_n)
    queue.to_csv(output_path, index=False)
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily retention outreach queue.")
    parser.add_argument("--input", dest="input_path", default=None, help="Customer CSV path. Defaults to sample customers.")
    parser.add_argument("--output", dest="output_path", default=None, help="Queue CSV path.")
    parser.add_argument("--top-n", dest="top_n", type=int, default=None, help="Limit queue rows.")
    args = parser.parse_args()

    queue = generate_retention_queue(args.input_path, args.output_path, args.top_n)
    print(f"Generated retention queue rows={len(queue)}")


if __name__ == "__main__":
    main()
