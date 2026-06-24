from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.predict import MODEL_DIR, load_artifacts, predict_batch as score_batch, predict_one
from app.schemas import BatchPredictionRequest, CustomerInput, PredictionResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="Telecom Customer Churn Risk Scoring API",
    version="1.0.0",
    description="Classical ML churn risk scoring with explanations and retention recommendations.",
)


@app.get("/health")
def health() -> dict[str, object]:
    model_ready = (MODEL_DIR / "churn_model.pkl").exists() and (MODEL_DIR / "preprocessor.pkl").exists()
    return {"status": "ok", "model_ready": model_ready}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput) -> dict[str, object]:
    try:
        return predict_one(customer)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/batch_predict", response_model=list[PredictionResponse])
def batch_predict(request: BatchPredictionRequest) -> list[dict[str, object]]:
    try:
        return score_batch(request.customers)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/model/metrics")
def model_metrics() -> dict[str, object]:
    try:
        return load_artifacts()["metrics"]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/customer/{customer_id}/risk", response_model=PredictionResponse)
def customer_risk(customer_id: str) -> dict[str, object]:
    sample_path = PROJECT_ROOT / "data" / "sample_inputs" / "sample_customers.csv"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample customer data missing. Run `make train` first.")

    sample_df = pd.read_csv(sample_path)
    matches = sample_df[sample_df["customer_id"].astype(str) == str(customer_id)]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Customer not found: {customer_id}")

    record = matches.iloc[0].to_dict()
    record["customer_id"] = str(record["customer_id"])
    try:
        return predict_one(CustomerInput(**record))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

