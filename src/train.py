from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.business_cost import total_business_cost
from src.calibration import calibration_summary
from src.data_cleaning import clean_telco_data
from src.data_ingestion import load_telco_data
from src.data_sources import merge_operational_sources
from src.experiment_tracking import write_experiment_run
from src.feature_engineering import (
    FEATURE_COLUMNS,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
    engineer_features,
    get_feature_columns,
    split_features_target,
)
from src.segment_thresholds import select_segment_thresholds, thresholds_for_segments
from src.synthetic_features import add_synthetic_features
from src.threshold_tuning import select_optimal_threshold
from src.uplift_modeling import train_uplift_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_inputs"


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def _candidate_models(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
    except Exception:
        models["gradient_boosting"] = GradientBoostingClassifier(random_state=random_state)
    return models


def _probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    scores = model.decision_function(features)
    return 1 / (1 + np.exp(-scores))


def _calibrate_prefit_model(model: Any, X_calibration: np.ndarray, y_calibration: pd.Series) -> CalibratedClassifierCV:
    try:
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    except (ImportError, AttributeError):
        calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrator.fit(X_calibration, y_calibration)
    return calibrator


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return FEATURE_COLUMNS


def _unwrap_estimator(model: Any) -> Any:
    if model.__class__.__name__ == "FrozenEstimator" and hasattr(model, "estimator"):
        return _unwrap_estimator(model.estimator)
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        estimator = getattr(calibrated[0], "estimator", None) or getattr(calibrated[0], "base_estimator", None)
        if estimator is not None:
            return _unwrap_estimator(estimator)
    return model


def global_feature_importance(model: Any, preprocessor: ColumnTransformer, top_n: int = 25) -> list[dict[str, Any]]:
    names = get_feature_names(preprocessor)
    estimator = _unwrap_estimator(model)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_[0], dtype=float))
    else:
        return []

    ranked_idx = np.argsort(values)[::-1][:top_n]
    return [
        {"feature": names[idx], "importance": float(values[idx])}
        for idx in ranked_idx
        if idx < len(names)
    ]


def prepare_training_data(data_path: str | Path | None = None, n_samples: int = 3000) -> pd.DataFrame:
    raw = load_telco_data(data_path=data_path, n_samples=n_samples)
    cleaned = clean_telco_data(raw)
    sourced = merge_operational_sources(cleaned)
    enriched = add_synthetic_features(sourced, overwrite=False)
    engineered = engineer_features(enriched)
    return engineered


def train_pipeline(
    data_path: str | Path | None = None,
    n_samples: int = 3000,
    random_state: int = 42,
) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_training_data(data_path=data_path, n_samples=n_samples)
    df.to_csv(PROCESSED_DIR / "training_dataset.csv", index=False)

    X, y = split_features_target(df)
    numeric_features, categorical_features = get_feature_columns()
    X_train_calibration, X_test, y_train_calibration, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_calibration, y_train, y_calibration = train_test_split(
        X_train_calibration,
        y_train_calibration,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_calibration,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_calibration_transformed = preprocessor.transform(X_calibration)
    X_test_transformed = preprocessor.transform(X_test)

    model_results = []
    best_model_name = ""
    best_model = None
    best_score = -np.inf
    for model_name, model in _candidate_models(random_state).items():
        model.fit(X_train_transformed, y_train)
        probability = _probabilities(model, X_calibration_transformed)
        prediction = (probability >= 0.5).astype(int)
        roc_auc = roc_auc_score(y_calibration, probability)
        result = {
            "model_name": model_name,
            "roc_auc": float(roc_auc),
            "selection_split": "calibration",
            "f1_at_0_50": float(f1_score(y_calibration, prediction, zero_division=0)),
            "recall_at_0_50": float(recall_score(y_calibration, prediction, zero_division=0)),
            "precision_at_0_50": float(precision_score(y_calibration, prediction, zero_division=0)),
        }
        model_results.append(result)
        selection_score = roc_auc + result["recall_at_0_50"] * 0.05
        if selection_score > best_score:
            best_score = selection_score
            best_model_name = model_name
            best_model = model

    if best_model is None:
        raise RuntimeError("No model trained successfully.")

    calibrated_model = _calibrate_prefit_model(best_model, X_calibration_transformed, y_calibration)
    calibration_probability = _probabilities(calibrated_model, X_calibration_transformed)
    selected_threshold, threshold_rows = select_optimal_threshold(y_calibration.to_numpy(), calibration_probability)
    segment_thresholds = select_segment_thresholds(
        y_calibration.to_numpy(),
        calibration_probability,
        X_calibration["customer_segment"],
        default_threshold=selected_threshold,
    )

    y_probability = _probabilities(calibrated_model, X_test_transformed)
    test_thresholds = thresholds_for_segments(
        X_test["customer_segment"],
        segment_thresholds,
        default_threshold=selected_threshold,
    )
    y_prediction = (y_probability >= test_thresholds).astype(int)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_probability)
    business_cost = total_business_cost(y_test.to_numpy(), y_prediction)
    uplift_artifact = train_uplift_model(X_train_transformed, X_train, y_train.to_numpy(), random_state=random_state)

    metrics: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": best_model_name,
        "calibrated_probabilities": True,
        "rows": {
            "train": int(len(X_train)),
            "calibration": int(len(X_calibration)),
            "test": int(len(X_test)),
            "total": int(len(X)),
        },
        "selected_threshold": float(selected_threshold),
        "threshold_policy": {
            "false_negative_cost": 5000,
            "false_positive_cost": 500,
            "min_recall_target": 0.65,
            "threshold_tuning_split": "calibration",
            "test_decision_threshold": "segment_specific",
        },
        "segment_thresholds": segment_thresholds,
        "accuracy": float(accuracy_score(y_test, y_prediction)),
        "roc_auc": float(roc_auc_score(y_test, y_probability)),
        "precision": float(precision_score(y_test, y_prediction, zero_division=0)),
        "recall": float(recall_score(y_test, y_prediction, zero_division=0)),
        "f1": float(f1_score(y_test, y_prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_prediction, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, y_prediction, output_dict=True, zero_division=0),
        "business_cost": business_cost,
        "calibration": {
            "calibration": calibration_summary(y_calibration.to_numpy(), calibration_probability),
            "test": calibration_summary(y_test.to_numpy(), y_probability),
        },
        "precision_recall_curve": {
            "precision": [float(x) for x in pr_precision],
            "recall": [float(x) for x in pr_recall],
            "thresholds": [float(x) for x in pr_thresholds],
        },
        "threshold_tuning": threshold_rows,
        "candidate_models": model_results,
        "uplift_model": {
            "available": bool(uplift_artifact.get("available")),
            "method": uplift_artifact.get("method"),
            "uses_synthetic_experiment": bool(uplift_artifact.get("uses_synthetic_experiment")),
            "treatment_rate": uplift_artifact.get("treatment_rate"),
        },
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_importance": global_feature_importance(calibrated_model, preprocessor),
    }

    sample_cols = ["customer_id"] + RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES
    existing_sample_cols = [col for col in sample_cols if col in df.columns]
    sample_customers = df.loc[X_test.index, existing_sample_cols].head(50)
    sample_customers.to_csv(SAMPLE_DIR / "sample_customers.csv", index=False)
    sample_customers.head(1).to_json(SAMPLE_DIR / "single_customer.json", orient="records", indent=2)

    background_size = min(100, X_train_transformed.shape[0])
    joblib.dump(
        {
            "features": np.asarray(X_train_transformed[:background_size]),
            "feature_names": get_feature_names(preprocessor),
        },
        MODEL_DIR / "explainer_background.pkl",
    )
    joblib.dump(uplift_artifact, MODEL_DIR / "uplift_model.pkl")
    joblib.dump(calibrated_model, MODEL_DIR / "churn_model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
    metrics["experiment_tracking"] = write_experiment_run(
        MODEL_DIR,
        metrics,
        params={
            "n_samples": n_samples,
            "random_state": random_state,
            "calibration_method": "sigmoid",
            "threshold_policy": metrics["threshold_policy"],
        },
        artifact_paths={
            "model": str(MODEL_DIR / "churn_model.pkl"),
            "preprocessor": str(MODEL_DIR / "preprocessor.pkl"),
            "metrics": str(MODEL_DIR / "metrics.json"),
            "uplift_model": str(MODEL_DIR / "uplift_model.pkl"),
            "explainer_background": str(MODEL_DIR / "explainer_background.pkl"),
        },
    )
    (MODEL_DIR / "metrics.json").write_text(json.dumps(_to_jsonable(metrics), indent=2))
    return metrics


def main() -> None:
    metrics = train_pipeline()
    print(
        "Training complete "
        f"model={metrics['model_name']} "
        f"roc_auc={metrics['roc_auc']:.3f} "
        f"threshold={metrics['selected_threshold']:.2f}"
    )


if __name__ == "__main__":
    main()
