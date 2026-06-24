from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from src.feature_engineering import split_features_target
from src.train import MODEL_DIR, PROJECT_ROOT, _probabilities, prepare_training_data
from src.segment_thresholds import thresholds_for_segments


def evaluate_saved_model() -> dict[str, object]:
    model_path = MODEL_DIR / "churn_model.pkl"
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    metrics_path = MODEL_DIR / "metrics.json"
    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError("Model artifacts missing. Run `make train` first.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    threshold = 0.5
    segment_thresholds = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        threshold = metrics.get("selected_threshold", 0.5)
        segment_thresholds = metrics.get("segment_thresholds")

    df = prepare_training_data()
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    probs = _probabilities(model, preprocessor.transform(X_test))
    thresholds = thresholds_for_segments(X_test["customer_segment"], segment_thresholds, default_threshold=threshold)
    preds = (probs >= thresholds).astype(int)
    report = {
        "threshold": float(threshold),
        "uses_segment_thresholds": bool(segment_thresholds),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }
    output_path = PROJECT_ROOT / "models" / "evaluation_latest.json"
    output_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    result = evaluate_saved_model()
    print(json.dumps(result, indent=2))
