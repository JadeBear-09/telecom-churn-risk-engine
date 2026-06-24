from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def write_experiment_run(
    model_dir: Path,
    metrics: dict[str, Any],
    params: dict[str, Any],
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    """Write lightweight local experiment tracking and registry metadata."""
    experiment_dir = model_dir / "experiment_runs"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "metrics": {
            "model_name": metrics.get("model_name"),
            "roc_auc": metrics.get("roc_auc"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "selected_threshold": metrics.get("selected_threshold"),
            "total_business_cost": metrics.get("business_cost", {}).get("total_cost"),
            "brier_score": metrics.get("calibration", {}).get("test", {}).get("brier_score"),
            "expected_calibration_error": metrics.get("calibration", {})
            .get("test", {})
            .get("expected_calibration_error"),
        },
        "artifact_paths": artifact_paths,
    }

    run_path = experiment_dir / f"{run_id}.json"
    run_path.write_text(json.dumps(run_record, indent=2))

    registry_path = model_dir / "model_registry.json"
    registry = {"models": []}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())

    registry_entry = {
        "model_name": "telecom_churn_risk_model",
        "version": len(registry.get("models", [])) + 1,
        "stage": "candidate",
        "run_id": run_id,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "business_cost_then_roc_auc",
        "metrics": run_record["metrics"],
        "artifact_paths": artifact_paths,
    }
    registry.setdefault("models", []).append(registry_entry)
    registry["latest_version"] = registry_entry["version"]
    registry["latest_run_id"] = run_id
    registry_path.write_text(json.dumps(registry, indent=2))

    return {
        "run_id": run_id,
        "run_path": str(run_path),
        "registry_path": str(registry_path),
        "model_version": registry_entry["version"],
    }
