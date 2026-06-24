from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def ensure_model_artifacts():
    model_path = PROJECT_ROOT / "models" / "churn_model.pkl"
    preprocessor_path = PROJECT_ROOT / "models" / "preprocessor.pkl"
    if not model_path.exists() or not preprocessor_path.exists():
        from src.train import train_pipeline

        train_pipeline(n_samples=800)
