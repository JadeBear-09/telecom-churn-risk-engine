from __future__ import annotations

import pandas as pd

from app.schemas import CustomerInput


OPTIONAL_BATCH_COLUMNS = {"customer_id"}
BATCH_COLUMNS = list(CustomerInput.model_fields)
BATCH_REQUIRED_COLUMNS = [col for col in BATCH_COLUMNS if col not in OPTIONAL_BATCH_COLUMNS]
BATCH_COLUMN_SET = set(BATCH_COLUMNS)


def validate_batch_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    provided = set(df.columns)
    missing = [col for col in BATCH_REQUIRED_COLUMNS if col not in provided]
    unexpected = [col for col in df.columns if col not in BATCH_COLUMN_SET]
    return missing, unexpected


def format_missing_columns_error(missing: list[str]) -> str:
    missing_preview = ", ".join(missing[:12])
    if len(missing) > 12:
        missing_preview += f", ... ({len(missing)} total)"
    return (
        "This app accepts only telecom customer churn CSV files. "
        f"Missing required columns: {missing_preview}"
    )
