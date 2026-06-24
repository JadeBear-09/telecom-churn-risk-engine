from __future__ import annotations

import pandas as pd

from app.batch_validation import BATCH_COLUMNS, format_missing_columns_error, validate_batch_columns


def test_validate_batch_columns_accepts_telecom_churn_csv():
    df = pd.DataFrame([{column: "value" for column in BATCH_COLUMNS}])

    missing, unexpected = validate_batch_columns(df)

    assert missing == []
    assert unexpected == []


def test_validate_batch_columns_rejects_unrelated_csv():
    df = pd.DataFrame([{"rainfall": 12.5, "humidity": 86, "temperature": 29}])

    missing, unexpected = validate_batch_columns(df)

    assert "tenure" in missing
    assert "monthly_charges" in missing
    assert unexpected == ["rainfall", "humidity", "temperature"]


def test_format_missing_columns_error_explains_input_contract():
    message = format_missing_columns_error(["tenure", "monthly_charges"])

    assert "telecom customer churn CSV" in message
    assert "tenure" in message
    assert "monthly_charges" in message
