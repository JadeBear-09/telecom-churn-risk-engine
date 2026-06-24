from __future__ import annotations

import pandas as pd

from src.batch_scoring import generate_retention_queue


def test_generate_retention_queue_writes_ranked_csv(tmp_path):
    input_path = tmp_path / "customers.csv"
    output_path = tmp_path / "queue.csv"
    pd.DataFrame(
        [
            {
                "customer_id": "QUEUE-1",
                "contract": "Month-to-month",
                "monthly_charges": 95,
                "tenure": 4,
            }
        ]
    ).to_csv(input_path, index=False)

    queue = generate_retention_queue(input_path=input_path, output_path=output_path)

    assert output_path.exists()
    assert "queue_score" in queue.columns
    assert queue.iloc[0]["customer_id"] == "QUEUE-1"
