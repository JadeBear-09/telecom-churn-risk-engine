from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IBM_TELCO_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
IBM_TELCO_ALTERNATE_FILENAME = "Telco-Customer-Churn.csv"
IBM_TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def create_sample_telco_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """Create IBM Telco-compatible sample data when public CSV is absent."""
    rng = np.random.default_rng(random_state)

    customer_ids = [f"CUST-{idx:06d}" for idx in range(1, n_samples + 1)]
    gender = rng.choice(["Female", "Male"], size=n_samples)
    senior = rng.binomial(1, 0.18, size=n_samples)
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.32, 0.68])
    tenure = rng.integers(1, 73, size=n_samples)
    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.9, 0.1])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n_samples, p=[0.44, 0.56]),
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.35, 0.48, 0.17])

    def internet_addon(yes_rate: float) -> np.ndarray:
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n_samples, p=[yes_rate, 1 - yes_rate]),
        )

    online_security = internet_addon(0.34)
    online_backup = internet_addon(0.43)
    device_protection = internet_addon(0.42)
    tech_support = internet_addon(0.34)
    streaming_tv = internet_addon(0.49)
    streaming_movies = internet_addon(0.49)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.24, 0.21])
    paperless = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.34, 0.22, 0.22, 0.22],
    )

    monthly = (
        20
        + (phone_service == "Yes") * 10
        + (internet_service == "DSL") * 25
        + (internet_service == "Fiber optic") * 45
        + (multiple_lines == "Yes") * 8
        + (online_security == "Yes") * 5
        + (online_backup == "Yes") * 5
        + (device_protection == "Yes") * 6
        + (tech_support == "Yes") * 6
        + (streaming_tv == "Yes") * 9
        + (streaming_movies == "Yes") * 9
        + senior * 2
        + rng.normal(0, 5, size=n_samples)
    )
    monthly = np.clip(monthly, 18, 125).round(2)
    total = np.maximum(monthly * tenure * rng.normal(1.0, 0.06, size=n_samples), monthly).round(2)

    risk_logit = (
        -3.10
        + (contract == "Month-to-month") * 1.45
        - (contract == "Two year") * 0.55
        + (internet_service == "Fiber optic") * 0.70
        + (payment == "Electronic check") * 0.60
        + (tech_support == "No") * 0.62
        + (online_security == "No") * 0.46
        + (tenure < 12) * 0.80
        + (tenure < 6) * 0.35
        + (monthly > 82) * 0.48
        + senior * 0.22
        + (paperless == "Yes") * 0.20
        + rng.normal(0, 0.22, size=n_samples)
    )
    churn_prob = _sigmoid(risk_logit)
    churn = rng.binomial(1, churn_prob, size=n_samples)

    total_as_text = total.astype(str)
    missing_total_idx = rng.choice(np.arange(n_samples), size=max(1, int(n_samples * 0.01)), replace=False)
    total_as_text[missing_total_idx] = " "

    return pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_as_text,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


def _is_url(value: str | Path) -> bool:
    parsed = urlparse(str(value))
    return parsed.scheme in {"http", "https"}


def _download_real_telco_data(destination: Path) -> Path | None:
    try:
        urlretrieve(IBM_TELCO_URL, destination)
    except Exception:
        return None
    return destination


def _real_data_candidates() -> list[Path]:
    return [
        RAW_DIR / IBM_TELCO_ALTERNATE_FILENAME,
        RAW_DIR / IBM_TELCO_FILENAME,
    ]


def load_telco_data(data_path: str | Path | None = None, n_samples: int = 3000) -> pd.DataFrame:
    """Load real IBM Telco CSV or URL if present, otherwise generate compatible sample data."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if data_path is not None:
        if _is_url(data_path):
            return pd.read_csv(str(data_path))
        candidate = Path(data_path)
        if candidate.exists():
            return pd.read_csv(candidate)

    for candidate in _real_data_candidates():
        if candidate.exists():
            return pd.read_csv(candidate)

    downloaded = _download_real_telco_data(RAW_DIR / IBM_TELCO_ALTERNATE_FILENAME)
    if downloaded is not None and downloaded.exists():
        return pd.read_csv(downloaded)

    synthetic_path = RAW_DIR / "synthetic_telco_churn.csv"
    if synthetic_path.exists():
        return pd.read_csv(synthetic_path)

    df = create_sample_telco_data(n_samples=n_samples)
    df.to_csv(synthetic_path, index=False)
    return df


if __name__ == "__main__":
    loaded = load_telco_data()
    print(f"Loaded rows={len(loaded)} columns={len(loaded.columns)}")
