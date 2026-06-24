from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_ingestion import RAW_DIR


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        for col in output.columns
    ]
    if "customerid" in output.columns and "customer_id" not in output.columns:
        output = output.rename(columns={"customerid": "customer_id"})
    return output


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return _normalize_columns(pd.read_csv(path))


def _coalesce_source_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for col in columns:
        source_col = f"{col}_source"
        if source_col in output.columns:
            if col in output.columns:
                output[col] = output[col].where(output[col].notna(), output[source_col])
            else:
                output[col] = output[source_col]
            output = output.drop(columns=[source_col])
    return output


def _filter_recent(df: pd.DataFrame, date_columns: list[str], days: int) -> pd.DataFrame:
    date_col = next((col for col in date_columns if col in df.columns), None)
    if date_col is None:
        return df
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.notna().sum() == 0:
        return df
    anchor = dates.max()
    return df.loc[(anchor - dates).dt.days.between(0, days, inclusive="both")].copy()


def _merge_crm(base: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    crm = _load_csv(raw_dir / "crm_customers.csv")
    if crm is None or "customer_id" not in crm.columns:
        return base

    keep = [
        col
        for col in ["customer_id", "customer_segment", "region", "last_interaction_sentiment", "plan_change_count_12m"]
        if col in crm.columns
    ]
    if len(keep) <= 1:
        return base
    merged = base.merge(crm[keep].drop_duplicates("customer_id"), on="customer_id", how="left", suffixes=("", "_source"))
    return _coalesce_source_columns(merged, keep[1:])


def _merge_support(base: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    support = _load_csv(raw_dir / "support_tickets.csv")
    if support is None or "customer_id" not in support.columns:
        return base

    support = _filter_recent(support, ["created_at", "ticket_created_at", "opened_at"], days=90)
    category = support.get("category", support.get("ticket_type", pd.Series("", index=support.index))).astype(str).str.lower()
    subject = support.get("subject", pd.Series("", index=support.index)).astype(str).str.lower()
    is_complaint = category.str.contains("complaint", na=False) | subject.str.contains("complaint", na=False)

    aggregates = support.groupby("customer_id").size().rename("support_ticket_count_90d").to_frame()
    aggregates["complaint_count_90d"] = is_complaint.groupby(support["customer_id"]).sum().astype(int)
    merged = base.merge(aggregates.reset_index(), on="customer_id", how="left", suffixes=("", "_source"))
    return _coalesce_source_columns(merged, ["support_ticket_count_90d", "complaint_count_90d"])


def _merge_billing(base: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    billing = _load_csv(raw_dir / "billing_events.csv")
    if billing is None or "customer_id" not in billing.columns:
        return base

    billing = _filter_recent(billing, ["event_date", "created_at", "invoice_date"], days=183)
    event_type = billing.get("event_type", billing.get("type", pd.Series("", index=billing.index))).astype(str).str.lower()
    late = event_type.str.contains("late|overdue|missed", regex=True, na=False)
    dispute = event_type.str.contains("dispute|chargeback|billing_issue", regex=True, na=False)

    aggregates = pd.DataFrame({"customer_id": billing["customer_id"]})
    aggregates["late_payment_event"] = late.astype(int)
    aggregates["billing_dispute_event"] = dispute.astype(int)
    aggregates = aggregates.groupby("customer_id").sum().rename(
        columns={
            "late_payment_event": "late_payment_count_6m",
            "billing_dispute_event": "billing_dispute_count_6m",
        }
    )
    merged = base.merge(aggregates.reset_index(), on="customer_id", how="left", suffixes=("", "_source"))
    return _coalesce_source_columns(merged, ["late_payment_count_6m", "billing_dispute_count_6m"])


def _network_minutes_column(network: pd.DataFrame) -> str | None:
    for col in ["network_downtime_minutes_30d", "downtime_minutes", "outage_minutes", "duration_minutes"]:
        if col in network.columns:
            return col
    return None


def _merge_network(base: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    network = _load_csv(raw_dir / "network_outages.csv")
    if network is None:
        return base

    minutes_col = _network_minutes_column(network)
    if minutes_col is None:
        return base

    network = _filter_recent(network, ["event_date", "started_at", "created_at"], days=30)
    network[minutes_col] = pd.to_numeric(network[minutes_col], errors="coerce").fillna(0.0)

    if "customer_id" in network.columns:
        aggregates = (
            network.groupby("customer_id")[minutes_col]
            .sum()
            .rename("network_downtime_minutes_30d")
            .reset_index()
        )
        merged = base.merge(aggregates, on="customer_id", how="left", suffixes=("", "_source"))
        return _coalesce_source_columns(merged, ["network_downtime_minutes_30d"])

    if "region" in network.columns and "region" in base.columns:
        aggregates = (
            network.groupby("region")[minutes_col]
            .sum()
            .rename("network_downtime_minutes_30d")
            .reset_index()
        )
        merged = base.merge(aggregates, on="region", how="left", suffixes=("", "_source"))
        return _coalesce_source_columns(merged, ["network_downtime_minutes_30d"])

    return base


def merge_operational_sources(df: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Merge optional CRM, support, billing, and network CSV extracts when present."""
    output = df.copy()
    if "customer_id" not in output.columns:
        return output

    for merge_step in (_merge_crm, _merge_support, _merge_billing, _merge_network):
        output = merge_step(output, raw_dir)

    numeric_source_cols = [
        "complaint_count_90d",
        "support_ticket_count_90d",
        "network_downtime_minutes_30d",
        "late_payment_count_6m",
        "billing_dispute_count_6m",
        "plan_change_count_12m",
    ]
    for col in numeric_source_cols:
        if col in output.columns:
            output[col] = pd.to_numeric(output[col], errors="coerce")

    output = output.replace({np.nan: None})
    return output
