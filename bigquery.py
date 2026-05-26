"""BigQuery helpers used by the surveillance pipeline."""

import os
from datetime import datetime, timezone
from typing import Sequence

import pandas as pd
from google.auth import default
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.cloud.bigquery import DmlStats

PROJECT = "ai-panel-of-experts"


def get_client() -> bigquery.Client:
    credentials, _ = default()
    return bigquery.Client(credentials=credentials, project=PROJECT)


def query_bq(query: str, *, use_bqstorage: bool = False) -> pd.DataFrame:
    """Run a query without requiring the BigQuery Storage Read API."""
    timeout_s = float(os.environ.get("LEAP_BQ_TIMEOUT", "120"))
    client = get_client()
    job = client.query(query)
    job.result(timeout=timeout_s)
    return job.to_dataframe(create_bqstorage_client=use_bqstorage, timeout=timeout_s)


def merge_bq(
    df: pd.DataFrame,
    pk: str,
    dataset: str,
    table: str,
    *,
    clock_col: str | None = None,
    update_cols: Sequence[str] | None = None,
    create_target_if_missing: bool = False,
) -> DmlStats | None:
    if pk not in df.columns:
        raise ValueError(f"DataFrame must include '{pk}'.")

    if df[pk].duplicated(keep=False).any():
        dup_ids = df.loc[df[pk].duplicated(keep=False), pk].head(10).tolist()
        raise ValueError(f"Duplicate '{pk}' values detected. Examples: {dup_ids}")

    if clock_col and clock_col not in df.columns:
        df = df.copy()
        df[clock_col] = datetime.now(timezone.utc)

    client = get_client()
    target = f"{PROJECT}.{dataset}.{table}"
    temp = f"{PROJECT}.temp.{dataset}_{table}"

    if create_target_if_missing:
        try:
            client.get_table(target)
        except NotFound:
            client.load_table_from_dataframe(
                df.head(0),
                target,
                job_config=bigquery.LoadJobConfig(write_disposition="WRITE_EMPTY"),
            ).result()

    client.load_table_from_dataframe(
        df,
        temp,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    ).result()

    temp_schema = client.get_table(temp).schema
    cols = [field.name for field in temp_schema]
    if pk not in cols:
        raise RuntimeError(f"TEMP table is missing '{pk}' after load.")

    non_pk_cols = [col for col in cols if col != pk]
    update_cols = [col for col in (update_cols or non_pk_cols) if col in non_pk_cols]

    def q(col: str) -> str:
        return f"`{col}`"

    update_assignments = ", ".join([f"{q(col)} = S.{q(col)}" for col in update_cols])
    insert_cols = [pk] + non_pk_cols
    insert_cols_csv = ", ".join(q(col) for col in insert_cols)
    insert_vals_csv = ", ".join(f"S.{q(col)}" for col in insert_cols)

    matched_clause = (
        f"WHEN MATCHED AND S.{q(clock_col)} > T.{q(clock_col)} THEN"
        if clock_col and clock_col in cols
        else "WHEN MATCHED THEN"
    )
    update_branch = (
        f"{matched_clause}\n  UPDATE SET {update_assignments}"
        if update_assignments
        else ""
    )

    merge_sql = f"""
    MERGE `{target}` T
    USING `{temp}` S
    ON T.{q(pk)} = S.{q(pk)}
    {update_branch}
    WHEN NOT MATCHED THEN
    INSERT ({insert_cols_csv}) VALUES ({insert_vals_csv})
    """

    job = client.query(merge_sql)
    job.result()

    client.query(f"DROP TABLE `{temp}`").result()

    return job.dml_stats
