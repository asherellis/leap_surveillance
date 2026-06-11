"""Local file and BigQuery storage for LEAP surveillance."""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import pandas as pd
from google.auth import default
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.cloud.bigquery import DmlStats

from .common import (
    DEFAULT_BQ_PROJECT,
    DEFAULT_SURVEILLANCE_DATASET,
    is_empty,
    make_result_id,
    resolution_status,
    safe_str,
    to_float,
)


def context_maps(response: dict) -> tuple[dict, dict, dict]:
    official = {
        item.get("dimension", "Overall"): item
        for item in response.get("last_official_values", []) or []
    }
    current = {
        item.get("dimension", "Overall"): item
        for item in response.get("current_estimates", []) or []
    }
    resolution = {
        (item.get("forecast_date"), item.get("dimension", "Overall")): item
        for item in response.get("resolution_values", []) or []
    }
    return official, current, resolution


def _none_if_empty(value):
    return None if is_empty(value) else value


def _first_non_empty(*values):
    for value in values:
        if not is_empty(value):
            return value
    return None


def _forecast_output_row(
    *,
    q_id: str,
    q_name: str,
    model_id: str,
    forecast: dict,
    response: dict,
    validation: dict | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
    result_id: str | None = None,
) -> dict:
    official, current, resolution = context_maps(response)
    dim = forecast.get("dimension", "Overall")
    forecast_date = forecast.get("forecast_date", "")
    official_value = official.get(dim) or official.get("Overall") or {}
    current_value = current.get(dim) or current.get("Overall") or {}
    resolution_value = resolution.get((forecast_date, dim)) or resolution.get((forecast_date, "Overall")) or {}
    # Route by whether the row resolved (black), not by date — past-dated unresolved rows are still gray estimates.
    color = forecast.get("color_code", "")
    is_resolved = getattr(color, "value", color) == "black"
    target_value_type = "resolution" if is_resolved else "forecast"
    forecast_value = forecast.get("forecast_value", "")
    if is_resolved:
        rv = resolution_value.get("value", "")
        resolution_target_value = forecast_value if is_empty(rv) else rv
        forecast_target_value = ""
    else:
        forecast_target_value = forecast_value
        resolution_target_value = ""

    row = {
        "model_id": model_id,
        "question_id": q_id,
        "question_name": q_name,
        "target_value_type": target_value_type,
        "target_date": forecast_date,
        "dimension": dim,
        "quantile": forecast.get("quantile", ""),
        "forecast_target_value": forecast_target_value,
        "resolution_target_value": resolution_target_value,
        "color_code": forecast.get("color_code", ""),
        "resolution_status": resolution_status(forecast_date, forecast.get("color_code", "")),
        "resolution_source_date": resolution_value.get("source_date", ""),
        "resolution_source": resolution_value.get("source", ""),
        "current_estimate_value": current_value.get("value", ""),
        "current_estimate_confidence": current_value.get("confidence", ""),
        "latest_official_value": official_value.get("value", ""),
        "latest_official_date": official_value.get("date", ""),
        "latest_official_source": official_value.get("source", ""),
    }
    if result_id is not None:
        row["result_id"] = result_id
    if run_id is not None:
        row["run_id"] = run_id
    if created_at is not None:
        row["created_at"] = created_at
    if validation is not None:
        row["validation_ok"] = str(validation.get("ok", False))
        row["usable_for_scoring"] = str(validation.get("usable_for_scoring", False))
    return row


def _serialize_model_result(model_result) -> dict | None:
    """Serialize a ModelRunResult into the JSON-friendly dict shape used in per_model blocks."""
    if model_result is None:
        return None
    out: dict = {}
    if model_result.response is not None:
        out["response"] = model_result.response.model_dump(mode="json")
    if model_result.evidence:
        out["evidence"] = [
            {
                "source_type": e.source_type,
                "url": e.url,
                "title": e.title,
                "snippet": e.snippet,
                "full_text": e.full_text,
            }
            for e in model_result.evidence
        ]
    if model_result.validation:
        out["validation"] = {
            "ok": model_result.validation.get("ok"),
            "usable_for_scoring": model_result.validation.get("usable_for_scoring"),
            "issues": model_result.validation.get("issues", []),
        }
    if model_result.quality is not None:
        qr = model_result.quality
        out["quality"] = {
            "confidence": qr.confidence,
            "adequate": qr.adequate,
            "missing_data": qr.missing_data,
            "browser_would_help": qr.browser_would_help,
            "browser_url": qr.browser_url,
            "browser_objective": qr.browser_objective,
            "reason": qr.reason,
        }
    out["browser_used"] = bool(model_result.browser_used)
    out["browser_status"] = model_result.browser_status or "not_proposed"
    if model_result.error:
        out["error"] = model_result.error
    return out


def build_run_data(
    run_id,
    questions,
    per_model_lists,
    *,
    mode: str = "gpt",
    models: dict | None = None,
    costs_list=None,
    consensus_blocks=None,
    errors_list=None,
) -> dict:
    """Build the run JSON with symmetric per_model blocks for each model that ran."""
    models = models or {}
    expected_len = len(questions)
    by_name = {
        "per_model_lists": per_model_lists,
        "costs_list": costs_list,
        "consensus_blocks": consensus_blocks,
        "errors_list": errors_list,
    }
    for name, values in by_name.items():
        if values is not None and len(values) != expected_len:
            raise ValueError(f"{name} length {len(values)} does not match questions length {expected_len}")

    costs_iter = costs_list or [None] * len(questions)
    consensus_iter = consensus_blocks or [None] * len(questions)
    errors_iter = errors_list or [None] * len(questions)
    results = []
    for q, per_model, costs, consensus, error in zip(
        questions, per_model_lists, costs_iter, consensus_iter, errors_iter,
    ):
        result = {
            "id": q.id,
            "name": q.name,
            "question_type": q.question_type,
            "unit": q.unit,
            "unit_min": q.unit_min,
            "unit_max": q.unit_max,
            "question_text": q.question_text,
            "resolution_criteria": q.resolution_criteria,
            "background_info": q.background_info,
        }
        if error:
            result["error"] = error
        # Per-model block (symmetric — gpt and claude are siblings)
        per_model_block = {}
        for tag, model_result in (per_model or {}).items():
            per_model_block[tag] = _serialize_model_result(model_result) or {}
        if per_model_block:
            result["per_model"] = per_model_block
        if consensus is not None:
            result["consensus"] = consensus
        if costs is not None:
            result["cost"] = costs.as_dict()
        results.append(result)

    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "models": models,
        "summary": _summarize_run(per_model_lists, costs_iter, errors_iter),
        "questions": results,
    }


def _summarize_run(per_model_lists, costs_iter, errors_iter) -> dict:
    """Question-level run counts. Prefers GPT when both models ran; falls back to Claude."""
    ok_count = 0
    quality_issue_count = 0
    browser_count = 0
    due_unresolved = 0

    for per_model in per_model_lists:
        # Question-level summary uses whichever model produced output (GPT preferred for stable counts).
        primary = (per_model or {}).get("gpt") or (per_model or {}).get("claude")
        if primary is None:
            continue
        if (primary.validation or {}).get("ok"):
            ok_count += 1
        if primary.quality is not None and primary.quality.adequate is False:
            quality_issue_count += 1
        if primary.browser_used:
            browser_count += 1
        if primary.response is not None:
            seen: set[tuple[str, str]] = set()
            for f in primary.response.forecasts:
                key = (f.forecast_date, f.dimension)
                if key in seen:
                    continue
                seen.add(key)
                if resolution_status(f.forecast_date, f.color_code) == "due_unresolved":
                    due_unresolved += 1

    return {
        "question_count": len(per_model_lists),
        "ok_count": ok_count,
        "quality_issue_count": quality_issue_count,
        "error_count": sum(1 for e in errors_iter if e),
        "browser_count": browser_count,
        "due_unresolved_count": due_unresolved,
        "total_cost": sum((c.total if c is not None else 0.0) for c in costs_iter),
    }


def write_json_output(run_data: dict, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"run_{run_data['run_id']}.json"
    with open(path, "w") as f:
        json.dump(run_data, f, indent=2, default=str)
    return str(path)


def write_csv_output(run_id, questions, per_model_lists, output_dir):
    """One CSV per run with model_id as the first column. Both models' rows live as siblings."""
    expected_len = len(questions)
    if len(per_model_lists) != expected_len:
        raise ValueError(f"per_model_lists length {len(per_model_lists)} does not match questions length {expected_len}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for q, per_model in zip(questions, per_model_lists):
        for model_id, model_result in (per_model or {}).items():
            if model_result is None or model_result.response is None:
                continue
            response = model_result.response.model_dump(mode="json")
            validation = model_result.validation
            for forecast in response.get("forecasts", []):
                row = _forecast_output_row(
                    q_id=q.id,
                    q_name=q.name,
                    model_id=model_id,
                    forecast=forecast,
                    response=response,
                    validation=validation,
                )
                row.update({
                    "question_type": q.question_type,
                    "unit": q.unit,
                    "unit_min": q.unit_min,
                    "unit_max": q.unit_max,
                    "validation_issues": ", ".join((validation or {}).get("issues", [])),
                })
                rows.append(row)
    if not rows:
        return None
    path = Path(output_dir) / f"run_{run_id}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


_bq_client_cache: bigquery.Client | None = None


def _get_client() -> bigquery.Client:
    global _bq_client_cache
    if _bq_client_cache is None:
        credentials, _ = default()
        _bq_client_cache = bigquery.Client(credentials=credentials, project=DEFAULT_BQ_PROJECT)
    return _bq_client_cache


def query_bq(
    query: str,
    *,
    use_bqstorage: bool = False,
    query_parameters: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Run a query without requiring the BigQuery Storage Read API."""
    timeout_s = float(os.environ.get("LEAP_BQ_TIMEOUT", "120"))
    client = _get_client()
    job_config = None
    if query_parameters:
        job_config = bigquery.QueryJobConfig(query_parameters=list(query_parameters))
    job = client.query(query, job_config=job_config)
    job.result(timeout=timeout_s)
    return job.to_dataframe(create_bqstorage_client=use_bqstorage, timeout=timeout_s)


def _merge_bq(
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

    client = _get_client()
    target = f"{DEFAULT_BQ_PROJECT}.{dataset}.{table}"
    temp = f"{DEFAULT_BQ_PROJECT}.{dataset}.{table}_staging_{uuid4().hex}"

    if create_target_if_missing:
        try:
            client.get_table(target)
        except NotFound:
            client.load_table_from_dataframe(
                df.head(0),
                target,
                job_config=bigquery.LoadJobConfig(write_disposition="WRITE_EMPTY"),
            ).result()

    try:
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
        return job.dml_stats
    finally:
        try:
            client.query(f"DROP TABLE IF EXISTS `{temp}`").result()
        except Exception:
            pass


def _try_merge_bigquery_rows(
    label: str,
    rows: list[dict],
    *,
    pk: str,
    dataset: str,
    table: str,
    clock_col: str,
    dtypes: dict | None = None,
):
    if not rows:
        return None

    df = pd.DataFrame(rows)

    if dtypes:
        for col, dtype in dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

    try:
        return _merge_bq(
            df,
            pk=pk,
            dataset=dataset,
            table=table,
            clock_col=clock_col,
            create_target_if_missing=True,
        )
    except Exception as e:
        print(f"  {label} write failed: {e}")
        return None


def write_surveillance_to_bigquery(run_data: dict) -> dict:
    run_id = run_data.get("run_id")
    created_at = run_data.get("created_at")
    mode = run_data.get("mode", "gpt")
    models = run_data.get("models") or {}

    result_rows = []
    evidence_rows = []
    for question in run_data.get("questions", []):
        q_id = question.get("id")
        per_model = question.get("per_model") or {}
        for model_id, model_block in per_model.items():
            if not model_block:
                continue
            validation = model_block.get("validation") or {}
            response = model_block.get("response") or {}
            quality = model_block.get("quality") or {}

            for forecast in response.get("forecasts", []):
                color = forecast.get("color_code")
                result_id = make_result_id(
                    run_id,
                    q_id,
                    forecast.get("forecast_date"),
                    forecast.get("dimension"),
                    forecast.get("quantile"),
                    model_id,
                )
                row = _forecast_output_row(
                    q_id=q_id,
                    q_name=question.get("name"),
                    model_id=model_id,
                    forecast=forecast,
                    response=response,
                    validation=validation,
                    run_id=run_id,
                    created_at=created_at,
                    result_id=result_id,
                )
                result_rows.append({
                    "result_id": result_id,
                    "run_id": run_id,
                    "model_id": model_id,
                    "question_id": q_id,
                    "question_name": question.get("name"),
                    "question_type": question.get("question_type"),
                    "unit": question.get("unit"),
                    "unit_min": question.get("unit_min"),
                    "unit_max": question.get("unit_max"),
                    # target_value_type chooses the numeric value column; resolution_status is the
                    # human review status derived from date + color. Related, but not interchangeable.
                    "target_value_type": row.get("target_value_type"),
                    "target_date": row.get("target_date"),
                    "dimension": forecast.get("dimension"),
                    "quantile": forecast.get("quantile"),
                    "forecast_target_value": _none_if_empty(row.get("forecast_target_value")),
                    "resolution_target_value": _none_if_empty(row.get("resolution_target_value")),
                    "color_code": color,
                    "resolution_status": row.get("resolution_status"),
                    "resolution_source_date": row.get("resolution_source_date") or None,
                    "resolution_source": row.get("resolution_source") or None,
                    "current_estimate_value": _none_if_empty(row.get("current_estimate_value")),
                    "current_estimate_confidence": _none_if_empty(row.get("current_estimate_confidence")),
                    "latest_official_value": _none_if_empty(row.get("latest_official_value")),
                    "latest_official_date": row.get("latest_official_date") or None,
                    "latest_official_source": row.get("latest_official_source") or None,
                    "rationale": (response.get("rationale") or "")[:2000],
                    "sources": ", ".join(response.get("sources") or [])[:1000],
                    "validation_ok": validation.get("ok", False),
                    "validation_issues": ", ".join(validation.get("issues") or [])[:1000] or None,
                    "usable_for_scoring": validation.get("usable_for_scoring", False),
                    "quality_confidence": quality.get("confidence"),
                    "quality_adequate": quality.get("adequate"),
                    "quality_missing_data": "; ".join(quality.get("missing_data") or [])[:1000] or None,
                    "quality_reason": (quality.get("reason") or "")[:2000] or None,
                    "browser_would_help": quality.get("browser_would_help"),
                    "browser_url": quality.get("browser_url") or None,
                    "browser_objective": quality.get("browser_objective") or None,
                    "browser_used": bool(model_block.get("browser_used")),
                    "review_status": None,
                    "override_value": None,
                    "override_color": None,
                    "final_value": _first_non_empty(row.get("forecast_target_value"), row.get("resolution_target_value")),
                    "final_color": color,
                    "notes": None,
                    "reviewed_at": None,
                    "created_at": created_at,
                    "ingestion_timestamp": datetime.now(timezone.utc),
                })

            for i, ev in enumerate(model_block.get("evidence") or []):
                evidence_rows.append({
                    "evidence_id": f"{run_id}_{q_id}_{model_id}_{i}",
                    "run_id": run_id,
                    "model_id": model_id,
                    "question_id": q_id,
                    "source_type": ev.get("source_type"),
                    "url": ev.get("url"),
                    "title": ev.get("title"),
                    "snippet": (ev.get("snippet") or "")[:1000],
                    "full_text": (ev.get("full_text") or "")[:5000],
                    "created_at": created_at,
                    "ingestion_timestamp": datetime.now(timezone.utc),
                })

    summary = run_data.get("summary") or {}
    run_rows = [{
        "run_id": run_id,
        "created_at": created_at,
        "mode": mode,
        "gpt_model": models.get("gpt"),
        "claude_model": models.get("claude"),
        "question_count": summary.get("question_count", 0),
        "success_count": summary.get("ok_count", 0),
        "error_count": summary.get("error_count", 0),
        "browser_count": summary.get("browser_count", 0),
        "total_cost": summary.get("total_cost", 0.0),
        "ingestion_timestamp": datetime.now(timezone.utc),
    }]

    result_dtypes = {
        "override_value": "Float64",
        "override_color": "string",
        "review_status": "string",
        "notes": "string",
        "quality_confidence": "Int64",
        "forecast_target_value": "Float64",
        "resolution_target_value": "Float64",
        "current_estimate_value": "Float64",
        "current_estimate_confidence": "Int64",
        "latest_official_value": "Float64",
        "unit_min": "Float64",
        "unit_max": "Float64",
    }

    return {
        "results": _try_merge_bigquery_rows(
            "BigQuery results", result_rows,
            pk="result_id", dataset=DEFAULT_SURVEILLANCE_DATASET, table="surveillance_result",
            clock_col="ingestion_timestamp", dtypes=result_dtypes,
        ),
        "evidence": _try_merge_bigquery_rows(
            "BigQuery evidence", evidence_rows,
            pk="evidence_id", dataset=DEFAULT_SURVEILLANCE_DATASET, table="surveillance_evidence",
            clock_col="ingestion_timestamp",
        ),
        "runs": _try_merge_bigquery_rows(
            "BigQuery run", run_rows,
            pk="run_id", dataset=DEFAULT_SURVEILLANCE_DATASET, table="surveillance_run",
            clock_col="ingestion_timestamp",
        ),
    }


def _get_existing_result_ids_for_groups(group_ids: list[str]) -> dict[str, list[tuple[str, int]]]:
    """Find existing result rows for reviewed Sheet groups."""
    if not group_ids:
        return {}

    prefixes = [f"{g}_" for g in group_ids]
    query = f"""
    SELECT result_id
    FROM `{DEFAULT_BQ_PROJECT}.{DEFAULT_SURVEILLANCE_DATASET}.surveillance_result`
    WHERE EXISTS (
        SELECT 1
        FROM UNNEST(@prefixes) AS prefix
        WHERE STARTS_WITH(result_id, prefix)
    )
    """
    by_group: dict[str, list[tuple[str, int]]] = {g: [] for g in group_ids}
    try:
        df = query_bq(
            query,
            query_parameters=[
                bigquery.ArrayQueryParameter("prefixes", "STRING", prefixes)
            ],
        )
    except Exception as e:
        print(f"  warning: could not read existing result_ids ({e}); no rows will be updated")
        return by_group

    for rid in df["result_id"].tolist():
        # Result IDs are {group_id}_{quantile}_{model_id}. If the trailing segment is
        # non-numeric, peel it off as the model_id and re-parse for {group_id}_{quantile}.
        head, _, tail = rid.rpartition("_")
        if not tail.isdigit():
            head, _, tail = head.rpartition("_")
        if not tail.isdigit():
            continue
        if head in by_group:
            by_group[head].append((rid, int(tail)))
    return by_group


def sync_reviews_to_bigquery(reviewed_items: list[dict]) -> dict:
    if not reviewed_items:
        return {"results": None, "skipped": 0}

    reviewed_at = datetime.now(timezone.utc)

    unique_groups = list({item.get("group_id", "") for item in reviewed_items if item.get("group_id")})
    existing_by_group = _get_existing_result_ids_for_groups(unique_groups)

    pending_rows = []
    missing_groups: list[str] = []
    for item in reviewed_items:
        group_id = item.get("group_id", "")
        row_type = item.get("type", "")
        override_color = item.get("review_color")
        override_color = _none_if_empty(override_color)
        free_notes = safe_str(item.get("review_notes")) or ""

        override_val = to_float(item.get("review_value"))
        if row_type == "resolved":
            score = safe_str(item.get("review_verdict", ""))
            note_parts = []
            if score:
                note_parts.append(f"score:{score}")
            if free_notes:
                note_parts.append(free_notes)
            notes = " ".join(note_parts) or None
        else:
            notes = free_notes or None

        existing = existing_by_group.get(group_id, [])
        if not existing:
            missing_groups.append(group_id)
            continue

        for result_id, q in existing:
            row = {
                "result_id": result_id,
                "review_status": "Reviewed",
                "usable_for_scoring": True,
                "notes": notes,
                "reviewed_at": reviewed_at,
                "ingestion_timestamp": datetime.now(timezone.utc),
            }

            if row_type == "resolved" and override_val is not None:
                row["final_value"] = override_val
                if q == 50:
                    row["override_value"] = override_val
            elif row_type == "forecast" and q == 50 and override_val is not None:
                row["final_value"] = override_val
                row["override_value"] = override_val

            if override_color:
                row["override_color"] = override_color
                row["final_color"] = override_color

            pending_rows.append(row)

    skipped = len(missing_groups)
    if skipped > 0:
        preview = f"{missing_groups[:3]}{'...' if skipped > 3 else ''}"
        print(
            f"  warning: {skipped} reviewed group(s) had no rows in "
            f"surveillance_result; skipped: {preview}"
        )

    if not pending_rows:
        return {"results": None, "skipped": skipped}

    return {
        "results": _try_merge_bigquery_rows(
            "BigQuery results (review update)", pending_rows,
            pk="result_id", dataset=DEFAULT_SURVEILLANCE_DATASET, table="surveillance_result",
            clock_col="ingestion_timestamp",
        ),
        "skipped": skipped,
    }
