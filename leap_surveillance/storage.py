"""Local file and BigQuery storage for LEAP surveillance."""

import csv
from collections import defaultdict
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import pandas as pd
from google.auth import default
from google.cloud import bigquery
from google.cloud.bigquery import DmlStats

from .common import (
    DEFAULT_BQ_PROJECT,
    DEFAULT_SURVEILLANCE_DATASET,
    PROBABILITY_TOLERANCE_POINTS,
    Q50_TOLERANCE,
    TIMING_FORECAST_DATE,
    WHEN_TOLERANCE_YR,
    enum_value,
    is_empty,
    is_truthy,
    row_resolution_status,
    within_relative_tolerance,
)


def _safe_num(value):
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return None


def _safe_float_or_none(value):
    try:
        return float(value) if value not in (None, "") else None
    except (ValueError, TypeError):
        return None


def _parse_date_or_none(value):
    if is_empty(value):
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def context_maps(response: dict) -> tuple[dict, dict, dict]:
    current = {
        item.get("dimension", "Overall"): item
        for item in response.get("current_estimates", []) or []
    }
    # A dimension can have multiple components (e.g. ratio numerator/denominator) — match by value, not last-wins.
    official: dict = {}
    for item in response.get("last_official_values", []) or []:
        dim = item.get("dimension", "Overall")
        if dim not in official:
            official[dim] = item
            continue
        anchor = _safe_num((current.get(dim) or {}).get("value"))
        if anchor is None:
            continue
        prev, new = _safe_num(official[dim].get("value")), _safe_num(item.get("value"))
        if new is not None and (prev is None or abs(new - anchor) < abs(prev - anchor)):
            official[dim] = item
    resolution = {
        (item.get("forecast_date"), item.get("dimension", "Overall")): item
        for item in response.get("resolution_values", []) or []
    }
    return official, current, resolution


def pick_by_dimension(value_map: dict, dim: str, single_dim: bool) -> dict:
    """Match official/current values to a row dimension; tolerate free-text labels on older runs."""
    if value_map.get(dim):
        return value_map[dim]
    if value_map.get("Overall"):
        return value_map["Overall"]
    for label, obj in value_map.items():  # older runs label by "{dim} (some detail)"
        if label.startswith(f"{dim} ") or label.startswith(f"{dim}("):
            return obj
    return next(iter(value_map.values())) if (single_dim and value_map) else {}


def _forecast_output_row(
    *,
    q_id: str,
    q_name: str,
    model_id: str,
    forecast: dict,
    response: dict,
    validation: dict | None = None,
) -> dict:
    official, current, resolution = context_maps(response)
    dim = forecast.get("dimension", "Overall")
    forecast_date = forecast.get("forecast_date", "")
    single_dim = len({f.get("dimension", "Overall") for f in response.get("forecasts", []) or []}) <= 1
    official_value = pick_by_dimension(official, dim, single_dim)
    current_value = pick_by_dimension(current, dim, single_dim)
    resolution_value = resolution.get((forecast_date, dim)) or resolution.get((forecast_date, "Overall")) or {}
    # Route by whether the row resolved (black), not by date — past-dated unresolved rows are still gray estimates.
    is_resolved = enum_value(forecast.get("color_code", "")) == "black"
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
        "resolution_status": row_resolution_status(forecast_date, forecast.get("color_code", "")),
        "resolution_source_date": resolution_value.get("source_date", ""),
        "resolution_source": resolution_value.get("source", ""),
        "current_estimate_value": current_value.get("value", ""),
        "current_estimate_confidence": current_value.get("confidence", ""),
        "latest_official_value": official_value.get("value", ""),
        "latest_official_date": official_value.get("date", ""),
        "latest_official_source": official_value.get("source", ""),
    }
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
                "source_role": e.source_role,
                "retrieval_status": e.retrieval_status,
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
    out["browser_url"] = model_result.browser_url or (model_result.quality.browser_url if model_result.quality else "")
    out["browser_objective"] = model_result.browser_objective or (model_result.quality.browser_objective if model_result.quality else "")
    out["browser_error"] = model_result.browser_error or ""
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
            "dim_question_map": q.dim_question_map,
            "evidence_plan": q.evidence_plan,
        }
        if error:
            result["error"] = error
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


def _stable_pair(a: float | None, b: float | None, color: str | None, q_type: str) -> bool:
    """Two q50s agree: exact for resolved (black) rows, tolerance band otherwise."""
    if a is None or b is None:
        return False
    if color == "black":
        return abs(a - b) < 1e-9
    if q_type == "when":
        return abs(a - b) <= WHEN_TOLERANCE_YR
    if q_type == "probability":
        return abs(a - b) <= PROBABILITY_TOLERANCE_POINTS  # absolute points, not relative - relative % is meaningless near 0
    return within_relative_tolerance(a, b, Q50_TOLERANCE)


def _value_changed(prior: float | None, current: float | None, q_type: str) -> bool:
    """Did a baseline fact move since the last comparable run? Both missing isn't a change - nothing to compare."""
    if prior is None and current is None:
        return False
    return not _stable_pair(prior, current, None, q_type)


def _classify_sequence(points: list[tuple], q_type: str) -> str:
    """points = [(q50, color), ...] oldest→newest. Tolerance uses the newer point's color."""
    pts = [p for p in points if p[0] is not None]
    if len(pts) <= 1:
        return "new"
    consistent = all(_stable_pair(pts[i][0], pts[i + 1][0], pts[i + 1][1], q_type) for i in range(len(pts) - 1))
    return "stable" if consistent else "volatile"


def _combine_stability(gpt_stab: str, claude_stab: str) -> str:
    if gpt_stab == "stable" and claude_stab == "stable":
        return "both_stable"
    if gpt_stab == "new" and claude_stab == "new":
        return "new"
    if "volatile" in (gpt_stab, claude_stab):
        return "volatile"
    return "one_stable"


def _combine_change_flags(tag_results: list[bool]) -> str:
    """One bool per model with a comparable prior run; blank (not "unchanged") if no model had one."""
    if not tag_results:
        return ""
    return "TRUE" if any(tag_results) else "FALSE"


def _summarize_run(per_model_lists, costs_iter, errors_iter) -> dict:
    """Question-level run counts. Quality and browser counts are per-model; ok_count uses GPT-preferred primary."""
    ok_count = 0
    quality_issue_count = 0
    browser_count = 0
    due_unresolved = 0
    model_error_count = 0
    quality_by_model: dict[str, int] = defaultdict(int)
    browser_by_model: dict[str, int] = defaultdict(int)

    for per_model in per_model_lists:
        primary = (per_model or {}).get("gpt") or (per_model or {}).get("claude")
        if primary is None:
            continue
        if (primary.validation or {}).get("ok"):
            ok_count += 1
        if primary.response is not None:
            seen: set[tuple[str, str]] = set()
            for f in primary.response.forecasts:
                key = (f.forecast_date, f.dimension)
                if key in seen:
                    continue
                seen.add(key)
                if row_resolution_status(f.forecast_date, f.color_code) == "due_unresolved":
                    due_unresolved += 1
        any_inadequate = False
        for tag, model_result in (per_model or {}).items():
            if model_result is None:
                continue
            if model_result.error:
                model_error_count += 1
            if model_result.quality is not None and model_result.quality.adequate is False:
                quality_by_model[tag] += 1
                any_inadequate = True
            if model_result.browser_used:
                browser_by_model[tag] += 1
                browser_count += 1
        if any_inadequate:
            quality_issue_count += 1

    return {
        "question_count": len(per_model_lists),
        "ok_count": ok_count,
        "quality_issue_count": quality_issue_count,
        "quality_by_model": dict(quality_by_model),
        "browser_by_model": dict(browser_by_model),
        "error_count": sum(1 for e in errors_iter if e),
        "model_error_count": model_error_count,
        "browser_count": browser_count,
        "due_unresolved_count": due_unresolved,
        "total_cost": sum((c.total if c is not None else 0.0) for c in costs_iter),
    }


def write_json_output(run_data: dict, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"run_{run_data.get('run_id', '')}.json"
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
    # Union of all row keys: rows are heterogeneous (validation fields are conditional),
    # so fieldnames=rows[0].keys() would crash order-dependently on mixed data.
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
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


def _merge_bigquery(
    df: pd.DataFrame,
    pk: str,
    dataset: str,
    table: str,
    *,
    clock_col: str | None = None,
    update_cols: Sequence[str] | None = None,
) -> DmlStats | None:
    """MERGE df into an existing BQ table. Assumes the table and its schema already exist - never creates or alters them (Jordan owns schema)."""
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

    try:
        client.load_table_from_dataframe(
            df,
            temp,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
        ).result()

        cols = [field.name for field in client.get_table(temp).schema]
        if pk not in cols:
            raise RuntimeError(f"TEMP table is missing '{pk}' after load.")

        non_pk_cols = [col for col in cols if col != pk]
        update_cols = [col for col in (update_cols or non_pk_cols) if col in non_pk_cols]

        update_assignments = ", ".join([f"`{col}` = S.`{col}`" for col in update_cols])
        insert_cols = [pk] + non_pk_cols
        insert_cols_csv = ", ".join(f"`{col}`" for col in insert_cols)
        insert_vals_csv = ", ".join(f"S.`{col}`" for col in insert_cols)

        matched_clause = (
            f"WHEN MATCHED AND S.`{clock_col}` > T.`{clock_col}` THEN"
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
        ON T.`{pk}` = S.`{pk}`
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


def _merge_bigquery_rows(
    rows: list[dict],
    *,
    pk: str,
    dataset: str,
    table: str,
    clock_col: str | None = None,
):
    """None for empty input (nothing to write) vs. a real write failure, which propagates - see _merge_bigquery."""
    if not rows:
        return None
    return _merge_bigquery(
        pd.DataFrame(rows),
        pk=pk,
        dataset=dataset,
        table=table,
        clock_col=clock_col,
    )


def _q50_forecast(model_block: dict | None, fdate: str, dim: str) -> dict | None:
    """The q50 forecast row for (fdate, dim) in one serialized model block, or None."""
    for fc in ((model_block or {}).get("response") or {}).get("forecasts", []) or []:
        if fc.get("quantile") == 50 and fc.get("forecast_date") == fdate:
            if (fc.get("dimension", "Overall") or "Overall") == dim:
                return fc
    return None


def write_to_fact_resolution(run_data: dict, all_items: list | None = None):
    """Write resolved/projected target-date values to fact.fact_resolution."""
    now = datetime.now(timezone.utc)

    item_map: dict[tuple, dict] = {}
    for item in (all_items or []):
        group_qid = item.get("group_question_id") or item.get("question_id", "")
        key = (group_qid, item.get("target_date", ""), item.get("dimension", "Overall") or "Overall")
        item_map[key] = item

    def effective_color(item: dict | None, per_model: dict, fdate: str, dim: str) -> str | None:
        review_color = (item or {}).get("review_color", "")
        if review_color not in (None, ""):
            # Normalize spacing variants ("dark gray" → "dark_gray") before writing to BQ.
            review_color = review_color.strip().lower().replace(" ", "_")
            return review_color
        for tag in ("gpt", "claude"):
            fc = _q50_forecast(per_model.get(tag), fdate, dim)
            if fc is not None:
                return enum_value(fc.get("color_code"))
        return None

    def black_q50_avg(per_model: dict, fdate: str, dim: str) -> float | None:
        vals = []
        for block in per_model.values():
            fc = _q50_forecast(block, fdate, dim)
            if fc is not None and enum_value(fc.get("color_code")) == "black":
                v = _safe_float_or_none(fc.get("forecast_value"))
                if v is not None:
                    vals.append(v)
        return sum(vals) / len(vals) if vals else None

    def model_resolution_source(per_model: dict, fdate: str, dim: str) -> tuple[str, date | None]:
        for tag in ("gpt", "claude"):
            resolution = {
                (item.get("forecast_date"), item.get("dimension", "Overall")): item
                for item in (per_model.get(tag) or {}).get("response", {}).get("resolution_values", []) or []
            }
            item = resolution.get((fdate, dim)) or resolution.get((fdate, "Overall")) or {}
            if item:
                return item.get("source") or "surveillance_projected", _parse_date_or_none(item.get("source_date"))
        return "surveillance_projected", None

    rows = []
    for q in run_data.get("questions", []):
        group_qid = q.get("id", "")
        per_model = q.get("per_model") or {}
        dim_q_map = q.get("dim_question_map") or {}

        for key, dq_id in dim_q_map.items():
            if "|" not in key:
                print(f"  warning: malformed dim_question_map key '{key}' (no '|'); skipping")
                continue
            fdate, dim = key.split("|", 1)
            # Timing rows aren't excluded here - the status gate below and the no-date skip further down already handle them.
            fdate_date = _parse_date_or_none(fdate)  # fallback resolution_date; None if key is non-ISO

            item = item_map.get((group_qid, fdate, dim))
            color = effective_color(item, per_model, fdate, dim)
            reviewed_status = ((item or {}).get("reviewed_question_resolution_status") or "").strip().lower()
            system_status = ((item or {}).get("question_resolution_status") or "").strip().lower()
            status = reviewed_status or system_status
            if color != "black" and status not in ("resolved", "projected"):
                continue

            reviewed_value = (
                _safe_float_or_none((item or {}).get("reviewed_question_resolution_value"))
                if is_truthy((item or {}).get("reviewed")) else None
            )
            system_value = _safe_float_or_none((item or {}).get("question_resolution_value"))
            if reviewed_value is not None:
                value = reviewed_value
                source = (item or {}).get("review_source") or "surveillance_reviewed"
                resolution_status_value = "confirmed" if reviewed_status == "resolved" else "projected"
                resolved_at = now if resolution_status_value == "confirmed" else None
                resolution_date = (
                    _parse_date_or_none((item or {}).get("question_resolution_source_date"))
                    or fdate_date
                )
            elif system_value is not None:
                value = system_value
                source = (item or {}).get("question_resolution_source") or "surveillance_projected"
                resolution_status_value = "projected"
                resolved_at = None
                resolution_date = (
                    _parse_date_or_none((item or {}).get("question_resolution_source_date"))
                    or fdate_date
                )
            else:
                value = black_q50_avg(per_model, fdate, dim)
                if value is None:
                    continue
                source, source_date = model_resolution_source(per_model, fdate, dim)
                resolution_status_value = "projected"
                resolved_at = None
                resolution_date = source_date or fdate_date

            if resolution_date is None:
                print(f"  warning: no parseable resolution date for '{key}'; skipping")
                continue

            rows.append({
                "question_id": dq_id,
                "resolution_value": value,
                "resolution_date": resolution_date,
                "resolution_source": source,
                "resolution_status": resolution_status_value,
                "resolved_at": resolved_at,
            })

    if not rows:
        return None
    return _merge_bigquery_rows(
        rows,
        pk="question_id", dataset="fact", table="fact_resolution",
    )


def _run_date_from_id(run_id: str) -> date:
    try:
        return datetime.strptime(run_id[:8], "%Y%m%d").date()
    except (ValueError, TypeError):
        return datetime.now(timezone.utc).date()


def write_to_dim_baseline(run_data: dict, all_sheet_rows: list[dict]):
    """Write historical/current baseline values to dim.dim_baseline."""
    run_id = run_data.get("run_id", "")
    current_date = _run_date_from_id(run_id)

    def first_float(*values) -> tuple[float | None, str]:
        for label, value in values:
            parsed = _safe_float_or_none(value)
            if parsed is not None:
                return parsed, label
        return None, ""

    rows = []
    seen: set[str] = set()
    for item in all_sheet_rows or []:
        group_qid = item.get("group_question_id") or item.get("question_id", "")
        if not group_qid:
            continue
        dim = item.get("dimension") or "Overall"

        reviewed_row = is_truthy(item.get("reviewed"))
        lov, lov_origin = first_float(
            ("reviewed", item.get("review_last_official_value") if reviewed_row else None),
            ("gpt", item.get("gpt_latest_official_value")),
            ("claude", item.get("claude_latest_official_value")),
        )
        lov_date = (
            current_date if lov_origin == "reviewed"  # the human's own review date, not the model's source date
            else _parse_date_or_none(item.get("gpt_latest_official_date"))
            or _parse_date_or_none(item.get("claude_latest_official_date"))
            or current_date
        )
        current, current_origin = first_float(
            ("reviewed", item.get("review_current_value") if reviewed_row else None),
            ("gpt", item.get("gpt_current_estimate")),
            ("claude", item.get("claude_current_estimate")),
        )

        candidates = [
            ("historical", lov, lov_date, lov_origin),
            ("current_day", current, current_date, current_origin),
        ]
        for baseline_type, value, baseline_date, origin in candidates:
            if value is None:
                continue
            baseline_id = f"{run_id}_{group_qid}_{dim}_{baseline_type}".replace(" ", "_")
            if baseline_id in seen:
                continue
            seen.add(baseline_id)
            rows.append({
                "baseline_id": baseline_id,
                "question_group_id": group_qid,
                "question_group_source_id": group_qid,
                "question_dimension": dim,
                "baseline_date": baseline_date,
                "baseline_value": value,
                "baseline_type": baseline_type,
                "baseline_source": "fri_research" if origin == "reviewed" else "llm_surveillance",
            })

    if not rows:
        return None
    return _merge_bigquery_rows(
        rows,
        pk="baseline_id", dataset="dim", table="dim_baseline",
    )


def write_to_surveillance_result(run_data: dict, all_sheet_rows: list[dict]):
    """Write full surveillance results (LLM forecasts + human review) to surveillance.surveillance_result."""
    now = datetime.now(timezone.utc)
    run_id = run_data.get("run_id", "")

    sheet_map: dict[tuple, dict] = {}
    for item in (all_sheet_rows or []):
        group_qid = item.get("group_question_id") or item.get("question_id", "")
        key = (group_qid, item.get("target_date", ""), item.get("dimension", "Overall") or "Overall")
        sheet_map[key] = item

    def q50_for(per_model: dict, model: str, fdate: str, dim: str) -> float | None:
        fc = _q50_forecast(per_model.get(model), fdate, dim)
        return _safe_float_or_none(fc.get("forecast_value")) if fc is not None else None

    def color_for(per_model: dict, model: str, fdate: str, dim: str) -> str | None:
        fc = _q50_forecast(per_model.get(model), fdate, dim)
        return enum_value(fc.get("color_code")) if fc is not None else None

    rows = []
    for q in run_data.get("questions", []):
        group_qid = q.get("id", "")
        per_model = q.get("per_model") or {}
        consensus_status = (q.get("consensus") or {}).get("status", "")
        dim_q_map = q.get("dim_question_map") or {}

        for key, dq_id in dim_q_map.items():
            if "|" not in key:
                print(f"  warning: malformed dim_question_map key '{key}' (no '|'); skipping")
                continue
            fdate, dim = key.split("|", 1)
            is_timing = fdate == TIMING_FORECAST_DATE
            sheet = sheet_map.get((group_qid, fdate, dim)) or {}

            rows.append({
                "run_question_id": f"{run_id}_{dq_id}",
                "run_id": run_id,
                "question_id": dq_id,
                "group_question_id": group_qid,
                "question_name": q.get("name", ""),
                "dimension": dim,
                "target_date": None if is_timing else fdate,
                "question_type": q.get("question_type", ""),
                "unit": q.get("unit", ""),
                "status": sheet.get("status", ""),
                "question_resolution_status": sheet.get("question_resolution_status") or None,
                "question_resolution_value": _safe_float_or_none(sheet.get("question_resolution_value")),
                "question_resolution_source": sheet.get("question_resolution_source") or None,
                "question_resolution_source_date": sheet.get("question_resolution_source_date") or None,
                "needs_review": sheet.get("needs_review") in (True, "TRUE", "true", "1", 1),
                "gpt_q50": q50_for(per_model, "gpt", fdate, dim),
                "claude_q50": q50_for(per_model, "claude", fdate, dim),
                "gpt_color": color_for(per_model, "gpt", fdate, dim),
                "claude_color": color_for(per_model, "claude", fdate, dim),
                "consensus_status": consensus_status,
                "reviewed": sheet.get("reviewed", False),
                "review_verdict": sheet.get("review_verdict") or None,
                "review_last_official_value": _safe_float_or_none(sheet.get("review_last_official_value")),
                "review_current_value": _safe_float_or_none(sheet.get("review_current_value")),
                "reviewed_question_resolution_status": sheet.get("reviewed_question_resolution_status") or None,
                "reviewed_question_resolution_value": _safe_float_or_none(sheet.get("reviewed_question_resolution_value")),
                "review_color": sheet.get("review_color") or None,
                "review_source": sheet.get("review_source") or None,
                "review_notes": sheet.get("review_notes") or None,
                "synced_at": now,
            })

    if not rows:
        return None
    return _merge_bigquery_rows(
        rows,
        pk="run_question_id",
        dataset=DEFAULT_SURVEILLANCE_DATASET,
        table="surveillance_result",
        clock_col="synced_at",
    )
