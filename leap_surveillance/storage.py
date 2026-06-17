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
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.cloud.bigquery import DmlStats

from .common import (
    DEFAULT_BQ_PROJECT,
    DEFAULT_SURVEILLANCE_DATASET,
    NEAR_ZERO_SUM,
    Q50_TOLERANCE,
    TIMING_FORECAST_DATE,
    is_empty,
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
            "dim_question_map": q.dim_question_map,
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


_STABILITY_MAX_RUNS = 10   # prior production runs to consider
_STABILITY_SCAN_CAP = 40   # files to scan looking for them (most are legacy/test runs)
_WHEN_TOLERANCE_YR = 2     # ±years for non-black when-type rows


def _stable_pair(a: float | None, b: float | None, color: str | None, q_type: str) -> bool:
    """Two q50s agree: exact for resolved (black) rows, tolerance band otherwise."""
    if a is None or b is None:
        return False
    if color == "black":
        return a == b
    if q_type == "when":
        return abs(a - b) <= _WHEN_TOLERANCE_YR
    s = abs(a) + abs(b)
    return s < NEAR_ZERO_SUM or abs(a - b) / (s / 2) <= Q50_TOLERANCE


def _classify_sequence(points: list[tuple], q_type: str) -> str:
    """points = [(q50, color), ...] oldest→newest. Tolerance uses the newer point's color."""
    pts = [p for p in points if p[0] is not None]
    if len(pts) <= 1:
        return "new"
    consistent = all(_stable_pair(pts[i][0], pts[i + 1][0], pts[i + 1][1], q_type) for i in range(len(pts) - 1))
    if not consistent:
        return "volatile"
    return "converging" if len(pts) == 2 else "stable"


def _worst_stability(labels: list[str]) -> str:
    """Roll a model's per-row stability up to one label — worst wins; new only if no data."""
    non_new = [label for label in labels if label != "new"]
    if not non_new:
        return "new"
    if "volatile" in non_new:
        return "volatile"
    if "converging" in non_new:
        return "converging"
    return "stable"


def _combine_stability(gpt_stab: str, claude_stab: str) -> str:
    if gpt_stab == "stable" and claude_stab == "stable":
        return "both_stable"
    if gpt_stab == "new" and claude_stab == "new":
        return "new"
    if "volatile" in (gpt_stab, claude_stab):
        return "volatile"
    if gpt_stab in ("stable", "converging") and claude_stab in ("stable", "converging"):
        return "converging"
    return "one_stable"


def _adequate_q50s(model_block: dict):
    """Yield (fdate, dim, q50, color) for each adequate q50 forecast in a model block."""
    if not model_block or model_block.get("error"):
        return
    if not (model_block.get("quality") or {}).get("adequate", False):
        return
    for fc in (model_block.get("response") or {}).get("forecasts", []):
        if fc.get("quantile") == 50 and fc.get("forecast_value") is not None:
            color = fc.get("color_code")
            color = color.get("value", color) if isinstance(color, dict) else color
            yield fc.get("forecast_date", ""), fc.get("dimension", "Overall"), fc.get("forecast_value"), color


def _read_production_history(output_dir: str, current_models: dict, current_run_id: str) -> list[dict]:
    """Most-recent prior runs whose models match the current production models, oldest→newest."""
    files = sorted(Path(output_dir).glob("run_*.json"), reverse=True)
    runs: list[dict] = []
    for path in files[:_STABILITY_SCAN_CAP]:
        if len(runs) >= _STABILITY_MAX_RUNS:
            break
        if "_partial" in path.stem:
            continue
        if path.stem[len("run_"):] == current_run_id:
            continue
        try:
            run = json.loads(path.read_text())
        except Exception:
            continue
        models = run.get("models") or {}
        if any(m and models.get(tag) == m for tag, m in current_models.items()):
            runs.append(run)
    runs.reverse()
    return runs


def enrich_with_run_stability(run_data: dict, output_dir: str) -> dict:
    """Annotate each question with cross-run stability, scanning prior production runs once."""
    current_models = run_data.get("models") or {}
    current_run_id = run_data.get("run_id", "")
    q_types = {q.get("id", ""): q.get("question_type", "quantile") for q in run_data.get("questions", [])}

    seq: dict[tuple, list] = defaultdict(list)   # (qid, tag, fdate, dim) -> [(q50, color), ...] oldest→newest
    runs_seen: dict[str, set] = defaultdict(set)  # qid -> {run_id, ...}

    def ingest(run: dict) -> None:
        rid = run.get("run_id", "")
        models = run.get("models") or {}
        for q in run.get("questions", []):
            qid = q.get("id", "")
            for tag, model_block in (q.get("per_model") or {}).items():
                if models.get(tag) != current_models.get(tag):  # only this model's own production history
                    continue
                for fdate, dim, q50, color in _adequate_q50s(model_block):
                    seq[(qid, tag, fdate, dim)].append((q50, color))
                    runs_seen[qid].add(rid)

    for run in _read_production_history(output_dir, current_models, current_run_id):
        ingest(run)
    ingest(run_data)  # current run is the newest point

    per_qtag: dict[tuple, list] = defaultdict(list)  # (qid, tag) -> [row labels]
    for (qid, tag, _fdate, _dim), points in seq.items():
        per_qtag[(qid, tag)].append(_classify_sequence(points, q_types.get(qid, "quantile")))

    for q in run_data.get("questions", []):
        qid = q.get("id", "")
        gpt_stab = _worst_stability(per_qtag.get((qid, "gpt"), []))
        claude_stab = _worst_stability(per_qtag.get((qid, "claude"), []))
        q["run_stability"] = {
            "gpt_run_stability": gpt_stab,
            "claude_run_stability": claude_stab,
            "run_stability": _combine_stability(gpt_stab, claude_stab),
            "runs_seen": len(runs_seen.get(qid, set())),
        }
    return run_data


def _summarize_run(per_model_lists, costs_iter, errors_iter) -> dict:
    """Question-level run counts. Prefers GPT when both models ran; falls back to Claude."""
    ok_count = 0
    quality_issue_count = 0
    browser_count = 0
    due_unresolved = 0
    model_error_count = 0

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
        for model_result in (per_model or {}).values():
            if model_result is not None and model_result.error:
                model_error_count += 1

    return {
        "question_count": len(per_model_lists),
        "ok_count": ok_count,
        "quality_issue_count": quality_issue_count,
        "error_count": sum(1 for e in errors_iter if e),
        "model_error_count": model_error_count,
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


def _try_merge_bigquery_rows(
    label: str,
    rows: list[dict],
    *,
    pk: str,
    dataset: str,
    table: str,
    clock_col: str | None = None,
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


def write_accepted_to_fact_resolution(run_data: dict, reviewed_items: list | None = None):
    """Write auto-accepted or human-approved forecasts to fact.fact_resolution."""
    now = datetime.now(timezone.utc)

    reviewed_map: dict[tuple, dict] = {}
    for item in (reviewed_items or []):
        key = (item.get("question_id", ""), item.get("target_date", ""), item.get("dimension", "Overall") or "Overall")
        reviewed_map[key] = item

    def q50_values(per_model: dict, fdate: str, dim: str) -> list[float]:
        vals = []
        for block in per_model.values():
            for fc in (block.get("response") or {}).get("forecasts", []):
                if fc.get("quantile") == 50 and fc.get("forecast_date") == fdate:
                    d = fc.get("dimension", "Overall") or "Overall"
                    v = fc.get("forecast_value")
                    if d == dim and v is not None:
                        vals.append(float(v))
        return vals

    def row_color(per_model: dict, fdate: str, dim: str) -> str | None:
        for tag in ("gpt", "claude"):
            for fc in (per_model.get(tag) or {}).get("response", {}).get("forecasts", []):
                if fc.get("quantile") == 50 and fc.get("forecast_date") == fdate:
                    if (fc.get("dimension", "Overall") or "Overall") == dim:
                        c = fc.get("color_code")
                        return c.get("value", c) if isinstance(c, dict) else c
        return None

    rows = []
    for q in run_data.get("questions", []):
        group_qid = q.get("id", "")
        per_model = q.get("per_model") or {}
        consensus_status = (q.get("consensus") or {}).get("status", "")
        dim_q_map = q.get("dim_question_map") or {}

        for key, dq_id in dim_q_map.items():
            fdate, dim = key.split("|", 1)
            if fdate == TIMING_FORECAST_DATE:
                continue  # "event_occurrence" is not a real date; fact_resolution.resolution_date is DATE
            reviewed = reviewed_map.get((group_qid, fdate, dim))

            if reviewed is not None:
                verdict = (reviewed.get("review_verdict") or "").lower().strip()
                if verdict not in ("correct", "close"):
                    continue
                raw = reviewed.get("review_value", "")
                if raw not in (None, ""):
                    try:
                        value = float(raw)
                    except (ValueError, TypeError):
                        continue
                else:
                    # require explicit value when models disagreed
                    if consensus_status != "auto_accepted":
                        continue
                    vals = q50_values(per_model, fdate, dim)
                    if not vals:
                        continue
                    value = sum(vals) / len(vals)
                source = "surveillance_reviewed"
            elif consensus_status == "auto_accepted":
                vals = q50_values(per_model, fdate, dim)
                if not vals:
                    continue
                value = sum(vals) / len(vals)
                source = "surveillance_consensus"
            else:
                continue

            color = row_color(per_model, fdate, dim)
            is_resolved = color == "black"
            rows.append({
                "question_id": dq_id,
                "resolution_value": value,
                "resolution_date": date.fromisoformat(fdate),
                "resolution_source": source,
                "resolution_status": "resolved" if is_resolved else "projected",
                "resolved_at": now if is_resolved else None,
            })

    if not rows:
        return None
    return _try_merge_bigquery_rows(
        "fact_resolution", rows,
        pk="question_id", dataset="fact", table="fact_resolution",
    )


