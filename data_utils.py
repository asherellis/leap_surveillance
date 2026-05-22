"""Data loading, local output, Sheets review, and BigQuery sync."""

import csv
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from schemas import ExpectedForecast, QuestionSpec

DEFAULT_OUTPUT_DIR = "output"
DEFAULT_SHEET_ID = os.environ.get(
    "LEAP_SHEET_ID", "1lT7zVfKAsVZU7bKaEALq1AWApfFmWMisprTK42l7RDo"
)

SHEET_TEXT_LIMIT = 500

SHEET_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]
CREDENTIALS_DIR = Path.home() / ".config" / "leap-surveillance"
SHEET_HEADERS = [
    "result_id",
    "run_id",
    "created_at",
    "question_id",
    "question_name",
    "forecast_date",
    "dimension",
    "quantile",
    "forecast_value",
    "color_code",
    "validation_ok",
    "usable_for_scoring",
    "rationale",
    "sources",
    "value_override",
    "color_override",
    "reviewed",
    "notes",
    "reviewed_at",
]

INSTRUCTIONS_CONTENT = [
    ["How to Review"],
    [""],
    ["1. Go to 'Pending Review' tab"],
    ["2. Look at each forecast"],
    ["3. If correct: Check the 'reviewed' box"],
    ["4. If wrong: Put the correct value in 'value_override'"],
]


def safe_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    if str(val) == "<NA>":
        return ""
    return str(val)


def to_float(val):
    if val is None:
        return None
    try:
        if math.isnan(val):
            return None
    except (TypeError, ValueError):
        pass
    if safe_str(val).strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


FULL_QUANTILES = [0, 5, 25, 50, 75, 95, 100]
TIMING_FORECAST_DATE = "event_occurrence"


def _is_empty(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return safe_str(val).strip() == ""


def _infer_surveillance_question_type(
    question_text: str,
    unit: str,
    dates: list[str],
    dimensions: list[str],
    source_percentiles: list[int],
) -> str:
    """Infer quantile/probability/timing shape from current warehouse fields."""
    text = question_text.lower()
    unit_lower = unit.lower()
    pct_set = set(source_percentiles)

    if not dates and pct_set:
        return "when"

    asks_probability = bool(
        re.search(r"\bprobability\b|\bprobabilit(y|ies)\b|\bwill\b", text)
    )
    probability_unit = "probability" in unit_lower
    scalar_probability = pct_set == {50} and (
        probability_unit
        or "what is the probability" in text
        or text.strip().startswith("will ")
        or text.strip().startswith("what is the probability")
        or text.strip().startswith("what's the probability")
    )
    distribution_over_options = pct_set == {50} and probability_unit and len(dimensions) > 1

    if asks_probability and (scalar_probability or distribution_over_options):
        return "probability"

    return "quantile"


def _build_prompt_context(row, dates: list[str], dimensions: list[str], source_percentiles: list[int]) -> str:
    prompt = safe_str(row.get("question_set_text"))
    bg = safe_str(row.get("question_set_background_information"))
    if bg:
        prompt += f"\n\nBackground:\n{bg}"
    res = safe_str(row.get("question_set_resolution_criteria"))
    if res:
        prompt += f"\n\nResolution:\n{res}"
    unit = safe_str(row.get("unit_display_text"))
    if unit:
        prompt += f"\n\nUnit: {unit}"

    unit_min = to_float(row.get("unit_min_value"))
    unit_max = to_float(row.get("unit_max_value"))
    bounds = []
    if unit_min is not None:
        bounds.append(f"minimum {unit_min:g}")
    if unit_max is not None:
        bounds.append(f"maximum {unit_max:g}")
    if bounds:
        prompt += f"\nUnit bounds: {', '.join(bounds)}"

    if dates:
        prompt += f"\nRequested resolution dates: {', '.join(dates)}"
    if dimensions != ["Overall"]:
        prompt += f"\nRequested dimensions: {', '.join(dimensions)}"
    if source_percentiles:
        prompt += (
            "\nPercentiles present in BigQuery for human forecasts: "
            f"{', '.join(str(p) for p in source_percentiles)}"
        )
    return prompt


def _expected_forecasts(
    question_type: str, dates: list[str], dimensions: list[str]
) -> list[ExpectedForecast]:
    if question_type == "probability":
        return [ExpectedForecast(d, dim, 50) for d in dates for dim in dimensions]

    if question_type == "when":
        return [
            ExpectedForecast(TIMING_FORECAST_DATE, dim, p)
            for dim in dimensions
            for p in FULL_QUANTILES
        ]

    return [
        ExpectedForecast(d, dim, p)
        for d in dates
        for dim in dimensions
        for p in FULL_QUANTILES
    ]


def load_questions(limit=None, prod=False) -> list[QuestionSpec]:
    from bigquery import query_bq

    prefix = "" if prod else "dev_"
    query = f"""
    WITH forecast_sets AS (
        SELECT qs.question_set_id, qs.question_set_name
        FROM `ai-panel-of-experts.{prefix}dim.question_set` qs
        JOIN `ai-panel-of-experts.{prefix}dim.question` q
            ON qs.question_set_id = q.question_set_id
        WHERE q.question_type = 'forecast'
        GROUP BY qs.question_set_id, qs.question_set_name
        ORDER BY qs.question_set_name
        {"LIMIT " + str(limit) if limit else ""}
    )
    SELECT qs.question_set_id, qs.question_set_name, qs.question_set_text,
           qs.question_set_background_information, qs.question_set_resolution_criteria,
           u.unit_display_text, u.unit_min_value, u.unit_max_value,
           q.question_resolution_date, q.question_percentile, q.question_dimension
    FROM `ai-panel-of-experts.{prefix}dim.question_set` qs
    JOIN forecast_sets fs ON qs.question_set_id = fs.question_set_id
    JOIN `ai-panel-of-experts.{prefix}dim.question` q
        ON qs.question_set_id = q.question_set_id
        AND q.question_type = 'forecast'
    LEFT JOIN `ai-panel-of-experts.{prefix}dim.unit` u ON qs.unit_id = u.unit_id
    ORDER BY qs.question_set_name, q.question_resolution_date, q.question_dimension,
             q.question_percentile"""

    df = query_bq(query)
    questions = []
    for _, group in df.groupby("question_set_id", sort=False):
        row = group.iloc[0]
        dates = sorted(
            {
                safe_str(d).strip()
                for d in group["question_resolution_date"].tolist()
                if not _is_empty(d)
            }
        )
        dimensions = sorted(
            {
                safe_str(d).strip()
                for d in group["question_dimension"].tolist()
                if not _is_empty(d)
            }
        ) or ["Overall"]
        source_percentiles = sorted(
            {
                int(float(p))
                for p in group["question_percentile"].tolist()
                if not _is_empty(p)
            }
        )

        question_text = safe_str(row.get("question_set_text"))
        unit = safe_str(row.get("unit_display_text"))
        question_type = _infer_surveillance_question_type(
            question_text, unit, dates, dimensions, source_percentiles
        )
        prompt = _build_prompt_context(row, dates, dimensions, source_percentiles)
        expected = _expected_forecasts(question_type, dates, dimensions)

        question_id = row.get("question_set_id")
        question_name = row.get("question_set_name")
        if not question_id or not question_name or not expected:
            continue
        questions.append(
            QuestionSpec(
                question_id,
                question_name,
                prompt,
                expected,
                question_type=question_type,
                unit=unit,
                unit_min=to_float(row.get("unit_min_value")),
                unit_max=to_float(row.get("unit_max_value")),
            )
        )

    return questions


def write_json_output(
    run_id,
    model,
    questions,
    responses,
    validations,
    evidences,
    quality_reports,
    output_dir,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for q, resp, val, ev, qr in zip(
        questions, responses, validations, evidences, quality_reports
    ):
        result = {
            "id": q.id,
            "name": q.name,
            "response": resp.model_dump() if resp else None,
        }
        if val:
            result["validation"] = {
                "ok": val["ok"],
                "usable_for_scoring": val["usable"],
                "issues": val.get("issues", []),
            }
        if ev:
            result["evidence"] = [
                {
                    "source_type": e.source_type,
                    "url": e.url,
                    "title": e.title,
                    "snippet": e.snippet,
                    "full_text": e.full_text,
                }
                for e in ev
            ]
        if qr:
            result["quality"] = {
                "confidence": qr.confidence,
                "adequate": qr.adequate,
                "missing_data": qr.missing_data,
                "reason": qr.reason,
            }
        results.append(result)

    path = Path(output_dir) / f"run_{run_id}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "questions": results,
            },
            f,
            indent=2,
            default=str,
        )
    return str(path)


def write_csv_output(run_id, questions, responses, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for q, resp in zip(questions, responses):
        if resp:
            for f in resp.forecasts:
                rows.append(
                    {
                        "question_id": q.id,
                        "question_name": q.name,
                        "forecast_date": f.forecast_date,
                        "dimension": f.dimension,
                        "quantile": f.quantile,
                        "forecast_value": f.forecast_value,
                        "color_code": f.color_code.value if f.color_code else None,
                    }
                )
    if not rows:
        return None
    path = Path(output_dir) / f"run_{run_id}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def get_sheets_client():
    try:
        import gspread
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        raise ImportError("Run: pip install gspread google-auth-oauthlib")

    token_path = CREDENTIALS_DIR / "token.json"
    secrets_path = CREDENTIALS_DIR / "client_secrets.json"
    service_path = CREDENTIALS_DIR / "service_account.json"

    if service_path.exists():
        return gspread.service_account(filename=str(service_path))

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SHEET_SCOPES)

    if creds and creds.valid:
        return gspread.authorize(creds)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not secrets_path.exists():
            raise FileNotFoundError(f"OAuth secrets not found at {secrets_path}")
        flow = InstalledAppFlow.from_client_secrets_file(
            str(secrets_path), SHEET_SCOPES
        )
        creds = flow.run_local_server(port=0)

    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    return gspread.authorize(creds)


def publish_to_sheet(run_data: dict, sheet_id: str = DEFAULT_SHEET_ID) -> int:
    import gspread

    client = get_sheets_client()
    sheet = client.open_by_key(sheet_id)

    try:
        ws = sheet.worksheet("Pending Review")
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet("Pending Review", rows=1000, cols=20)
        ws.update("A1", [SHEET_HEADERS])
        ws.freeze(rows=1)

    rows = []
    run_id = run_data.get("run_id", "unknown")
    created_at = run_data.get("created_at", "")

    for question in run_data.get("questions", []):
        validation = question.get("validation") or {}
        response = question.get("response") or {}
        rationale = response.get("rationale", "")
        sources = ", ".join(response.get("sources", []))

        for forecast in response.get("forecasts", []):
            q_id = question.get("id", "")
            result_id = f"{run_id}_{q_id}_{forecast.get('forecast_date')}_{forecast.get('dimension')}_{forecast.get('quantile')}"
            rows.append(
                [
                    result_id,
                    run_id,
                    created_at,
                    q_id,
                    question.get("name", ""),
                    forecast.get("forecast_date", ""),
                    forecast.get("dimension", ""),
                    forecast.get("quantile", ""),
                    forecast.get("forecast_value", ""),
                    forecast.get("color_code", ""),
                    str(validation.get("ok", False)),
                    str(validation.get("usable_for_scoring", False)),
                    rationale[:SHEET_TEXT_LIMIT],
                    sources[:SHEET_TEXT_LIMIT],
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

    if rows:
        ws.append_rows(rows, value_input_option="USER_ENTERED")
    return len(rows)


def setup_sheet(sheet_id: str = DEFAULT_SHEET_ID) -> None:
    import gspread

    client = get_sheets_client()
    sheet = client.open_by_key(sheet_id)

    try:
        pending_ws = sheet.worksheet("Pending Review")
        pending_ws.clear()
        pending_ws.update("A1", [SHEET_HEADERS])
        pending_ws.freeze(rows=1)
    except gspread.WorksheetNotFound:
        pending_ws = sheet.add_worksheet("Pending Review", rows=1000, cols=20)
        pending_ws.update("A1", [SHEET_HEADERS])
        pending_ws.freeze(rows=1)

    try:
        reviewed_ws = sheet.worksheet("Reviewed")
        reviewed_ws.clear()
        reviewed_ws.update("A1", [SHEET_HEADERS])
        reviewed_ws.freeze(rows=1)
    except gspread.WorksheetNotFound:
        reviewed_ws = sheet.add_worksheet("Reviewed", rows=1000, cols=20)
        reviewed_ws.update("A1", [SHEET_HEADERS])
        reviewed_ws.freeze(rows=1)

    reviewed_col_index = SHEET_HEADERS.index("reviewed")
    for ws in [pending_ws, reviewed_ws]:
        sheet.batch_update({
            "requests": [{
                "setDataValidation": {
                    "range": {
                        "sheetId": ws.id,
                        "startRowIndex": 1,
                        "endRowIndex": 1000,
                        "startColumnIndex": reviewed_col_index,
                        "endColumnIndex": reviewed_col_index + 1,
                    },
                    "rule": {
                        "condition": {"type": "BOOLEAN"},
                        "showCustomUi": True,
                    },
                }
            }]
        })

    try:
        instructions_ws = sheet.worksheet("Instructions")
        instructions_ws.clear()
    except gspread.WorksheetNotFound:
        instructions_ws = sheet.add_worksheet("Instructions", rows=50, cols=5)

    instructions_ws.update("A1", INSTRUCTIONS_CONTENT)

    print(f"Sheet setup complete: {sheet_id}")


def get_reviewed_items(sheet_id: str = DEFAULT_SHEET_ID) -> tuple[list[dict], list[int]]:
    import gspread

    client = get_sheets_client()
    sheet = client.open_by_key(sheet_id)
    try:
        ws = sheet.worksheet("Pending Review")
    except gspread.WorksheetNotFound:
        return [], []

    reviewed_items = []
    row_numbers = []

    for i, row in enumerate(ws.get_all_records(), start=2):
        reviewed_raw = row.get("reviewed", "")
        if isinstance(reviewed_raw, bool):
            reviewed = reviewed_raw
        else:
            reviewed = str(reviewed_raw).strip().lower() in ("true", "1", "yes")

        val_override = row.get("value_override")
        color_override = row.get("color_override")
        has_override = val_override not in (None, "") or color_override not in (None, "")

        if reviewed or has_override:
            reviewed_items.append(
                {
                    "result_id": row.get("result_id"),
                    "run_id": row.get("run_id"),
                    "question_id": row.get("question_id"),
                    "question_name": row.get("question_name"),
                    "forecast_date": row.get("forecast_date"),
                    "dimension": row.get("dimension"),
                    "quantile": row.get("quantile"),
                    "original_value": row.get("forecast_value"),
                    "original_color": row.get("color_code"),
                    "override_value": row.get("value_override"),
                    "override_color": row.get("color_override"),
                    "validation_ok": row.get("validation_ok"),
                    "usable_for_scoring": row.get("usable_for_scoring"),
                    "rationale": row.get("rationale"),
                    "sources": row.get("sources"),
                    "notes": row.get("notes"),
                    "review_status": "Reviewed",
                    "created_at": row.get("created_at"),
                }
            )
            row_numbers.append(i)
    return reviewed_items, row_numbers


def move_to_reviewed(
    sheet_id: str,
    reviewed_items: list[dict],
    row_numbers: list[int]
) -> int:
    import gspread

    if not reviewed_items:
        return 0

    client = get_sheets_client()
    sheet = client.open_by_key(sheet_id)

    try:
        reviewed_ws = sheet.worksheet("Reviewed")
    except gspread.WorksheetNotFound:
        reviewed_ws = sheet.add_worksheet("Reviewed", rows=1000, cols=20)
        reviewed_ws.update("A1", [SHEET_HEADERS])
        reviewed_ws.freeze(rows=1)

    reviewed_at = datetime.now(timezone.utc).isoformat()
    rows_to_add = []
    for item in reviewed_items:
        rows_to_add.append([
            item.get("result_id", ""),
            item.get("run_id", ""),
            item.get("created_at", ""),
            item.get("question_id", ""),
            item.get("question_name", ""),
            item.get("forecast_date", ""),
            item.get("dimension", ""),
            item.get("quantile", ""),
            item.get("original_value", ""),
            item.get("original_color", ""),
            item.get("validation_ok", ""),
            item.get("usable_for_scoring", ""),
            item.get("rationale", "")[:SHEET_TEXT_LIMIT] if item.get("rationale") else "",
            item.get("sources", "")[:SHEET_TEXT_LIMIT] if item.get("sources") else "",
            item.get("override_value", ""),
            item.get("override_color", ""),
            "TRUE",
            item.get("notes", ""),
            reviewed_at,
        ])

    if rows_to_add:
        reviewed_ws.append_rows(rows_to_add, value_input_option="USER_ENTERED")

    try:
        pending_ws = sheet.worksheet("Pending Review")
        for row_num in sorted(row_numbers, reverse=True):
            pending_ws.delete_rows(row_num)
    except Exception as e:
        print(f"Warning: Could not delete rows from Pending Review: {e}")

    return len(rows_to_add)


def try_merge_bigquery_rows(
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

    import pandas as pd
    from bigquery import merge_bq

    df = pd.DataFrame(rows)

    if dtypes:
        for col, dtype in dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

    try:
        return merge_bq(
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


def write_surveillance_to_bigquery(run_data: dict, prod: bool = False) -> dict:
    prefix = "" if prod else "dev_"
    run_id = run_data.get("run_id")
    created_at = run_data.get("created_at")
    model = run_data.get("model")

    result_rows = []
    for question in run_data.get("questions", []):
        validation = question.get("validation") or {}
        response = question.get("response") or {}
        quality = question.get("quality") or {}
        q_id = question.get("id")

        for forecast in response.get("forecasts", []):
            forecast_val = forecast.get("forecast_value")
            color = forecast.get("color_code")
            result_rows.append({
                "result_id": f"{run_id}_{q_id}_{forecast.get('forecast_date')}_{forecast.get('dimension')}_{forecast.get('quantile')}",
                "run_id": run_id,
                "model": model,
                "question_id": q_id,
                "question_name": question.get("name"),
                "forecast_date": forecast.get("forecast_date"),
                "dimension": forecast.get("dimension"),
                "quantile": forecast.get("quantile"),
                "forecast_value": forecast_val,
                "color_code": color,
                "rationale": (response.get("rationale") or "")[:2000],
                "sources": ", ".join(response.get("sources") or [])[:1000],
                "validation_ok": validation.get("ok", False),
                "usable_for_scoring": validation.get("usable_for_scoring", False),
                "quality_confidence": quality.get("confidence"),
                "quality_adequate": quality.get("adequate"),
                "review_status": None,
                "override_value": None,
                "override_color": None,
                "final_value": forecast_val,
                "final_color": color,
                "notes": None,
                "reviewed_at": None,
                "created_at": created_at,
                "ingestion_timestamp": datetime.now(timezone.utc),
            })

    evidence_rows = []
    for question in run_data.get("questions", []):
        for i, ev in enumerate(question.get("evidence") or []):
            evidence_rows.append({
                "evidence_id": f"{run_id}_{question.get('id')}_{i}",
                "run_id": run_id,
                "question_id": question.get("id"),
                "source_type": ev.get("source_type"),
                "url": ev.get("url"),
                "title": ev.get("title"),
                "snippet": (ev.get("snippet") or "")[:1000],
                "full_text": (ev.get("full_text") or "")[:5000],
                "created_at": created_at,
                "ingestion_timestamp": datetime.now(timezone.utc),
            })

    run_rows = [{
        "run_id": run_id,
        "created_at": created_at,
        "model": model,
        "question_count": len(run_data.get("questions", [])),
        "success_count": sum(1 for q in run_data.get("questions", []) if q.get("validation", {}).get("ok")),
        "error_count": sum(1 for q in run_data.get("questions", []) if not q.get("response")),
        "ingestion_timestamp": datetime.now(timezone.utc),
    }]

    result_dtypes = {
        "override_value": "Float64",
        "override_color": "string",
        "review_status": "string",
        "notes": "string",
        "quality_confidence": "Int64",
    }

    return {
        "results": try_merge_bigquery_rows(
            "BigQuery results", result_rows,
            pk="result_id", dataset=f"{prefix}fact", table="surveillance_result",
            clock_col="ingestion_timestamp", dtypes=result_dtypes,
        ),
        "evidence": try_merge_bigquery_rows(
            "BigQuery evidence", evidence_rows,
            pk="evidence_id", dataset=f"{prefix}fact", table="surveillance_evidence",
            clock_col="ingestion_timestamp",
        ),
        "runs": try_merge_bigquery_rows(
            "BigQuery run", run_rows,
            pk="run_id", dataset=f"{prefix}fact", table="surveillance_run",
            clock_col="ingestion_timestamp",
        ),
    }


def get_existing_result_ids(result_ids: list[str], prod: bool = False) -> set[str]:
    if not result_ids:
        return set()

    from bigquery import query_bq

    prefix = "" if prod else "dev_"
    ids_str = ", ".join(f"'{rid}'" for rid in result_ids)
    query = f"""
    SELECT result_id
    FROM `ai-panel-of-experts.{prefix}fact.surveillance_result`
    WHERE result_id IN ({ids_str})
    """
    try:
        df = query_bq(query)
        return set(df["result_id"].tolist())
    except Exception:
        return set()


def sync_reviews_to_bigquery(reviewed_items: list[dict], prod: bool = False) -> dict:
    if not reviewed_items:
        return {"results": None, "skipped": 0}

    prefix = "" if prod else "dev_"
    reviewed_at = datetime.now(timezone.utc)

    pending_rows = []
    for item in reviewed_items:
        result_id = item.get("result_id")

        forecast_val = to_float(item.get("original_value"))
        override_val = to_float(item.get("override_value"))
        final_value = override_val if override_val is not None else forecast_val

        override_color = item.get("override_color")
        original_color = item.get("original_color")
        final_color = override_color if override_color not in (None, "") else original_color

        pending_rows.append({
            "result_id": result_id,
            "review_status": "Reviewed",
            "override_value": override_val,
            "override_color": override_color if override_color not in (None, "") else None,
            "final_value": final_value,
            "final_color": final_color,
            "usable_for_scoring": True,
            "notes": item.get("notes"),
            "reviewed_at": reviewed_at,
            "ingestion_timestamp": datetime.now(timezone.utc),
        })

    all_ids = [r["result_id"] for r in pending_rows]
    existing_ids = get_existing_result_ids(all_ids, prod=prod)

    rows = [r for r in pending_rows if r["result_id"] in existing_ids]
    skipped = len(pending_rows) - len(rows)

    if skipped > 0:
        print(f"  Warning: {skipped} items skipped (raw result row missing in BigQuery)")

    if not rows:
        return {"results": None, "skipped": skipped}

    return {
        "results": try_merge_bigquery_rows(
            "BigQuery results (review update)", rows,
            pk="result_id", dataset=f"{prefix}fact", table="surveillance_result", clock_col="ingestion_timestamp",
        ),
        "skipped": skipped,
    }
