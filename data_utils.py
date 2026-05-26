"""Data loading, local output, Sheets review, and BigQuery sync."""

import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from schemas import ExpectedForecast, QuestionSpec

DEFAULT_OUTPUT_DIR = "output"
DEFAULT_SHEET_ID = os.environ.get(
    "LEAP_SHEET_ID", "1lT7zVfKAsVZU7bKaEALq1AWApfFmWMisprTK42l7RDo"
)

SHEET_TEXT_LIMIT = 1000

SHEET_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]
CREDENTIALS_DIR = Path.home() / ".config" / "leap-surveillance"

# Single review tab; one row per question / target_date / dimension.
REVIEW_HEADERS = [
    # ── Identity (frozen; cols 0–3) ─────────────────────────────────────────
    "question_name",    # 0
    "target_date",      # 1  the date the question asks about
    "dimension",        # 2  sub-metric breakdown; "Overall" if none
    "type",             # 3  resolved (past) or forecast (future)
    # ── LLM output — read-only context (cols 4–12, blue header) ─────────────
    "color_code",       # 4   LLM's classification (black/dark gray/light gray/white)
    "confidence",       # 5   judge's confidence 0–100
    "llm_value",        # 6   resolved value (type=resolved) or median q50 (type=forecast)
    "q25",              # 7   25th-percentile forecast (forecast rows only)
    "q75",              # 8   75th-percentile forecast (forecast rows only)
    "source_date",      # 9   date of the data the LLM used (resolved rows)
    "data_source",      # 10  citation for that data (resolved rows)
    "rationale",        # 11  LLM's reasoning summary
    "sources",          # 12  URLs the LLM searched
    # ── Reviewer fills in (cols 13–18, green header) ─────────────────────────
    "actual_value",     # 13  true resolved value (resolved rows)
    "verified_source",  # 14  citation for actual_value (resolved rows)
    "score",            # 15  correct / close / wrong / confidently_wrong (resolved rows)
    "corrected_value",  # 16  your corrected median estimate if LLM is wrong (forecast rows)
    "corrected_color",  # 17  your corrected color classification if LLM is wrong (either type)
    "notes",            # 18  free-form observations
    # ── Status (cols 19–20, yellow header) ───────────────────────────────────
    "reviewed",         # 19  checkbox — check when done
    "reviewed_at",      # 20  auto-stamped by sync command
    # ── Pipeline metadata — hidden (cols 21–24) ───────────────────────────────
    "group_id",         # 21  BQ sync key (one sheet row → 7 BQ rows, one per quantile)
    "question_id",      # 22
    "run_id",           # 23
    "created_at",       # 24
]

SCORE_OPTIONS = ["correct", "close", "wrong", "confidently_wrong"]
COLOR_OPTIONS = ["black", "dark gray", "light gray", "white"]

INSTRUCTIONS_CONTENT = [
    ["LEAP Surveillance Review"],
    [""],
    ["Each row is one question / target date / dimension. An LLM used web search to either"],
    ["find the resolved answer (type=resolved) or produce a forecast (type=forecast)."],
    [""],
    ["FOR RESOLVED ROWS (type = resolved)"],
    ["Read the llm_value and rationale, then fill in:"],
    ["  actual_value    — the true answer from your own research"],
    ["  verified_source — your source"],
    ["  score           — correct / close / wrong / confidently_wrong"],
    [""],
    ["FOR FORECAST ROWS (type = forecast)"],
    ["llm_value is the median estimate; q25/q75 is the uncertainty range."],
    ["Only fill in if something seems clearly wrong:"],
    ["  corrected_value — your corrected median"],
    ["  corrected_color — black/dark gray/light gray/white (how open is the question?)"],
    [""],
    ["Check 'reviewed' when done. Then run:  python run_surveillance.py sync"],
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


def _date_value_type(forecast_date: str, today: date | None = None) -> str:
    today = today or datetime.now(timezone.utc).date()
    try:
        return "resolution" if date.fromisoformat(forecast_date) < today else "forecast"
    except ValueError:
        return "forecast"


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
        return [
            ExpectedForecast(d, dim, 50, _date_value_type(d))
            for d in dates
            for dim in dimensions
        ]

    if question_type == "when":
        return [
            ExpectedForecast(TIMING_FORECAST_DATE, dim, p)
            for dim in dimensions
            for p in FULL_QUANTILES
        ]

    return [
        ExpectedForecast(d, dim, p, _date_value_type(d))
        for d in dates
        for dim in dimensions
        for p in FULL_QUANTILES
    ]


def _context_maps(response: dict) -> tuple[dict, dict, dict]:
    official = {
        item.get("dimension", "Overall"): item
        for item in response.get("last_official_values", []) or []
    }
    current = {
        item.get("dimension", "Overall"): item
        for item in response.get("current_resolution_values", []) or []
    }
    resolution = {
        (item.get("forecast_date"), item.get("dimension", "Overall")): item
        for item in response.get("resolution_values", []) or []
    }
    return official, current, resolution


def _forecast_output_row(
    *,
    q_id: str,
    q_name: str,
    forecast: dict,
    response: dict,
    validation: dict | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
    result_id: str | None = None,
) -> dict:
    official, current, resolution = _context_maps(response)
    dim = forecast.get("dimension", "Overall")
    forecast_date = forecast.get("forecast_date", "")
    official_value = official.get(dim) or official.get("Overall") or {}
    current_value = current.get(dim) or current.get("Overall") or {}
    resolution_value = resolution.get((forecast_date, dim)) or resolution.get((forecast_date, "Overall")) or {}
    target_value_type = forecast.get("value_type", "forecast")
    forecast_target_value = forecast.get("forecast_value", "") if target_value_type == "forecast" else ""
    resolution_target_value = resolution_value.get("value", "") if target_value_type == "resolution" else ""

    row = {
        "question_id": q_id,
        "question_name": q_name,
        "target_value_type": target_value_type,
        "target_date": forecast_date,
        "dimension": dim,
        "quantile": forecast.get("quantile", ""),
        "forecast_target_value": forecast_target_value,
        "resolution_target_value": resolution_target_value,
        "color_code": forecast.get("color_code", ""),
        "resolution_source_date": resolution_value.get("source_date", ""),
        "resolution_source": resolution_value.get("source", ""),
        "current_resolution_value": current_value.get("value", ""),
        "current_resolution_confidence": current_value.get("confidence", ""),
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


def _last_content_row(values: list[list], headers: list[str]) -> int:
    reviewed_idx = headers.index("reviewed") if "reviewed" in headers else -1
    last_row = 1
    for row_number, row in enumerate(values[1:], start=2):
        has_content = any(
            safe_str(cell).strip()
            for idx, cell in enumerate(row)
            if idx != reviewed_idx
        )
        if has_content:
            last_row = row_number
    return last_row


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


def build_run_data(
    run_id,
    model,
    questions,
    responses,
    validations,
    evidences,
    quality_reports,
    costs_list=None,
) -> dict:
    costs_iter = costs_list or [None] * len(questions)
    results = []
    for q, resp, val, ev, qr, costs in zip(
        questions, responses, validations, evidences, quality_reports, costs_iter
    ):
        result = {
            "id": q.id,
            "name": q.name,
            "response": resp.model_dump(mode="json") if resp else None,
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
                    "url_verified": e.url_verified,
                    "http_status": e.http_status,
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
        if costs is not None:
            result["cost"] = costs.as_dict()
        results.append(result)

    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "questions": results,
    }


def write_json_output(run_data: dict, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"run_{run_data['run_id']}.json"
    with open(path, "w") as f:
        json.dump(run_data, f, indent=2, default=str)
    return str(path)


def write_csv_output(run_id, questions, responses, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for q, resp in zip(questions, responses):
        if resp:
            response = resp.model_dump(mode="json")
            for forecast in response.get("forecasts", []):
                rows.append(
                    _forecast_output_row(
                        q_id=q.id,
                        q_name=q.name,
                        forecast=forecast,
                        response=response,
                    )
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
        ws = sheet.worksheet("Review")
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet("Review", rows=1000, cols=len(REVIEW_HEADERS) + 2)
        ws.update("A1", [REVIEW_HEADERS])
        ws.freeze(rows=1)

    rows: list[list] = []
    run_id = run_data.get("run_id", "unknown")
    created_at = run_data.get("created_at", "")

    for question in run_data.get("questions", []):
        q_id = question.get("id", "")
        q_name = question.get("name", "")
        response = question.get("response") or {}
        quality = question.get("quality") or {}
        confidence = safe_str(quality.get("confidence", ""))
        rationale = safe_str(response.get("rationale", ""))[:SHEET_TEXT_LIMIT]
        sources = ", ".join(response.get("sources", []))[:SHEET_TEXT_LIMIT]

        # Group forecasts by (forecast_date, dimension) → {quantile: forecast}
        groups: dict[tuple, dict] = defaultdict(dict)
        for forecast in response.get("forecasts", []):
            fdate = forecast.get("forecast_date", "")
            dim = forecast.get("dimension", "Overall")
            q = forecast.get("quantile")
            groups[(fdate, dim)][q] = forecast

        _, _, resolution_map = _context_maps(response)

        for (fdate, dim), quants in groups.items():
            group_id = f"{run_id}_{q_id}_{fdate}_{dim}"
            any_forecast = next(iter(quants.values()))
            color_code = any_forecast.get("color_code", "")
            row_type = "resolved" if color_code == "black" else "forecast"

            res_val = resolution_map.get((fdate, dim)) or resolution_map.get((fdate, "Overall")) or {}

            if row_type == "resolved":
                llm_val = safe_str(any_forecast.get("forecast_value", ""))
                if not llm_val:
                    llm_val = safe_str(res_val.get("value", ""))
                q25_val = ""
                q75_val = ""
                res_source_date = safe_str(res_val.get("source_date", ""))
                res_source = safe_str(res_val.get("source", ""))[:SHEET_TEXT_LIMIT]
            else:
                llm_val = safe_str(quants.get(50, {}).get("forecast_value", ""))
                q25_val = safe_str(quants.get(25, {}).get("forecast_value", ""))
                q75_val = safe_str(quants.get(75, {}).get("forecast_value", ""))
                res_source_date = ""
                res_source = ""

            row = {
                "question_name": q_name,
                "target_date": fdate,
                "dimension": dim,
                "type": row_type,
                "color_code": color_code,
                "confidence": confidence,
                "llm_value": llm_val,
                "q25": q25_val,
                "q75": q75_val,
                "source_date": res_source_date,
                "data_source": res_source,
                "rationale": rationale,
                "sources": sources,
                "actual_value": "",
                "verified_source": "",
                "score": "",
                "corrected_value": "",
                "corrected_color": "",
                "notes": "",
                "reviewed": "",
                "reviewed_at": "",
                "group_id": group_id,
                "question_id": q_id,
                "run_id": run_id,
                "created_at": created_at,
            }
            rows.append([row.get(h, "") for h in REVIEW_HEADERS])

    if rows:
        existing = ws.get_all_values()
        next_row = _last_content_row(existing, REVIEW_HEADERS) + 1
        ws.update(f"A{next_row}", rows, value_input_option="USER_ENTERED")
        # Apply validation only to the rows just written — no phantom arrows on empty rows.
        _apply_row_validation(sheet, ws.id, next_row, next_row + len(rows) - 1)

    return len(rows)


def _apply_row_validation(sheet, ws_id: int, start_row: int, end_row: int) -> None:
    """Apply checkbox and dropdown validation to a specific 1-indexed inclusive row range."""
    def _val(col_name: str, rule: dict) -> dict:
        idx = REVIEW_HEADERS.index(col_name)
        return {
            "setDataValidation": {
                "range": {
                    "sheetId": ws_id,
                    "startRowIndex": start_row - 1,  # 0-indexed
                    "endRowIndex": end_row,           # exclusive
                    "startColumnIndex": idx,
                    "endColumnIndex": idx + 1,
                },
                "rule": rule,
            }
        }
    sheet.batch_update({"requests": [
        _val("reviewed", {"condition": {"type": "BOOLEAN"}, "showCustomUi": True}),
        _val("score", {"condition": {"type": "ONE_OF_LIST", "values": [{"userEnteredValue": v} for v in SCORE_OPTIONS]}, "showCustomUi": True}),
        _val("corrected_color", {"condition": {"type": "ONE_OF_LIST", "values": [{"userEnteredValue": v} for v in COLOR_OPTIONS]}, "showCustomUi": True}),
    ]})


def setup_sheet(sheet_id: str = DEFAULT_SHEET_ID) -> None:
    """Create or reset the Review tab and Instructions tab.

    Removes legacy tabs from earlier sheet designs.
    """
    import gspread

    client = get_sheets_client()
    sheet = client.open_by_key(sheet_id)

    legacy_tabs = (
        "Pending Review",
        "Reviewed",
        "Resolution Review",
        "Forecast Review",
    )
    for old_name in legacy_tabs:
        try:
            sheet.del_worksheet(sheet.worksheet(old_name))
        except gspread.WorksheetNotFound:
            pass

    try:
        ws = sheet.worksheet("Review")
        ws.clear()
        ws.update("A1", [REVIEW_HEADERS])
        ws.freeze(rows=1)
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet("Review", rows=1000, cols=len(REVIEW_HEADERS) + 2)
        ws.update("A1", [REVIEW_HEADERS])
        ws.freeze(rows=1)

    # ── Formatting ────────────────────────────────────────────────────────────
    # Validation is applied per-publish in _apply_row_validation so it covers
    # only actual data rows and never produces phantom arrows on empty rows.
    def _color_header(start_col: int, end_col: int, r: float, g: float, b: float) -> dict:
        return {
            "repeatCell": {
                "range": {
                    "sheetId": ws.id,
                    "startRowIndex": 0, "endRowIndex": 1,
                    "startColumnIndex": start_col, "endColumnIndex": end_col,
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {"red": r, "green": g, "blue": b},
                        "textFormat": {"bold": True},
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat)",
            }
        }

    # Column widths (pixels), one per REVIEW_HEADERS entry
    col_widths = [
        220,  # question_name
        90,   # target_date
        130,  # dimension
        80,   # type
        85,   # color_code
        80,   # confidence
        90,   # llm_value
        70,   # q25
        70,   # q75
        90,   # source_date
        160,  # data_source
        260,  # rationale
        200,  # sources
        100,  # actual_value
        180,  # verified_source
        120,  # score
        100,  # corrected_value
        100,  # corrected_color
        180,  # notes
        80,   # reviewed
        110,  # reviewed_at
        160,  # group_id   (hidden)
        120,  # question_id (hidden)
        120,  # run_id      (hidden)
        140,  # created_at  (hidden)
    ]

    format_requests = [
        # Header color groups
        _color_header(0, 4, 0.91, 0.91, 0.91),      # identity: neutral gray
        _color_header(4, 13, 0.79, 0.90, 0.97),     # LLM output: light blue
        _color_header(13, 19, 0.83, 0.94, 0.83),    # reviewer: light green
        _color_header(19, 21, 1.00, 0.95, 0.80),    # status: light yellow
        _color_header(21, 25, 0.95, 0.95, 0.95),    # metadata: light gray (hidden)
        # Freeze header row + first 4 identity columns
        {
            "updateSheetProperties": {
                "properties": {
                    "sheetId": ws.id,
                    "gridProperties": {"frozenRowCount": 1, "frozenColumnCount": 4},
                },
                "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount",
            }
        },
        # Hide pipeline-internal metadata columns
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": ws.id, "dimension": "COLUMNS",
                    "startIndex": 21, "endIndex": 25,
                },
                "properties": {"hiddenByUser": True},
                "fields": "hiddenByUser",
            }
        },
    ]
    for i, width in enumerate(col_widths):
        format_requests.append({
            "updateDimensionProperties": {
                "range": {
                    "sheetId": ws.id, "dimension": "COLUMNS",
                    "startIndex": i, "endIndex": i + 1,
                },
                "properties": {"pixelSize": width},
                "fields": "pixelSize",
            }
        })
    sheet.batch_update({"requests": format_requests})

    try:
        instructions_ws = sheet.worksheet("Instructions")
        instructions_ws.clear()
    except gspread.WorksheetNotFound:
        instructions_ws = sheet.add_worksheet("Instructions", rows=50, cols=5)

    instructions_ws.update("A1", INSTRUCTIONS_CONTENT)
    print(f"Sheet setup complete: {sheet_id}")


def get_reviewed_items(sheet_id: str = DEFAULT_SHEET_ID) -> tuple[list[dict], list[int]]:
    """Read rows where 'reviewed' is checked or any correction field is filled."""
    import gspread

    client = get_sheets_client()
    sheet = client.open_by_key(sheet_id)

    try:
        ws = sheet.worksheet("Review")
    except gspread.WorksheetNotFound:
        return [], []

    reviewed_items: list[dict] = []
    row_numbers: list[int] = []

    for i, row in enumerate(ws.get_all_records(), start=2):
        reviewed_raw = row.get("reviewed", "")
        if isinstance(reviewed_raw, bool):
            reviewed = reviewed_raw
        else:
            reviewed = str(reviewed_raw).strip().lower() in ("true", "1", "yes")

        has_override = any(
            row.get(col) not in (None, "")
            for col in ("corrected_value", "corrected_color", "actual_value")
        )

        group_id = safe_str(row.get("group_id", "")).strip()
        if not group_id:
            continue

        if not (reviewed or has_override):
            continue

        reviewed_items.append({
            "group_id": group_id,
            "run_id": row.get("run_id"),
            "question_id": row.get("question_id"),
            "question_name": row.get("question_name"),
            "target_date": row.get("target_date"),
            "dimension": row.get("dimension"),
            "type": row.get("type"),
            "color_code": row.get("color_code"),
            "confidence": row.get("confidence"),
            "llm_value": row.get("llm_value"),
            "q25": row.get("q25"),
            "q75": row.get("q75"),
            "actual_value": row.get("actual_value"),
            "verified_source": row.get("verified_source"),
            "score": row.get("score"),
            "corrected_value": row.get("corrected_value"),
            "corrected_color": row.get("corrected_color"),
            "notes": row.get("notes"),
        })
        row_numbers.append(i)

    return reviewed_items, row_numbers


def stamp_reviewed_rows(
    sheet_id: str,
    reviewed_items: list[dict],
    row_numbers: list[int],
) -> int:
    """Stamp reviewed_at on each reviewed row in-place (single API call)."""
    if not reviewed_items:
        return 0

    import gspread
    from gspread.utils import rowcol_to_a1

    client = get_sheets_client()
    sheet = client.open_by_key(sheet_id)

    try:
        ws = sheet.worksheet("Review")
    except gspread.WorksheetNotFound:
        return 0

    reviewed_at = datetime.now(timezone.utc).isoformat()
    reviewed_at_col = REVIEW_HEADERS.index("reviewed_at") + 1
    updates = [
        {"range": rowcol_to_a1(r, reviewed_at_col), "values": [[reviewed_at]]}
        for r in row_numbers
    ]
    if updates:
        ws.batch_update(updates, value_input_option="USER_ENTERED")
    return len(row_numbers)


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
            color = forecast.get("color_code")
            result_id = f"{run_id}_{q_id}_{forecast.get('forecast_date')}_{forecast.get('dimension')}_{forecast.get('quantile')}"
            row = _forecast_output_row(
                q_id=q_id,
                q_name=question.get("name"),
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
                "model": model,
                "question_id": q_id,
                "question_name": question.get("name"),
                "target_value_type": row.get("target_value_type"),
                "target_date": row.get("target_date"),
                "dimension": forecast.get("dimension"),
                "quantile": forecast.get("quantile"),
                "forecast_target_value": row.get("forecast_target_value") if row.get("forecast_target_value") not in (None, "") else None,
                "resolution_target_value": row.get("resolution_target_value") if row.get("resolution_target_value") not in (None, "") else None,
                "color_code": color,
                "resolution_source_date": row.get("resolution_source_date") or None,
                "resolution_source": row.get("resolution_source") or None,
                "current_resolution_value": row.get("current_resolution_value") if row.get("current_resolution_value") not in (None, "") else None,
                "current_resolution_confidence": row.get("current_resolution_confidence") if row.get("current_resolution_confidence") not in (None, "") else None,
                "latest_official_value": row.get("latest_official_value") if row.get("latest_official_value") not in (None, "") else None,
                "latest_official_date": row.get("latest_official_date") or None,
                "latest_official_source": row.get("latest_official_source") or None,
                "rationale": (response.get("rationale") or "")[:2000],
                "sources": ", ".join(response.get("sources") or [])[:1000],
                "validation_ok": validation.get("ok", False),
                "usable_for_scoring": validation.get("usable_for_scoring", False),
                "quality_confidence": quality.get("confidence"),
                "quality_adequate": quality.get("adequate"),
                "review_status": None,
                "override_value": None,
                "override_color": None,
                "final_value": next((v for v in (row.get("forecast_target_value"), row.get("resolution_target_value")) if v not in (None, "")), None),
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
                "url_verified": ev.get("url_verified"),
                "http_status": ev.get("http_status"),
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
        "forecast_target_value": "Float64",
        "resolution_target_value": "Float64",
        "current_resolution_value": "Float64",
        "current_resolution_confidence": "Int64",
        "latest_official_value": "Float64",
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
    """Fan each reviewed sheet row (group_id) out to 7 BQ result_ids (one per quantile)."""
    if not reviewed_items:
        return {"results": None, "skipped": 0}

    prefix = "" if prod else "dev_"
    reviewed_at = datetime.now(timezone.utc)

    pending_rows = []
    for item in reviewed_items:
        group_id = item.get("group_id", "")
        row_type = item.get("type", "")
        override_color = item.get("corrected_color")
        override_color = override_color if override_color not in (None, "") else None
        free_notes = safe_str(item.get("notes")) or ""

        if row_type == "resolved":
            override_val = to_float(item.get("actual_value"))
            score = safe_str(item.get("score", ""))
            note_parts = []
            if score:
                note_parts.append(f"score:{score}")
            if free_notes:
                note_parts.append(free_notes)
            notes = " ".join(note_parts) or None
        else:
            override_val = to_float(item.get("corrected_value"))
            notes = free_notes or None

        for q in FULL_QUANTILES:
            result_id = f"{group_id}_{q}"
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

    all_ids = [r["result_id"] for r in pending_rows]
    existing_ids = get_existing_result_ids(all_ids, prod=prod)

    rows = [r for r in pending_rows if r["result_id"] in existing_ids]
    skipped = len(pending_rows) - len(rows)

    if skipped > 0:
        print(f"  Warning: {skipped} result_ids skipped (row missing in BigQuery)")

    if not rows:
        return {"results": None, "skipped": skipped}

    return {
        "results": try_merge_bigquery_rows(
            "BigQuery results (review update)", rows,
            pk="result_id", dataset=f"{prefix}fact", table="surveillance_result",
            clock_col="ingestion_timestamp",
        ),
        "skipped": skipped,
    }
