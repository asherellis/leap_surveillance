"""Google Sheets review UI for LEAP surveillance."""

from collections import defaultdict
from pathlib import Path

from .common import (
    DEFAULT_SHEET_ID,
    SHEET_TEXT_LIMIT,
    make_review_group_id,
    resolution_status,
    safe_str,
)
from .storage import context_maps


SHEET_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
CREDENTIALS_DIR = Path.home() / ".config" / "leap-surveillance"

REVIEW_HEADERS = [
    "question_name",
    "question_text",
    "resolution_criteria",
    "target_date",
    "dimension",
    "status",
    "question_type",
    "unit",
    "llm_answer",
    "q0",
    "q5",
    "q25",
    "q75",
    "q95",
    "q100",
    "judge_confidence",
    "judge_reason",
    "validation_issues",
    "missing_data",
    "browser_used",
    "rationale",
    "all_sources",
    "resolution_source",
    "resolution_source_date",
    "latest_official_value",
    "latest_official_date",
    "latest_official_source",
    "current_estimate",
    "current_estimate_confidence",
    "llm_color",
    "review_verdict",
    "review_value",
    "review_source",
    "review_color",
    "review_notes",
    "reviewed",
    "group_id",
    "question_id",
    "run_id",
]

VERDICT_OPTIONS = ["correct", "close", "wrong", "confidently wrong"]
COLOR_OPTIONS = ["black", "dark gray", "light gray", "white"]

INSTRUCTIONS_CONTENT = [
    ["LEAP Surveillance Review"],
    ["Use this sheet to review the results from each LEAP surveillance run."],
    ["Each run gets its own run_<run_id> tab. Each row is one question result for a target date and dimension."],
    [""],
    ["How to review"],
    ["Open the newest run tab."],
    ["Read the question, resolution criteria, answer, rationale, sources, and any validation issues."],
    ["Fill review_verdict for each row you review."],
    ["If you change the answer, fill review_value and review_source."],
    ["Tick reviewed when the row is done."],
    ["When finished, run leap-surveillance sync. To preview first, run leap-surveillance sync --no-bq."],
    [""],
    ["Column groups"],
    ["A-H", "Question, resolution criteria, target date, status, type, and unit."],
    ["I-AD", "The LLM's answer, the full forecast distribution, judge evaluation, rationale, and cited sources."],
    ["AE-AJ", "Your review."],
    ["AK-AM", "Sync identifiers — do not edit."],
    [""],
    ["Key columns"],
    ["status", "What the row needs: resolved (verify), due_unresolved (find the value), forecast (usually leave alone), resolved_early (verify)."],
    ["llm_answer", "The value to review. For forecast rows, this is q50."],
    ["current_estimate", "The LLM's value as of today — not the same as llm_answer (which is at the target date)."],
    ["q25 / q75", "50% confidence interval (IQR) around q50. Forecast rows only."],
    ["judge_confidence", "Confidence in the research, not confidence that the forecast will happen."],
    ["validation_issues", "Problems to check before approving."],
    ["missing_data", "The judge's specific list of gaps or weak spots in the response."],
    ["review_verdict", "correct / close / wrong / confidently wrong."],
    ["reviewed", "Only reviewed rows are synced."],
]


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


RUN_TAB_PREFIX = "run_"


def run_tab_name(run_id: str) -> str:
    return f"{RUN_TAB_PREFIX}{run_id}"


def _latest_run_tab(sheet):
    """Return the latest run_* worksheet, or None."""
    run_tabs = [ws for ws in sheet.worksheets() if ws.title.startswith(RUN_TAB_PREFIX)]
    if not run_tabs:
        return None
    return max(run_tabs, key=lambda ws: ws.title)


def _reorder_tabs(sheet) -> None:
    """Instructions leftmost, then run_* tabs newest-first, then anything else."""
    all_ws = sheet.worksheets()
    instructions = [w for w in all_ws if w.title == "Instructions"]
    run_tabs = sorted(
        (w for w in all_ws if w.title.startswith(RUN_TAB_PREFIX)),
        key=lambda w: w.title,
        reverse=True,
    )
    handled = set(instructions) | set(run_tabs)
    others = [w for w in all_ws if w not in handled]
    ordered = instructions + run_tabs + others
    if ordered != all_ws:
        sheet.reorder_worksheets(ordered)


def build_review_rows(run_data: dict) -> list[list]:
    """Collapse raw forecast rows into human-review rows ordered by REVIEW_HEADERS."""
    run_id = run_data.get("run_id", "unknown")
    rows: list[list] = []

    for question in run_data.get("questions", []):
        q_id = question.get("id", "")
        q_name = question.get("name", "")
        q_type = question.get("question_type", "")
        q_unit = question.get("unit", "")
        q_text = safe_str(question.get("question_text", ""))[:SHEET_TEXT_LIMIT]
        q_criteria = safe_str(question.get("resolution_criteria", ""))[:SHEET_TEXT_LIMIT]
        response = question.get("response") or {}
        quality = question.get("quality") or {}
        validation = question.get("validation") or {}
        confidence = safe_str(quality.get("confidence", ""))
        confidence_reason = safe_str(quality.get("reason", ""))[:SHEET_TEXT_LIMIT]
        validation_issues = ", ".join(validation.get("issues", []) or [])[:SHEET_TEXT_LIMIT]
        missing_data = ", ".join(quality.get("missing_data", []) or [])[:SHEET_TEXT_LIMIT]
        browser_used_str = "TRUE" if question.get("browser_used") else ""
        rationale = safe_str(response.get("rationale", ""))[:SHEET_TEXT_LIMIT]
        sources = ", ".join(response.get("sources", []))[:SHEET_TEXT_LIMIT]

        # Group forecasts by (forecast_date, dimension) -> {quantile: forecast}
        groups: dict[tuple, dict] = defaultdict(dict)
        for forecast in response.get("forecasts", []):
            fdate = forecast.get("forecast_date", "")
            dim = forecast.get("dimension", "Overall")
            q = forecast.get("quantile")
            groups[(fdate, dim)][q] = forecast

        official_map, current_map, resolution_map = context_maps(response)

        for (fdate, dim), quants in groups.items():
            group_id = make_review_group_id(run_id, q_id, fdate, dim)
            display_forecast = quants.get(50) or next(iter(quants.values()))
            color_code = display_forecast.get("color_code", "")
            row_type = "resolved" if color_code == "black" else "forecast"

            res_val = resolution_map.get((fdate, dim)) or resolution_map.get((fdate, "Overall")) or {}

            if row_type == "resolved":
                td_val = safe_str(display_forecast.get("forecast_value", ""))
                if not td_val:
                    td_val = safe_str(res_val.get("value", ""))
                q0_val = q5_val = q25_val = q75_val = q95_val = q100_val = ""
                res_source_date = safe_str(res_val.get("source_date", ""))
                res_source = safe_str(res_val.get("source", ""))[:SHEET_TEXT_LIMIT]
            else:
                td_val = safe_str(quants.get(50, {}).get("forecast_value", ""))
                q0_val = safe_str(quants.get(0, {}).get("forecast_value", ""))
                q5_val = safe_str(quants.get(5, {}).get("forecast_value", ""))
                q25_val = safe_str(quants.get(25, {}).get("forecast_value", ""))
                q75_val = safe_str(quants.get(75, {}).get("forecast_value", ""))
                q95_val = safe_str(quants.get(95, {}).get("forecast_value", ""))
                q100_val = safe_str(quants.get(100, {}).get("forecast_value", ""))
                res_source_date = ""
                res_source = ""

            cur_val_obj = current_map.get(dim) or current_map.get("Overall") or {}
            cur_val = safe_str(cur_val_obj.get("value", ""))
            cur_conf = safe_str(cur_val_obj.get("confidence", ""))

            off_val_obj = official_map.get(dim) or official_map.get("Overall") or {}
            off_val = safe_str(off_val_obj.get("value", ""))
            off_date = safe_str(off_val_obj.get("date", ""))
            off_source = safe_str(off_val_obj.get("source", ""))[:SHEET_TEXT_LIMIT]

            row = {
                "question_name": q_name,
                "target_date": fdate,
                "dimension": dim,
                "question_type": q_type,
                "unit": q_unit,
                "status": resolution_status(fdate, color_code),
                "llm_color": color_code,
                "llm_answer": td_val,
                "current_estimate": cur_val,
                "current_estimate_confidence": cur_conf,
                "q0": q0_val,
                "q5": q5_val,
                "q25": q25_val,
                "q75": q75_val,
                "q95": q95_val,
                "q100": q100_val,
                "resolution_source_date": res_source_date,
                "resolution_source": res_source,
                "latest_official_value": off_val,
                "latest_official_date": off_date,
                "latest_official_source": off_source,
                "rationale": rationale,
                "all_sources": sources,
                "judge_confidence": confidence,
                "validation_issues": validation_issues,
                "missing_data": missing_data,
                "browser_used": browser_used_str,
                "review_value": "",
                "review_source": "",
                "review_verdict": "",
                "review_notes": "",
                "review_color": "",
                "reviewed": "",
                "question_text": q_text,
                "resolution_criteria": q_criteria,
                "judge_reason": confidence_reason,
                "group_id": group_id,
                "question_id": q_id,
                "run_id": run_id,
            }
            rows.append([row.get(h, "") for h in REVIEW_HEADERS])

    return rows


def publish_to_sheet(run_data: dict, sheet_id: str = DEFAULT_SHEET_ID) -> int:
    """Publish a run to a formatted `run_<run_id>` tab."""
    client = get_sheets_client()
    sheet = client.open_by_key(sheet_id)
    run_id = run_data.get("run_id", "unknown")
    tab_name = run_tab_name(run_id)
    ws = _create_run_tab(sheet, tab_name)
    rows = build_review_rows(run_data)

    if rows:
        start_row = 2
        end_row = start_row + len(rows) - 1
        ws.update(f"A{start_row}", rows, value_input_option="USER_ENTERED")
        _apply_row_validation(sheet, ws.id, start_row, end_row)
        _sort_review_rows(sheet, ws.id)

    _reorder_tabs(sheet)
    return len(rows)


def _sort_review_rows(sheet, ws_id: int) -> None:
    """Sort review rows (excluding header) by question_name, target_date, dimension."""
    cols = [
        REVIEW_HEADERS.index("question_name"),
        REVIEW_HEADERS.index("target_date"),
        REVIEW_HEADERS.index("dimension"),
    ]
    sheet.batch_update({"requests": [{
        "sortRange": {
            "range": {
                "sheetId": ws_id,
                "startRowIndex": 1,
                "startColumnIndex": 0,
                "endColumnIndex": len(REVIEW_HEADERS),
            },
            "sortSpecs": [{"dimensionIndex": c, "sortOrder": "ASCENDING"} for c in cols],
        }
    }]})


def _apply_row_validation(sheet, ws_id: int, start_row: int, end_row: int) -> None:
    def _val(col_name: str, rule: dict | None) -> dict:
        idx = REVIEW_HEADERS.index(col_name)
        req = {
            "setDataValidation": {
                "range": {
                    "sheetId": ws_id,
                    "startRowIndex": start_row - 1,  # 0-indexed
                    "endRowIndex": end_row,           # exclusive
                    "startColumnIndex": idx,
                    "endColumnIndex": idx + 1,
                },
            }
        }
        if rule is not None:
            req["setDataValidation"]["rule"] = rule
        return req

    def _one_of(values: list[str]) -> dict:
        return {
            "condition": {
                "type": "ONE_OF_LIST",
                "values": [{"userEnteredValue": value} for value in values],
            },
            "showCustomUi": True,
        }

    sheet.batch_update({"requests": [
        _val("reviewed", {"condition": {"type": "BOOLEAN"}, "showCustomUi": True}),
        _val("review_verdict", _one_of(VERDICT_OPTIONS)),
        _val("review_color", _one_of(COLOR_OPTIONS)),
    ]})


def _format_instructions(sheet, ws) -> None:
    """Style the Instructions tab."""
    section_headers = ("How to review", "Column groups", "Key columns")

    def _bold(row_idx: int, col_end: int = 1) -> dict:
        return {"repeatCell": {
            "range": {"sheetId": ws.id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                      "startColumnIndex": 0, "endColumnIndex": col_end},
            "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
            "fields": "userEnteredFormat.textFormat"}}

    def _section_header(row_idx: int) -> dict:
        return {"repeatCell": {
            "range": {"sheetId": ws.id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                      "startColumnIndex": 0, "endColumnIndex": 2},
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93},
                "textFormat": {"bold": True, "fontSize": 12},
                "padding": {"top": 4, "bottom": 4, "left": 6, "right": 6}}},
            "fields": "userEnteredFormat(backgroundColor,textFormat,padding)"}}

    reqs = [
        {"updateDimensionProperties": {
            "range": {"sheetId": ws.id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 1},
            "properties": {"pixelSize": 165}, "fields": "pixelSize"}},
        {"updateDimensionProperties": {
            "range": {"sheetId": ws.id, "dimension": "COLUMNS", "startIndex": 1, "endIndex": 2},
            "properties": {"pixelSize": 700}, "fields": "pixelSize"}},
        {"repeatCell": {
            "range": {"sheetId": ws.id},
            "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP", "verticalAlignment": "TOP"}},
            "fields": "userEnteredFormat(wrapStrategy,verticalAlignment)"}},
        {"repeatCell": {
            "range": {"sheetId": ws.id, "startRowIndex": 0, "endRowIndex": 1,
                      "startColumnIndex": 0, "endColumnIndex": 2},
            "cell": {"userEnteredFormat": {"textFormat": {"bold": True, "fontSize": 16}}},
            "fields": "userEnteredFormat.textFormat"}},
    ]

    def _merge(row_idx: int) -> dict:
        return {"mergeCells": {
            "range": {"sheetId": ws.id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                      "startColumnIndex": 0, "endColumnIndex": 2},
            "mergeType": "MERGE_ALL"}}

    for i, row in enumerate(INSTRUCTIONS_CONTENT):
        first = (row[0] if row else "").strip() if row else ""
        if i == 0:
            reqs.append(_merge(i))
            continue
        if not first:
            continue
        if len(row) >= 2:
            reqs.append(_bold(i))
        elif first in section_headers:
            reqs.append(_merge(i))
            reqs.append(_section_header(i))
        else:
            reqs.append(_merge(i))

    sheet.batch_update({"requests": reqs})


_REVIEW_COL_WIDTHS = [
    180, 260, 260, 85, 120, 115, 95, 90,
    90, 55, 55, 55, 55, 55, 55,
    80, 200, 170, 170, 70, 220, 160,
    150, 130, 100, 110, 150, 100, 90, 80,
    110, 90, 150, 90, 180, 75,
    140, 110, 110,
]


def _create_run_tab(sheet, tab_name: str):
    """Create a per-run review tab."""
    import gspread

    try:
        existing = sheet.worksheet(tab_name)
        sheet.del_worksheet(existing)
    except gspread.WorksheetNotFound:
        pass
    ws = sheet.add_worksheet(tab_name, rows=1000, cols=len(REVIEW_HEADERS) + 2)
    ws.update("A1", [REVIEW_HEADERS])

    def _header_band(start_col: int, end_col: int, r: float, g: float, b: float) -> dict:
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

    format_requests = [
        {
            "updateSheetProperties": {
                "properties": {
                    "sheetId": ws.id,
                    "gridProperties": {"frozenRowCount": 1, "frozenColumnCount": 0},
                },
                "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount",
            }
        },
        {
            "repeatCell": {
                "range": {"sheetId": ws.id},
                "cell": {"userEnteredFormat": {"wrapStrategy": "CLIP"}},
                "fields": "userEnteredFormat.wrapStrategy",
            }
        },
        {
            "updateDimensionProperties": {
                "range": {"sheetId": ws.id, "dimension": "ROWS"},
                "properties": {"pixelSize": 21},
                "fields": "pixelSize",
            }
        },
        _header_band(0, 8, 0.93, 0.93, 0.93),
        _header_band(8, 30, 0.85, 0.92, 0.98),
        _header_band(30, 36, 0.86, 0.94, 0.86),
        _header_band(36, len(REVIEW_HEADERS), 0.96, 0.96, 0.96),
        {
            "setBasicFilter": {
                "filter": {
                    "range": {
                        "sheetId": ws.id,
                        "startRowIndex": 0,
                        "startColumnIndex": 0,
                        "endColumnIndex": len(REVIEW_HEADERS),
                    }
                }
            }
        },
        {
            "setDataValidation": {
                "range": {
                    "sheetId": ws.id,
                    "startRowIndex": 0,
                    "endRowIndex": 1000,
                    "startColumnIndex": 0,
                    "endColumnIndex": len(REVIEW_HEADERS) + 4,
                },
            }
        },
    ]
    for i, width in enumerate(_REVIEW_COL_WIDTHS):
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
    return ws


def setup_sheet(sheet_id: str = DEFAULT_SHEET_ID) -> None:
    """Refresh the Instructions tab without touching run tabs."""
    import gspread

    client = get_sheets_client()
    sheet = client.open_by_key(sheet_id)

    try:
        existing = sheet.worksheet("Instructions")
        sheet.del_worksheet(existing)
    except gspread.WorksheetNotFound:
        pass
    instructions_ws = sheet.add_worksheet("Instructions", rows=60, cols=4)

    rows2 = [list(r) + [""] * (2 - len(r)) for r in INSTRUCTIONS_CONTENT]
    instructions_ws.update("A1", rows2)
    _format_instructions(sheet, instructions_ws)
    _reorder_tabs(sheet)
    print(f"Instructions tab refreshed: {sheet_id}")


def get_reviewed_items(
    sheet_id: str = DEFAULT_SHEET_ID,
    tab_name: str | None = None,
) -> tuple[list[dict], list[int]]:
    """Read reviewed rows from a per-run tab. Defaults to the most-recent `run_*` tab."""
    import gspread

    client = get_sheets_client()
    sheet = client.open_by_key(sheet_id)

    if tab_name:
        try:
            ws = sheet.worksheet(tab_name)
        except gspread.WorksheetNotFound:
            print(f"  warning: no tab named '{tab_name}'")
            return [], []
    else:
        ws = _latest_run_tab(sheet)
        if ws is None:
            print(f"  warning: no run_* tabs found in sheet")
            return [], []
        print(f"  reading reviewed rows from '{ws.title}'")

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
            for col in ("review_value", "review_color")
        )

        group_id = safe_str(row.get("group_id", "")).strip()
        if not group_id:
            continue

        if not (reviewed or has_override):
            continue

        status = row.get("status", "")
        reviewed_items.append({
            "group_id": group_id,
            "run_id": row.get("run_id"),
            "question_id": row.get("question_id"),
            "question_name": row.get("question_name"),
            "target_date": row.get("target_date"),
            "dimension": row.get("dimension"),
            "question_type": row.get("question_type"),
            "unit": row.get("unit"),
            "type": "resolved" if str(status).strip().lower() in ("resolved", "resolved_early", "due_unresolved") else "forecast",
            "status": status,
            "judge_confidence": row.get("judge_confidence"),
            "validation_issues": row.get("validation_issues"),
            "llm_answer": row.get("llm_answer"),
            "current_estimate": row.get("current_estimate"),
            "q25": row.get("q25"),
            "q75": row.get("q75"),
            "review_value": row.get("review_value"),
            "review_source": row.get("review_source"),
            "review_verdict": row.get("review_verdict"),
            "review_color": row.get("review_color"),
            "review_notes": row.get("review_notes"),
        })
        row_numbers.append(i)

    return reviewed_items, row_numbers
