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
    # Context band (cols 0-5): identify the row + its status at a glance.
    "question_name",         # 0
    "target_date",           # 1  the date the question asks about
    "dimension",             # 2  sub-metric breakdown; "Overall" if none
    "status",                # 3  DERIVED: resolved / due_unresolved / resolved_early / forecast
    "question_type",         # 4  quantile / probability / when
    "unit",                  # 5  display unit for numeric values
    # LLM/judge band (cols 6-17): the answer, the uncertainty, and the trust signals.
    "llm_answer",            # 6  value AT target_date (resolved value if resolved, else q50 estimate)
    "q25",                   # 7  25th-percentile estimate (non-resolved rows)
    "q75",                   # 8  75th-percentile estimate (non-resolved rows)
    "judge_confidence",      # 9  judge's methodology/evidence confidence (0-100)
    "validation_issues",     # 10 deterministic validator issues, if any
    "judge_reason",          # 11 judge explanation for the confidence score
    "rationale",             # 12 LLM's reasoning summary
    "all_sources",           # 13 every URL the LLM consulted
    "resolution_source",     # 14 citation for the resolved value (resolved rows)
    "resolution_source_date",  # 15 date the resolved value represents (resolved rows)
    "current_estimate",      # 16 LLM's value as of today (context, not scored)
    "llm_color",             # 17 raw LLM color_code (already encoded in status)
    # Reviewer band (cols 18-23): fill left-to-right.
    "review_verdict",        # 18 correct / close / wrong / confidently wrong
    "review_value",          # 19 corrected value (if verdict != correct)
    "review_source",         # 20 citation for review_value
    "review_color",          # 21 color override if you disagree with the LLM
    "review_notes",          # 22 free-form observations
    "reviewed",              # 23 checkbox - check when done
    # Reference band (cols 24-28): question text + sync keys.
    "question_text",         # 24 full question sentence from BQ
    "resolution_criteria",   # 25 how FRI says to score it
    "group_id",              # 26 BQ sync key
    "question_id",           # 27
    "run_id",                # 28
]

VERDICT_OPTIONS = ["correct", "close", "wrong", "confidently wrong"]
COLOR_OPTIONS = ["black", "dark gray", "light gray", "white"]

# Two-column glossary: [term, definition]. Single-cell rows are the title, section
# headers, intro, and the closing line. Formatting is applied in _format_instructions().
INSTRUCTIONS_CONTENT = [
    ["LEAP Surveillance Review"],
    [""],
    ["Each surveillance run lives in its own `run_<run_id>` tab. Open the most recent tab to review the latest run."],
    ["One row per question / target date / dimension. Confirm or correct the LLM's value, tick `reviewed`, then run sync."],
    [""],
    ["STATUS"],
    ["resolved", "Date passed, LLM found a published value. Confirm it."],
    ["due_unresolved", "Date passed, no published value yet. Track down the real value."],
    ["forecast", "Future date. Leave it unless the value or rationale looks wrong."],
    ["resolved_early", "Future date, outcome already settled. Confirm the value."],
    [""],
    ["FILL IN"],
    ["review_value", "Correct value from your research, or a forecast correction."],
    ["review_source", "Link or citation for that value."],
    ["review_verdict", "correct / close / wrong / confidently wrong."],
    ["review_color", "Override color if you'd classify the certainty differently."],
    ["review_notes", "Anything worth flagging."],
    ["reviewed", "Tick once the row is done."],
    [""],
    ["Then run: leap-surveillance sync"],
    [""],
    ["KEY LLM COLUMNS"],
    ["llm_answer", "Resolved value, or q50 forecast."],
    ["q25 / q75", "Uncertainty range around q50 (forecast rows)."],
    ["judge_confidence", "Second model's confidence in evidence + methodology (0-100)."],
    ["validation_issues", "Mechanical checks: missing rows, mixed colors, non-increasing quantiles."],
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


def _run_tab_name(run_id: str) -> str:
    return f"{RUN_TAB_PREFIX}{run_id}"


def _latest_run_tab(sheet):
    """Return the latest run_* worksheet, or None."""
    run_tabs = [ws for ws in sheet.worksheets() if ws.title.startswith(RUN_TAB_PREFIX)]
    if not run_tabs:
        return None
    return max(run_tabs, key=lambda ws: ws.title)


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
        rationale = safe_str(response.get("rationale", ""))[:SHEET_TEXT_LIMIT]
        sources = ", ".join(response.get("sources", []))[:SHEET_TEXT_LIMIT]

        # Group forecasts by (forecast_date, dimension) -> {quantile: forecast}
        groups: dict[tuple, dict] = defaultdict(dict)
        for forecast in response.get("forecasts", []):
            fdate = forecast.get("forecast_date", "")
            dim = forecast.get("dimension", "Overall")
            q = forecast.get("quantile")
            groups[(fdate, dim)][q] = forecast

        _, current_map, resolution_map = context_maps(response)

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
                q25_val = ""
                q75_val = ""
                res_source_date = safe_str(res_val.get("source_date", ""))
                res_source = safe_str(res_val.get("source", ""))[:SHEET_TEXT_LIMIT]
            else:
                td_val = safe_str(quants.get(50, {}).get("forecast_value", ""))
                q25_val = safe_str(quants.get(25, {}).get("forecast_value", ""))
                q75_val = safe_str(quants.get(75, {}).get("forecast_value", ""))
                res_source_date = ""
                res_source = ""

            cur_val_obj = current_map.get(dim) or current_map.get("Overall") or {}
            cur_val = safe_str(cur_val_obj.get("value", ""))

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
                "q25": q25_val,
                "q75": q75_val,
                "resolution_source_date": res_source_date,
                "resolution_source": res_source,
                "rationale": rationale,
                "all_sources": sources,
                "judge_confidence": confidence,
                "validation_issues": validation_issues,
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
    tab_name = _run_tab_name(run_id)
    ws = _create_run_tab(sheet, tab_name)
    rows = build_review_rows(run_data)

    if rows:
        start_row = 2
        end_row = start_row + len(rows) - 1
        ws.update(f"A{start_row}", rows, value_input_option="USER_ENTERED")
        _apply_row_validation(sheet, ws.id, start_row, end_row)
        _sort_review_rows(sheet, ws.id)

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
        _val("reviewed", None),
        _val("review_verdict", None),
        _val("review_color", None),
        _val("reviewed", {"condition": {"type": "BOOLEAN"}, "showCustomUi": True}),
        _val("review_verdict", _one_of(VERDICT_OPTIONS)),
        _val("review_color", _one_of(COLOR_OPTIONS)),
    ]})


def _format_instructions(sheet, ws) -> None:
    """Style the Instructions tab: bold title, section headers, and terms; set column widths."""
    header_prefixes = ("STATUS", "FILL IN", "KEY LLM COLUMNS")

    def _bold(row_idx: int, col_end: int = 1) -> dict:
        return {"repeatCell": {
            "range": {"sheetId": ws.id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                      "startColumnIndex": 0, "endColumnIndex": col_end},
            "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
            "fields": "userEnteredFormat.textFormat"}}

    reqs = [
        {"updateDimensionProperties": {
            "range": {"sheetId": ws.id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 1},
            "properties": {"pixelSize": 165}, "fields": "pixelSize"}},
        {"updateDimensionProperties": {
            "range": {"sheetId": ws.id, "dimension": "COLUMNS", "startIndex": 1, "endIndex": 2},
            "properties": {"pixelSize": 680}, "fields": "pixelSize"}},
        {"repeatCell": {
            "range": {"sheetId": ws.id, "startRowIndex": 0, "endRowIndex": 1,
                      "startColumnIndex": 0, "endColumnIndex": 1},
            "cell": {"userEnteredFormat": {"textFormat": {"bold": True, "fontSize": 13}}},
            "fields": "userEnteredFormat.textFormat"}},
    ]

    for i, row in enumerate(INSTRUCTIONS_CONTENT):
        if i == 0:
            continue
        first = (row[0] if row else "").strip()
        if not first:
            continue
        if len(row) >= 2:                          # term / definition row
            reqs.append(_bold(i))                  # bold the term in col A
        elif first.startswith(header_prefixes):    # section header
            reqs.append(_bold(i))

    sheet.batch_update({"requests": reqs})


# Column widths (pixels), one per REVIEW_HEADERS entry, in the same order.
_REVIEW_COL_WIDTHS = [
    180,  # question_name
    85,   # target_date
    120,  # dimension
    115,  # status (derived)
    95,   # question_type
    90,   # unit
    90,   # llm_answer
    60,   # q25
    60,   # q75
    80,   # judge_confidence
    170,  # validation_issues
    200,  # judge_reason
    220,  # rationale
    160,  # all_sources
    150,  # resolution_source
    130,  # resolution_source_date
    100,  # current_estimate
    80,   # llm_color
    110,  # review_verdict
    90,   # review_value
    150,  # review_source
    90,   # review_color
    180,  # review_notes
    75,   # reviewed
    150,  # question_text
    150,  # resolution_criteria
    140,  # group_id
    110,  # question_id
    110,  # run_id
]


def _create_run_tab(sheet, tab_name: str):
    """Create a per-run review tab with bold headers.

    If a tab with this name already exists, it is deleted and recreated so any prior
    formatting (backgrounds, conditional rules, freezes) does not persist."""
    import gspread

    try:
        existing = sheet.worksheet(tab_name)
        sheet.del_worksheet(existing)
    except gspread.WorksheetNotFound:
        pass
    ws = sheet.add_worksheet(tab_name, rows=1000, cols=len(REVIEW_HEADERS) + 2)
    ws.update("A1", [REVIEW_HEADERS])

    # Header-row category bands: cue the reader about what each column group is for.
    # Light backgrounds only; data rows stay plain white.
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
        _header_band(0, 6, 0.93, 0.93, 0.93),       # Context: light gray
        _header_band(6, 18, 0.85, 0.92, 0.98),      # LLM + judge output: light blue
        _header_band(18, 24, 0.86, 0.94, 0.86),     # Reviewer edits: light green
        _header_band(24, len(REVIEW_HEADERS), 0.96, 0.96, 0.96),  # Reference: very light gray
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
