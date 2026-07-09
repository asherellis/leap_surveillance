"""Google Sheets review UI for LEAP surveillance."""

import re
from collections import defaultdict
from pathlib import Path

from .common import (
    DEFAULT_SHEET_ID,
    NEVER_YEAR,
    Q50_TOLERANCE,
    SHEET_TEXT_LIMIT,
    enum_value,
    is_empty,
    make_review_group_id,
    row_resolution_status,
    safe_str,
    to_float,
    within_relative_tolerance,
)
from .storage import context_maps, pick_by_dimension


SHEET_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
CREDENTIALS_DIR = Path.home() / ".config" / "leap-surveillance"

_MODEL_FIELDS = [
    "q50", "color",
    "q0", "q5", "q25", "q75", "q95", "q100",
    "latest_official_value", "latest_official_date", "latest_official_source",
    "current_estimate", "current_estimate_confidence",
    "resolution_value", "resolution_source", "resolution_source_date",
    "judge_confidence", "browser_status", "browser_url", "browser_objective", "browser_error",
    "missing_data", "validation_issues", "judge_reason",
    "rationale", "sources",
]

_METADATA_COLS = ["question_name", "needs_review", "status", "target_date", "dimension",
                  "question_type", "unit",
                  "question_resolution_status", "question_resolution_value",
                  "question_resolution_source", "question_resolution_source_date",
                  "question_text", "resolution_criteria"]
# Diagnostics apply to any run; consensus columns are dual-model-only.
_CONSENSUS_COLS = ["model_consensus", "consensus_q50_delta_pct", "confidence_tier"]
_DIAGNOSTIC_COLS = [
    "run_stability", "gpt_run_stability", "claude_run_stability", "runs_seen",
    "has_official_value", "has_current_value", "value_has_changed",
]
_REVIEWER_COLS = [
    "review_verdict", "review_last_official_value", "review_current_value",
    "reviewed_question_resolution_status", "reviewed_question_resolution_value",
    "review_source", "review_color", "review_notes", "reviewed",
]
_SYNC_COLS = ["review_row_id", "group_question_id", "question_id", "surveillance_timestamp"]


def _interleaved_model_cols() -> list[str]:
    """Generate (gpt_X, claude_X) interleaved pairs for every per-model field."""
    cols = []
    for f in _MODEL_FIELDS:
        cols.append(f"gpt_{f}")
        cols.append(f"claude_{f}")
    return cols


def _review_headers(mode: str) -> list[str]:
    """Build the review-tab layout for dual-model or single-model runs."""
    if mode == "both":
        return [*_METADATA_COLS, *_interleaved_model_cols(), *_CONSENSUS_COLS, *_DIAGNOSTIC_COLS, *_REVIEWER_COLS, *_SYNC_COLS]
    tag = "gpt" if mode == "gpt" else "claude"
    return [*_METADATA_COLS, *[f"{tag}_{f}" for f in _MODEL_FIELDS], *_DIAGNOSTIC_COLS, *_REVIEWER_COLS, *_SYNC_COLS]


VERDICT_OPTIONS = ["correct", "close", "partially right", "wrong", "confidently wrong", "unknown"]
COLOR_OPTIONS = ["black", "dark gray", "light gray", "white"]
RESOLUTION_STATUS_OPTIONS = ["open", "resolved", "failed_to_resolve", "projected"]

INSTRUCTIONS_CONTENT = [
    ["LEAP Surveillance Review"],
    ["Use this sheet to review the results from each LEAP surveillance run."],
    ["Each run gets its own run_<run_id> tab. Each row is one question x target_date x dimension."],
    [""],
    ["How to review"],
    ["Open the newest run tab."],
    ["Read the question and resolution criteria. Check the answer, rationale, and sources."],
    ["For rows you review: set review_verdict, fill review_last_official_value/current_value for baselines, fill reviewed_question_resolution_status/value for resolved outcomes, add review_source, and tick reviewed."],
    ["Skip rows you don't review — they sync using the model q50 as the projected value."],
    ["When finished, run leap-surveillance sync."],
    [""],
    ["Column groups"],
    ["Metadata", "Question name, status, target date, dimension, resolution state, question type/unit, full question text, and resolution criteria."],
    ["Model columns", "--both mode has GPT/Claude interleaved pairs. Forecast values come first, then official/current values and sources, then diagnostics."],
    ["Consensus/stability", "Model agreement, q50 delta, run stability, runs seen, confidence tier, value-change flags."],
    ["Reviewer columns", "Fill these in."],
    ["Sync IDs", "review_row_id, group_question_id, question_id, surveillance_timestamp — leave alone."],
    [""],
    ["Key columns"],
    ["status", "resolved (check the value), due_unresolved (look it up), forecast (usually leave), resolved_early (check the value)."],
    ["question_resolution_status", "open / resolved / failed_to_resolve / projected. This is the model/system view of whether a target row has a resolution value."],
    ["question_resolution_value", "Model/system resolution value for this target row, if one exists. Human-confirmed values belong in reviewed_question_resolution_value."],
    ["needs_review", "TRUE when the row likely needs human attention because it is resolved/due, changed, disagreed, or had extraction/quality issues."],
    ["gpt_q50 / claude_q50", "Each model's forecast median. Blank on resolved rows, which use gpt_resolution_value / claude_resolution_value instead."],
    ["gpt_resolution_value / claude_resolution_value", "Each model's confirmed value at the target date, for resolved (black) rows. Blank otherwise."],
    ["gpt_color / claude_color", "black = resolved/known, white = no defensible estimate, gray = forecast/partial information."],
    ["gpt_latest_official_value / claude_latest_official_value", "The most recent published official figure the model found."],
    ["gpt_current_estimate / claude_current_estimate", "The model's estimate of the current value (as of the run date). Different from gpt_q50/claude_q50, which is at the target date."],
    ["gpt_q25 / gpt_q75 / claude_q25 / claude_q75", "50% confidence interval (IQR) around gpt_q50 / claude_q50."],
    ["gpt_judge_confidence / claude_judge_confidence", "Judge's confidence in its adequacy assessment for this model's research."],
    ["gpt_missing_data / claude_missing_data", "Specific gaps the per-model judge flagged (e.g. STALE DATA, EXTRACTION FAILURE)."],
    ["model_consensus", "auto_accepted / disagreement / single_model_only / both_failed. Auto-accept requires model adequacy, validation, color agreement, and agreement on shared official/current values."],
    ["confidence_tier", "high = auto_accepted + both_stable. medium = auto_accepted + partially stable. low = everything else."],
    ["has_official_value", "TRUE if an authoritative last official value exists for this question."],
    ["has_current_value", "TRUE if a current-day estimate exists for this question."],
    ["value_has_changed", "TRUE if either model's last official value or current estimate changed since the prior run. Blank on the first run."],
    ["review_verdict", "correct = both models right. close = roughly right. partially right = one model right or right direction/wrong magnitude. wrong = clearly off. confidently wrong = model was certain and wrong. unknown = reviewed but can't assess (future question, insufficient data)."],
    ["review_last_official_value", "Human-verified last official value, from the cited source or a manual lookup."],
    ["review_current_value", "Human-verified current value as of the run date."],
    ["reviewed_question_resolution_status", "Human-verified resolution status: open / resolved / failed_to_resolve / projected."],
    ["reviewed_question_resolution_value", "Human-verified value at the question's resolution date. Use this for black/resolved rows instead of review_last_official_value."],
    ["review_source", "Source URL or citation for the verified values above."],
    ["reviewed", "Tick when done with your review. All rows sync regardless — reviewed rows carry human values, unreviewed rows carry model projections."],
    [""],
    ["All columns"],
    ["question_name", "Short name identifying the question."],
    ["status", "resolved / due_unresolved / forecast / resolved_early."],
    ["question_resolution_status", "open / resolved / failed_to_resolve / projected, derived from model outputs and row status."],
    ["question_resolution_value", "Model/system resolution value at the target date, when available."],
    ["question_resolution_source", "Source for question_resolution_value."],
    ["question_resolution_source_date", "Date the resolution source value represents."],
    ["needs_review", "TRUE if this row should be manually inspected before treating it as confirmed."],
    ["target_date", "The date this forecast row is predicting. One question may have multiple target dates."],
    ["dimension", "Sub-dimension (e.g. 'Overall', 'US', 'EU'). Most questions have only 'Overall'."],
    ["question_text", "Full question text as written in the LEAP panel."],
    ["resolution_criteria", "The exact condition that determines when and how this question resolves."],
    ["question_type", "quantile / probability / when."],
    ["unit", "Unit of the forecast value (e.g. '% GDP growth', 'USD billions')."],
    ["gpt_q50 / claude_q50", "Each model's forecast median. Blank on resolved (black) rows."],
    ["gpt_resolution_value / claude_resolution_value", "Each model's confirmed value at the target date, for resolved (black) rows only."],
    ["gpt_color / claude_color", "black / dark gray / light gray / white — resolved, hard bound, directional signal, or no narrowing information."],
    ["gpt_q0 / claude_q0", "Lowest feasible bound (0th percentile-style bound, not an ordinary credible interval endpoint)."],
    ["gpt_q5 / claude_q5", "5th percentile."],
    ["gpt_q25 / claude_q25", "25th percentile (lower IQR bound)."],
    ["gpt_q75 / claude_q75", "75th percentile (upper IQR bound)."],
    ["gpt_q95 / claude_q95", "95th percentile."],
    ["gpt_q100 / claude_q100", "Highest feasible bound (natural upper bound when one exists, otherwise a practical extreme-tail bound)."],
    ["gpt_judge_confidence / claude_judge_confidence", "Judge's confidence in its adequacy assessment."],
    ["gpt_browser_status / claude_browser_status", "not_proposed / not_useful / proposed_no_url / extract_failed / refinement_rejected / accepted. Browser runs only when the judge flags an issue."],
    ["gpt_browser_url / claude_browser_url", "URL proposed for browser extraction, when any."],
    ["gpt_browser_objective / claude_browser_objective", "Exact extraction instruction sent to the browser agent."],
    ["gpt_browser_error / claude_browser_error", "Browser extraction or refinement rejection reason, when any."],
    ["gpt_missing_data / claude_missing_data", "Data gaps the judge flagged."],
    ["gpt_judge_reason / claude_judge_reason", "Judge's explanation of any adequacy issues. Blank if adequate."],
    ["gpt_validation_issues / claude_validation_issues", "Deterministic-validator findings (monotonicity, missing rows, out-of-bounds values, etc.)."],
    ["gpt_latest_official_value / claude_latest_official_value", "Most recent published official figure the model found."],
    ["gpt_latest_official_date / claude_latest_official_date", "Date that official figure was published or measured."],
    ["gpt_latest_official_source / claude_latest_official_source", "URL or citation for the latest official value."],
    ["gpt_current_estimate / claude_current_estimate", "Model's estimate of the current value as of the run date."],
    ["gpt_current_estimate_confidence / claude_current_estimate_confidence", "Confidence in the current estimate (high / medium / low)."],
    ["gpt_rationale / claude_rationale", "Model's written explanation for its forecast."],
    ["gpt_sources / claude_sources", "Research URLs the model cited."],
    ["gpt_resolution_source / claude_resolution_source", "Source used for a resolved (black) row's resolution value."],
    ["gpt_resolution_source_date / claude_resolution_source_date", "Date of the resolution source."],
    ["model_consensus", "auto_accepted / disagreement / single_model_only / both_failed."],
    ["consensus_q50_delta_pct", "|gpt_q50 − claude_q50| / mean(gpt_q50, claude_q50). Informational for non-timing questions; official/current values are the primary match signal."],
    ["run_stability", "Aggregate cross-run stability: both_stable / one_stable / converging / volatile / new."],
    ["gpt_run_stability / claude_run_stability", "Per-model q50 stability over the last 10 production runs."],
    ["runs_seen", "Number of production runs that included this question (denominator for run_stability)."],
    ["confidence_tier", "high / medium / low. Derived from consensus status and run stability."],
    ["has_official_value", "TRUE if GPT or Claude found a published last official value for this question. FALSE for question types where no official figure exists."],
    ["has_current_value", "TRUE if GPT or Claude produced a current-day estimate for this question. FALSE where not applicable."],
    ["value_has_changed", "TRUE if either model's last official value or current estimate changed since the prior run. FALSE = unchanged. Blank = first run."],
    ["review_verdict", "correct = both models right. close = roughly right. partially right = one model right or right direction/wrong magnitude. wrong = clearly off. confidently wrong = model was certain and wrong. unknown = reviewed but can't assess (future question, insufficient data)."],
    ["review_last_official_value", "Human-verified last official value."],
    ["review_current_value", "Human-verified current value as of the run date."],
    ["reviewed_question_resolution_status", "Human-verified resolution status for the target row."],
    ["reviewed_question_resolution_value", "Human-verified resolution value for the target row."],
    ["review_source", "Source URL or citation for the verified values above."],
    ["review_color", "Color override. Same options as gpt_color/claude_color."],
    ["review_notes", "Free-text notes."],
    ["reviewed", "Tick when done with your review. All rows sync regardless — reviewed rows carry human values, unreviewed rows carry model projections."],
    ["review_row_id", "Stable Sheet row ID: surveillance_timestamp + group_question_id + target_date + dimension."],
    ["group_question_id", "BigQuery ID for the question group (dim_question_group)."],
    ["question_id", "BigQuery ID for the date/dimension question row (dim_question)."],
    ["surveillance_timestamp", "UTC timestamp of this pipeline run (YYYYMMDD_HHMMSS). Matches the tab name suffix; sync uses it to locate the run's JSON."],
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


def _question_resolution_fields(status: str, gpt_row: dict, claude_row: dict) -> dict:
    """Derive target-date resolution fields without overloading baseline values."""
    status_norm = str(status or "").strip().lower()
    for row in (gpt_row, claude_row):
        value = row.get("resolution_value")
        if not is_empty(value):
            return {
                "question_resolution_status": "resolved",
                "question_resolution_value": value,
                "question_resolution_source": row.get("resolution_source", ""),
                "question_resolution_source_date": row.get("resolution_source_date", ""),
            }
    if status_norm in ("resolved", "resolved_early"):
        return {
            "question_resolution_status": "resolved",
            "question_resolution_value": "",
            "question_resolution_source": "",
            "question_resolution_source_date": "",
        }
    if status_norm == "due_unresolved":
        return {
            "question_resolution_status": "failed_to_resolve",
            "question_resolution_value": "",
            "question_resolution_source": "",
            "question_resolution_source_date": "",
        }
    return {
        "question_resolution_status": "open",
        "question_resolution_value": "",
        "question_resolution_source": "",
        "question_resolution_source_date": "",
    }


def _needs_review(
    *,
    row_status: str,
    consensus_status: str,
    value_changed,
    gpt_view: dict,
    claude_view: dict,
    q_type: str = "",
    gpt_q50=None,
    claude_q50=None,
) -> str:
    """Flag rows where human review is likely valuable before DWH confirmation."""
    status_norm = str(row_status or "").strip().lower()
    if status_norm in ("resolved", "resolved_early", "due_unresolved"):
        return "TRUE"
    if value_changed is True:
        return "TRUE"
    if consensus_status and consensus_status != "auto_accepted":
        return "TRUE"
    # Quantile auto-accept intentionally skips q50 (differing forecasts are expected) - still flag wild divergence for review.
    if q_type == "quantile":
        a, b = to_float(gpt_q50), to_float(claude_q50)
        if a is not None and b is not None and not within_relative_tolerance(a, b, Q50_TOLERANCE):
            return "TRUE"
    for view in (gpt_view, claude_view):
        if view.get("browser_status") in ("extract_failed", "refinement_rejected"):
            return "TRUE"
        if view.get("missing_data") or view.get("validation_issues"):
            return "TRUE"
    return "FALSE"


_RUN_TAB_RE = re.compile(r"^run_\d{8}_\d{6}$")


def _latest_run_tab(sheet):
    """Return the latest run_YYYYMMDD_HHMMSS worksheet, or None.

    Strict format match: a manual tab like run_backup would otherwise sort after every
    timestamped tab ('b' > '2') and silently become "latest" for sync.
    """
    run_tabs = [ws for ws in sheet.worksheets() if _RUN_TAB_RE.match(ws.title)]
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


def _render_when_year(value: str) -> str:
    """Display the 'never' sentinel as a human-readable label."""
    if not value:
        return value
    try:
        if int(float(value)) == NEVER_YEAR:
            return "never"
    except (TypeError, ValueError):
        pass
    return value


def _extract_model_view(model_block: dict) -> dict:
    """Extract everything one model contributes to a Sheet row, ready for per-row indexing."""
    if not model_block:
        return {
            "quality_confidence": "", "missing_data": "", "judge_reason": "",
            "browser_status": "", "browser_url": "", "browser_objective": "", "browser_error": "",
            "rationale": "", "sources": "",
            "validation_issues": "",
            "groups": {}, "official_map": {}, "current_map": {}, "resolution_map": {},
            "absent": True,
        }
    response = model_block.get("response") or {}
    quality = model_block.get("quality") or {}
    validation = model_block.get("validation") or {}

    groups: dict[tuple, dict] = defaultdict(dict)
    for forecast in response.get("forecasts", []) or []:
        fdate = forecast.get("forecast_date", "")
        dim = forecast.get("dimension", "Overall")
        q = forecast.get("quantile")
        groups[(fdate, dim)][q] = forecast

    official_map, current_map, resolution_map = context_maps(response)

    return {
        "quality_confidence": safe_str(quality.get("confidence", "")),
        "missing_data": ", ".join(quality.get("missing_data", []) or [])[:SHEET_TEXT_LIMIT],
        "judge_reason": safe_str(quality.get("reason", ""))[:SHEET_TEXT_LIMIT],
        "browser_status": model_block.get("browser_status") or (
            "accepted" if model_block.get("browser_used") else "not_proposed"
        ),
        "browser_url": safe_str(model_block.get("browser_url", ""))[:SHEET_TEXT_LIMIT],
        "browser_objective": safe_str(model_block.get("browser_objective", ""))[:SHEET_TEXT_LIMIT],
        "browser_error": safe_str(model_block.get("browser_error", ""))[:SHEET_TEXT_LIMIT],
        "rationale": safe_str(response.get("rationale", ""))[:SHEET_TEXT_LIMIT],
        "sources": ", ".join(response.get("sources", []) or [])[:SHEET_TEXT_LIMIT],
        "validation_issues": ", ".join(validation.get("issues", []) or [])[:SHEET_TEXT_LIMIT],
        "groups": dict(groups),
        "official_map": official_map,
        "current_map": current_map,
        "resolution_map": resolution_map,
        "absent": False,
    }


def _row_fields_for_model(view: dict, fdate: str, dim: str, q_type: str) -> dict:
    """Compute the per-row fields for one model (q50, color, distribution, official values, etc)."""
    if view.get("absent"):
        return {k: "" for k in (
            "q50", "color",
            "q0", "q5", "q25", "q75", "q95", "q100",
            "latest_official_value", "latest_official_date", "latest_official_source",
            "current_estimate", "current_estimate_confidence",
            "resolution_value", "resolution_source", "resolution_source_date",
        )}

    quants = view["groups"].get((fdate, dim)) or {}
    display_forecast = quants.get(50) or (next(iter(quants.values())) if quants else {})
    color_code = enum_value(display_forecast.get("color_code", "")) if display_forecast else ""
    row_type = "resolved" if color_code == "black" else "forecast"

    res_val = view["resolution_map"].get((fdate, dim)) or view["resolution_map"].get((fdate, "Overall")) or {}

    # Read as-is: the pipeline already nulls quantiles for resolved non-timing rows at the source.
    q0_val, q5_val, q25_val, q75_val, q95_val, q100_val = (
        safe_str(quants.get(q, {}).get("forecast_value", "")) for q in (0, 5, 25, 75, 95, 100)
    )
    q50_val = safe_str(quants.get(50, {}).get("forecast_value", ""))
    if q_type == "when":
        q0_val = _render_when_year(q0_val)
        q5_val = _render_when_year(q5_val)
        q25_val = _render_when_year(q25_val)
        q50_val = _render_when_year(q50_val)
        q75_val = _render_when_year(q75_val)
        q95_val = _render_when_year(q95_val)
        q100_val = _render_when_year(q100_val)

    if row_type == "resolved":
        resolution_value = safe_str(res_val.get("value", ""))
        res_source_date = safe_str(res_val.get("source_date", ""))
        res_source = safe_str(res_val.get("source", ""))[:SHEET_TEXT_LIMIT]
    else:
        resolution_value = ""
        res_source_date = ""
        res_source = ""

    # Dimension labels are free text on older runs, so fall back to the sole dimension when it doesn't match.
    single_dim = len({d for _, d in view["groups"]}) <= 1

    cur_val_obj = pick_by_dimension(view["current_map"], dim, single_dim)
    cur_val = safe_str(cur_val_obj.get("value", ""))
    cur_conf = safe_str(cur_val_obj.get("confidence", ""))

    off_val_obj = pick_by_dimension(view["official_map"], dim, single_dim)
    off_val = safe_str(off_val_obj.get("value", ""))
    off_date = safe_str(off_val_obj.get("date", ""))
    off_source = safe_str(off_val_obj.get("source", ""))[:SHEET_TEXT_LIMIT]

    return {
        "q50": q50_val, "color": color_code,
        "q0": q0_val, "q5": q5_val, "q25": q25_val,
        "q75": q75_val, "q95": q95_val, "q100": q100_val,
        "latest_official_value": off_val, "latest_official_date": off_date,
        "latest_official_source": off_source,
        "current_estimate": cur_val, "current_estimate_confidence": cur_conf,
        "resolution_value": resolution_value,
        "resolution_source": res_source, "resolution_source_date": res_source_date,
    }


def build_review_rows(run_data: dict) -> tuple[list[list], list[str]]:
    """Build Sheet rows from run_data; returns (rows, headers) matched to the run's mode."""
    mode = run_data.get("mode", "both")
    headers = _review_headers(mode)
    run_id = run_data.get("run_id", "unknown")
    rows: list[list] = []

    for question in run_data.get("questions", []):
        q_name = question.get("name", "")
        q_type = question.get("question_type", "")
        q_unit = question.get("unit", "")
        q_text = safe_str(question.get("question_text", ""))[:SHEET_TEXT_LIMIT]
        q_criteria = safe_str(question.get("resolution_criteria", ""))[:SHEET_TEXT_LIMIT]
        per_model = question.get("per_model") or {}

        gpt_view = _extract_model_view(per_model.get("gpt") or {})
        claude_view = _extract_model_view(per_model.get("claude") or {})

        # Driving set of (date, dim) keys: union of both models' forecast groups.
        all_keys = set(gpt_view["groups"].keys()) | set(claude_view["groups"].keys())

        consensus_block = question.get("consensus") or {}
        consensus_status_val = consensus_block.get("status", "") if consensus_block else ""
        row_diff_by_key: dict[tuple, dict] = {
            (rd.get("forecast_date", ""), rd.get("dimension", "Overall")): rd
            for rd in consensus_block.get("row_diffs", []) or []
        }
        stab = question.get("run_stability") or {}
        run_stab = stab.get("run_stability", "")
        gpt_run_stab = stab.get("gpt_run_stability", "")
        claude_run_stab = stab.get("claude_run_stability", "")
        runs_seen = stab.get("runs_seen", "")

        for (fdate, dim) in sorted(all_keys):
            group_question_id = question.get("id", "")
            dim_question_id = (question.get("dim_question_map") or {}).get(f"{fdate}|{dim}", "")
            review_row_id = make_review_group_id(run_id, group_question_id, fdate, dim)

            gpt_row = _row_fields_for_model(gpt_view, fdate, dim, q_type)
            claude_row = _row_fields_for_model(claude_view, fdate, dim, q_type)

            # Driving color for the status column: prefer GPT, fall back to Claude.
            display_color = gpt_row.get("color") or claude_row.get("color") or ""
            row_status = row_resolution_status(fdate, display_color)
            resolution_fields = _question_resolution_fields(row_status, gpt_row, claude_row)

            row_diff = row_diff_by_key.get((fdate, dim)) or {}
            delta_pct = row_diff.get("delta_pct")
            delta_pct_str = "" if delta_pct is None else f"{delta_pct:.3f}"
            value_changed = question.get("value_changed")

            row = {
                "question_name": q_name,
                "status": row_status,
                "target_date": fdate,
                "dimension": dim,
                **resolution_fields,
                "needs_review": _needs_review(
                    row_status=row_status,
                    consensus_status=consensus_status_val,
                    value_changed=value_changed,
                    gpt_view=gpt_view,
                    claude_view=claude_view,
                    q_type=q_type,
                    gpt_q50=gpt_row.get("q50"),
                    claude_q50=claude_row.get("q50"),
                ),
                "question_text": q_text,
                "resolution_criteria": q_criteria,
                "question_type": q_type,
                "unit": q_unit,
                "gpt_judge_confidence": gpt_view["quality_confidence"],
                "gpt_browser_status": gpt_view["browser_status"],
                "gpt_browser_url": gpt_view["browser_url"],
                "gpt_browser_objective": gpt_view["browser_objective"],
                "gpt_browser_error": gpt_view["browser_error"],
                "gpt_missing_data": gpt_view["missing_data"],
                "gpt_judge_reason": gpt_view["judge_reason"],
                "gpt_validation_issues": gpt_view["validation_issues"],
                "gpt_rationale": gpt_view["rationale"],
                "gpt_sources": gpt_view["sources"],
                "claude_judge_confidence": claude_view["quality_confidence"],
                "claude_browser_status": claude_view["browser_status"],
                "claude_browser_url": claude_view["browser_url"],
                "claude_browser_objective": claude_view["browser_objective"],
                "claude_browser_error": claude_view["browser_error"],
                "claude_missing_data": claude_view["missing_data"],
                "claude_judge_reason": claude_view["judge_reason"],
                "claude_validation_issues": claude_view["validation_issues"],
                "claude_rationale": claude_view["rationale"],
                "claude_sources": claude_view["sources"],
                "gpt_q50": gpt_row["q50"], "claude_q50": claude_row["q50"],
                "gpt_color": gpt_row["color"], "claude_color": claude_row["color"],
                "gpt_q0": gpt_row["q0"], "claude_q0": claude_row["q0"],
                "gpt_q5": gpt_row["q5"], "claude_q5": claude_row["q5"],
                "gpt_q25": gpt_row["q25"], "claude_q25": claude_row["q25"],
                "gpt_q75": gpt_row["q75"], "claude_q75": claude_row["q75"],
                "gpt_q95": gpt_row["q95"], "claude_q95": claude_row["q95"],
                "gpt_q100": gpt_row["q100"], "claude_q100": claude_row["q100"],
                "gpt_latest_official_value": gpt_row["latest_official_value"],
                "claude_latest_official_value": claude_row["latest_official_value"],
                "gpt_latest_official_date": gpt_row["latest_official_date"],
                "claude_latest_official_date": claude_row["latest_official_date"],
                "gpt_latest_official_source": gpt_row["latest_official_source"],
                "claude_latest_official_source": claude_row["latest_official_source"],
                "gpt_current_estimate": gpt_row["current_estimate"],
                "claude_current_estimate": claude_row["current_estimate"],
                "gpt_current_estimate_confidence": gpt_row["current_estimate_confidence"],
                "claude_current_estimate_confidence": claude_row["current_estimate_confidence"],
                "gpt_resolution_value": gpt_row["resolution_value"],
                "claude_resolution_value": claude_row["resolution_value"],
                "gpt_resolution_source": gpt_row["resolution_source"],
                "claude_resolution_source": claude_row["resolution_source"],
                "gpt_resolution_source_date": gpt_row["resolution_source_date"],
                "claude_resolution_source_date": claude_row["resolution_source_date"],
                "model_consensus": consensus_status_val,
                "consensus_q50_delta_pct": delta_pct_str,
                "run_stability": run_stab,
                "gpt_run_stability": gpt_run_stab,
                "claude_run_stability": claude_run_stab,
                "runs_seen": runs_seen,
                "confidence_tier": _confidence_tier(consensus_status_val, run_stab),
                "has_official_value": _has_official_value(gpt_row, claude_row),
                "has_current_value": _has_current_value(gpt_row, claude_row),
                "value_has_changed": "" if question.get("value_changed") is None else ("TRUE" if question.get("value_changed") else "FALSE"),
                "review_verdict": "",
                "review_last_official_value": "",
                "review_current_value": "",
                "reviewed_question_resolution_status": "",
                "reviewed_question_resolution_value": "",
                "review_source": "",
                "review_color": "",
                "review_notes": "",
                "reviewed": "",
                "review_row_id": review_row_id,
                "group_question_id": group_question_id,
                "question_id": dim_question_id,
                "surveillance_timestamp": run_id,
            }
            rows.append([row.get(h, "") for h in headers])

    return rows, headers


def publish_to_sheet(run_data: dict, sheet_id: str = DEFAULT_SHEET_ID) -> int:
    """Publish a run to a formatted `run_<run_id>` tab."""
    client = get_sheets_client()
    sheet = client.open_by_key(sheet_id)
    run_id = run_data.get("run_id", "unknown")
    tab_name = run_tab_name(run_id)
    rows, headers = build_review_rows(run_data)
    ws = _create_run_tab(sheet, tab_name, headers)

    if rows:
        start_row = 2
        end_row = start_row + len(rows) - 1
        ws.update(f"A{start_row}", rows, value_input_option="USER_ENTERED")
        _apply_row_validation(sheet, ws.id, start_row, end_row, headers)
        _sort_review_rows(sheet, ws.id, headers)

    _reorder_tabs(sheet)
    return len(rows)


def _sort_review_rows(sheet, ws_id: int, headers: list[str]) -> None:
    """Sort review rows (excluding header) by question_name, target_date, dimension."""
    cols = [
        headers.index("question_name"),
        headers.index("target_date"),
        headers.index("dimension"),
    ]
    sheet.batch_update({"requests": [{
        "sortRange": {
            "range": {
                "sheetId": ws_id,
                "startRowIndex": 1,
                "startColumnIndex": 0,
                "endColumnIndex": len(headers),
            },
            "sortSpecs": [{"dimensionIndex": c, "sortOrder": "ASCENDING"} for c in cols],
        }
    }]})


def _apply_row_validation(sheet, ws_id: int, start_row: int, end_row: int, headers: list[str]) -> None:
    def _val(col_name: str, rule: dict) -> dict:
        idx = headers.index(col_name)
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
        _val("reviewed_question_resolution_status", _one_of(RESOLUTION_STATUS_OPTIONS)),
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


_N, _M = 80, 140

_FIELD_WIDTH = {
    "q50": _M, "color": _N,
    "q0": _N, "q5": _N, "q25": _N, "q75": _N, "q95": _N, "q100": _N,
    "judge_confidence": _N, "browser_status": _M,
    "missing_data": _M, "judge_reason": _M, "validation_issues": _M,
    "latest_official_value": _M, "latest_official_date": _M, "latest_official_source": _M,
    "current_estimate": _M, "current_estimate_confidence": _N,
    "rationale": _M, "sources": _M,
    "resolution_value": _M, "resolution_source": _M, "resolution_source_date": _M,
}


def _has_official_value(gpt_row: dict, claude_row: dict) -> bool:
    lov = gpt_row.get("latest_official_value", "") or claude_row.get("latest_official_value", "")
    return bool(lov and "not applicable" not in str(lov).lower())


def _has_current_value(gpt_row: dict, claude_row: dict) -> bool:
    ce = gpt_row.get("current_estimate", "") or claude_row.get("current_estimate", "")
    return bool(ce and "not applicable" not in str(ce).lower())


def _confidence_tier(model_consensus: str, run_stability: str) -> str:
    if model_consensus == "auto_accepted" and run_stability == "both_stable":
        return "high"
    if model_consensus == "auto_accepted" and run_stability in ("converging", "one_stable"):
        return "medium"
    return "low"


def _header_width(col: str) -> int:
    """Return pixel width for any review column."""
    for prefix in ("gpt_", "claude_"):
        if col.startswith(prefix):
            return _FIELD_WIDTH.get(col[len(prefix):], _M)
    if col in ("consensus_q50_delta_pct", "reviewed", "runs_seen", "needs_review",
               "gpt_run_stability", "claude_run_stability", "has_official_value", "has_current_value", "value_has_changed"):
        return _N
    return _M


def _create_run_tab(sheet, tab_name: str, headers: list[str]):
    """Create a per-run review tab sized and colored for the given header list."""
    import gspread

    try:
        existing = sheet.worksheet(tab_name)
        sheet.del_worksheet(existing)
    except gspread.WorksheetNotFound:
        pass
    ws = sheet.add_worksheet(tab_name, rows=1000, cols=len(headers))
    ws.update("A1", [headers])

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

    meta_end = len(_METADATA_COLS)
    consensus_start = headers.index("model_consensus") if "model_consensus" in headers else None
    reviewer_start = headers.index("review_verdict")
    sync_start = headers.index("review_row_id")
    n = len(headers)

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
        _header_band(0, meta_end, 0.93, 0.93, 0.93),                                   # gray   — metadata
        _header_band(meta_end, consensus_start or reviewer_start, 0.90, 0.92, 0.96),    # blue   — model cols
    ]
    if consensus_start is not None:
        format_requests.append(_header_band(consensus_start, reviewer_start, 0.99, 0.95, 0.80))  # yellow — consensus
    format_requests += [
        _header_band(reviewer_start, sync_start, 0.86, 0.94, 0.86),   # green    — reviewer
        _header_band(sync_start, n, 0.96, 0.96, 0.96),                 # gray-dim — sync IDs
        {
            "setBasicFilter": {
                "filter": {
                    "range": {
                        "sheetId": ws.id,
                        "startRowIndex": 0,
                        "startColumnIndex": 0,
                        "endColumnIndex": n,
                    }
                }
            }
        },
    ]
    for i, col in enumerate(headers):
        format_requests.append({
            "updateDimensionProperties": {
                "range": {
                    "sheetId": ws.id, "dimension": "COLUMNS",
                    "startIndex": i, "endIndex": i + 1,
                },
                "properties": {"pixelSize": _header_width(col)},
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
    instructions_ws = sheet.add_worksheet("Instructions", rows=max(100, len(INSTRUCTIONS_CONTENT) + 5), cols=4)

    rows2 = [list(r) + [""] * (2 - len(r)) for r in INSTRUCTIONS_CONTENT]
    instructions_ws.update("A1", rows2)
    _format_instructions(sheet, instructions_ws)
    _reorder_tabs(sheet)
    print(f"Instructions tab refreshed: {sheet_id}")


def get_reviewed_items(
    sheet_id: str = DEFAULT_SHEET_ID,
    tab_name: str | None = None,
    reviewed_only: bool = True,
) -> tuple[list[dict], list[int]]:
    """Read rows from a per-run tab. Defaults to the most-recent `run_*` tab."""
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
            print("  warning: no run_* tabs found in sheet")
            return [], []
        print(f"  reading rows from '{ws.title}'")

    items: list[dict] = []
    row_numbers: list[int] = []

    for i, row in enumerate(ws.get_all_records(), start=2):
        reviewed_raw = row.get("reviewed", "")
        if isinstance(reviewed_raw, bool):
            reviewed = reviewed_raw
        else:
            reviewed = str(reviewed_raw).strip().lower() in ("true", "1", "yes")

        run_id = safe_str(row.get("surveillance_timestamp", "")).strip()
        group_question_id = safe_str(row.get("group_question_id", "")).strip()
        target_date = row.get("target_date", "")
        dimension = row.get("dimension", "Overall") or "Overall"
        fallback_row_id = make_review_group_id(run_id, group_question_id, target_date, dimension)
        review_row_id = safe_str(row.get("review_row_id") or fallback_row_id).strip()
        if not review_row_id:
            continue

        if reviewed_only and not reviewed:
            continue

        status = row.get("status", "")
        items.append({
            "review_row_id": review_row_id,
            "group_question_id": group_question_id,
            "run_id": run_id,  # internal key stays run_id so sync.py grouping/file lookup is untouched
            "question_id": row.get("question_id"),
            "question_name": row.get("question_name"),
            "target_date": row.get("target_date"),
            "dimension": row.get("dimension"),
            "question_resolution_status": row.get("question_resolution_status"),
            "question_resolution_value": row.get("question_resolution_value"),
            "question_resolution_source": row.get("question_resolution_source"),
            "question_resolution_source_date": row.get("question_resolution_source_date"),
            "needs_review": row.get("needs_review"),
            "question_type": row.get("question_type"),
            "unit": row.get("unit"),
            "type": "resolved" if str(status).strip().lower() in ("resolved", "resolved_early", "due_unresolved") else "forecast",
            "status": status,
            "reviewed": reviewed,
            "review_last_official_value": row.get("review_last_official_value"),
            "review_current_value": row.get("review_current_value"),
            "reviewed_question_resolution_status": row.get("reviewed_question_resolution_status"),
            "reviewed_question_resolution_value": row.get("reviewed_question_resolution_value"),
            "review_source": row.get("review_source"),
            "review_verdict": row.get("review_verdict"),
            "review_color": row.get("review_color"),
            "review_notes": row.get("review_notes"),
            "gpt_latest_official_value": row.get("gpt_latest_official_value"),
            "claude_latest_official_value": row.get("claude_latest_official_value"),
            "gpt_latest_official_date": row.get("gpt_latest_official_date"),
            "claude_latest_official_date": row.get("claude_latest_official_date"),
            "gpt_current_estimate": row.get("gpt_current_estimate"),
            "claude_current_estimate": row.get("claude_current_estimate"),
        })
        row_numbers.append(i)

    return items, row_numbers
