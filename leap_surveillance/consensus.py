"""GPT vs Claude current/official value + color agreement, with 5% numeric tolerance."""

from collections import defaultdict
from typing import Any

from .common import NEAR_ZERO_SUM, NEVER_YEAR, Q50_TOLERANCE, WHEN_TOLERANCE_YR, enum_value, within_relative_tolerance

OFFICIAL_TOLERANCE = 0.05


def _field(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _extract_values_by_dim(response, field_name: str) -> dict[str, Any]:
    """Index response value lists by dimension -> value."""
    values = _field(response, field_name) or []
    result = {}
    for item in values:
        dim = _field(item, "dimension") or "Overall"
        val = _field(item, "value")
        skip_markers = ("not applicable", "-999", "n/a")
        if val is not None and not any(m in str(val).lower() for m in skip_markers):
            result[dim] = val
    return result


def _values_match(a, b) -> bool:
    """Numeric ±5% tolerance; fall back to normalized string match."""
    try:
        fa, fb = float(str(a).replace(",", "")), float(str(b).replace(",", ""))
        denom = (abs(fa) + abs(fb)) / 2
        return abs(fa - fb) / denom <= OFFICIAL_TOLERANCE if denom > 0 else True
    except (ValueError, TypeError):
        return str(a).strip().lower() == str(b).strip().lower()


def _all_shared_values_match(a_by_dim: dict[str, Any], b_by_dim: dict[str, Any]) -> bool:
    shared_dims = set(a_by_dim) & set(b_by_dim)
    return all(_values_match(a_by_dim[d], b_by_dim[d]) for d in shared_dims) if shared_dims else True


def _extract_q50_by_row(forecasts) -> dict[tuple[str, str], dict[str, Any]]:
    """Index forecasts by (forecast_date, dimension) → {"q50", "color"} dict."""
    by_row: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"q50": None, "color": None})
    for f in forecasts or []:
        date = _field(f, "forecast_date")
        dim = _field(f, "dimension") or "Overall"
        quantile = _field(f, "quantile")
        value = _field(f, "forecast_value")
        color = enum_value(_field(f, "color_code"))
        key = (date, dim)
        if by_row[key]["color"] is None:
            by_row[key]["color"] = color
        if quantile == 50:
            by_row[key]["q50"] = value
    return dict(by_row)


def _extract_resolution_by_row(response) -> dict[tuple[str, str], dict]:
    """Index a model's resolution_values by (forecast_date, dimension) → {value, status}."""
    result: dict[tuple[str, str], dict] = {}
    for rv in (_field(response, "resolution_values") or []):
        date = _field(rv, "forecast_date")
        dim = _field(rv, "dimension") or "Overall"
        result[(date, dim)] = {
            "value": _field(rv, "value"),
            "status": _field(rv, "resolution_status") or "unresolved",
        }
    return result


def _resolution_value_match(gpt_res: dict | None, claude_res: dict | None) -> tuple[bool, float | None]:
    """Compare two models' resolved values for a row. Two rows with no authoritative value agree."""
    gv = (gpt_res or {}).get("value")
    cv = (claude_res or {}).get("value")
    if gv is None and cv is None:
        return True, None          # both failed/unresolved → agreement
    if gv is None or cv is None:
        return False, None         # one resolved, the other not → disagreement
    match = _values_match(gv, cv)  # observed values compared with ±5% tolerance
    denom = (abs(gv) + abs(cv)) / 2
    delta_pct = (abs(gv - cv) / denom) if denom > 0 else 0.0
    return match, delta_pct


def _q50_within_tolerance(a: float | None, b: float | None) -> bool:
    """For non-black rows: q50 must agree within 10% relative-to-mean."""
    return within_relative_tolerance(a, b, Q50_TOLERANCE)


def _row_match(gpt_row: dict, claude_row: dict, question_type: str) -> tuple[bool, bool, bool, float | None]:
    """Return (color_match, q50_match, both_match, delta_pct)."""
    color_match = gpt_row["color"] == claude_row["color"] and gpt_row["color"] is not None
    gpt_q50 = gpt_row["q50"]
    claude_q50 = claude_row["q50"]
    if gpt_q50 is None or claude_q50 is None:
        q50_match = False
        delta_pct = None
    elif gpt_row["color"] == "black":
        q50_match = abs(gpt_q50 - claude_q50) < 1e-9
        delta_pct = 0.0 if q50_match else None
    elif question_type == "when":
        # q50s are years; relative tolerance is meaningless. Both "never" agree; otherwise ±2 years.
        both_never = gpt_q50 >= NEVER_YEAR and claude_q50 >= NEVER_YEAR
        q50_match = both_never or abs(gpt_q50 - claude_q50) <= WHEN_TOLERANCE_YR
        delta_pct = None
    else:
        q50_match = _q50_within_tolerance(gpt_q50, claude_q50)
        abs_sum = abs(gpt_q50) + abs(claude_q50)
        delta_pct = (abs(gpt_q50 - claude_q50) / (abs_sum / 2)) if abs_sum >= NEAR_ZERO_SUM else 0.0
    return color_match, q50_match, color_match and q50_match, delta_pct


def compute_consensus(per_model: dict, question_type: str = "quantile") -> dict:
    """Compare gpt vs claude outputs; return status and value/color agreement diagnostics."""
    gpt = per_model.get("gpt")
    claude = per_model.get("claude")

    gpt_errored = gpt is None or gpt.error is not None or gpt.response is None
    claude_errored = claude is None or claude.error is not None or claude.response is None

    if gpt_errored and claude_errored:
        return {"status": "both_failed", "color_agreement": False, "q50_agreement": False,
                "row_diffs": [], "reason": "Both models errored or produced no response."}
    if gpt_errored or claude_errored:
        survivor = "claude" if gpt_errored else "gpt"
        return {"status": "single_model_only", "color_agreement": False, "q50_agreement": False,
                "row_diffs": [], "reason": f"Only {survivor} produced a response."}

    gpt_adequate = bool(getattr(gpt.quality, "adequate", False))
    claude_adequate = bool(getattr(claude.quality, "adequate", False))
    both_valid = bool((gpt.validation or {}).get("ok")) and bool((claude.validation or {}).get("ok"))

    gpt_rows = _extract_q50_by_row(gpt.response.forecasts)
    claude_rows = _extract_q50_by_row(claude.response.forecasts)
    gpt_res = _extract_resolution_by_row(gpt.response)
    claude_res = _extract_resolution_by_row(claude.response)
    all_keys = sorted(set(gpt_rows.keys()) | set(claude_rows.keys()))

    # Missing LOV/current values don't block agreement — probability/when/panel questions may not have them.
    gpt_official = _extract_values_by_dim(gpt.response, "last_official_values")
    claude_official = _extract_values_by_dim(claude.response, "last_official_values")
    official_all_match = _all_shared_values_match(gpt_official, claude_official)
    gpt_current = _extract_values_by_dim(gpt.response, "current_estimates")
    claude_current = _extract_values_by_dim(claude.response, "current_estimates")
    current_all_match = _all_shared_values_match(gpt_current, claude_current)

    row_diffs = []
    color_all_match = True
    q50_all_match = True
    resolution_all_match = True
    for key in all_keys:
        date, dim = key
        gpt_row = gpt_rows.get(key, {"q50": None, "color": None})
        claude_row = claude_rows.get(key, {"q50": None, "color": None})

        # A row whose q50 is nulled in both models and has a resolution entry is a resolution
        # row: compare the resolved value, not q50 (which was intentionally nulled).
        is_resolution_row = (
            gpt_row["q50"] is None and claude_row["q50"] is None
            and (key in gpt_res or key in claude_res)
        )
        if is_resolution_row:
            q50_match, delta_pct = _resolution_value_match(gpt_res.get(key), claude_res.get(key))
            both_failed = (gpt_res.get(key) or {}).get("value") is None and (claude_res.get(key) or {}).get("value") is None
            # Color isn't meaningful when neither model resolved a value.
            color_match = True if both_failed else (gpt_row["color"] == claude_row["color"] and gpt_row["color"] is not None)
            row_match_ok = color_match and official_all_match and current_all_match and q50_match
            if not q50_match:
                resolution_all_match = False
        else:
            color_match, q50_match, _, delta_pct = _row_match(gpt_row, claude_row, question_type)
            q50_required = question_type in ("when", "probability")
            # when/probability rows have no objective official/current value, so q50 is the substantive match signal.
            row_match_ok = color_match and official_all_match and current_all_match and (q50_match if q50_required else True)
            if not q50_match:
                q50_all_match = False

        row_diffs.append({
            "forecast_date": date,
            "dimension": dim,
            "gpt_q50": gpt_row["q50"],
            "claude_q50": claude_row["q50"],
            "gpt_color": gpt_row["color"],
            "claude_color": claude_row["color"],
            "color_match": color_match,
            "q50_match": q50_match,
            "resolution_row": is_resolution_row,
            "delta_pct": delta_pct,
            "match": row_match_ok,
        })
        if not color_match:
            color_all_match = False

    both_adequate = gpt_adequate and claude_adequate
    all_rows_agree = color_all_match and official_all_match and current_all_match and resolution_all_match
    if question_type in ("when", "probability"):
        all_rows_agree = all_rows_agree and q50_all_match  # q50 is the substantive signal here

    if both_adequate and both_valid and all_rows_agree:
        status = "auto_accepted"
        reason = "Both models adequate, valid, and agree on color, official values, and current values."
    else:
        status = "disagreement"
        reasons = []
        if not gpt_adequate:
            reasons.append("gpt inadequate")
        if not claude_adequate:
            reasons.append("claude inadequate")
        if not both_valid:
            reasons.append("validation failed on a model")
        if not color_all_match:
            reasons.append("color mismatch on some rows")
        if not official_all_match:
            reasons.append("official value mismatch across dimensions")
        if not current_all_match:
            reasons.append("current value mismatch across dimensions")
        if question_type in ("when", "probability") and not q50_all_match:
            reasons.append("q50 mismatch")
        if not resolution_all_match:
            reasons.append("resolution value mismatch on some rows")
        reason = "; ".join(reasons) if reasons else "models disagree"

    return {
        "status": status,
        "color_agreement": color_all_match,
        "q50_agreement": q50_all_match,
        "resolution_agreement": resolution_all_match,
        "official_agreement": official_all_match,
        "current_agreement": current_all_match,
        "row_diffs": row_diffs,
        "reason": reason,
    }
