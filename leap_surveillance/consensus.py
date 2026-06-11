"""Compute agreement between GPT and Claude model outputs.

The auto-accept rule (when both models pass adequacy):
- Same color_code on every (forecast_date, dimension) row, AND
- For black rows: q50 must match exactly
- For non-black rows: q50 must be within 10% relative-to-mean
  (`abs(a-b) / ((|a|+|b|)/2) <= 0.10`)

q50 comparison only — not all 7 quantiles.
"""

from collections import defaultdict
from typing import Any


# Two q50 values are "near zero" and treated as agreed if their absolute-value sum
# falls below this threshold (avoids divide-by-zero noise on degenerate metrics).
_NEAR_ZERO_SUM = 1e-9
_Q50_TOLERANCE = 0.10  # 10% relative-to-mean


def _extract_q50_by_row(forecasts) -> dict[tuple[str, str], dict[str, Any]]:
    """Index forecasts by (forecast_date, dimension), keeping the q50 row.

    Returns: {(date, dim): {"q50": float | None, "color": str | None}}.
    """
    by_row: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"q50": None, "color": None})
    for f in forecasts or []:
        # `f` may be a Pydantic Forecast object or a plain dict (when reading from JSON).
        date = getattr(f, "forecast_date", None) or (f.get("forecast_date") if isinstance(f, dict) else None)
        dim = getattr(f, "dimension", None) or (f.get("dimension") if isinstance(f, dict) else None) or "Overall"
        quantile = getattr(f, "quantile", None)
        if quantile is None and isinstance(f, dict):
            quantile = f.get("quantile")
        value = getattr(f, "forecast_value", None)
        if value is None and isinstance(f, dict):
            value = f.get("forecast_value")
        color = getattr(f, "color_code", None)
        if color is None and isinstance(f, dict):
            color = f.get("color_code")
        # color_code may be an enum on the Pydantic side.
        color = getattr(color, "value", color)
        key = (date, dim)
        if by_row[key]["color"] is None:
            by_row[key]["color"] = color
        if quantile == 50:
            by_row[key]["q50"] = value
    return dict(by_row)


def _q50_within_tolerance(a: float | None, b: float | None) -> bool:
    """For non-black rows: q50 must agree within 10% relative-to-mean."""
    if a is None or b is None:
        return False
    abs_sum = abs(a) + abs(b)
    if abs_sum < _NEAR_ZERO_SUM:
        return True  # both effectively zero
    return abs(a - b) / (abs_sum / 2) <= _Q50_TOLERANCE


def _row_match(gpt_row: dict, claude_row: dict) -> tuple[bool, bool, bool, float | None]:
    """Return (color_match, q50_match, both_match, delta_pct)."""
    color_match = gpt_row["color"] == claude_row["color"] and gpt_row["color"] is not None
    gpt_q50 = gpt_row["q50"]
    claude_q50 = claude_row["q50"]
    if gpt_q50 is None or claude_q50 is None:
        q50_match = False
        delta_pct = None
    elif gpt_row["color"] == "black":
        q50_match = gpt_q50 == claude_q50
        delta_pct = 0.0 if q50_match else None
    else:
        q50_match = _q50_within_tolerance(gpt_q50, claude_q50)
        abs_sum = abs(gpt_q50) + abs(claude_q50)
        delta_pct = (abs(gpt_q50 - claude_q50) / (abs_sum / 2)) if abs_sum >= _NEAR_ZERO_SUM else 0.0
    return color_match, q50_match, color_match and q50_match, delta_pct


def compute_consensus(per_model: dict) -> dict:
    """Compare gpt vs claude ModelRunResult outputs and return the agreement block.

    `per_model` is a dict with keys like {"gpt": ModelRunResult, "claude": ModelRunResult}.
    A model that errored has `.error` set and may have `response=None`.
    """
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

    gpt_rows = _extract_q50_by_row(gpt.response.forecasts)
    claude_rows = _extract_q50_by_row(claude.response.forecasts)
    all_keys = sorted(set(gpt_rows.keys()) | set(claude_rows.keys()))

    row_diffs = []
    color_all_match = True
    q50_all_match = True
    for key in all_keys:
        date, dim = key
        gpt_row = gpt_rows.get(key, {"q50": None, "color": None})
        claude_row = claude_rows.get(key, {"q50": None, "color": None})
        color_match, q50_match, both_match, delta_pct = _row_match(gpt_row, claude_row)
        row_diffs.append({
            "forecast_date": date,
            "dimension": dim,
            "gpt_q50": gpt_row["q50"],
            "claude_q50": claude_row["q50"],
            "gpt_color": gpt_row["color"],
            "claude_color": claude_row["color"],
            "color_match": color_match,
            "q50_match": q50_match,
            "delta_pct": delta_pct,
            "match": both_match,
        })
        if not color_match:
            color_all_match = False
        if not q50_match:
            q50_all_match = False

    all_rows_agree = color_all_match and q50_all_match
    both_adequate = gpt_adequate and claude_adequate

    if both_adequate and all_rows_agree:
        status = "auto_accepted"
        reason = "Both models adequate and all rows agree on color and q50."
    else:
        status = "disagreement"
        reasons = []
        if not gpt_adequate:
            reasons.append("gpt inadequate")
        if not claude_adequate:
            reasons.append("claude inadequate")
        if not color_all_match:
            reasons.append("color mismatch on some rows")
        if not q50_all_match:
            reasons.append("q50 mismatch on some rows")
        reason = "; ".join(reasons) if reasons else "models disagree"

    return {
        "status": status,
        "color_agreement": color_all_match,
        "q50_agreement": q50_all_match,
        "row_diffs": row_diffs,
        "reason": reason,
    }
