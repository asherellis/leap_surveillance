"""Shared constants and small helpers."""

from datetime import date, datetime, timezone
import math
import os

import pandas as pd

DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_BQ_PROJECT = os.environ.get("LEAP_BQ_PROJECT") or "data-warehouse-dev-492608"
DEFAULT_SURVEILLANCE_DATASET = os.environ.get("LEAP_SURVEILLANCE_DATASET") or "surveillance"
DEFAULT_SHEET_ID = (
    os.environ.get("LEAP_SHEET_ID")
    or "1lT7zVfKAsVZU7bKaEALq1AWApfFmWMisprTK42l7RDo"
)
DEFAULT_MODEL = os.environ.get("LEAP_MODEL") or "gpt-5.5"
DEFAULT_EVALUATOR_MODEL = os.environ.get("LEAP_EVALUATOR_MODEL") or "gpt-4.1-mini"
DEFAULT_BROWSER_MODEL = os.environ.get("LEAP_BROWSER_MODEL") or "gpt-4o"
CLAUDE_RESEARCH_MODEL = os.environ.get("LEAP_CLAUDE_MODEL") or "claude-opus-4-8"
CLAUDE_EVALUATOR_MODEL = os.environ.get("LEAP_CLAUDE_EVALUATOR_MODEL") or "claude-haiku-4-5-20251001"
TEST_CLAUDE_MODEL = os.environ.get("LEAP_TEST_CLAUDE_MODEL") or "claude-haiku-4-5-20251001"
TEST_CLAUDE_EVALUATOR_MODEL = os.environ.get("LEAP_TEST_CLAUDE_EVALUATOR_MODEL") or "claude-haiku-4-5-20251001"
TEST_MODEL = os.environ.get("LEAP_TEST_MODEL") or "gpt-4o-mini"
TEST_EVALUATOR_MODEL = os.environ.get("LEAP_TEST_EVALUATOR_MODEL") or "gpt-4o-mini"

# Approximate pricing ($/1M tokens) for cost estimates — informational only.
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.5": (5.00, 30.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def provider_for_model(model: str) -> str:
    """Return 'anthropic' for Claude models, 'openai' otherwise."""
    return "anthropic" if strip_provider_prefix(model).startswith("claude") else "openai"


def strip_provider_prefix(model: str) -> str:
    """Strip 'openai/' or 'anthropic/' prefix if present (backwards-compat with old .env files)."""
    for prefix in ("openai/", "anthropic/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def cost_for_tokens(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost from token counts using the local price table."""
    bare = strip_provider_prefix(model)
    for key, (inp_price, out_price) in _MODEL_PRICES.items():
        if key in bare:
            return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000
    return 0.0


SHEET_TEXT_LIMIT = 10000
FULL_QUANTILES = [0, 5, 25, 50, 75, 95, 100]
TIMING_FORECAST_DATE = "event_occurrence"
Q50_TOLERANCE = 0.10   # 10% relative-to-mean tolerance for non-black q50 agreement
NEAR_ZERO_SUM = 1e-9   # treat |a| + |b| below this as "both effectively zero"


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


def date_value_type(forecast_date: str, today: date | None = None) -> str:
    today = today or datetime.now(timezone.utc).date()
    try:
        return "resolution" if date.fromisoformat(forecast_date) < today else "forecast"
    except (TypeError, ValueError):
        return "forecast"


def resolution_status(forecast_date: str, color_code, today: date | None = None) -> str:
    is_black = getattr(color_code, "value", color_code) == "black"
    today = today or datetime.now(timezone.utc).date()
    try:
        is_past = date.fromisoformat(forecast_date) < today
    except (ValueError, TypeError):
        is_past = False
    if is_black:
        return "resolved" if is_past else "resolved_early"
    return "due_unresolved" if is_past else "forecast"


def is_empty(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    s = safe_str(val).strip()
    return s == "" or s.lower() in ("nat", "nan", "none")


def make_review_group_id(run_id: str, question_id: str, forecast_date: str, dimension: str) -> str:
    return f"{run_id}_{question_id}_{forecast_date}_{dimension}"


def make_result_id(
    run_id: str,
    question_id: str,
    forecast_date: str,
    dimension: str,
    quantile: int,
    model_id: str,
) -> str:
    return f"{make_review_group_id(run_id, question_id, forecast_date, dimension)}_{quantile}_{model_id}"
