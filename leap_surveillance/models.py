"""Schemas and deterministic validation for LEAP surveillance."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

STRICT_CONFIG = ConfigDict(extra="forbid")


@dataclass
class ExpectedForecast:
    forecast_date: str
    dimension: str
    quantile: Optional[int]
    value_type: str = "forecast"


@dataclass
class QuestionSpec:
    id: str
    name: str
    prompt: str
    expected_forecasts: list[ExpectedForecast] = field(default_factory=list)
    question_type: str = "quantile"
    unit: str = ""
    unit_min: Optional[float] = None
    unit_max: Optional[float] = None
    question_text: str = ""
    resolution_criteria: str = ""
    background_info: str = ""
    dim_question_map: dict = field(default_factory=dict)  # {"fdate|dim": dim_question.question_id}
    rc_source: dict | None = None  # filled by extract_rc_source() before research; None = not yet extracted
    evidence_plan: dict | None = None  # serialized EvidencePlan shared across research/judge/browser stages
    is_conditional: bool = False  # scenario_id IS NOT NULL: elicit forecasts only, no LOV/current/resolution
    scenario_name: str = ""
    scenario_description: str = ""


@dataclass
class EvidenceItem:
    source_type: str
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    full_text: Optional[str] = None
    source_role: str = ""
    retrieval_status: str = ""


@dataclass
class RunCost:
    research: float = 0.0
    judge_stage1: float = 0.0
    judge_stage2: float = 0.0
    refinement: float = 0.0
    claude_research: float = 0.0
    claude_judge_stage1: float = 0.0
    claude_judge_stage2: float = 0.0
    claude_refinement: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.research + self.judge_stage1 + self.judge_stage2 + self.refinement
            + self.claude_research + self.claude_judge_stage1 + self.claude_judge_stage2 + self.claude_refinement
        )

    def as_dict(self) -> dict:
        return {
            "research": self.research,
            "judge_stage1": self.judge_stage1,
            "judge_stage2": self.judge_stage2,
            "refinement": self.refinement,
            "claude_research": self.claude_research,
            "claude_judge_stage1": self.claude_judge_stage1,
            "claude_judge_stage2": self.claude_judge_stage2,
            "claude_refinement": self.claude_refinement,
            "total": self.total,
        }


@dataclass
class BrowserEvidence:
    url: str
    objective: str
    extracted_text: str
    success: bool
    error: Optional[str] = None


class ResearchQualityReport(BaseModel):
    model_config = STRICT_CONFIG
    adequate: bool = True
    confidence: int = Field(default=50, ge=0, le=100)
    missing_data: list[str] = Field(default_factory=list)
    browser_would_help: bool = False
    browser_url: str = ""
    browser_objective: str = ""
    reason: str = ""


class AdequacyAssessment(BaseModel):
    model_config = STRICT_CONFIG
    adequate: bool
    confidence: int = Field(ge=0, le=100)
    issues: list[str] = Field(default_factory=list)
    reason: str = ""


class BrowserDecision(BaseModel):
    model_config = STRICT_CONFIG
    browser_would_help: bool
    browser_url: str = ""
    browser_objective: str = ""
    reason: str = ""


class ColorCode(str, Enum):
    black = "black"
    dark_gray = "dark gray"
    light_gray = "light gray"
    white = "white"


class ValueType(str, Enum):
    forecast = "forecast"
    resolution = "resolution"


class ResolutionStatus(str, Enum):
    resolved = "resolved"      # authoritative value found
    failed = "failed"          # resolution date passed, no value findable (date-based only)
    unresolved = "unresolved"  # not resolved and not failed (e.g. timing event not yet occurred)


class OfficialValue(BaseModel):
    dimension: str
    value: Optional[float]
    date: str
    source: str


class CurrentValue(BaseModel):
    dimension: str
    value: Optional[float]
    confidence: int = Field(ge=0, le=100)


class ResolutionValue(BaseModel):
    forecast_date: str
    dimension: str
    value: Optional[float]
    source_date: str
    source: str
    confidence: int = Field(ge=0, le=100)
    resolution_status: str = "unresolved"          # resolved / failed / unresolved
    best_guess_resolution: Optional[float] = None   # model estimate when value is unavailable


class ForecastValue(BaseModel):
    value_type: ValueType
    forecast_date: str
    dimension: str
    quantile: Optional[int]
    forecast_value: Optional[float]
    color_code: ColorCode


class SurveillanceResponse(BaseModel):
    last_official_values: list[OfficialValue]
    current_estimates: list[CurrentValue]
    resolution_values: list[ResolutionValue]
    forecasts: list[ForecastValue]
    rationale: str
    sources: list[str]


# Strict* models describe the LLM JSON output; runtime models above allow nullable/flattened fields the LLM doesn't set.
class StrictOfficialValue(BaseModel):
    model_config = STRICT_CONFIG
    dimension: str
    value: float
    date: str
    source: str


class StrictCurrentValue(BaseModel):
    model_config = STRICT_CONFIG
    dimension: str
    value: float
    confidence: int = Field(ge=0, le=100)


class StrictResolutionValue(BaseModel):
    model_config = STRICT_CONFIG
    forecast_date: str
    dimension: str
    resolution_status: ResolutionStatus
    value: float                    # authoritative value; use -999 when none was found
    best_guess_resolution: float    # best point estimate (equals value when resolved; your estimate when failed)
    source_date: str
    source: str
    confidence: int = Field(ge=0, le=100)


class StrictForecastValue(BaseModel):
    model_config = STRICT_CONFIG
    forecast_date: str
    dimension: str
    quantile: int
    forecast_value: float
    color_code: ColorCode


class StrictSource(BaseModel):
    model_config = STRICT_CONFIG
    url: str
    title: str
    snippet: str


class StrictSurveillanceResponse(BaseModel):
    model_config = STRICT_CONFIG
    last_official_values: list[StrictOfficialValue]
    current_estimates: list[StrictCurrentValue]
    resolution_values: list[StrictResolutionValue]
    forecasts: list[StrictForecastValue]
    rationale: str
    sources: list[StrictSource]


class StrictEventSurveillanceResponse(BaseModel):
    """LLM output for probability/timing questions, where LOV/current are not meaningful."""
    model_config = STRICT_CONFIG
    resolution_values: list[StrictResolutionValue]
    forecasts: list[StrictForecastValue]
    rationale: str
    sources: list[StrictSource]


class StrictConditionalResponse(BaseModel):
    """LLM output for conditional (scenario) questions: forecasts only.

    A hypothetical scenario never resolves, and LOV/current are not meaningful, so
    resolution_values/last_official_values/current_estimates are all omitted.
    """
    model_config = STRICT_CONFIG
    forecasts: list[StrictForecastValue]
    rationale: str
    sources: list[StrictSource]


def make_strict_schema(
    model: type[BaseModel],
    allowed_dimensions: Optional[list[str]] = None,
    allowed_quantiles: Optional[list[int]] = None,
    allowed_forecast_dates: Optional[list[str]] = None,
    unit_min: Optional[float] = None,
    unit_max: Optional[float] = None,
) -> dict:
    schema = model.model_json_schema()

    def fix_schema(obj: dict) -> dict:
        if not isinstance(obj, dict):
            return obj
        obj.pop("default", None)
        if "properties" in obj:
            obj["additionalProperties"] = False
            obj["required"] = list(obj["properties"].keys())
        for value in obj.values():
            if isinstance(value, dict):
                fix_schema(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        fix_schema(item)
        if "$defs" in obj:
            for def_schema in obj["$defs"].values():
                fix_schema(def_schema)
        return obj

    schema = fix_schema(schema)
    forecast_def = schema.get("$defs", {}).get("StrictForecastValue", {})
    properties = forecast_def.get("properties", {})
    if allowed_forecast_dates and "forecast_date" in properties:
        properties["forecast_date"]["enum"] = allowed_forecast_dates
    if allowed_dimensions and "dimension" in properties:
        properties["dimension"]["enum"] = allowed_dimensions
    if allowed_quantiles and "quantile" in properties:
        properties["quantile"]["enum"] = allowed_quantiles
    resolution_def = schema.get("$defs", {}).get("StrictResolutionValue", {})
    resolution_properties = resolution_def.get("properties", {})
    if allowed_forecast_dates and "forecast_date" in resolution_properties:
        resolution_properties["forecast_date"]["enum"] = allowed_forecast_dates
    if allowed_dimensions and "dimension" in resolution_properties:
        resolution_properties["dimension"]["enum"] = allowed_dimensions
    if allowed_dimensions:
        for def_name in ("StrictOfficialValue", "StrictCurrentValue"):
            props = schema.get("$defs", {}).get(def_name, {}).get("properties", {})
            if "dimension" in props:
                props["dimension"]["enum"] = allowed_dimensions
    if unit_min is not None or unit_max is not None:
        fv_prop = properties.get("forecast_value", {})
        if unit_min is not None:
            fv_prop["minimum"] = unit_min
        if unit_max is not None:
            fv_prop["maximum"] = unit_max
    return schema


def _convert_sentinel_value(v: float) -> Optional[float]:
    return None if v == -999 else v


def _resolution_status_str(rv) -> str:
    """Normalize a resolution entry's status (enum or str) to a plain string."""
    status = getattr(rv, "resolution_status", "unresolved")
    return getattr(status, "value", status) or "unresolved"


def strict_to_regular_response(
    strict: StrictSurveillanceResponse | StrictEventSurveillanceResponse | StrictConditionalResponse | str,
    expected_forecasts: Optional[list[ExpectedForecast]] = None,
    question_type: Optional[str] = None,
    is_conditional: bool = False,
) -> tuple[SurveillanceResponse, list[EvidenceItem]]:
    # Conditional (scenario) questions elicit forecasts only: no LOV/current, no resolution.
    no_official = is_conditional or question_type in ("probability", "when")
    no_resolution = is_conditional
    if isinstance(strict, str):
        if is_conditional:
            strict = StrictConditionalResponse.model_validate_json(strict)
        elif no_official:
            strict = StrictEventSurveillanceResponse.model_validate_json(strict)
        else:
            strict = StrictSurveillanceResponse.model_validate_json(strict)

    seen_urls: set[str] = set()
    evidence = []
    for s in strict.sources:
        if s.url not in seen_urls:
            seen_urls.add(s.url)
            evidence.append(EvidenceItem(source_type="web_search", url=s.url, title=s.title, snippet=s.snippet))
    expected_value_types = {
        (e.forecast_date, e.dimension, e.quantile): e.value_type
        for e in expected_forecasts or []
    }
    strict_resolutions = [] if no_resolution else getattr(strict, "resolution_values", [])
    resolution_by_key = {
        (rv.forecast_date, rv.dimension): rv for rv in strict_resolutions
    }
    expected_date_dim_keys = {
        (e.forecast_date, e.dimension)
        for e in expected_forecasts or []
    }

    forecasts = []
    for fv in strict.forecasts:
        key = (fv.forecast_date, fv.dimension, fv.quantile)
        value_type = ValueType(expected_value_types.get(key, "forecast"))
        if value_type == ValueType.resolution:
            # No forecast distribution on a past date — the value lives in resolution_values.
            forecast_value = None
            rv = resolution_by_key.get((fv.forecast_date, fv.dimension))
            resolved = rv is not None and _resolution_status_str(rv) == "resolved"
            color_code = ColorCode.black if resolved else fv.color_code
        else:
            forecast_value = _convert_sentinel_value(fv.forecast_value)
            color_code = fv.color_code
        forecasts.append(
            ForecastValue(
                value_type=value_type,
                forecast_date=fv.forecast_date,
                dimension=fv.dimension,
                quantile=fv.quantile,
                forecast_value=forecast_value,
                color_code=color_code,
            )
        )

    response = SurveillanceResponse(
        last_official_values=[
            OfficialValue(
                dimension=ov.dimension,
                value=_convert_sentinel_value(ov.value),
                date=ov.date,
                source=ov.source,
            )
            for ov in ([] if no_official else getattr(strict, "last_official_values", []))
        ],
        current_estimates=[
            CurrentValue(
                dimension=cv.dimension,
                value=_convert_sentinel_value(cv.value),
                confidence=cv.confidence,
            )
            for cv in ([] if no_official else getattr(strict, "current_estimates", []))
        ],
        resolution_values=[
            ResolutionValue(
                forecast_date=rv.forecast_date,
                dimension=rv.dimension,
                value=_convert_sentinel_value(rv.value),
                source_date=rv.source_date,
                source=rv.source,
                confidence=rv.confidence,
                resolution_status=_resolution_status_str(rv),
                best_guess_resolution=_convert_sentinel_value(getattr(rv, "best_guess_resolution", None))
                if getattr(rv, "best_guess_resolution", None) is not None else None,
            )
            for rv in strict_resolutions
            if expected_forecasts is None
            or (rv.forecast_date, rv.dimension) in expected_date_dim_keys
        ],
        forecasts=forecasts,
        rationale=strict.rationale,
        sources=[s.url for s in strict.sources],
    )
    return response, evidence


def _parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _contains_leap_forecast_anchor(response: "SurveillanceResponse") -> bool:
    """Detect actual LEAP forecast sourcing without flagging explicit non-use statements."""
    source_text = " ".join(response.sources or []).lower()
    if "leap.forecastingresearch.org" in source_text or "leap wave" in source_text:
        return True

    rationale = response.rationale or ""
    anchor_patterns = ("leap.forecastingresearch.org", "leap wave")
    negation_re = re.compile(
        r"\b(did not|didn't|do not|don't|not|never|without|excluded?|skip(?:ped)?|avoid(?:ed)?)\b"
        r".{0,120}\b(use|read|cite|consult|anchor|rely|include|draw)\b"
        r"|\b(use|read|cite|consult|anchor|rely|include|draw)\b.{0,120}"
        r"\b(did not|didn't|do not|don't|not|never|without|excluded?|skip(?:ped)?|avoid(?:ed)?)\b",
        re.IGNORECASE,
    )
    exclusion_re = re.compile(r"\b(did not|didn't|do not|don't|never|without|excluded?|skip(?:ped)?|avoid(?:ed)?)\b", re.IGNORECASE)
    for sentence in re.split(r"(?<=[.!?])\s+", rationale):
        lowered = sentence.lower()
        if not any(pattern in lowered for pattern in anchor_patterns):
            continue
        if negation_re.search(sentence) or exclusion_re.search(sentence):
            continue
        return True
    return False


def validate_response(
    response: SurveillanceResponse, expected: list[ExpectedForecast] | None = None,
    unit_min: float | None = None, unit_max: float | None = None,
) -> dict:
    issues = []
    expected_has_q50 = bool(expected and any(e.quantile == 50 for e in expected))

    if not response.forecasts:
        issues.append("no_forecasts")
    if not response.sources:
        issues.append("no_sources")

    # Flag actual anchoring on LEAP's own panel forecasts, not rationale sentences denying it.
    if _contains_leap_forecast_anchor(response):
        issues.append("leap_forecast_anchoring")

    if expected:
        expected_keys = {(e.forecast_date, e.dimension) for e in expected}
        expected_resolution_keys = {
            (e.forecast_date, e.dimension) for e in expected if e.value_type == "resolution"
        }
        forecast_keys = {(f.forecast_date, f.dimension) for f in response.forecasts}
        missing_keys = expected_keys - forecast_keys
        if missing_keys:
            issues.append(f"missing_{len(missing_keys)}_date_dims")
        unexpected_keys = forecast_keys - expected_keys
        if unexpected_keys:
            issues.append(f"unexpected_{len(unexpected_keys)}_date_dims")

        if expected_has_q50:
            q50s = {
                (f.forecast_date, f.dimension)
                for f in response.forecasts
                if f.quantile == 50 and f.forecast_value is not None
            }
            # Resolution rows intentionally null their q50 (the value lives in resolution_values).
            missing_q50s = expected_keys - q50s - expected_resolution_keys
            if missing_q50s:
                issues.append(f"missing_{len(missing_q50s)}_q50s")

        expected_triplets = {(e.forecast_date, e.dimension, e.quantile) for e in expected}
        forecast_triplets = {(f.forecast_date, f.dimension, f.quantile) for f in response.forecasts}
        missing_triplets = expected_triplets - forecast_triplets
        if missing_triplets:
            issues.append(f"missing_{len(missing_triplets)}_forecast_rows")
        unexpected_triplets = forecast_triplets - expected_triplets
        if unexpected_triplets:
            issues.append(f"unexpected_{len(unexpected_triplets)}_forecast_rows")
        if len(forecast_triplets) < len(response.forecasts):
            issues.append("duplicate_forecast_rows")

        for r in response.resolution_values:
            target_date = _parse_iso_date(r.forecast_date)
            source_date = _parse_iso_date(r.source_date)
            if target_date and source_date and source_date > target_date:
                issues.append(f"resolution_source_after_target_{r.forecast_date}_{r.dimension}")

        if expected_resolution_keys:
            # A resolution row counts as returned if it carries an authoritative value OR a best guess.
            returned_resolution_keys = {
                (r.forecast_date, r.dimension) for r in response.resolution_values
                if r.value is not None or r.best_guess_resolution is not None
            }
            missing_resolutions = expected_resolution_keys - returned_resolution_keys
            if missing_resolutions:
                issues.append(f"missing_{len(missing_resolutions)}_resolution_values")

    # Nulls on resolution rows are expected (no forecast on a past date); only count forecast rows.
    forecast_rows = [f for f in response.forecasts if getattr(f.value_type, "value", f.value_type) != "resolution"]
    null_count = sum(1 for f in forecast_rows if f.forecast_value is None)
    if forecast_rows and null_count > len(forecast_rows) // 2:
        issues.append(f"{null_count}_null_values")

    if unit_min is not None or unit_max is not None:
        for f in response.forecasts:
            v = f.forecast_value
            if v is None:
                continue
            if unit_min is not None and v < unit_min:
                issues.append(f"unit_bound_violation_below_min_{f.forecast_date}_{f.dimension}_q{f.quantile}")
                break
            if unit_max is not None and v > unit_max:
                issues.append(f"unit_bound_violation_above_max_{f.forecast_date}_{f.dimension}_q{f.quantile}")
                break

    quantile_groups = defaultdict(list)
    for f in response.forecasts:
        if f.forecast_value is not None and f.quantile is not None:
            quantile_groups[(f.forecast_date, f.dimension)].append((f.quantile, f.forecast_value))
    for key, forecasts in quantile_groups.items():
        sorted_forecasts = sorted(forecasts, key=lambda x: x[0])
        for i in range(len(sorted_forecasts) - 1):
            if sorted_forecasts[i][1] > sorted_forecasts[i + 1][1]:
                issues.append(f"quantiles_not_increasing_{key[0]}_{key[1]}")
                break

    resolution_keys = {
        (r.forecast_date, r.dimension)
        for r in response.resolution_values
        if r.value is not None and r.source
    }
    black_groups = defaultdict(list)
    black_color_groups = set()
    group_colors = defaultdict(set)
    for f in response.forecasts:
        group_colors[(f.forecast_date, f.dimension)].add(f.color_code.value)
        if f.color_code.value == "black":
            key = (f.forecast_date, f.dimension)
            black_color_groups.add(key)
            if f.forecast_value is not None:
                black_groups[key].append(f.forecast_value)

    for key, values in black_groups.items():
        if len({round(v, 12) for v in values}) > 1:
            issues.append(f"black_quantiles_differ_{key[0]}_{key[1]}")
    for key in black_color_groups:
        if key not in resolution_keys:
            issues.append(f"black_without_resolution_value_{key[0]}_{key[1]}")
    for key, colors in group_colors.items():
        if len(colors) > 1:
            issues.append(f"mixed_colors_{key[0]}_{key[1]}_{'/'.join(sorted(colors))}")

    # A failed/unresolved resolution row legitimately isn't black (no authoritative value).
    resolution_status_by_key = {
        (r.forecast_date, r.dimension): (r.resolution_status or "unresolved")
        for r in response.resolution_values
    }
    seen_resolution_not_black: set[tuple] = set()
    for f in response.forecasts:
        if getattr(f.value_type, "value", f.value_type) == "resolution" and f.color_code.value != "black":
            key = (f.forecast_date, f.dimension)
            if resolution_status_by_key.get(key, "unresolved") in ("failed", "unresolved"):
                continue
            if key not in seen_resolution_not_black:
                seen_resolution_not_black.add(key)
                issues.append(f"resolution_not_black_{f.forecast_date}_{f.dimension}")

    if expected_has_q50:
        usable_for_scoring = any(
            f.quantile == 50 and f.forecast_value is not None for f in response.forecasts
        )
    else:
        usable_for_scoring = any(f.forecast_value is not None for f in response.forecasts)
    return {"ok": len(issues) == 0, "usable_for_scoring": usable_for_scoring, "issues": issues}
