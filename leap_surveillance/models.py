"""Schemas and deterministic validation for LEAP surveillance."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
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


@dataclass
class EvidenceItem:
    source_type: str
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    full_text: Optional[str] = None


@dataclass
class RunCost:
    research: float = 0.0
    judge_stage1: float = 0.0
    judge_stage2: float = 0.0
    refinement: float = 0.0
    # Claude path (dual-model mode).
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


# Strict* models describe the LLM JSON output. The runtime models above use
# nullable values, flattened source URLs, and system-assigned forecast value_type.
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
    value: float
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


def make_strict_schema(
    model: type[BaseModel],
    allowed_dimensions: Optional[list[str]] = None,
    allowed_quantiles: Optional[list[int]] = None,
    allowed_forecast_dates: Optional[list[str]] = None,
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
    return schema


def _convert_sentinel_value(v: float) -> Optional[float]:
    return None if v == -999 else v


def strict_to_regular_response(
    strict: StrictSurveillanceResponse | str,
    expected_forecasts: Optional[list[ExpectedForecast]] = None,
) -> tuple[SurveillanceResponse, list[EvidenceItem]]:
    if isinstance(strict, str):
        strict = StrictSurveillanceResponse.model_validate_json(strict)

    evidence = [
        EvidenceItem(source_type="web_search", url=s.url, title=s.title, snippet=s.snippet)
        for s in strict.sources
    ]
    expected_value_types = {
        (e.forecast_date, e.dimension, e.quantile): e.value_type
        for e in expected_forecasts or []
    }
    resolution_by_key = {
        (rv.forecast_date, rv.dimension): _convert_sentinel_value(rv.value)
        for rv in strict.resolution_values
    }
    expected_date_dim_keys = {
        (e.forecast_date, e.dimension)
        for e in expected_forecasts or []
    }

    forecasts = []
    for fv in strict.forecasts:
        key = (fv.forecast_date, fv.dimension, fv.quantile)
        value_type = ValueType(expected_value_types.get(key, "forecast"))
        resolution_value = resolution_by_key.get((fv.forecast_date, fv.dimension))
        if value_type == ValueType.resolution and resolution_value is not None:
            forecast_value = resolution_value
            color_code = ColorCode.black
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
            for ov in strict.last_official_values
        ],
        current_estimates=[
            CurrentValue(
                dimension=cv.dimension,
                value=_convert_sentinel_value(cv.value),
                confidence=cv.confidence,
            )
            for cv in strict.current_estimates
        ],
        resolution_values=[
            ResolutionValue(
                forecast_date=rv.forecast_date,
                dimension=rv.dimension,
                value=_convert_sentinel_value(rv.value),
                source_date=rv.source_date,
                source=rv.source,
                confidence=rv.confidence,
            )
            for rv in strict.resolution_values
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

    if expected:
        expected_keys = {(e.forecast_date, e.dimension) for e in expected}
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
            missing_q50s = expected_keys - q50s
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

    null_count = sum(1 for f in response.forecasts if f.forecast_value is None)
    if null_count > len(response.forecasts) // 2:
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

    if expected_has_q50:
        usable_for_scoring = any(
            f.quantile == 50 and f.forecast_value is not None for f in response.forecasts
        )
    else:
        usable_for_scoring = any(f.forecast_value is not None for f in response.forecasts)
    return {"ok": len(issues) == 0, "usable_for_scoring": usable_for_scoring, "issues": issues}
