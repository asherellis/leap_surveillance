"""Data schemas for surveillance questions, evidence, and responses."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


from pydantic import BaseModel, ConfigDict, Field

# Reject unexpected fields in strict structured outputs.
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
    question_type: str = "quantile"  # "quantile", "probability", or "when"
    unit: str = ""
    unit_min: Optional[float] = None
    unit_max: Optional[float] = None


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

    @property
    def total(self) -> float:
        return self.research + self.judge_stage1 + self.judge_stage2 + self.refinement

    def as_dict(self) -> dict:
        return {
            "research": self.research,
            "judge_stage1": self.judge_stage1,
            "judge_stage2": self.judge_stage2,
            "refinement": self.refinement,
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
    current_resolution_values: list[CurrentValue]
    resolution_values: list[ResolutionValue]
    forecasts: list[ForecastValue]
    rationale: str
    sources: list[str]


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
    current_resolution_values: list[StrictCurrentValue]
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


def convert_sentinel_value(v: float) -> Optional[float]:
    """Convert the strict-schema no-estimate sentinel back to None."""
    return None if v == -999 else v


def strict_to_regular_response(
    strict: StrictSurveillanceResponse,
    expected_forecasts: Optional[list[ExpectedForecast]] = None,
) -> tuple[SurveillanceResponse, list[EvidenceItem]]:
    evidence = [
        EvidenceItem(
            source_type="web_search", url=s.url, title=s.title, snippet=s.snippet
        )
        for s in strict.sources
    ]

    expected_value_types = {}
    if expected_forecasts:
        expected_value_types = {
            (e.forecast_date, e.dimension, e.quantile): e.value_type
            for e in expected_forecasts
        }
    resolution_by_key = {
        (rv.forecast_date, rv.dimension): convert_sentinel_value(rv.value)
        for rv in strict.resolution_values
    }
    expected_resolution_keys = {
        (e.forecast_date, e.dimension)
        for e in expected_forecasts or []
        if e.value_type == "resolution"
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
            forecast_value = convert_sentinel_value(fv.forecast_value)
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
                value=convert_sentinel_value(ov.value),
                date=ov.date,
                source=ov.source,
            )
            for ov in strict.last_official_values
        ],
        current_resolution_values=[
            CurrentValue(
                dimension=cv.dimension,
                value=convert_sentinel_value(cv.value),
                confidence=cv.confidence,
            )
            for cv in strict.current_resolution_values
        ],
        resolution_values=[
            ResolutionValue(
                forecast_date=rv.forecast_date,
                dimension=rv.dimension,
                value=convert_sentinel_value(rv.value),
                source_date=rv.source_date,
                source=rv.source,
                confidence=rv.confidence,
            )
            for rv in strict.resolution_values
            if expected_forecasts is None
            or (rv.forecast_date, rv.dimension) in expected_resolution_keys
        ],
        forecasts=forecasts,
        rationale=strict.rationale,
        sources=[s.url for s in strict.sources],
    )
    return response, evidence
