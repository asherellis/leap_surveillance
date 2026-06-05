"""Load and shape LEAP question metadata from BigQuery."""

import re

from .common import (
    DEFAULT_BQ_PROJECT,
    FULL_QUANTILES,
    TIMING_FORECAST_DATE,
    date_value_type,
    is_empty,
    safe_str,
    to_float,
)
from .models import ExpectedForecast, QuestionSpec
from .storage import query_bq


def _infer_surveillance_question_type(
    question_text: str,
    unit: str,
    dates: list[str],
    dimensions: list[str],
    source_percentiles: list[int],
) -> str:
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


def _build_prompt_context(row, dates: list[str], dimensions: list[str]) -> str:
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
    return prompt


def _expected_forecasts(
    question_type: str, dates: list[str], dimensions: list[str]
) -> list[ExpectedForecast]:
    if question_type == "probability":
        return [
            ExpectedForecast(d, dim, 50, date_value_type(d))
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
        ExpectedForecast(d, dim, p, date_value_type(d))
        for d in dates
        for dim in dimensions
        for p in FULL_QUANTILES
    ]


def load_questions(limit=None) -> list[QuestionSpec]:

    query = f"""
    WITH forecast_groups AS (
        SELECT qg.question_group_id, qg.question_group_name
        FROM `{DEFAULT_BQ_PROJECT}.dim.dim_question_group` qg
        JOIN `{DEFAULT_BQ_PROJECT}.dim.dim_question` q
            ON qg.question_group_id = q.question_group_id
        WHERE q.project_id = 'leap' AND q.question_type = 'forecast'
        GROUP BY qg.question_group_id, qg.question_group_name
        ORDER BY qg.question_group_name
        {"LIMIT " + str(limit) if limit else ""}
    )
    SELECT
        qg.question_group_id AS question_set_id,
        qg.question_group_name AS question_set_name,
        q.question_text AS question_set_text,
        qg.question_group_background_info AS question_set_background_information,
        qg.question_group_resolution_criteria AS question_set_resolution_criteria,
        u.unit_display_text, u.unit_min_value, u.unit_max_value,
        q.question_horizon_date AS question_resolution_date,
        q.question_percentile,
        q.question_dimension
    FROM `{DEFAULT_BQ_PROJECT}.dim.dim_question_group` qg
    JOIN forecast_groups fg ON qg.question_group_id = fg.question_group_id
    JOIN `{DEFAULT_BQ_PROJECT}.dim.dim_question` q
        ON qg.question_group_id = q.question_group_id
        AND q.project_id = 'leap'
        AND q.question_type = 'forecast'
    LEFT JOIN `{DEFAULT_BQ_PROJECT}.dim.dim_unit` u ON qg.unit_id = u.unit_id
    ORDER BY qg.question_group_name, q.question_horizon_date, q.question_dimension, q.question_percentile"""

    df = query_bq(query)
    questions = []
    for _, group in df.groupby("question_set_id", sort=False):
        row = group.iloc[0]
        dates = sorted(
            {
                safe_str(d).strip()
                for d in group["question_resolution_date"].tolist()
                if not is_empty(d)
            }
        )
        dimensions = sorted(
            {
                safe_str(d).strip()
                for d in group["question_dimension"].tolist()
                if not is_empty(d)
            }
        ) or ["Overall"]
        source_percentiles = sorted(
            {
                int(float(p))
                for p in group["question_percentile"].tolist()
                if not is_empty(p)
            }
        )

        question_text = safe_str(row.get("question_set_text"))
        unit = safe_str(row.get("unit_display_text"))

        # Points-allocation groups are not quantile, timing, or probability questions.
        if not dates and not source_percentiles:
            print(f"  warning: skipping '{row.get('question_set_name')}' - points-allocation type not supported")
            continue

        question_type = _infer_surveillance_question_type(
            question_text, unit, dates, dimensions, source_percentiles
        )
        prompt = _build_prompt_context(row, dates, dimensions)
        expected = _expected_forecasts(question_type, dates, dimensions)

        question_id = row.get("question_set_id")
        question_name = row.get("question_set_name")
        if not question_id or not question_name or not expected:
            continue
        if not safe_str(row.get("question_set_resolution_criteria")):
            print(f"  warning: no resolution_criteria for '{question_name}'")
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
                question_text=question_text,
                resolution_criteria=safe_str(row.get("question_set_resolution_criteria")),
                background_info=safe_str(row.get("question_set_background_information")),
            )
        )

    return questions
