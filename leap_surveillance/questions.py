"""Load and shape LEAP question metadata from BigQuery."""

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
    unit_name: str,
    dates: list[str],
    source_percentiles: list[int],
) -> str:

    if unit_name.strip().lower() == "probability":
        if set(source_percentiles) == {50}:
            return "probability"
        # Explicit return so a no-dates probability-unit question can't fall through to "when".
        print(f"  warning: '{question_text}' - probability unit question with non-50 percentiles not supported; handled as a quantile question instead.")
        return "quantile"
    if unit_name.strip().lower() == "year" or not dates:
        return "when"
    return "quantile"

def _build_prompt_context(row, dates: list[str], dimensions: list[str]) -> str:
    prompt = safe_str(row.get("question_set_text"))
    bg = safe_str(row.get("question_set_background_information"))
    if bg:
        prompt += f"\n\nBackground:\n{bg}"
    res = safe_str(row.get("question_set_resolution_criteria"))
    if res:
        prompt += f"\n\nResolution:\n{res}"
    unit = safe_str(row.get("unit_name"))
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


DEV_QUESTION_IDS = (
    "b08de26c49ee71b135534e09057cc8323168477d0383c505fac80f50504345c3",  # LiveCodeBench Pro (quantile, browser)
    "5010797e20c65f4635321a01a81fe08e297b15c0abaddc5a4be474824a435856",  # Labor Share (quantile, big grid)
    "6bf9bae3e8b4d3c1a5fe0c48af99b1941bbf1aa61f2173e75e4b6f2db0e3a9ef",  # U.S. and China Military Agreement (probability)
    "147346ac17bb49d1d0ae6b0894f1d96b803ed5d1d315f7dddb5130799cb52714",  # Coffee Test (when / timing)
    "10a514657ba6e075f831ef8d5e4635ffd9003f24e0f6a2e4db8f95999c98bfeb",  # U.S. versus China Polarity (multi-dim)
)


def load_questions(limit=None, dev=False) -> list[QuestionSpec]:

    dev_filter = ""
    if dev:
        ids = ", ".join(f'"{qid}"' for qid in DEV_QUESTION_IDS)
        dev_filter = f"AND qg.question_group_id IN ({ids})"

    # Conditional (scenario) questions are excluded: scenario_id IS NULL is filtered per-row,
    # before any GROUP BY, so mixed groups keep only their unconditional questions and
    # fully-conditional groups drop out entirely.
    query = f"""
    WITH forecast_groups AS (
        SELECT qg.question_group_id, qg.question_group_name
        FROM `{DEFAULT_BQ_PROJECT}.dim.dim_question_group` qg
        JOIN `{DEFAULT_BQ_PROJECT}.dim.dim_question` q
            ON qg.question_group_id = q.question_group_id
        WHERE q.project_id = 'leap' AND q.question_type = 'forecast'
        AND q.scenario_id IS NULL
        {dev_filter}
        GROUP BY qg.question_group_id, qg.question_group_name
        ORDER BY qg.question_group_name
        {"LIMIT " + str(limit) if limit else ""}
    )
    SELECT
        qg.question_group_id AS question_set_id,
        qg.question_group_name AS question_set_name,
        qg.question_group_text AS question_set_text,
        qg.question_group_background_info AS question_set_background_information,
        qg.question_group_resolution_criteria AS question_set_resolution_criteria,
        u.unit_name, u.unit_min_value, u.unit_max_value,
        q.question_id,
        q.question_horizon_date AS question_resolution_date,
        q.question_percentile,
        q.question_dimension
    FROM `{DEFAULT_BQ_PROJECT}.dim.dim_question_group` qg
    JOIN forecast_groups fg ON qg.question_group_id = fg.question_group_id
    JOIN `{DEFAULT_BQ_PROJECT}.dim.dim_question` q
        ON qg.question_group_id = q.question_group_id
        AND q.project_id = 'leap'
        AND q.question_type = 'forecast'
        AND q.scenario_id IS NULL
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
        unit_name = safe_str(row.get("unit_name"))

        if not dates and not source_percentiles:
            print(f"  warning: skipping '{row.get('question_set_name')}' - no date or percentile data, cannot build a forecast")
            continue

        question_type = _infer_surveillance_question_type(
            question_text, unit_name, dates, source_percentiles
        )
        prompt = _build_prompt_context(row, dates, dimensions)
        expected = _expected_forecasts(question_type, dates, dimensions)

        question_id = row.get("question_set_id")
        question_name = row.get("question_set_name")
        if not question_id or not question_name or not expected:
            continue
        if not safe_str(row.get("question_set_resolution_criteria")):
            print(f"  warning: no resolution_criteria for '{question_name}'")

        # Map "fdate|dim" -> dim_question.question_id for q50 rows only (when-type dates are NULL, use TIMING_FORECAST_DATE).
        dim_q_map = {}
        for _, r in group.iterrows():
            pct = r.get("question_percentile")
            if is_empty(pct) or int(float(pct)) != 50:
                continue
            raw_date = r.get("question_resolution_date")
            if not is_empty(raw_date):
                fdate = str(raw_date).strip()
            else:
                fdate = TIMING_FORECAST_DATE
            # NaN is truthy, so `or "Overall"` alone would produce a "nan" key; use is_empty.
            raw_dim = r.get("question_dimension")
            dim = (safe_str(raw_dim).strip() if not is_empty(raw_dim) else "") or "Overall"
            dq_id = r.get("question_id")
            if not is_empty(dq_id):
                dim_q_map[f"{fdate}|{dim}"] = str(dq_id)

        questions.append(
            QuestionSpec(
                question_id,
                question_name,
                prompt,
                expected,
                question_type=question_type,
                unit=unit_name,
                unit_min=to_float(row.get("unit_min_value")),
                unit_max=to_float(row.get("unit_max_value")),
                question_text=question_text,
                resolution_criteria=safe_str(row.get("question_set_resolution_criteria")),
                background_info=safe_str(row.get("question_set_background_information")),
                dim_question_map=dim_q_map,
            )
        )

    return questions
