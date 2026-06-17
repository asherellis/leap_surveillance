"""LLM calls: research, adequacy judge, browser decision, and refinement."""

import json
import os
import re
import time
from datetime import datetime, timezone

import anthropic
import openai
from dotenv import load_dotenv

from .browser import BROWSER_EVIDENCE_LIMIT, is_safe_url
from .common import (
    DEFAULT_EVALUATOR_MODEL,
    DEFAULT_MODEL,
    TEST_CLAUDE_MODEL,
    TEST_EVALUATOR_MODEL,
    TEST_MODEL,
    _env_float,
    _env_int,
    cost_for_tokens,
    provider_for_model,
    strip_provider_prefix,
)
from .models import (
    AdequacyAssessment,
    BrowserDecision,
    BrowserEvidence,
    EvidenceItem,
    ExpectedForecast,
    QuestionSpec,
    ResearchQualityReport,
    RunCost,
    StrictSurveillanceResponse,
    SurveillanceResponse,
    make_strict_schema,
    strict_to_regular_response,
)

load_dotenv()

OPENAI_SAFETY_IDENTIFIER = os.environ.get("OPENAI_SAFETY_IDENTIFIER", "")

DEFAULT_REASONING_EFFORT = "high"

RESEARCH_TIMEOUT = _env_float("LEAP_RESEARCH_TIMEOUT", 1800.0)
EVALUATION_TIMEOUT = _env_float("LEAP_EVALUATION_TIMEOUT", 120.0)
MAX_TIMEOUT_RETRIES = _env_int("LEAP_MAX_TIMEOUT_RETRIES", 2)
MAX_SEARCH_ROUNDS = _env_int("LEAP_MAX_SEARCH_ROUNDS", 20)
MIN_ADEQUATE_CONFIDENCE = _env_int("LEAP_MIN_ADEQUATE_CONFIDENCE", 65)


# Collapse degenerate LLM floats (0.000000...0) that bloat JSON past token limit.
_COLLAPSE_FLOATS_RE = re.compile(r'(\d+\.\d{4})\d{4,}')


QUANTILE_INTERPRETATION_FULL = """Quantile interpretation:
Return exactly seven forecast entries for each date/dimension: q0, q5, q25, q50, q75, q95, and q100.

q0 and q100 are feasibility bounds, not ordinary probabilistic quantiles.
- q0 is the lowest value still possible given current constraints. Use the natural unit lower bound (e.g., 0 for a percent or count). For a cumulative or never-decreasing metric, use the latest known value as the floor.
- q100 is the highest value still possible given current constraints. Use the natural unit upper bound if one exists (e.g., 100 for a percent). If there is no natural upper bound, use a high but coherent practical-tail value (a 99.99th-percentile scenario).

Use q5/q25/q50/q75/q95 as the probability distribution, with q50 as the median. Values must be non-decreasing: q0 <= q5 <= q25 <= q50 <= q75 <= q95 <= q100. Adjacent quantiles may be equal when justified by a feasibility bound or high-confidence point mass (e.g., q0 = q5 = current value for a cumulative metric near its floor).

Use -999 only when no reasonable estimate is possible for a specific value."""


QUANTILE_INTERPRETATION_BRIEF = """Quantile interpretation:
Return exactly seven forecast entries for each date/dimension combination, one for each quantile: 0, 5, 25, 50, 75, 95, and 100.

All seven quantile forecast values must be valid numbers and non-decreasing: q0 <= q5 <= q25 <= q50 <= q75 <= q95 <= q100. Adjacent quantiles may be equal when justified by feasibility bounds or high confidence.

Quantile meanings:
- q=0: Lowest value still possible. Use the natural unit lower bound, or the latest known value for a cumulative metric that cannot decrease.
- q=5, 25, 50, 75, 95: Probability distribution (q=50 is your median/best estimate).
- q=100: Highest value still possible. Use the natural unit upper bound, or a 99.99th-percentile scenario if no natural bound exists.

Use -999 only when no reasonable estimate is possible for a specific value."""


COLOR_CODE_SYSTEM_FULL = """Color coding:
Assign one color_code per date/dimension combination, using the same color for all quantiles in that group.

- black: resolved - use only when the group has a source-backed value in resolution_values, or a development definitively resolves it. A past date alone is not enough; if no authoritative value exists yet, use the non-black color that reflects your remaining uncertainty.
- dark gray: new evidence changes the feasible range; some previously possible values are now impossible
- light gray: the full natural range is still feasible, but monotonic change makes part of it implausible for the interior quantiles
- white: the full natural range remains feasible and there is not enough information to narrow it

Color reflects how much uncertainty has collapsed, not whether the date is past or future. If color_code is black for a date/dimension group, all quantile values in that group must be identical.

Use the same gray shade across non-black future dates unless a physical or definitional reason restricts an earlier date that does not apply to later ones.

Explain the color choice in the rationale. For dark gray or light gray, name the specific report, data point, or finding that justifies the choice."""


COLOR_CODE_SYSTEM_BRIEF = """Color coding:
- black: resolved - authoritative value available in resolution_values (a past date alone is not enough)
- dark gray: feasible range changed; part of the previous range is now impossible
- light gray: full range remains feasible, but part is implausible because of monotonic expectations
- white: full natural range remains feasible; no monotonicity assumptions

For dark gray or light gray, name the specific report, data point, or finding in the rationale."""


RESEARCH_PRINCIPLES = """Research principles:
- Match the period. When a value is tied to a specific date or period, find the value as of that period and prefer sources from that period; include the period in your searches. Do not substitute a value from a different time than the one asked about.
- Use central estimates. Report a point estimate or stated central value, not an interval bound or range endpoint. If a source gives a range or confidence interval, use its midpoint or stated central value, and say which you used in your rationale.
- Match the unit. Report all values in the question's stated unit. If units are percent, report percentages (e.g. 58.3), not decimals (0.583). For dollar metrics, respect the denomination - USD ($), USD ($ Millions), or USD ($ Billions).
- Respect scope. If the metric is limited to a particular category, track, subset, or population, confirm each source matches that scope, and note in your rationale what you included and excluded.
- Use base rates where available. When the metric has historical data or a clear reference class, use it as an anchor for the forecast. If your forecast departs materially from that history or reference class, say why.
- Build aggregates exactly as the question defines them. If the metric is an average, sum, or index over components, gather each component and compute it yourself rather than copying a headline figure; list the components used. Combine them using the method the question's resolution criteria specifies; impose no default weighting of your own. When the component structure is ambiguous (e.g., a benchmark reports multiple sub-scores or tiers and the criteria doesn't say how they roll up), state the assumption you made and flag it explicitly in your rationale.
- Don't manufacture data. Ground each value in a source that actually reports it. When you cannot, prefer -999 (or appropriately wide uncertainty for forecast rows) over inferring a number from out-of-scope or out-of-period material. Never present an extrapolation as if it were observed data."""


RATIONALE_REQUIREMENTS = """Rationale requirements:
- State the source basis for the latest official value, current estimate, and each resolved target-date value.
- For each target date/dimension, state the status decision: resolved, unresolved past date, future forecast, or resolved early.
- Explain the color_code choice for each target date/dimension.
- State any unit conversion, aggregation formula, or scope assumption used.
- For forecast rows, explain the q50 and the main drivers of the q25/q75 spread. Do not explain every quantile separately unless q0/q100 need special justification."""


def _expected_forecast_lines(expected_forecasts: list[ExpectedForecast]) -> str:
    return "\n".join(
        f"  - {ef.forecast_date}, {ef.dimension}, q={ef.quantile}, value_type={ef.value_type}"
        for ef in expected_forecasts
    )


def _resolution_guidance(question: QuestionSpec) -> str:
    if not any(f.value_type == "resolution" for f in question.expected_forecasts):
        return "No requested forecast rows are past resolution dates. Return an empty resolution_values list unless a future target date has already resolved early."

    return """Resolution value guidance:
Some requested rows have value_type="resolution": their target date has passed, so try to find the metric's authoritative value as of that exact target date. A past date does not guarantee that a resolved value exists.

- If you find an authoritative value for the target date: report it in resolution_values (with source and source_date), set all quantiles for that group to that single value, and use color_code="black".
- If no authoritative value exists for the target date (the metric was never measured for it, the data is not published yet, or only post-target-date sources exist): leave that group out of resolution_values, give your best-estimate distribution across the quantiles, and use the non-black color that reflects your remaining uncertainty. Put your single best guess in current_estimates so the estimate is not lost. A non-black distribution for a past target date where no official value has been published is correct behavior and will not be penalized.
- resolution_values.source_date = the date the value represents, not the source's publication date.
- current_estimates reflect today's best guess and must never overwrite or substitute for a past resolution value."""


def _question_type_guidance(question: QuestionSpec, full: bool = True) -> str:
    if question.question_type == "probability":
        return """Probability question guidance:
This is a probability question. Return exactly one forecast entry for each expected date/dimension row, with quantile=50.

Forecast values are probabilities on a 0 to 100 scale. Do not return the 0, 5, 25, 75, 95, or 100 quantiles for probability questions.

Latest official values and current estimates are not applicable for this question type. For those value fields, use -999 and set confidence=0.

Use color_code="white" if the event remains unresolved and the full probability range is still open. Use color_code="black" if the event has occurred or the question is already resolved."""

    if question.question_type == "when":
        return """Timing question guidance:
This is a timing question asking when an event will first occur. The forecast_date value is a placeholder label; the forecast_value itself must be a year.

Return exactly seven forecast entries for each expected row: quantiles 0, 5, 25, 50, 75, 95, and 100. Forecast values must be years only, with no ranges, dates, or extra text.

All seven timing quantiles must be present, non-null, and non-decreasing. If the event has not yet occurred, q=0 should usually be the current year as the earliest feasible occurrence year.

If p(never) is at least 5%, use 9999 as the q100 year. 9999 is a sentinel meaning "never", not a literal year.
For events that are long-tail-but-finite, use a real far-future year such as 2200, 2300, or 2500.
Do not use 9999 as a default upper bound. Use it only when the event might never occur.
Multiple upper quantiles can be 9999 only if they all represent that same "never" judgment and monotonicity is preserved (e.g., q95=9999 only if q100=9999 too).

Latest official values and current estimates are not applicable for this question type. For those value fields, use -999 and set confidence=0.

Use color_code="black" if the event has already occurred. If it has not occurred, use color_code="dark gray" because first occurrence in past years is no longer feasible."""

    return QUANTILE_INTERPRETATION_FULL if full else QUANTILE_INTERPRETATION_BRIEF


def _research_schema(question: QuestionSpec) -> dict:
    return make_strict_schema(
        StrictSurveillanceResponse,
        allowed_dimensions=sorted({f.dimension for f in question.expected_forecasts}),
        allowed_quantiles=sorted({f.quantile for f in question.expected_forecasts if f.quantile is not None}),
        allowed_forecast_dates=sorted({f.forecast_date for f in question.expected_forecasts}),
        unit_min=question.unit_min,
        unit_max=question.unit_max,
    )


def _claude_research(
    question: QuestionSpec,
    model: str,
    prompt: str,
    *,
    attempt: int = 1,
    costs: RunCost | None = None,
    cost_bucket: str = "research",
) -> tuple[SurveillanceResponse, list[EvidenceItem]]:
    """Research via Anthropic SDK with server-side web search and adaptive thinking."""
    schema = _research_schema(question)
    produce_tool = {
        "name": "produce_forecast",
        "description": "Output the structured surveillance forecast.",
        "input_schema": schema,
    }
    messages = [{"role": "user", "content": prompt}]
    model_id = strip_provider_prefix(model)

    timeout_attempt = 0
    search_round = 0
    while True:
        try:
            thinking = {"thinking": {"type": "adaptive"}} if "haiku" not in model_id else {}
            response = anthropic.Anthropic().messages.create(
                model=model_id,
                max_tokens=64000,
                tools=[{"type": "web_search_20250305", "name": "web_search"}, produce_tool],
                messages=messages,
                timeout=RESEARCH_TIMEOUT,
                **thinking,
            )
        except Exception as e:
            error_str = str(e).lower()
            if ("timeout" in error_str or "timed out" in error_str) and timeout_attempt < MAX_TIMEOUT_RETRIES:
                timeout_attempt += 1
                print(f"    timeout retry {timeout_attempt}/{MAX_TIMEOUT_RETRIES}...")
                continue
            if isinstance(e, anthropic.RateLimitError):
                print("    rate limit; sleeping 60s...")
                time.sleep(60)
                continue
            raise

        if costs is not None:
            setattr(costs, cost_bucket, getattr(costs, cost_bucket) + cost_for_tokens(model_id, response.usage.input_tokens, response.usage.output_tokens))

        if response.stop_reason == "pause_turn":
            search_round += 1
            if search_round >= MAX_SEARCH_ROUNDS:
                raise RuntimeError(f"Claude search exceeded {MAX_SEARCH_ROUNDS} rounds without producing a forecast")
            messages = messages + [{"role": "assistant", "content": [b.model_dump() for b in response.content]}]
            continue

        tool_input = next(
            (b.input for b in response.content if getattr(b, "type", None) == "tool_use" and b.name == "produce_forecast"),
            None,
        )
        if tool_input is not None:
            text = json.dumps(tool_input)
        else:
            text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
            text = _COLLAPSE_FLOATS_RE.sub(r'\1', text)
            if text and text[0] != '{':
                idx = text.find('{')
                if idx > 0:
                    text = text[idx:]

        try:
            return strict_to_regular_response(text, question.expected_forecasts)
        except Exception as e:
            retryable = "EOF while parsing" in str(e) or "expected ident" in str(e) or "validation error" in str(e).lower()
            if attempt < 3 and retryable:
                print(f"    parse retry {attempt + 1}/3 ({len(text)} chars): {e}")
                return _claude_research(question, model, prompt, attempt=attempt + 1, costs=costs, cost_bucket=cost_bucket)
            print(f"    validation error (raw text[:500]): {text[:500]}")
            raise


def _gpt_research(
    question: QuestionSpec,
    model: str,
    prompt: str,
    *,
    attempt: int = 1,
    test_mode: bool = False,
    costs: RunCost | None = None,
    cost_bucket: str = "research",
) -> tuple[SurveillanceResponse, list[EvidenceItem]]:
    """Research via OpenAI Responses API with server-side web search and json_schema output."""
    schema = _research_schema(question)
    model_id = strip_provider_prefix(model)
    extra = {} if test_mode else {"reasoning": {"effort": DEFAULT_REASONING_EFFORT}}

    timeout_attempt = 0
    while True:
        try:
            response = openai.OpenAI().responses.create(
                model=model_id,
                input=[{"role": "user", "content": prompt}],
                tools=[{"type": "web_search"}],
                text={"format": {"type": "json_schema", "name": "StrictSurveillanceResponse", "schema": schema, "strict": True}},
                max_output_tokens=64000,
                timeout=RESEARCH_TIMEOUT,
                safety_identifier=OPENAI_SAFETY_IDENTIFIER or None,
                **extra,
            )
            break
        except Exception as e:
            error_str = str(e).lower()
            if ("timeout" in error_str or "timed out" in error_str) and timeout_attempt < MAX_TIMEOUT_RETRIES:
                timeout_attempt += 1
                print(f"    timeout retry {timeout_attempt}/{MAX_TIMEOUT_RETRIES}...")
                continue
            if isinstance(e, openai.RateLimitError):
                print("    rate limit; sleeping 60s...")
                time.sleep(60)
                continue
            raise

    if costs is not None:
        setattr(costs, cost_bucket, getattr(costs, cost_bucket) + cost_for_tokens(model_id, response.usage.input_tokens, response.usage.output_tokens))

    text = _COLLAPSE_FLOATS_RE.sub(r'\1', response.output_text)

    try:
        return strict_to_regular_response(text, question.expected_forecasts)
    except Exception as e:
        if attempt < 3 and "EOF while parsing" in str(e):
            print(f"    truncation retry {attempt + 1}/3 ({len(text)} chars)...")
            return _gpt_research(question, model, prompt, attempt=attempt + 1, test_mode=test_mode, costs=costs, cost_bucket=cost_bucket)
        print(f"    validation error (raw text[:500]): {text[:500]}")
        raise


def research_question(
    question: QuestionSpec,
    model: str = DEFAULT_MODEL,
    *,
    attempt: int = 1,
    test_mode: bool = False,
    costs: RunCost | None = None,
    cost_bucket: str = "research",
) -> tuple[SurveillanceResponse, list[EvidenceItem]]:
    if test_mode:
        model = TEST_CLAUDE_MODEL if provider_for_model(model) == "anthropic" else TEST_MODEL

    run_date = datetime.now(timezone.utc).date().isoformat()
    prompt = f"""You are a research analyst. Search for evidence and produce a structured forecast.

Question: {question.name}
Question type: {question.question_type}
Run date: {run_date}

Context:
{question.prompt}

Expected forecast rows:
{_expected_forecast_lines(question.expected_forecasts)}

Task:
1. Find the latest official value for the metric, including date and source. Do not guess. If no exact official value exists, use the closest quasi-official value that the background or resolution criteria clearly point to, and say so in the rationale.
2. Estimate the current value as of the run date: if the question resolved today, what value would you score the forecast against? This may differ from the latest official value and can combine the latest data point, current reporting, and reasonable extrapolation.
3. For past target dates, report a resolution value only when an authoritative source supports it. If no such value exists, leave it unresolved rather than guessing one.
4. Generate forecast rows for every expected date/dimension/quantile listed above.
5. Provide structured sources with url, title, and snippet.

{RESEARCH_PRINCIPLES}

{_question_type_guidance(question, full=True)}

{_resolution_guidance(question)}

{COLOR_CODE_SYSTEM_FULL}

{RATIONALE_REQUIREMENTS}

Requirements:
- Stay faithful to the question text, resolution criteria, unit, dates, dimensions, and quantiles. Do not reinterpret the metric or change units based on search results.
- Return forecasts for exactly the expected rows listed above. The system assigns value_type from those rows.
- Use -999 only when no defensible estimate is possible.
- Quantile forecasts must be non-decreasing.
- If unit bounds are provided, forecast values must respect them unless the question text explicitly overrides them.
- For each source: include url, title, and a key snippet.
- Do not anchor on existing LEAP forecasts. If LEAP forecasts or analyses appear in web search results, do not review or copy them; form your estimates independently."""

    if provider_for_model(model) == "anthropic":
        return _claude_research(question, model, prompt, attempt=attempt, costs=costs, cost_bucket=cost_bucket)
    return _gpt_research(question, model, prompt, attempt=attempt, test_mode=test_mode, costs=costs, cost_bucket=cost_bucket)


def _format_evidence_for_judge(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return "No sources."
    lines = []
    for i, e in enumerate(evidence, 1):
        lines.append(f"[{i}] URL: {e.url}")
        if e.title:
            lines.append(f"    Title: {e.title}")
        if e.snippet:
            lines.append(f"    Snippet: {e.snippet}")
        if e.full_text:
            lines.append(f"    Extracted ({e.source_type}): {e.full_text[:500]}")
    return "\n".join(lines)


def _format_forecasts_for_judge(response: SurveillanceResponse) -> str:
    if not response.forecasts:
        return "No forecasts."
    return "\n".join(
        f"- {f.forecast_date} {f.dimension} q{f.quantile}: {f.forecast_value} "
        f"({f.color_code.value}, type={getattr(f.value_type, 'value', f.value_type)})"
        for f in response.forecasts
    )


def _format_context_values_for_judge(response: SurveillanceResponse) -> str:
    parts = []
    if response.last_official_values:
        parts.append("Latest official values:")
        for v in response.last_official_values:
            parts.append(f"- {v.dimension}: {v.value} as of {v.date}; source={v.source}")
    if response.current_estimates:
        parts.append("Current estimates:")
        for v in response.current_estimates:
            parts.append(f"- {v.dimension}: {v.value}; confidence={v.confidence}")
    if response.resolution_values:
        parts.append("Resolution values:")
        for v in response.resolution_values:
            parts.append(
                f"- {v.forecast_date} {v.dimension}: {v.value}; "
                f"source_date={v.source_date}; source={v.source}; confidence={v.confidence}"
            )
    return "\n".join(parts) if parts else "No official/current/resolution values."


def _format_expected_for_judge(expected: list[ExpectedForecast]) -> str:
    if not expected:
        return "No expected forecasts specified."
    return "\n".join(
        f"- {e.forecast_date} {e.dimension} q{e.quantile} (type={e.value_type})"
        for e in expected
    )


def _call_structured_judge(
    prompt: str,
    schema_class: type,
    eval_model: str,
    max_output_tokens: int,
    cost_bucket: str,
    costs: RunCost | None,
    attempt: int = 1,
):
    """Call a judge LLM and return the parsed schema. Routes to Anthropic or OpenAI SDK by provider."""
    from pydantic import ValidationError as PydanticValidationError
    schema = make_strict_schema(schema_class)
    model_id = strip_provider_prefix(eval_model)
    if provider_for_model(eval_model) == "anthropic":
        tool_name = "produce_judgment"
        response = anthropic.Anthropic().messages.create(
            model=model_id,
            max_tokens=max_output_tokens,
            tools=[{"name": tool_name, "description": "Output structured judgment.", "input_schema": schema}],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
            timeout=EVALUATION_TIMEOUT,
        )
        tool_block = next(b for b in response.content if getattr(b, "type", None) == "tool_use" and b.name == tool_name)
        text = json.dumps(tool_block.input)
    else:
        response = openai.OpenAI().responses.create(
            model=model_id,
            input=[{"role": "user", "content": prompt}],
            text={"format": {"type": "json_schema", "name": schema_class.__name__, "schema": schema, "strict": True}},
            max_output_tokens=max_output_tokens,
            timeout=EVALUATION_TIMEOUT,
            safety_identifier=OPENAI_SAFETY_IDENTIFIER or None,
        )
        text = response.output_text
    if costs is not None:
        setattr(costs, cost_bucket, getattr(costs, cost_bucket) + cost_for_tokens(model_id, response.usage.input_tokens, response.usage.output_tokens))
    try:
        return schema_class.model_validate_json(text)
    except PydanticValidationError:
        if attempt < 3:
            return _call_structured_judge(prompt, schema_class, eval_model, max_output_tokens, cost_bucket, costs, attempt=attempt + 1)
        raise


def evaluate_adequacy(
    response: SurveillanceResponse,
    question: QuestionSpec,
    evidence: list[EvidenceItem],
    test_mode: bool = False,
    costs: RunCost | None = None,
    eval_model: str | None = None,
    cost_bucket: str = "judge_stage1",
) -> AdequacyAssessment:
    expected_summary = _format_expected_for_judge(question.expected_forecasts)
    sources_summary = _format_evidence_for_judge(evidence)
    forecasts_summary = _format_forecasts_for_judge(response)
    context_values_summary = _format_context_values_for_judge(response)

    prompt = f"""Find concrete problems with this surveillance response.

Do not give a general quality rating. Look for specific review-blocking issues. If you find a problem, add a concise item to issues[] naming the failure mode and the affected value, source, date, or forecast row. If you find no concrete problems, leave issues[] empty.

Question: {question.name}
Question details:
{question.prompt}

Expected forecast rows:
{expected_summary}

Response rationale:
{response.rationale}

Official/current/resolution values:
{context_values_summary}

Generated forecasts:
{forecasts_summary}

Sources:
{sources_summary}

Data unavailability is not a defect: If the rationale explicitly states that the required value is not published by any authoritative source — and this is plausible given the nature of the metric — do not flag it as STALE DATA or EXTRACTION FAILURE. A response that identifies absent data, explains why, and provides a best-estimate distribution is adequate. Only flag if the model could plausibly have found the data but didn't try.

Look for these failure modes:
1. STALE DATA: The cited source value is older than the source's own update cadence. Compare the source's date against how often that source updates, not against the question's resolution date. Examples: monthly data last reported 12+ months ago, quarterly data last reported a year ago, a live leaderboard cited from an old archive snapshot, or an annual figure where the newer year's value has already been released. Do not flag a value as stale simply because it is recent, or because the resolution date is far in the future. Do not flag annual data merely because the latest year has not been published yet.
2. EXTRACTION FAILURE: The rationale says a specific dashboard, page, leaderboard, table, or live source could not be read, rendered, retrieved, or extracted, and no adequate alternative source gives the same metric, period, and scope. Treat varied wording as evidence of this problem, including "JavaScript-rendered", "not directly retrievable", "returned 404", "would not load", "could not retrieve", "could not see", "did not expose", "not visible", or "used an archive snapshot instead". Do not flag if the rationale failed to retrieve Source A but obtained the same metric, period, and scope from Source B — the extraction succeeded via an alternative.
3. SCOPE MISMATCH: A cited source reports a different period, category, benchmark split, population, geography, unit, or subset than the question asks for.
4. UNSUPPORTED CLAIM: A specific numeric claim in the rationale is not supported by the source snippets or listed evidence.
5. STRUCTURAL DEFECT: Expected forecast rows are missing, quantiles within the same (date, dimension) group are non-monotonic, probability values are not on a 0-100 scale, unit bounds are violated, or the response changes the requested dates/dimensions/quantiles. By design, q0 and q100 are feasibility bounds, not credible-interval extremes — q0 at the natural unit floor (e.g., 0 for a percent or count), q100 at the natural unit ceiling (e.g., 100 for a percent), and q100=9999 for timing questions (the "never" sentinel) are all expected and must not be flagged.
6. RESOLUTION DEFECT: A past target-date row is black/resolved without an authoritative as-of-date value, uses a post-target-date value as if it were the target-date value, or fabricates a resolution. If no authoritative as-of-date value exists, a non-black estimate distribution is acceptable — this is not a defect. Example: a target date of 2025-12-31 that has already passed, where no official annual figure has been published yet, should not be flagged merely for being non-black.

Adequacy rule:
- Set adequate=false if and only if issues[] is non-empty. Every problem that makes the response inadequate must appear in issues[] — do not leave issues[] empty and set adequate=false.
- Set adequate=true if and only if issues[] is empty.

Confidence (0-100) is how sure you are about your own review, not about whether the forecast will come true. Start at 90 for a clean response with authoritative sources and lower it for each of:
- The staleness or scope call required judgment (no clear update cadence, ambiguous geography or subset) — lower by 10-15.
- Only some of the rationale's numeric claims are traceable to listed sources — lower by 10-20.
- Sources are sparse (one or two), or you cannot tell whether a cited dashboard was actually read — lower by 10-15.
- The metric is fuzzy or the question's resolution criteria leave room for interpretation — lower by 5-10.
Use the full 0-100 range. Identical numbers across questions suggest you are not actually distinguishing them.

Put only concrete problems in issues[]. Keep each issue to one sentence. Do not include praise or affirmative observations such as "sources are authoritative", "forecasts are complete", or "rationale is well-grounded". Provide a brief reason explaining the issue list and adequacy decision."""

    if eval_model is None:
        eval_model = TEST_EVALUATOR_MODEL if test_mode else DEFAULT_EVALUATOR_MODEL
    try:
        return _call_structured_judge(prompt, AdequacyAssessment, eval_model, 6000, cost_bucket, costs)
    except Exception as e:
        # On evaluator failure, mark inadequate so a human reviewer is alerted.
        return AdequacyAssessment(
            adequate=False,
            confidence=0,
            issues=["Adequacy evaluation failed"],
            reason=f"Evaluation failed: {e}",
        )


def decide_browser(
    response: SurveillanceResponse,
    question: QuestionSpec,
    evidence: list[EvidenceItem],
    adequacy: AdequacyAssessment,
    test_mode: bool = False,
    costs: RunCost | None = None,
    eval_model: str | None = None,
    cost_bucket: str = "judge_stage2",
) -> BrowserDecision:
    sources_summary = _format_evidence_for_judge(evidence)
    issues_list = "\n".join(f"- {i}" for i in adequacy.issues) or "- (no specific issues listed)"

    prompt = f"""The surveillance response was flagged as inadequate. Decide whether browser automation on a specific URL would address the problem.

Question: {question.name}
Question details:
{question.prompt}

Issues from adequacy review:
{issues_list}

Adequacy reviewer reason: {adequacy.reason}

Response rationale:
{response.rationale}

Sources already consulted:
{sources_summary}

Browser automation is useful for:
- JavaScript-heavy dashboards (e.g., METR time horizons, lmarena.ai, livecodebenchpro.com, Kaggle leaderboards)
- Interactive tables or charts that need clicking/scrolling to reveal data
- Pages where web search returns the URL but not the specific value
- Cases where the rationale says a referenced page or dashboard did not expose a needed value. Phrases like "the fetched text does not expose the values", "the page would not load the chart", or "I could not retrieve the table" are extraction problems, not methodology problems.

Browser automation is not useful for:
- Genuine methodological problems (LLM misinterpreted the question, used the wrong metric, applied wrong aggregation)
- Issues that more careful reading of existing sources would fix
- PDF documents (separate path)
- Paywalled content (cannot bypass)
- Search engines (do not propose google.com or similar)

Decide:
- Set browser_would_help=true only if browser scraping a specific URL would plausibly fix the identified issue.
- If yes, propose a specific browser_url and a browser_objective that names the exact metric or value, the exact column or section label, and — if the page has multiple tabs — which tab to navigate to. Bad: "Extract data from the leaderboard." Good: "Extract the Pass@1 score from the Hard difficulty column on the Leaderboard tab — not the overall Pass@1 or count columns."
- If no, set browser_would_help=false and leave browser_url empty. Explain briefly in reason."""

    if eval_model is None:
        eval_model = TEST_EVALUATOR_MODEL if test_mode else DEFAULT_EVALUATOR_MODEL
    try:
        decision = _call_structured_judge(prompt, BrowserDecision, eval_model, 800, cost_bucket, costs)
        if decision.browser_would_help:
            url = decision.browser_url.strip()
            if not url:
                return BrowserDecision(
                    browser_would_help=False,
                    reason=f"Browser recommended but no URL provided (judge said: {decision.reason})",
                )
            safe, reason = is_safe_url(url)
            if not safe:
                return BrowserDecision(
                    browser_would_help=False,
                    browser_url=url,
                    browser_objective=decision.browser_objective,
                    reason=f"Browser URL rejected by safety filter ({reason}); judge had said: {decision.reason}",
                )
        return decision
    except Exception as e:
        return BrowserDecision(
            browser_would_help=False,
            reason=f"Browser decision failed: {e}",
        )


def judge_response(
    response: SurveillanceResponse,
    question: QuestionSpec,
    evidence: list[EvidenceItem],
    test_mode: bool = False,
    costs: RunCost | None = None,
    propose_browser: bool = True,
    eval_model: str | None = None,
    cost_bucket_stage1: str = "judge_stage1",
    cost_bucket_stage2: str = "judge_stage2",
) -> ResearchQualityReport:
    # Pass propose_browser=False on a re-judge to skip the second judge call.
    adequacy = evaluate_adequacy(
        response, question, evidence, test_mode=test_mode, costs=costs,
        eval_model=eval_model, cost_bucket=cost_bucket_stage1,
    )

    if adequacy.adequate and adequacy.confidence < MIN_ADEQUATE_CONFIDENCE:
        adequacy = AdequacyAssessment(
            adequate=False,
            confidence=adequacy.confidence,
            issues=[
                *(adequacy.issues or []),
                f"low_confidence_below_{MIN_ADEQUATE_CONFIDENCE}",
            ],
            reason=(
                f"Evaluator marked adequate but confidence was {adequacy.confidence}%, "
                f"below the {MIN_ADEQUATE_CONFIDENCE}% minimum. Original reason: {adequacy.reason}"
            ),
        )

    if adequacy.adequate or not propose_browser:
        return ResearchQualityReport(
            adequate=adequacy.adequate,
            confidence=adequacy.confidence,
            missing_data=adequacy.issues,
            browser_would_help=False,
            browser_url="",
            browser_objective="",
            reason=adequacy.reason,
        )

    browser_dec = decide_browser(
        response, question, evidence, adequacy, test_mode=test_mode, costs=costs,
        eval_model=eval_model, cost_bucket=cost_bucket_stage2,
    )
    combined_reason = adequacy.reason
    if browser_dec.reason:
        combined_reason = f"{adequacy.reason} | Browser: {browser_dec.reason}"
    return ResearchQualityReport(
        adequate=False,
        confidence=adequacy.confidence,
        missing_data=adequacy.issues,
        browser_would_help=browser_dec.browser_would_help,
        browser_url=browser_dec.browser_url,
        browser_objective=browser_dec.browser_objective,
        reason=combined_reason,
    )


def refine_with_browser(
    question: QuestionSpec,
    original_response: SurveillanceResponse,
    browser_evidence: BrowserEvidence,
    evidence: list[EvidenceItem] | None = None,
    test_mode: bool = False,
    costs: RunCost | None = None,
    model: str | None = None,
    cost_bucket: str = "refinement",
) -> SurveillanceResponse:
    original_sources = [e for e in (evidence or []) if e.source_type != "browser"]
    sources_block = (
        _format_evidence_for_judge(original_sources)
        if original_sources
        else ", ".join(original_response.sources)
    )
    original_values = _format_context_values_for_judge(original_response)
    original_forecasts = _format_forecasts_for_judge(original_response)
    prompt = f"""Update the surveillance response with new browser-extracted data.

Question: {question.name}
Original prompt: {question.prompt}

Original response:
- Rationale: {original_response.rationale}

Original official/current/resolution values:
{original_values}

Original forecasts:
{original_forecasts}

Sources already consulted (with title and snippet):
{sources_block}

New browser data from {browser_evidence.url}:
{browser_evidence.extracted_text[:BROWSER_EVIDENCE_LIMIT]}

Expected forecast rows: {_expected_forecast_lines(question.expected_forecasts)}

{RESEARCH_PRINCIPLES}

{_question_type_guidance(question, full=False)}

{_resolution_guidance(question)}

{COLOR_CODE_SYSTEM_BRIEF}

Instructions:
- Use the browser data when it is more relevant than, or directly contradicts, the original response.
- Preserve original values when the browser data does not address them.
- If the browser data only shows that a dashboard does not expose the needed value, say so in the rationale and keep the original forecast distribution unless it was directly contradicted.
- Return exactly the expected rows. The system assigns value_type from those rows.
- Use -999 only when no defensible estimate is possible.

{RATIONALE_REQUIREMENTS}"""

    if model is None:
        model = TEST_MODEL if test_mode else DEFAULT_MODEL
    schema = _research_schema(question)
    model_id = strip_provider_prefix(model)

    try:
        if provider_for_model(model) == "anthropic":
            produce_tool = {"name": "produce_forecast", "description": "Output the refined surveillance forecast.", "input_schema": schema}
            response = anthropic.Anthropic().messages.create(
                model=model_id,
                max_tokens=64000,
                tools=[produce_tool],
                tool_choice={"type": "tool", "name": "produce_forecast"},
                messages=[{"role": "user", "content": prompt}],
                timeout=RESEARCH_TIMEOUT,
            )
            tool_block = next(b for b in response.content if getattr(b, "type", None) == "tool_use" and b.name == "produce_forecast")
            text = json.dumps(tool_block.input)
            if costs is not None:
                setattr(costs, cost_bucket, getattr(costs, cost_bucket) + cost_for_tokens(model_id, response.usage.input_tokens, response.usage.output_tokens))
        else:
            extra = {} if test_mode else {"reasoning": {"effort": DEFAULT_REASONING_EFFORT}}
            response = openai.OpenAI().responses.create(
                model=model_id,
                input=[{"role": "user", "content": prompt}],
                text={"format": {"type": "json_schema", "name": "StrictSurveillanceResponse", "schema": schema, "strict": True}},
                max_output_tokens=64000,
                timeout=RESEARCH_TIMEOUT,
                safety_identifier=OPENAI_SAFETY_IDENTIFIER or None,
                **extra,
            )
            text = response.output_text
            if costs is not None:
                setattr(costs, cost_bucket, getattr(costs, cost_bucket) + cost_for_tokens(model_id, response.usage.input_tokens, response.usage.output_tokens))
        text = _COLLAPSE_FLOATS_RE.sub(r'\1', text)
        refined_response, _ = strict_to_regular_response(text, question.expected_forecasts)
        return refined_response
    except Exception as e:
        print(f"  Refinement failed, keeping original: {e}")
        return original_response
