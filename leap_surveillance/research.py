"""LLM research, quality checks, and browser-use."""

import asyncio
import ipaddress
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import litellm
from dotenv import load_dotenv

from .common import (
    DEFAULT_BROWSER_MODEL,
    DEFAULT_EVALUATOR_MODEL,
    DEFAULT_MODEL,
    TEST_EVALUATOR_MODEL,
    TEST_MODEL,
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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return int(raw)


DEFAULT_REASONING_EFFORT = "high"

import threading

BROWSER_TIMEOUT = _env_float("LEAP_BROWSER_TIMEOUT", 180.0)
MAX_BROWSER_STEPS = _env_int("LEAP_BROWSER_MAX_STEPS", 15)
BROWSER_EVIDENCE_LIMIT = 4000
RESEARCH_TIMEOUT = _env_float("LEAP_RESEARCH_TIMEOUT", 300.0)
EVALUATION_TIMEOUT = _env_float("LEAP_EVALUATION_TIMEOUT", 60.0)
MAX_TIMEOUT_RETRIES = _env_int("LEAP_MAX_TIMEOUT_RETRIES", 1)
MIN_ADEQUATE_CONFIDENCE = _env_int("LEAP_MIN_ADEQUATE_CONFIDENCE", 50)

# Limit browser-use Agent invocations to one at a time across worker threads.
# Each Agent spawns a Chromium process; running >1 concurrently risks resource contention.
_BROWSER_SEMAPHORE = threading.Semaphore(1)


QUANTILE_INTERPRETATION_FULL = """Quantile interpretation:
Return exactly seven forecast entries for each date/dimension: q0, q5, q25, q50, q75, q95, and q100.

Use q0 and q100 as feasibility bounds, not ordinary probabilistic quantiles. q0 is the lowest coherent value given current constraints; q100 is the highest coherent value. For bounded metrics, use natural bounds where appropriate. For cumulative metrics, q0 should not be below the latest known value.

Use q5/q25/q50/q75/q95 as the probability distribution, with q50 as the median. Values must be non-decreasing: q0 <= q5 <= q25 <= q50 <= q75 <= q95 <= q100. Adjacent quantiles may be equal when justified by a feasibility bound or high-confidence point mass (e.g., q0 = q5 = current value for a cumulative metric near its floor).

Do not return all -999 values because the future is uncertain. Use -999 only when no reasonable estimate is possible for a specific value."""


QUANTILE_INTERPRETATION_BRIEF = """Quantile interpretation:
You must return exactly seven forecast entries for each date/dimension combination, one for each quantile: 0, 5, 25, 50, 75, 95, and 100.

All seven quantile forecast values must be valid numbers and non-decreasing: q0 <= q5 <= q25 <= q50 <= q75 <= q95 <= q100. Adjacent quantiles may be equal when justified by feasibility bounds or high confidence.

Quantile meanings:
- q=0: Absolute minimum feasible value (physical bound, or current value for a cumulative metric)
- q=5, 25, 50, 75, 95: Probability distribution (q=50 is your median/best estimate)
- q=100: Absolute maximum feasible value (natural upper bound or practical limit)"""


COLOR_CODE_SYSTEM_FULL = """Color coding:
Assign one color_code per date/dimension combination, using the same color for all quantiles in that group.

- black: resolved - an authoritative value for this date/dimension is available, or a development definitively resolves it. A past date alone does not make a group black; if the date has passed but no authoritative value exists yet (never measured, not published, or only later sources), use the non-black color that reflects your remaining uncertainty.
- dark gray: new evidence changes the feasible range; some previously possible values are now impossible
- light gray: the full natural range is still feasible, but monotonic change makes part of it implausible for the interior quantiles
- white: the full natural range remains feasible and there is not enough information to narrow it

Color reflects how much uncertainty has collapsed, not whether the date is past or future. Use black only when the row has a resolved value. If color_code is black for a date/dimension group, all quantile values in that group must be identical.

Explain the color choice in the rationale."""


COLOR_CODE_SYSTEM_BRIEF = """Color coding:
- black: resolved - authoritative value available (a past date alone is not enough)
- dark gray: feasible range changed; part of the previous range is now impossible
- light gray: full range remains feasible, but part is implausible because of monotonic expectations
- white: full natural range remains feasible; no monotonicity assumptions"""


RESEARCH_PRINCIPLES = """Research principles:
- Match the period. When a value is tied to a specific date or period, find the value as of that period and prefer sources from that period; include the period in your searches. Do not substitute a value from a different time than the one asked about.
- Use central estimates. Report a point estimate or stated central value, not an interval bound or range endpoint. If a source gives a range or confidence interval, use its midpoint or stated central value, and say which you used in your rationale.
- Respect scope. If the metric is limited to a particular category, track, subset, or population, confirm each source matches that scope, and note in your rationale what you included and excluded.
- Build aggregates exactly as the question defines them. If the metric is an average, sum, or index over components, gather each component and compute it yourself rather than copying a headline figure; list the components used. Combine them using the method the question's resolution criteria specifies; impose no default weighting of your own. When the component structure is ambiguous (e.g., a benchmark reports multiple sub-scores or tiers and the criteria doesn't say how they roll up), state the assumption you made and flag it explicitly in your rationale.
- Don't manufacture data. Ground each value in a source that actually reports it. When you cannot, prefer -999 (or appropriately wide uncertainty for forecast rows) over inferring a number from out-of-scope or out-of-period material. Never present an extrapolation as if it were observed data."""


RATIONALE_REQUIREMENTS = """Rationale requirements:
- State the source basis for the latest official value, current estimate, and each resolved target-date value.
- For each target date/dimension, state the status decision: resolved, unresolved past date, future forecast, or resolved early.
- Explain the color_code choice for each target date/dimension.
- State any unit conversion, aggregation formula, or scope assumption used.
- For forecast rows, explain the q50 and the main drivers of the q25/q75 spread. Do not explain every quantile separately unless q0/q100 need special justification."""


def _response_cost(response, model: str) -> float:
    try:
        if getattr(getattr(response, "usage", None), "cost", None) is not None:
            return float(response.usage.cost)
        return litellm.completion_cost(completion_response=response, model=model) or 0.0
    except Exception:
        return 0.0


def _extract_text_from_response(response) -> str:
    if getattr(response, "output_text", None):
        return response.output_text
    for item in response.output:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type == "message":
            content_list = item.get("content") if isinstance(item, dict) else item.content
            for content in content_list:
                content_type = content.get("type") if isinstance(content, dict) else getattr(content, "type", None)
                if content_type == "output_text":
                    return content.get("text") if isinstance(content, dict) else content.text
    raise RuntimeError("No text output found in response")


def _expected_forecast_lines(expected_forecasts: list[ExpectedForecast]) -> str:
    return "\n".join(
        f"  - {ef.forecast_date}, {ef.dimension}, q={ef.quantile}, value_type={ef.value_type}"
        for ef in expected_forecasts
    )


def _resolution_guidance(question: QuestionSpec) -> str:
    if not any(f.value_type == "resolution" for f in question.expected_forecasts):
        return "No requested forecast rows are past resolution dates. Return resolution_values as an empty list."

    return """Resolution value guidance:
Some requested rows have value_type="resolution": their target date has passed, so try to find the metric's authoritative value as of that exact target date. A past date does not guarantee that a resolved value exists.

- If you find an authoritative value for the target date: report it in resolution_values (with source and source_date), set all quantiles for that group to that single value, and use color_code="black".
- If no authoritative value exists for the target date (the metric was never measured for it, the data is not published yet, or only post-target-date sources exist): leave that group out of resolution_values, give your best-estimate distribution across the quantiles, and use the non-black color that reflects your remaining uncertainty. Put your single best guess in current_estimates so the estimate is not lost.
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

All seven timing quantiles must be present, non-null, and non-decreasing. If the event has not yet occurred, q=0 should usually be the current year as the earliest feasible occurrence year. If the event may never occur, represent that tail risk with distant future years rather than -999; use 2500 only as an extreme upper-tail year when needed.

Latest official values and current estimates are not applicable for this question type. For those value fields, use -999 and set confidence=0.

Use color_code="black" if the event has already occurred. If it has not occurred, use color_code="dark gray" because first occurrence in past years is no longer feasible."""

    return QUANTILE_INTERPRETATION_FULL if full else QUANTILE_INTERPRETATION_BRIEF


def _research_schema(question: QuestionSpec) -> dict:
    return make_strict_schema(
        StrictSurveillanceResponse,
        allowed_dimensions=sorted({f.dimension for f in question.expected_forecasts}),
        allowed_quantiles=sorted({f.quantile for f in question.expected_forecasts if f.quantile is not None}),
        allowed_forecast_dates=sorted({f.forecast_date for f in question.expected_forecasts}),
    )


def research(
    question: QuestionSpec,
    model: str = DEFAULT_MODEL,
    retry_on_truncation: bool = True,
    test_mode: bool = False,
    costs: RunCost | None = None,
) -> tuple[SurveillanceResponse, list[EvidenceItem]]:
    if test_mode:
        model = TEST_MODEL
    return _do_research(
        question, model, retry_on_truncation, attempt=1, test_mode=test_mode, costs=costs
    )


def _do_research(
    question: QuestionSpec,
    model: str,
    retry_on_truncation: bool,
    attempt: int,
    test_mode: bool = False,
    costs: RunCost | None = None,
) -> tuple[SurveillanceResponse, list[EvidenceItem]]:
    run_date = datetime.now(timezone.utc).date().isoformat()
    prompt = f"""You are a research analyst. Use web search to find the relevant evidence, then produce a structured forecast.

Question: {question.name}
Question type: {question.question_type}
Run date: {run_date}

Context:
{question.prompt}

Expected forecast rows:
{_expected_forecast_lines(question.expected_forecasts)}

Task:
1. Find the latest official value for the metric, including date and source.
2. Estimate the current value as of the run date. This may differ from the latest official value if the metric has changed.
3. For past target dates, report a resolution value only when an authoritative source supports it. If no such value exists, leave it unresolved rather than guessing one.
4. Generate forecast rows for every expected date/quantile listed above.
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
- Do not anchor on existing LEAP forecasts; form independent estimates."""

    schema = _research_schema(question)

    timeout_attempt = 0
    while True:
        try:
            params = {
                "model": model,
                "input": [{"role": "user", "content": prompt}],
                "tools": [{"type": "web_search"}],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "StrictSurveillanceResponse",
                        "schema": schema,
                        "strict": True,
                    }
                },
                "max_output_tokens": 30000,
                "timeout": RESEARCH_TIMEOUT,
                "safety_identifier": OPENAI_SAFETY_IDENTIFIER or None,
            }
            # Some lower-cost test models do not support reasoning_effort.
            if not test_mode:
                params["reasoning_effort"] = DEFAULT_REASONING_EFFORT
            response = litellm.responses(**params)
            break
        except Exception as e:
            error_str = str(e).lower()
            is_timeout = "timeout" in error_str or "timed out" in error_str
            if is_timeout and timeout_attempt < MAX_TIMEOUT_RETRIES:
                timeout_attempt += 1
                print(f"    timeout retry {timeout_attempt}/{MAX_TIMEOUT_RETRIES}...")
                continue
            raise

    if costs is not None:
        costs.research += _response_cost(response, model)

    text = _extract_text_from_response(response)

    # Collapse degenerate LLM floats (0.000000...0) that bloat JSON past token limit.
    text = re.sub(r'(\d+\.\d{4})\d{4,}', r'\1', text)

    try:
        strict_response = StrictSurveillanceResponse.model_validate_json(text)
    except Exception as e:
        if retry_on_truncation and attempt < 3 and "EOF while parsing" in str(e):
            print(f"    truncation retry {attempt + 1}/3 ({len(text)} chars)...")
            return _do_research(
                question,
                model,
                retry_on_truncation,
                attempt + 1,
                test_mode=test_mode,
                costs=costs,
            )
        print(f"    validation error (raw text[:500]): {text[:500]}")
        raise

    return strict_to_regular_response(strict_response, question.expected_forecasts)


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
):
    """Call a judge LLM with a strict-JSON schema and return the parsed model."""
    schema = make_strict_schema(schema_class)
    result = litellm.responses(
        model=eval_model,
        input=[{"role": "user", "content": prompt}],
        text={"format": {"type": "json_schema", "name": schema_class.__name__, "schema": schema, "strict": True}},
        max_output_tokens=max_output_tokens,
        timeout=EVALUATION_TIMEOUT,
        safety_identifier=OPENAI_SAFETY_IDENTIFIER or None,
    )
    if costs is not None:
        setattr(costs, cost_bucket, getattr(costs, cost_bucket) + _response_cost(result, eval_model))
    text = _extract_text_from_response(result)
    return schema_class.model_validate_json(text)


def evaluate_adequacy(
    response: SurveillanceResponse,
    question: QuestionSpec,
    evidence: list[EvidenceItem],
    test_mode: bool = False,
    costs: RunCost | None = None,
) -> AdequacyAssessment:
    expected_summary = _format_expected_for_judge(question.expected_forecasts)
    sources_summary = _format_evidence_for_judge(evidence)
    forecasts_summary = _format_forecasts_for_judge(response)
    context_values_summary = _format_context_values_for_judge(response)

    prompt = f"""Evaluate whether this surveillance response is adequate for review.

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

Resolution rows (value_type=resolution) have two valid outcomes:
- Resolved: an authoritative value for the target date was found. The group should be color_code=black with all quantiles equal to that value. Do not flag identical black quantiles as a problem.
- Unresolved: no authoritative as-of-date value exists because the metric was never measured, is not yet published, or only post-target-date sources exist. A non-black estimate distribution with no resolution value is acceptable.

Flag a resolution row when it is black but unsupported, fabricated, based on a post-target-date source, or otherwise appears wrong.

Evaluation criteria:
- Are the sources authoritative and appropriate for this question?
- Do the cited sources report the value for the period and scope the question asks about? A relevant source is not enough if it reports a different time period, category, or subset.
- Do the source snippets actually support the specific claims in the rationale?
- Are the unit, scale, and bounds correct? Probabilities must be on a 0-100 scale.
- Is the rationale well-grounded in the cited evidence (not just plausible-sounding)?
- If the rationale says a specific dashboard or page did not expose needed values, did the response use an adequate alternative source for the same metric, period, and scope? If not, treat it as a data gap.
- Are all expected forecast rows present (no missing date/dimension/quantile combinations)?
- For forecast rows: are quantiles non-decreasing and internally consistent across time horizons?
- Are there material data gaps?

Confidence means confidence in the response's evidence, interpretation, row completeness, and methodology. It is not confidence that a future forecast will come true. A well-supported long-range forecast can have high methodology confidence even though the future outcome is uncertain.

Set adequate=true only if the response meets the criteria above. Set adequate=false if any criterion fails. Base confidence on whether the evidence and rationale substantiate the method and values well enough for review, not on the mere presence of sources or inherent future uncertainty. If methodology/evidence confidence is below {MIN_ADEQUATE_CONFIDENCE}, set adequate=false even when the answer is structurally complete. List concrete problems in issues[] and provide a brief reason for the overall judgment."""

    eval_model = TEST_EVALUATOR_MODEL if test_mode else DEFAULT_EVALUATOR_MODEL
    try:
        return _call_structured_judge(prompt, AdequacyAssessment, eval_model, 1500, "judge_stage1", costs)
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
- If yes, propose a specific browser_url (not a search engine, not a private/local IP) and a concrete browser_objective ("Extract the X value from the Y table").
- If no, set browser_would_help=false and leave browser_url empty. Explain briefly in reason.

Do not propose search engines (google.com, bing.com, etc.), localhost, or private/internal IPs. These are blocked by the pipeline's URL safety filter."""

    eval_model = TEST_EVALUATOR_MODEL if test_mode else DEFAULT_EVALUATOR_MODEL
    try:
        decision = _call_structured_judge(prompt, BrowserDecision, eval_model, 800, "judge_stage2", costs)
        if decision.browser_would_help:
            url = decision.browser_url.strip()
            if not url:
                return BrowserDecision(
                    browser_would_help=False,
                    reason=f"Browser was recommended but no URL was provided. Original reason: {decision.reason}",
                )
            safe, reason = is_safe_url(url)
            if not safe:
                return BrowserDecision(
                    browser_would_help=False,
                    browser_url=url,
                    browser_objective=decision.browser_objective,
                    reason=f"Browser URL rejected by safety filter ({reason}). Original reason: {decision.reason}",
                )
        return decision
    except Exception as e:
        return BrowserDecision(
            browser_would_help=False,
            reason=f"Browser decision failed: {e}",
        )


def evaluate_response_quality(
    response: SurveillanceResponse,
    question: QuestionSpec,
    evidence: list[EvidenceItem],
    test_mode: bool = False,
    costs: RunCost | None = None,
) -> ResearchQualityReport:
    adequacy = evaluate_adequacy(response, question, evidence, test_mode=test_mode, costs=costs)

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

    if adequacy.adequate:
        return ResearchQualityReport(
            adequate=True,
            confidence=adequacy.confidence,
            missing_data=adequacy.issues,
            browser_would_help=False,
            browser_url="",
            browser_objective="",
            reason=adequacy.reason,
        )

    browser_dec = decide_browser(
        response, question, evidence, adequacy, test_mode=test_mode, costs=costs
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
Integrate the new browser data to improve the forecasts and resolution values. Preserve original values unless the browser data is more relevant or contradicts them. If the browser data only shows that a dashboard does not expose the needed value, say that in the rationale and keep the original forecast distribution unless the original was directly contradicted. Return exactly the expected rows. The system assigns value_type from those rows. Use -999 only when no defensible estimate is possible.

{RATIONALE_REQUIREMENTS}"""

    schema = _research_schema(question)

    try:
        model = TEST_MODEL if test_mode else DEFAULT_MODEL
        params = {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "StrictSurveillanceResponse",
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_output_tokens": 30000,
            "timeout": RESEARCH_TIMEOUT,
            "safety_identifier": OPENAI_SAFETY_IDENTIFIER or None,
        }
        if not test_mode:
            params["reasoning_effort"] = DEFAULT_REASONING_EFFORT
        response = litellm.responses(**params)
        if costs is not None:
            costs.refinement += _response_cost(response, model)
        text = _extract_text_from_response(response)
        text = re.sub(r'(\d+\.\d{4})\d{4,}', r'\1', text)
        strict_resp = StrictSurveillanceResponse.model_validate_json(text)
        refined_response, _ = strict_to_regular_response(strict_resp, question.expected_forecasts)
        return refined_response
    except Exception as e:
        print(f"  Refinement failed, keeping original: {e}")
        return original_response


def is_safe_url(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False, f"Invalid scheme: {parsed.scheme}"

        host = parsed.hostname or ""

        # Avoid letting the agent wander into search engines / CAPTCHA loops.
        search_domains = (
            "google.com",
            "googleusercontent.com",
            "bing.com",
            "duckduckgo.com",
            "yahoo.com",
            "baidu.com",
            "yandex.com",
            "search.brave.com",
        )
        if any(host == d or host.endswith(f".{d}") for d in search_domains):
            return False, "Search engine domain blocked"

        if host in ("localhost", "127.0.0.1", "::1"):
            return False, "Localhost blocked"

        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local:
                return False, f"Private/reserved IP blocked: {host}"
        except ValueError:
            pass

        if host == "metadata.google.internal":
            return False, "Metadata endpoint blocked"

        return True, ""
    except Exception as e:
        return False, f"URL parse error: {e}"


def _get_browser_llm(model: str):
    from browser_use.llm.openai.chat import ChatOpenAI as BrowserChatOpenAI
    bare_model = model.split("/", 1)[-1] if "/" in model else model
    return BrowserChatOpenAI(model=bare_model, api_key=os.environ.get("OPENAI_API_KEY"))


def browser_extract(
    url: str, objective: str, test_mode: bool = False, model_override: str | None = None
) -> BrowserEvidence:
    # PDF viewers make browser-use extraction unreliable.
    if url.lower().endswith(".pdf"):
        return BrowserEvidence(
            url=url,
            objective=objective,
            extracted_text="",
            success=False,
            error="PDF URL not supported by browser_extract (use web_search evidence instead)",
        )

    safe, reason = is_safe_url(url)
    if not safe:
        return BrowserEvidence(
            url=url,
            objective=objective,
            extracted_text="",
            success=False,
            error=f"Unsafe URL: {reason}",
        )

    async def _extract():
        from browser_use import Agent, Browser

        model = model_override or (TEST_MODEL if test_mode else DEFAULT_BROWSER_MODEL)
        llm = _get_browser_llm(model)
        browser = Browser(headless=True)
        try:
            agent = Agent(
                task=f"Go to {url} and {objective}. Return only the extracted data.",
                llm=llm,
                browser=browser,
            )
            return await asyncio.wait_for(
                agent.run(max_steps=MAX_BROWSER_STEPS), timeout=BROWSER_TIMEOUT
            )
        finally:
            await browser.stop()

    try:
        with _BROWSER_SEMAPHORE:
            result = asyncio.run(_extract())
        # Prefer final result; full histories include transient errors.
        extracted = getattr(result, "final_result", lambda: None)() or ""

        failure_markers = [
            "Invalid schema for response_format",
            "Stopping due to 3 consecutive failures",
            "LLM API call failed",
            '"error":',
            "was not successful",
            "Unfinished",
            "CAPTCHA",
            "recaptcha",
            "ERR_CERT",
        ]
        if not extracted or any(m in extracted for m in failure_markers):
            return BrowserEvidence(
                url=url,
                objective=objective,
                extracted_text="",
                success=False,
                error="browser-use run failed (see logs)",
            )

        return BrowserEvidence(url=url, objective=objective, extracted_text=extracted, success=True)
    except ImportError:
        return BrowserEvidence(
            url=url,
            objective=objective,
            extracted_text="",
            success=False,
            error="browser-use not installed",
        )
    except Exception as e:
        return BrowserEvidence(
            url=url, objective=objective, extracted_text="", success=False, error=str(e)
        )
