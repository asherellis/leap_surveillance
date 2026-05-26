"""LLM research, quality checks, and browser-use."""

import asyncio
import dataclasses
import ipaddress
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
import litellm
from dotenv import load_dotenv

from schemas import (
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


DEFAULT_MODEL = "openai/gpt-5.5"
DEFAULT_EVALUATOR_MODEL = "openai/gpt-4.1-mini"
DEFAULT_REASONING_EFFORT = "high"

TEST_MODEL = "openai/gpt-4o-mini"
TEST_EVALUATOR_MODEL = "openai/gpt-4o-mini"

BROWSER_TIMEOUT = _env_float("LEAP_BROWSER_TIMEOUT", 180.0)
MAX_BROWSER_STEPS = _env_int("LEAP_BROWSER_MAX_STEPS", 15)
BROWSER_EVIDENCE_LIMIT = 4000
RESEARCH_TIMEOUT = _env_float("LEAP_RESEARCH_TIMEOUT", 1800.0)
EVALUATION_TIMEOUT = _env_float("LEAP_EVALUATION_TIMEOUT", 60.0)
MAX_TIMEOUT_RETRIES = _env_int("LEAP_MAX_TIMEOUT_RETRIES", 2)


QUANTILE_INTERPRETATION_FULL = """QUANTILE INTERPRETATION:
Return exactly seven forecast entries for each date/dimension: q0, q5, q25, q50, q75, q95, and q100.

Use q0 and q100 as feasibility bounds, not ordinary probabilistic quantiles. q0 is the lowest coherent value given current constraints; q100 is the highest coherent value. For bounded metrics, use natural bounds where appropriate. For cumulative metrics, q0 should not be below the latest known value.

Use q5/q25/q50/q75/q95 as the probability distribution, with q50 as the median. Values must be non-decreasing: q0 <= q5 <= q25 <= q50 <= q75 <= q95 <= q100. Adjacent quantiles MAY be equal when justified by a feasibility bound or high-confidence point mass (e.g., q0 = q5 = current value for a cumulative metric near its floor).

Do not return all -999 values because the future is uncertain. Use -999 only when no reasonable estimate is possible for a specific value."""


QUANTILE_INTERPRETATION_BRIEF = """QUANTILE INTERPRETATION:
You must return exactly seven forecast entries for each date/dimension combination, one for each quantile: 0, 5, 25, 50, 75, 95, and 100.

All seven quantile forecast values must be valid numbers. Quantile values MUST be in non-decreasing order: q0 <= q5 <= q25 <= q50 <= q75 <= q95 <= q100. Adjacent quantiles may be equal when justified by feasibility bounds or high confidence.

Quantile meanings:
- q=0: Absolute minimum feasible value (physical bounds, or current value if cumulative metric)
- q=5, 25, 50, 75, 95: Probability distribution (q=50 is your median/best estimate)
- q=100: Absolute maximum feasible value (natural upper bound or practical limit)"""


COLOR_CODE_SYSTEM_FULL = """COLOR CODE SYSTEM:
Assign one color_code per date/dimension combination, using the same color for all quantiles in that group.

- black: resolved because the date has passed and official data is available, or because a development definitively resolves it
- dark gray: new evidence changes the feasible range; some previously possible values are now impossible
- light gray: the full natural range is still feasible, but monotonic change makes part of it implausible for the interior quantiles
- white: the full natural range remains feasible and there is not enough information to narrow it

Use black only when uncertainty has collapsed to a resolved value. If color_code is black for a date/dimension group, all quantile values in that group must be identical.

Explain the color choice in the rationale."""


COLOR_CODE_SYSTEM_BRIEF = """COLOR CODE GUIDANCE:
- BLACK: Question resolved (date passed, data available)
- DARK GRAY: Feasible range changed (part of previous range now impossible)
- LIGHT GRAY: Monotonic expectation (full range feasible but part implausible)
- WHITE: Full natural range feasible, no monotonicity assumptions"""


def _response_cost(response, model: str) -> float:
    try:
        if getattr(getattr(response, "usage", None), "cost", None) is not None:
            return float(response.usage.cost)
        return litellm.completion_cost(completion_response=response, model=model) or 0.0
    except Exception:
        return 0.0


async def _verify_url(url: str, client: httpx.AsyncClient) -> tuple[bool, int]:
    try:
        r = await client.head(url, follow_redirects=True, timeout=5.0)
        return r.status_code < 400, r.status_code
    except Exception:
        return False, 0


async def _verify_sources_async(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[_verify_url(e.url, client) for e in evidence],
            return_exceptions=True,
        )
    verified = []
    for e, res in zip(evidence, results):
        ok, status = res if isinstance(res, tuple) else (False, 0)
        verified.append(dataclasses.replace(e, url_verified=ok, http_status=status))
    return verified


def verify_sources(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    if not evidence:
        return evidence
    return asyncio.run(_verify_sources_async(evidence))


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


def expected_forecast_lines(expected_forecasts: list[ExpectedForecast]) -> str:
    return "\n".join(
        [
            f"  - {ef.forecast_date}, {ef.dimension}, q={ef.quantile}, value_type={ef.value_type}"
            for ef in expected_forecasts
        ]
    )


def resolution_guidance(question: QuestionSpec) -> str:
    if not any(f.value_type == "resolution" for f in question.expected_forecasts):
        return "No requested forecast rows are past resolution dates. Return resolution_values as an empty list."

    return """RESOLUTION VALUE GUIDANCE:
Some requested forecast rows have value_type="resolution" because their target dates are in the past.

For each past forecast_date/dimension pair, report one fixed resolution value in resolution_values. This value should be the metric value as of that target date, not the current value today. The resolution_values.source_date field means the date the metric value represents, not the publication date of the source.

Do not substitute the latest/current value for a past resolution date. A source published after the target date is only valid for resolution_values if it explicitly reports the historical value for that target date or target period. If the metric has changed after the target date, the post-target-date value belongs in current_resolution_values, not resolution_values. If you cannot find or reconstruct the target-date value, use -999 for that resolution value rather than using the current value.

In forecasts, still return every requested quantile row. For value_type="resolution" rows, set all quantiles for that forecast_date/dimension to the same fixed resolution value and use color_code="black".

Keep current_resolution_values separate. They are live as-of-today monitoring estimates and should not overwrite fixed past resolution values."""


def question_type_guidance(question: QuestionSpec, full: bool = True) -> str:
    if question.question_type == "probability":
        return """PROBABILITY QUESTION GUIDANCE:
This is a probability question. Return exactly one forecast entry for each expected date/dimension row, with quantile=50.

Forecast values are probabilities on a 0 to 100 scale. Do not return the 0, 5, 25, 75, 95, or 100 quantiles for probability questions.

Latest official values and current resolve-today values are not applicable for this question type. For those value fields, use -999 and set confidence=0.

Use color_code="white" if the event remains unresolved and the full probability range is still open. Use color_code="black" if the event has occurred or the question is already resolved."""

    if question.question_type == "when":
        return """TIMING QUESTION GUIDANCE:
This is a timing question asking when an event will first occur. The forecast_date value is a placeholder label; the forecast_value itself must be a year.

Return exactly seven forecast entries for each expected row: quantiles 0, 5, 25, 50, 75, 95, and 100. Forecast values must be years only, with no ranges, dates, or extra text.

All seven timing quantiles must be present, non-null, and non-decreasing. If the event has not yet occurred, q=0 should usually be the current year as the earliest feasible occurrence year. If the event may never occur, represent that tail risk with distant future years rather than -999; use 2500 only as an extreme upper-tail year when needed.

Latest official values and current resolve-today values are not applicable for this question type. For those value fields, use -999 and set confidence=0.

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
    response, evidence = _do_research(
        question, model, retry_on_truncation, attempt=1, test_mode=test_mode, costs=costs
    )
    evidence = verify_sources(evidence)
    return response, evidence


def _do_research(
    question: QuestionSpec,
    model: str,
    retry_on_truncation: bool,
    attempt: int,
    test_mode: bool = False,
    costs: RunCost | None = None,
) -> tuple[SurveillanceResponse, list[EvidenceItem]]:
    run_date = datetime.now(timezone.utc).date().isoformat()
    prompt = f"""You are a research analyst. Use web search to find current data, then produce a structured forecast.

Question: {question.name}
Question type: {question.question_type}
Run date: {run_date}

Context:
{question.prompt}

Expected forecast rows:
{expected_forecast_lines(question.expected_forecasts)}

TASK OVERVIEW:
1. Find the latest OFFICIAL VALUE for this metric (most recent published data with date and source)
2. Estimate the CURRENT RESOLUTION VALUE (what you would use if resolving today, may differ from official if metric has changed)
3. For past target dates, estimate fixed RESOLUTION VALUES as of those target dates
4. Generate FORECAST rows for every expected date/quantile listed above
5. Provide structured sources with url, title, and snippet

{question_type_guidance(question, full=True)}

{resolution_guidance(question)}

{COLOR_CODE_SYSTEM_FULL}

REQUIREMENTS:
- The question text, resolution criteria, unit, dates, dimensions, and quantiles above are the source of truth. Do not reinterpret the metric or change units based on search results.
- Find the latest OFFICIAL published value when applicable (with date and source)
- Estimate CURRENT RESOLUTION value when applicable (if resolving today, based on latest data plus reasonable extrapolation)
- Return forecasts for exactly the expected rows listed above. The system assigns value_type from those rows.
- Return one resolution_values entry for each past forecast_date/dimension pair, and no resolution_values entries for future dates
- Do not use a post-target-date current value as a resolution value unless the source specifically reports the target-date value
- Use -999 ONLY for individual values you truly cannot estimate (this should be rare)
- Ensure quantile forecasts form an increasing sequence when this is a quantile or timing question
- For each source: include url, title, and a key snippet/excerpt from that source
- Do NOT anchor on existing LEAP forecasts if you encounter them - form independent estimates based on your research"""

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
            # gpt-4o-mini does not support reasoning_effort
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


def _format_expected_for_judge(expected: list[ExpectedForecast]) -> str:
    if not expected:
        return "No expected forecasts specified."
    return "\n".join(
        f"- {e.forecast_date} {e.dimension} q{e.quantile} (type={e.value_type})"
        for e in expected
    )


def evaluate_adequacy(
    response: SurveillanceResponse,
    question: QuestionSpec,
    evidence: list[EvidenceItem],
    test_mode: bool = False,
    costs: RunCost | None = None,
) -> AdequacyAssessment:
    """Stage 1 judge call: assess whether the surveillance response is adequate."""
    expected_summary = _format_expected_for_judge(question.expected_forecasts)
    sources_summary = _format_evidence_for_judge(evidence)
    forecasts_summary = _format_forecasts_for_judge(response)

    prompt = f"""Evaluate whether this surveillance response adequately answers the forecasting question.

Question: {question.name}
Question details:
{question.prompt}

Expected forecast rows (what the response was asked to produce):
{expected_summary}

Response rationale (full):
{response.rationale}

Forecasts generated:
{forecasts_summary}

Sources consulted (with title and snippet):
{sources_summary}

Evaluation criteria:
- Are the sources authoritative and recent for this question?
- Do the source snippets actually support the specific claims in the rationale?
- Is the rationale well-grounded in the cited evidence (not just plausible-sounding)?
- Are all expected forecast rows present (no missing date/dimension/quantile combinations)?
- Are the forecasts internally consistent (quantiles non-decreasing per date/dimension, values coherent across time horizons)?
- Are there material data gaps where the LLM clearly lacked information?

Set adequate=true ONLY if the response is broadly trustworthy on all of the above. Set adequate=false if any criterion fails. List specific problems in issues[] (each item should be one concrete issue). Provide a brief reason explaining the overall judgment."""

    schema = make_strict_schema(AdequacyAssessment)
    eval_model = TEST_EVALUATOR_MODEL if test_mode else DEFAULT_EVALUATOR_MODEL

    try:
        result = litellm.responses(
            model=eval_model,
            input=[{"role": "user", "content": prompt}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "AdequacyAssessment",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=1500,
            timeout=EVALUATION_TIMEOUT,
            safety_identifier=OPENAI_SAFETY_IDENTIFIER or None,
        )
        if costs is not None:
            costs.judge_stage1 += _response_cost(result, eval_model)
        text = _extract_text_from_response(result)
        return AdequacyAssessment.model_validate_json(text)
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
    """Stage 2 judge call: given an inadequate response, decide whether browser-use would help."""
    sources_summary = _format_evidence_for_judge(evidence)
    issues_list = "\n".join(f"- {i}" for i in adequacy.issues) or "- (no specific issues listed)"

    prompt = f"""The surveillance response was flagged as inadequate. Decide whether using browser automation to scrape a specific URL would meaningfully help fix the gap.

Question: {question.name}
Question details:
{question.prompt}

Issues identified by adequacy review:
{issues_list}

Adequacy reviewer reason: {adequacy.reason}

Sources already consulted (via web search):
{sources_summary}

Browser automation is useful for:
- JavaScript-heavy dashboards (e.g., METR time horizons, lmarena.ai, livecodebenchpro.com, Kaggle leaderboards)
- Interactive tables or charts that need clicking/scrolling to reveal data
- Pages where web search returns the URL but not the specific value

Browser automation is NOT useful for:
- Methodological problems (LLM misinterpreted the question)
- Issues that more careful reading of existing sources would fix
- PDF documents (separate path)
- Paywalled content (cannot bypass)
- Search engines (do not propose google.com or similar)

Decide:
- Set browser_would_help=true only if browser scraping a specific URL would plausibly fix the identified issues.
- If yes, propose a specific browser_url (must NOT be a search engine, must NOT be a private/local IP) and a concrete browser_objective ("Extract the X value from the Y table").
- If no, set browser_would_help=false and leave browser_url empty. Explain briefly in reason.

Note: Do not propose search engines (google.com, bing.com, etc.), localhost, or private/internal IPs — these are blocked by the pipeline's URL safety filter and will silently fail."""

    schema = make_strict_schema(BrowserDecision)
    eval_model = TEST_EVALUATOR_MODEL if test_mode else DEFAULT_EVALUATOR_MODEL

    try:
        result = litellm.responses(
            model=eval_model,
            input=[{"role": "user", "content": prompt}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "BrowserDecision",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=800,
            timeout=EVALUATION_TIMEOUT,
            safety_identifier=OPENAI_SAFETY_IDENTIFIER or None,
        )
        if costs is not None:
            costs.judge_stage2 += _response_cost(result, eval_model)
        text = _extract_text_from_response(result)
        return BrowserDecision.model_validate_json(text)
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
    test_mode: bool = False,
    costs: RunCost | None = None,
) -> SurveillanceResponse:
    prompt = f"""Update the surveillance response with new browser-extracted data.

Question: {question.name}
Original prompt: {question.prompt}

Original response:
- Rationale: {original_response.rationale}
- Sources: {', '.join(original_response.sources)}

New browser data from {browser_evidence.url}:
{browser_evidence.extracted_text[:BROWSER_EVIDENCE_LIMIT]}

Expected forecast rows: {expected_forecast_lines(question.expected_forecasts)}

{question_type_guidance(question, full=False)}

{resolution_guidance(question)}

{COLOR_CODE_SYSTEM_BRIEF}

INSTRUCTIONS:
Integrate the new browser data to improve the forecasts and resolution values. Update values if the browser data provides better/more relevant information. Return exactly the expected rows. The system assigns value_type from those rows. Use -999 only for individual values you truly cannot estimate (should be rare)."""

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
        if host.endswith("google.com") or host.endswith("googleusercontent.com"):
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


_BROWSER_LLM_CLASS = None


def _browser_llm_class():
    global _BROWSER_LLM_CLASS
    if _BROWSER_LLM_CLASS is not None:
        return _BROWSER_LLM_CLASS
    from langchain_litellm import ChatLiteLLM
    from pydantic import ConfigDict

    # browser-use reads llm.provider / llm.model_name and monkey-patches ainvoke;
    # ChatLiteLLM(extra='ignore') blocks setattr, so we open the door with extra='allow'.
    class _ChatLiteLLMForBrowser(ChatLiteLLM):
        model_config = ConfigDict(extra="allow", protected_namespaces=(), arbitrary_types_allowed=True)
        provider: str = "litellm"

    _BROWSER_LLM_CLASS = _ChatLiteLLMForBrowser
    return _BROWSER_LLM_CLASS


def _get_browser_llm(model: str):
    """Return the LiteLLM adapter object browser-use expects."""
    cls = _browser_llm_class()
    model_kwargs = {}
    if OPENAI_SAFETY_IDENTIFIER:
        model_kwargs["safety_identifier"] = OPENAI_SAFETY_IDENTIFIER
    return cls(model=model, model_name=model, model_kwargs=model_kwargs)


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

        model = model_override or (TEST_MODEL if test_mode else DEFAULT_MODEL)
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
        result = asyncio.run(_extract())
        # Prefer final result; full histories include transient errors.
        extracted = getattr(result, "final_result", lambda: None)() or ""

        # Do not treat browser/tooling failures as usable evidence.
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
