from datetime import date
import importlib

import pytest

from leap_surveillance import common
from leap_surveillance.common import (
    DEFAULT_OUTPUT_DIR,
    TIMING_FORECAST_DATE,
    make_result_id,
    make_review_group_id,
    resolution_status,
)
from leap_surveillance import research, run_surveillance, sheets, storage
from leap_surveillance.models import (
    AdequacyAssessment,
    BrowserDecision,
    ColorCode,
    CurrentValue,
    ExpectedForecast,
    ForecastValue,
    OfficialValue,
    QuestionSpec,
    ResolutionValue,
    ResearchQualityReport,
    RunCost,
    SurveillanceResponse,
    ValueType,
)
from leap_surveillance.storage import build_run_data, write_csv_output
from leap_surveillance.models import validate_response


def _question(
    *,
    qid: str = "q1",
    qtype: str = "quantile",
    expected: list[ExpectedForecast] | None = None,
    unit: str = "Percent (%)",
) -> QuestionSpec:
    return QuestionSpec(
        id=qid,
        name="Question",
        prompt="Prompt",
        expected_forecasts=expected or [ExpectedForecast("2030-12-31", "Overall", 50)],
        question_type=qtype,
        unit=unit,
        unit_min=0,
        unit_max=100,
        question_text="Question text",
        resolution_criteria="Criteria",
    )


def _response(
    forecasts: list[ForecastValue],
    *,
    resolution_values: list[ResolutionValue] | None = None,
    sources: list[str] | None = None,
) -> SurveillanceResponse:
    return SurveillanceResponse(
        last_official_values=[
            OfficialValue(dimension="Overall", value=1.0, date="2025-01-01", source="https://example.com/off")
        ],
        current_estimates=[
            CurrentValue(dimension="Overall", value=2.0, confidence=70)
        ],
        resolution_values=resolution_values or [],
        forecasts=forecasts,
        rationale="Rationale",
        sources=sources if sources is not None else ["https://example.com/source"],
    )


def _forecast(
    forecast_date: str = "2030-12-31",
    dimension: str = "Overall",
    quantile: int = 50,
    value: float = 10.0,
    color: ColorCode = ColorCode.white,
) -> ForecastValue:
    return ForecastValue(
        value_type=ValueType.forecast,
        forecast_date=forecast_date,
        dimension=dimension,
        quantile=quantile,
        forecast_value=value,
        color_code=color,
    )


def test_validate_response_accepts_core_question_shapes():
    quantile_expected = [ExpectedForecast("2030-12-31", "Overall", q) for q in [0, 5, 25, 50, 75, 95, 100]]
    quantile_resp = _response([
        _forecast(quantile=q, value=float(q), color=ColorCode.white)
        for q in [0, 5, 25, 50, 75, 95, 100]
    ])
    assert validate_response(quantile_resp, quantile_expected)["ok"]

    probability_expected = [ExpectedForecast("2030-12-31", "Overall", 50)]
    probability_resp = _response([_forecast(quantile=50, value=35.0, color=ColorCode.white)])
    assert validate_response(probability_resp, probability_expected)["ok"]

    when_expected = [ExpectedForecast(TIMING_FORECAST_DATE, "Overall", q) for q in [0, 5, 25, 50, 75, 95, 100]]
    when_resp = _response([
        _forecast(TIMING_FORECAST_DATE, quantile=q, value=2030 + q / 10, color=ColorCode.dark_gray)
        for q in [0, 5, 25, 50, 75, 95, 100]
    ])
    assert validate_response(when_resp, when_expected)["ok"]


def test_validate_response_catches_mechanical_failures():
    expected = [ExpectedForecast("2030-12-31", "Overall", q) for q in [25, 50, 75]]
    mixed = _response([
        _forecast(quantile=25, value=1, color=ColorCode.white),
        _forecast(quantile=50, value=2, color=ColorCode.light_gray),
        _forecast(quantile=75, value=3, color=ColorCode.white),
    ])
    assert any(issue.startswith("mixed_colors_") for issue in validate_response(mixed, expected)["issues"])

    non_increasing = _response([
        _forecast(quantile=25, value=10),
        _forecast(quantile=50, value=5),
        _forecast(quantile=75, value=15),
    ])
    assert any(issue.startswith("quantiles_not_increasing_") for issue in validate_response(non_increasing, expected)["issues"])

    late_resolution = _response(
        [_forecast("2025-12-31", quantile=50, value=10, color=ColorCode.black)],
        resolution_values=[
            ResolutionValue(
                forecast_date="2025-12-31",
                dimension="Overall",
                value=10,
                source_date="2026-01-01",
                source="https://example.com/late",
                confidence=90,
            )
        ],
    )
    issues = validate_response(late_resolution, [ExpectedForecast("2025-12-31", "Overall", 50)])["issues"]
    assert any(issue.startswith("resolution_source_after_target_") for issue in issues)


def test_resolution_status_date_and_color_combinations():
    today = date(2026, 6, 2)
    assert resolution_status("2025-12-31", ColorCode.black, today=today) == "resolved"
    assert resolution_status("2025-12-31", ColorCode.white, today=today) == "due_unresolved"
    assert resolution_status("2030-12-31", ColorCode.white, today=today) == "forecast"
    assert resolution_status("2030-12-31", ColorCode.black, today=today) == "resolved_early"


def test_default_output_dir_is_root_outputs_folder():
    assert DEFAULT_OUTPUT_DIR == "outputs"


def test_empty_sheet_id_env_uses_default(monkeypatch):
    monkeypatch.setenv("LEAP_SHEET_ID", "")
    reloaded = importlib.reload(common)
    try:
        assert reloaded.DEFAULT_SHEET_ID == "1lT7zVfKAsVZU7bKaEALq1AWApfFmWMisprTK42l7RDo"
    finally:
        importlib.reload(common)


def test_result_id_extends_review_group_id_contract():
    group_id = make_review_group_id("run1", "q1", "2030-12-31", "Overall")
    assert group_id == "run1_q1_2030-12-31_Overall"
    assert make_result_id("run1", "q1", "2030-12-31", "Overall", 50) == f"{group_id}_50"


def test_cli_module_imports_from_package():
    assert callable(run_surveillance.main)


def test_question_filter_preserves_requested_order_and_limits_after_filtering():
    questions = [
        _question(qid="alpha"),
        _question(qid="bravo"),
        _question(qid="charlie"),
    ]
    filtered, missing = run_surveillance._filter_questions_by_id(
        questions,
        ["charlie", "missing", "alpha"],
        limit=1,
    )

    assert [q.id for q in filtered] == ["charlie"]
    assert missing == ["missing"]


def test_build_run_data_summary_counts_errors_browser_due_unresolved_and_cost():
    q1 = _question(qid="q1", expected=[ExpectedForecast("2025-12-31", "Overall", 50)])
    q2 = _question(qid="q2")
    resp = _response([_forecast("2025-12-31", value=10, color=ColorCode.white)])
    cost = RunCost(research=1.0, judge_stage1=0.25, judge_stage2=0.5, refinement=0.75)

    run_data = build_run_data(
        "run1",
        "model",
        [q1, q2],
        [resp, None],
        [{"ok": True, "usable_for_scoring": True, "issues": []}, None],
        [[], None],
        [
            ResearchQualityReport(
                confidence=80,
                adequate=False,
                browser_would_help=True,
                browser_url="https://example.com/data",
                browser_objective="Extract the value",
            ),
            None,
        ],
        costs_list=[cost, None],
        browser_useds=[True, False],
        errors_list=[None, {"type": "RuntimeError", "message": "boom", "traceback": "..."}],
    )

    assert run_data["summary"] == {
        "question_count": 2,
        "ok_count": 1,
        "error_count": 1,
        "browser_count": 1,
        "due_unresolved_count": 1,
        "total_cost": 2.5,
    }
    assert run_data["questions"][0]["quality"]["browser_would_help"] is True
    assert run_data["questions"][0]["quality"]["browser_url"] == "https://example.com/data"
    assert run_data["questions"][0]["quality"]["browser_objective"] == "Extract the value"


def test_build_run_data_rejects_misaligned_inputs():
    q = _question()
    with pytest.raises(ValueError, match="responses length"):
        build_run_data(
            "run1",
            "model",
            [q],
            [],
            [None],
            [None],
            [None],
        )


def test_write_csv_output_includes_question_context_and_validation_issues(tmp_path):
    q = _question(qtype="probability")
    resp = _response([_forecast(value=40.0)])
    path = write_csv_output(
        "run1",
        [q],
        [resp],
        str(tmp_path),
        validations=[{"ok": False, "usable_for_scoring": True, "issues": ["mixed_colors_x"]}],
    )

    assert path.endswith("run_run1.csv")
    text = (tmp_path / "run_run1.csv").read_text()
    assert "question_type" in text
    assert "unit_min" in text
    assert "mixed_colors_x" in text


def test_sync_reviews_to_bigquery_updates_actual_existing_rows(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        storage,
        "get_existing_result_ids_for_groups",
        lambda group_ids: {
            "prob": [("prob_50", 50)],
            "quant": [(f"quant_{q}", q) for q in [0, 5, 25, 50, 75, 95, 100]],
        },
    )

    def fake_merge(label, rows, **kwargs):
        captured["rows"] = rows
        captured["kwargs"] = kwargs

        class Stats:
            inserted_row_count = 0
            updated_row_count = len(rows)

        return Stats()

    monkeypatch.setattr(storage, "try_merge_bigquery_rows", fake_merge)

    result = storage.sync_reviews_to_bigquery([
        {"group_id": "prob", "type": "forecast", "review_value": "0.42", "review_notes": "median fix"},
        {"group_id": "quant", "type": "resolved", "review_value": "12", "review_verdict": "close", "review_color": "black"},
    ])

    assert result["skipped"] == 0
    rows = captured["rows"]
    assert len(rows) == 8
    prob_rows = [r for r in rows if r["result_id"].startswith("prob")]
    assert len(prob_rows) == 1
    assert prob_rows[0]["result_id"] == "prob_50"
    assert prob_rows[0]["final_value"] == pytest.approx(0.42)
    assert prob_rows[0]["override_value"] == pytest.approx(0.42)
    quant_rows = [r for r in rows if r["result_id"].startswith("quant")]
    assert len(quant_rows) == 7
    assert all(r["final_value"] == 12 for r in quant_rows)
    assert [r for r in quant_rows if r["result_id"] == "quant_50"][0]["override_value"] == 12
    assert all(r["final_color"] == "black" for r in quant_rows)


def test_low_confidence_adequate_judgment_is_downgraded(monkeypatch):
    q = _question()
    resp = _response([_forecast(value=40.0)])

    monkeypatch.setattr(
        research,
        "evaluate_adequacy",
        lambda *args, **kwargs: AdequacyAssessment(
            adequate=True,
            confidence=8,
            issues=[],
            reason="Structurally complete but weakly supported.",
        ),
    )
    monkeypatch.setattr(
        research,
        "decide_browser",
        lambda *args, **kwargs: BrowserDecision(
            browser_would_help=False,
            reason="No specific browser target would fix the low confidence.",
        ),
    )

    quality = research.evaluate_response_quality(resp, q, [])
    assert quality.adequate is False
    assert quality.confidence == 8
    assert f"low_confidence_below_{research.MIN_ADEQUATE_CONFIDENCE}" in quality.missing_data


def test_browser_refinement_rejects_structural_regression():
    original_validation = {"ok": True, "usable_for_scoring": True, "issues": []}
    refined_validation = {
        "ok": False,
        "usable_for_scoring": False,
        "issues": ["missing_1_q50s", "7_null_values"],
    }
    original_quality = ResearchQualityReport(adequate=False, confidence=70)
    refined_quality = ResearchQualityReport(adequate=False, confidence=40)

    accept, reason = run_surveillance._should_accept_browser_refinement(
        original_validation,
        refined_validation,
        original_quality,
        refined_quality,
    )

    assert accept is False
    assert "not usable" in reason


def test_browser_refinement_accepts_validation_fix():
    original_validation = {
        "ok": False,
        "usable_for_scoring": True,
        "issues": ["missing_1_forecast_rows"],
    }
    refined_validation = {"ok": True, "usable_for_scoring": True, "issues": []}
    original_quality = ResearchQualityReport(adequate=False, confidence=45)
    refined_quality = ResearchQualityReport(adequate=True, confidence=80)

    accept, reason = run_surveillance._should_accept_browser_refinement(
        original_validation,
        refined_validation,
        original_quality,
        refined_quality,
    )

    assert accept is True
    assert "fixed deterministic validation" in reason


def test_decide_browser_rejects_empty_or_unsafe_url(monkeypatch):
    q = _question()
    resp = _response([_forecast(value=40.0)])
    adequacy = AdequacyAssessment(
        adequate=False,
        confidence=20,
        issues=["needs browser"],
        reason="Needs browser extraction.",
    )

    monkeypatch.setattr(research.litellm, "responses", lambda **kwargs: object())
    monkeypatch.setattr(
        research,
        "_extract_text_from_response",
        lambda response: '{"browser_would_help": true, "browser_url": "", "browser_objective": "Extract data", "reason": "A dashboard would help."}',
    )
    empty = research.decide_browser(resp, q, [], adequacy)
    assert empty.browser_would_help is False
    assert "no URL" in empty.reason

    monkeypatch.setattr(
        research,
        "_extract_text_from_response",
        lambda response: '{"browser_would_help": true, "browser_url": "https://www.google.com/search?q=x", "browser_objective": "Search", "reason": "Search would help."}',
    )
    unsafe = research.decide_browser(resp, q, [], adequacy)
    assert unsafe.browser_would_help is False
    assert "safety filter" in unsafe.reason


def test_is_safe_url_blocks_search_local_and_private_targets():
    assert research.is_safe_url("https://example.com/data")[0] is True
    assert research.is_safe_url("https://www.bing.com/search?q=x")[0] is False
    assert research.is_safe_url("https://search.brave.com/search?q=x")[0] is False
    assert research.is_safe_url("http://localhost:8000")[0] is False
    assert research.is_safe_url("http://10.0.0.5/data")[0] is False


def test_build_review_rows_collapses_quantiles_to_review_groups():
    expected = [
        ExpectedForecast("2030-12-31", dim, q)
        for dim in ["A", "B"]
        for q in [0, 5, 25, 50, 75, 95, 100]
    ]
    q = _question(qtype="quantile", expected=expected, unit="Index")
    forecasts = [
        _forecast("2030-12-31", dimension=dim, quantile=quantile, value=value)
        for dim, offset in [("A", 0), ("B", 10)]
        for quantile, value in [(0, offset), (5, offset + 1), (25, offset + 2), (50, offset + 3), (75, offset + 4), (95, offset + 5), (100, offset + 6)]
    ]
    forecasts[2].color_code = ColorCode.black
    resp = _response(forecasts)
    run_data = build_run_data(
        "run1",
        "model",
        [q],
        [resp],
        [{"ok": True, "usable_for_scoring": True, "issues": []}],
        [[]],
        [ResearchQualityReport(confidence=80)],
    )

    rows = sheets.build_review_rows(run_data)
    headers = sheets.REVIEW_HEADERS
    assert len(rows) == 2
    by_dim = {row[headers.index("dimension")]: row for row in rows}
    assert by_dim["A"][headers.index("llm_answer")] == "3.0"
    assert by_dim["A"][headers.index("q25")] == "2.0"
    assert by_dim["A"][headers.index("q75")] == "4.0"
    assert by_dim["A"][headers.index("status")] == "forecast"
    assert by_dim["B"][headers.index("llm_answer")] == "13.0"
