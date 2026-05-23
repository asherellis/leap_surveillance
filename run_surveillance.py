"""CLI for LEAP surveillance runs and review sync."""

import sys
from pathlib import Path

if "__file__" in dir():
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import argparse
import json
import os
from datetime import date, datetime
from typing import Optional

from data_utils import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHEET_ID,
    get_reviewed_items,
    load_questions,
    move_to_reviewed,
    publish_to_sheet,
    setup_sheet,
    sync_reviews_to_bigquery,
    write_csv_output,
    write_json_output,
    write_surveillance_to_bigquery,
)
from research import (
    DEFAULT_MODEL,
    TEST_MODEL,
    browser_extract,
    evaluate_response_quality,
    refine_with_browser,
    research,
)
from schemas import (
    EvidenceItem,
    ExpectedForecast,
    QuestionSpec,
    SurveillanceResponse,
)


def _parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def validate_response(
    response: SurveillanceResponse, expected: Optional[list[ExpectedForecast]] = None
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

        expected_triplets = {
            (e.forecast_date, e.dimension, e.quantile) for e in expected
        }
        expected_value_types = {
            (e.forecast_date, e.dimension, e.quantile): e.value_type for e in expected
        }
        forecast_triplets = {
            (f.forecast_date, f.dimension, f.quantile) for f in response.forecasts
        }
        missing_triplets = expected_triplets - forecast_triplets
        if missing_triplets:
            issues.append(f"missing_{len(missing_triplets)}_forecast_rows")

        unexpected_triplets = forecast_triplets - expected_triplets
        if unexpected_triplets:
            issues.append(f"unexpected_{len(unexpected_triplets)}_forecast_rows")

        if len(forecast_triplets) < len(response.forecasts):
            issues.append("duplicate_forecast_rows")

        for f in response.forecasts:
            key = (f.forecast_date, f.dimension, f.quantile)
            expected_type = expected_value_types.get(key)
            actual_type = getattr(f.value_type, "value", f.value_type)
            if expected_type and actual_type != expected_type:
                issues.append(f"value_type_mismatch_{f.forecast_date}_{f.dimension}_{f.quantile}")

        resolution_keys = {
            (e.forecast_date, e.dimension)
            for e in expected
            if e.value_type == "resolution"
        }
        returned_resolution_keys = {
            (r.forecast_date, r.dimension)
            for r in response.resolution_values
            if r.value is not None
        }
        missing_resolution_keys = resolution_keys - returned_resolution_keys
        if missing_resolution_keys:
            issues.append(f"missing_{len(missing_resolution_keys)}_resolution_values")

        unexpected_resolution_keys = returned_resolution_keys - resolution_keys
        if unexpected_resolution_keys:
            issues.append(f"unexpected_{len(unexpected_resolution_keys)}_resolution_values")

        for r in response.resolution_values:
            target_date = _parse_iso_date(r.forecast_date)
            source_date = _parse_iso_date(r.source_date)
            if target_date and source_date and source_date > target_date:
                issues.append(f"resolution_source_after_target_{r.forecast_date}_{r.dimension}")

    null_count = sum(1 for f in response.forecasts if f.forecast_value is None)
    if null_count > len(response.forecasts) // 2:
        issues.append(f"{null_count}_null_values")

    from collections import defaultdict
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

    resolution_groups = defaultdict(list)
    black_groups = defaultdict(list)
    for f in response.forecasts:
        actual_type = getattr(f.value_type, "value", f.value_type)
        if f.color_code.value == "black" and f.forecast_value is not None:
            black_groups[(f.forecast_date, f.dimension)].append(f.forecast_value)
        if actual_type == "resolution":
            if f.color_code.value != "black":
                issues.append(f"resolution_not_black_{f.forecast_date}_{f.dimension}")
            if f.forecast_value is not None:
                resolution_groups[(f.forecast_date, f.dimension)].append(f.forecast_value)

    for key, values in resolution_groups.items():
        if len({round(v, 12) for v in values}) > 1:
            issues.append(f"resolution_quantiles_differ_{key[0]}_{key[1]}")

    for key, values in black_groups.items():
        if len({round(v, 12) for v in values}) > 1:
            issues.append(f"black_quantiles_differ_{key[0]}_{key[1]}")

    if expected_has_q50:
        usable = any(
            f.quantile == 50 and f.forecast_value is not None
            for f in response.forecasts
        )
    else:
        usable = any(f.forecast_value is not None for f in response.forecasts)
    return {"ok": len(issues) == 0, "usable": usable, "issues": issues}


def process_question(
    q: QuestionSpec, model: str, use_browser: bool = True, test_mode: bool = False
):
    print("  -> Researching with web search...", flush=True)
    response, evidence = research(q, model=model, test_mode=test_mode)
    print(f"  Research: {len(evidence)} sources, {len(response.forecasts)} forecasts", flush=True)

    print("  -> Evaluating quality...", flush=True)
    quality = evaluate_response_quality(response, q, test_mode=test_mode)
    browser_used = False

    if use_browser and not quality.adequate and quality.browser_would_help and quality.browser_url:
        print(f"  -> Browser: {quality.browser_url}", flush=True)
        browser_result = browser_extract(
            quality.browser_url,
            quality.browser_objective or "Extract data",
            test_mode=test_mode,
        )
        if browser_result.success:
            evidence.append(
                EvidenceItem(
                    source_type="browser",
                    url=browser_result.url,
                    full_text=browser_result.extracted_text,
                )
            )
            response = refine_with_browser(q, response, browser_result, test_mode=test_mode)
            quality = evaluate_response_quality(response, q, test_mode=test_mode)
            print(f"  -> Refined with browser data ({quality.confidence}% confidence)")
            browser_used = True
        else:
            print(f"  -> Browser failed: {browser_result.error}")
    else:
        print(f"  -> Quality: {quality.confidence}% confidence")

    validation = validate_response(response, q.expected_forecasts)
    color = response.forecasts[0].color_code.value if response.forecasts else "N/A"
    print(f"  -> {color} | {'ok' if validation['ok'] else 'warning'}", flush=True)
    return response, evidence, validation, quality, browser_used


def cmd_run(args):
    active_model = TEST_MODEL if args.test_mode else DEFAULT_MODEL
    if not args.yes:
        print(f"Model: {active_model}")
        print(f"Limit: {args.limit or 'all'}")
        print(f"BigQuery writes: {'skip' if args.no_bq else 'yes'}")
        print(f"Sheet: {'skip' if args.no_sheet else 'yes'}")
        if input("Continue? [y/N] ").lower() != "y":
            return

    print("Loading questions from BigQuery...")
    questions = load_questions(args.limit, prod=False)

    if args.questions:
        if os.path.isfile(args.questions):
            with open(args.questions) as f:
                filter_ids = {line.strip() for line in f if line.strip()}
        else:
            filter_ids = {qid.strip() for qid in args.questions.split(",")}
        questions = [q for q in questions if q.id in filter_ids]
        print(f"Filtered to {len(questions)} questions (from {len(filter_ids)} IDs)")
    else:
        print(f"Loaded {len(questions)} questions")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    responses, evidences, validations, quality_reports = [], [], [], []
    ok_count = 0
    browser_count = 0

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q.name}", flush=True)
        try:
            response, evidence, validation, quality, browser_used = process_question(
                q, active_model, use_browser=not args.no_browser, test_mode=args.test_mode
            )
            responses.append(response)
            evidences.append(evidence)
            validations.append(validation)
            quality_reports.append(quality)
            ok_count += 1 if validation["ok"] else 0
            browser_count += 1 if browser_used else 0
        except Exception as e:
            print(f"  -> error: {e}")
            responses.append(None)
            evidences.append(None)
            quality_reports.append(None)
            validations.append(None)

    json_path = write_json_output(
        run_id,
        active_model,
        questions,
        responses,
        validations,
        evidences,
        quality_reports,
        DEFAULT_OUTPUT_DIR,
    )
    write_csv_output(run_id, questions, responses, DEFAULT_OUTPUT_DIR)
    print(f"\nSaved: {json_path}")
    print(f"Summary: {ok_count}/{len(questions)} OK, {browser_count} browser extractions")

    with open(json_path) as f:
        run_data = json.load(f)

    if not args.no_bq:
        try:
            bq_stats = write_surveillance_to_bigquery(run_data, prod=False)
            results_ok = bq_stats.get("results") is not None
            evidence_ok = bq_stats.get("evidence") is not None
            runs_ok = bq_stats.get("runs") is not None

            if results_ok:
                print(f"BigQuery: {bq_stats['results'].inserted_row_count} results written")
            if evidence_ok:
                print(f"BigQuery: {bq_stats['evidence'].inserted_row_count} evidence rows written")
            if runs_ok:
                print(f"BigQuery: {bq_stats['runs'].inserted_row_count} run metadata written")

            if not results_ok:
                print("BigQuery: results write failed (continuing to Sheet)")
        except Exception as e:
            print(f"BigQuery write failed: {e} (continuing to Sheet)")

    if not args.no_sheet:
        try:
            n = publish_to_sheet(run_data, DEFAULT_SHEET_ID)
            print(f"Published {n} rows to Sheet")
        except Exception as e:
            print(f"Sheet publishing failed: {e}")


def cmd_sync(args):
    reviewed_items, row_numbers = get_reviewed_items(DEFAULT_SHEET_ID)
    print(f"Found {len(reviewed_items)} reviewed items")

    if not reviewed_items:
        print("Nothing to sync.")
        return

    bq_attempted = False
    bq_success = False
    if not args.no_bq:
        bq_attempted = True
        stats = sync_reviews_to_bigquery(reviewed_items, prod=False)
        if stats.get("results"):
            print(
                f"BigQuery: {stats['results'].inserted_row_count} inserted, "
                f"{stats['results'].updated_row_count} updated"
            )
            bq_success = True
        else:
            print("BigQuery sync failed (no write access?)")
        if stats.get("skipped", 0) > 0:
            print(f"  ({stats['skipped']} items skipped - raw data missing in BQ)")

    if bq_success or not bq_attempted:
        moved = move_to_reviewed(DEFAULT_SHEET_ID, reviewed_items, row_numbers)
        print(f"Moved {moved} rows to 'Reviewed' tab")
    else:
        if args.force_move:
            moved = move_to_reviewed(DEFAULT_SHEET_ID, reviewed_items, row_numbers)
            print(f"Moved {moved} rows to 'Reviewed' tab (--force-move)")
        else:
            print("Rows not moved (BQ failed). Use --force-move to move anyway, or --no-bq to skip BQ entirely.")


def cmd_setup(args):
    if not args.yes:
        confirm = input("This will clear all data in the sheet. Continue? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return
    setup_sheet(DEFAULT_SHEET_ID)


def main():
    parser = argparse.ArgumentParser(
        description="LEAP Surveillance - LLM forecasts for expert panel questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run --limit 5        Run surveillance on 5 questions
  %(prog)s run --limit 5 -y     Same, skip confirmation
  %(prog)s sync                 Sync reviewed items from Sheet
  %(prog)s setup                Reset the Google Sheet
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    run_parser = subparsers.add_parser("run", help="Run surveillance on questions")
    run_parser.add_argument("--limit", "-n", type=int, help="Limit to N questions")
    run_parser.add_argument("--questions", "-q", type=str, help="Comma-separated question IDs to run (or path to file with one ID per line)")
    run_parser.add_argument("--no-sheet", action="store_true", help="Skip publishing to Sheet")
    run_parser.add_argument("--no-bq", action="store_true", help="Skip BigQuery writes")
    run_parser.add_argument("--no-browser", action="store_true", help="Skip browser extraction (faster, more reliable)")
    run_parser.add_argument("--test-mode", "-t", action="store_true", help="Use lower-cost models for validation (gpt-4o-mini, no reasoning effort)")
    run_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    sync_parser = subparsers.add_parser("sync", help="Sync reviewed items from Sheet to BigQuery")
    sync_parser.add_argument("--no-bq", action="store_true", help="Skip BigQuery sync (Sheet-only mode)")
    sync_parser.add_argument("--force-move", action="store_true", help="Move rows even if BigQuery sync fails")

    setup_parser = subparsers.add_parser("setup", help="Reset the Google Sheet")
    setup_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "sync":
        cmd_sync(args)
    elif args.command == "setup":
        cmd_setup(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
