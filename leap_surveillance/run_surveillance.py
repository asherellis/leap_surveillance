"""CLI for LEAP surveillance runs and review sync."""

import argparse
import os
import traceback
from datetime import datetime

from .common import DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_SHEET_ID, TEST_MODEL, safe_str
from .models import (
    EvidenceItem,
    QuestionSpec,
    RunCost,
    validate_response,
)
from .storage import (
    build_run_data,
    sync_reviews_to_bigquery,
    write_csv_output,
    write_json_output,
    write_surveillance_to_bigquery,
)
from .sheets import (
    _run_tab_name,
    get_reviewed_items,
    publish_to_sheet,
    setup_sheet,
)
from .questions import load_questions
from .research import (
    browser_extract,
    evaluate_response_quality,
    refine_with_browser,
    research,
)


def _read_requested_question_ids(raw: str) -> list[str]:
    if os.path.isfile(raw):
        with open(raw) as f:
            return [line.strip() for line in f if line.strip()]
    return [qid.strip() for qid in raw.split(",") if qid.strip()]


def _filter_questions_by_id(
    questions: list[QuestionSpec],
    requested_ids: list[str],
    limit: int | None = None,
) -> tuple[list[QuestionSpec], list[str]]:
    by_id = {q.id: q for q in questions}
    filtered = [by_id[qid] for qid in requested_ids if qid in by_id]
    missing = [qid for qid in requested_ids if qid not in by_id]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered, missing


def _validation_score(validation: dict) -> int:
    issues = validation.get("issues", []) or []
    score = len(issues)
    if not validation.get("ok", False):
        score += 10
    if not validation.get("usable_for_scoring", False):
        score += 100
    return score


def _should_accept_browser_refinement(
    original_validation: dict,
    refined_validation: dict,
    original_quality,
    refined_quality,
) -> tuple[bool, str]:
    if not refined_validation.get("usable_for_scoring", False):
        return False, "refined response is not usable for scoring"

    if original_validation.get("ok", False) and not refined_validation.get("ok", False):
        issues = ", ".join(refined_validation.get("issues", []) or [])
        return False, f"refinement introduced validation issues: {issues}"

    if refined_validation.get("ok", False):
        if not original_validation.get("ok", False):
            return True, "refinement fixed deterministic validation issues"
        if refined_quality.adequate:
            return True, "refinement passed validation and quality review"
        if refined_quality.confidence >= original_quality.confidence:
            return True, "refinement preserved validation and improved confidence"
        return False, "refinement did not improve quality confidence"

    if _validation_score(refined_validation) < _validation_score(original_validation):
        return True, "refinement reduced deterministic validation issues"

    return False, "refinement did not improve deterministic validation"


def _try_browser_repair(
    q: QuestionSpec,
    response,
    evidence,
    validation: dict,
    quality,
    costs: RunCost,
    *,
    test_mode: bool = False,
):
    print(f"  -> Browser: {quality.browser_url}", flush=True)
    browser_result = browser_extract(
        quality.browser_url,
        quality.browser_objective or "Extract data",
        test_mode=test_mode,
    )
    if not browser_result.success:
        print(f"  -> Browser failed: {browser_result.error}")
        return response, evidence, validation, quality, False

    browser_evidence = EvidenceItem(
        source_type="browser",
        url=browser_result.url,
        full_text=browser_result.extracted_text,
    )
    evidence = [*evidence, browser_evidence]

    refined_response = refine_with_browser(
        q,
        response,
        browser_result,
        evidence=evidence,
        test_mode=test_mode,
        costs=costs,
    )
    refined_quality = evaluate_response_quality(
        refined_response, q, evidence, test_mode=test_mode, costs=costs
    )
    refined_validation = validate_response(refined_response, q.expected_forecasts)
    accept_refinement, accept_reason = _should_accept_browser_refinement(
        validation,
        refined_validation,
        quality,
        refined_quality,
    )
    if accept_refinement:
        print(f"  -> Refined with browser data ({refined_quality.confidence}% confidence)")
        return refined_response, evidence, refined_validation, refined_quality, True

    print(f"  -> Browser data kept as evidence; original response retained ({accept_reason})")
    return response, evidence, validation, quality, True


def process_question(
    q: QuestionSpec, model: str, use_browser: bool = True, test_mode: bool = False
):
    costs = RunCost()
    print("  -> Researching with web search...", flush=True)
    response, evidence = research(q, model=model, test_mode=test_mode, costs=costs)
    print(f"  Research: {len(evidence)} sources, {len(response.forecasts)} forecasts", flush=True)

    print("  -> Evaluating quality...", flush=True)
    quality = evaluate_response_quality(response, q, evidence, test_mode=test_mode, costs=costs)
    validation = validate_response(response, q.expected_forecasts)
    browser_used = False

    if not quality.adequate:
        if not use_browser:
            print(f"  -> Browser skipped: disabled ({quality.confidence}% confidence)")
        elif not quality.browser_would_help:
            print(f"  -> Browser skipped: not useful ({quality.confidence}% confidence)")
        elif not quality.browser_url:
            print(f"  -> Browser skipped: no URL proposed ({quality.confidence}% confidence)")
        else:
            response, evidence, validation, quality, browser_used = _try_browser_repair(
                q,
                response,
                evidence,
                validation,
                quality,
                costs,
                test_mode=test_mode,
            )
    else:
        print(f"  -> Quality: {quality.confidence}% confidence")

    color = response.forecasts[0].color_code.value if response.forecasts else "N/A"
    print(f"  -> {color} | {'ok' if validation['ok'] else 'warning'}", flush=True)
    return response, evidence, validation, quality, browser_used, costs


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
    questions = load_questions(None if args.questions else args.limit)

    if args.questions:
        requested_ids = _read_requested_question_ids(args.questions)
        questions, missing_ids = _filter_questions_by_id(
            questions, requested_ids, limit=args.limit
        )
        print(f"Filtered to {len(questions)} questions (from {len(requested_ids)} IDs)")
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            suffix = "..." if len(missing_ids) > 5 else ""
            print(f"  warning: {len(missing_ids)} requested ID(s) were not found: {preview}{suffix}")
    else:
        print(f"Loaded {len(questions)} questions")

    if not questions:
        print("No questions to run. Check --questions IDs, --limit, or the BigQuery source.")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    responses, evidences, validations, quality_reports, costs_list, browser_useds, errors_list = [], [], [], [], [], [], []

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q.name}", flush=True)
        try:
            response, evidence, validation, quality, browser_used, costs = process_question(
                q, active_model, use_browser=not args.no_browser, test_mode=args.test_mode
            )
            responses.append(response)
            evidences.append(evidence)
            validations.append(validation)
            quality_reports.append(quality)
            costs_list.append(costs)
            browser_useds.append(browser_used)
            errors_list.append(None)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  -> error: {e}")
            responses.append(None)
            evidences.append(None)
            quality_reports.append(None)
            validations.append(None)
            costs_list.append(None)
            browser_useds.append(False)
            errors_list.append({
                "type": type(e).__name__,
                "message": str(e)[:1000],
                "traceback": tb[-2000:],
            })

    run_data = build_run_data(
        run_id,
        active_model,
        questions,
        responses,
        validations,
        evidences,
        quality_reports,
        costs_list=costs_list,
        browser_useds=browser_useds,
        errors_list=errors_list,
    )
    json_path = write_json_output(run_data, DEFAULT_OUTPUT_DIR)
    write_csv_output(run_id, questions, responses, DEFAULT_OUTPUT_DIR, validations=validations)
    s = run_data["summary"]
    print(f"\nSaved: {json_path}")
    print(f"Summary: {s['ok_count']}/{s['question_count']} OK, {s['error_count']} errors, {s['browser_count']} browser extractions")
    print(f"Due/unresolved (past-dated, no authoritative value found): {s['due_unresolved_count']}")
    print(f"Estimated cost: ${s['total_cost']:.4f}")

    if not args.no_bq:
        try:
            bq_stats = write_surveillance_to_bigquery(run_data)
            for key, label in [("results", "results"), ("evidence", "evidence rows"), ("runs", "run metadata")]:
                stats = bq_stats.get(key)
                if stats is not None:
                    print(f"BigQuery: {stats.inserted_row_count} {label} written")
                else:
                    print(f"BigQuery: {key} write failed")
        except Exception as e:
            print(f"BigQuery write failed: {e} (continuing to Sheet)")

    if not args.no_sheet:
        try:
            n = publish_to_sheet(run_data, DEFAULT_SHEET_ID)
            print(f"Published {n} rows to Sheet tab '{_run_tab_name(run_id)}'")
        except Exception as e:
            print(f"Sheet publishing failed: {e}")


def cmd_sync(args):
    reviewed_items, _ = get_reviewed_items(DEFAULT_SHEET_ID, tab_name=args.tab)
    print(f"Found {len(reviewed_items)} reviewed items")

    if not reviewed_items:
        print("Nothing to sync.")
        return

    if args.no_bq:
        print("\nDRY RUN (--no-bq): would sync the following to BigQuery:")
        print(f"  {'#':>3}  {'question':40s} {'date':12s} {'dim':20s} {'unit':12s} {'verdict':18s} review_value")
        for i, item in enumerate(reviewed_items, 1):
            name = (item.get("question_name") or "")[:38]
            tdate = (item.get("target_date") or "")[:10]
            dim = (item.get("dimension") or "")[:18]
            unit = (item.get("unit") or "")[:10]
            verdict = (item.get("review_verdict") or "")[:16]
            rval = safe_str(item.get("review_value") or "")
            print(f"  {i:>3}  {name:40s} {tdate:12s} {dim:20s} {unit:12s} {verdict:18s} {rval}")
        print(f"\n{len(reviewed_items)} item(s) would be synced. Re-run without --no-bq to write.")
        return

    stats = sync_reviews_to_bigquery(reviewed_items)
    if stats.get("results"):
        print(
            f"BigQuery: {stats['results'].inserted_row_count} inserted, "
            f"{stats['results'].updated_row_count} updated"
        )
    else:
        print("BigQuery sync failed (no write access?)")
        print("Use --no-bq to preview reviewed items without writing.")
    if stats.get("skipped", 0) > 0:
        print(f"  ({stats['skipped']} items skipped - raw data missing in BQ)")


def cmd_setup(args):
    if not args.yes:
        confirm = input("Rebuild the Instructions tab from current code? Per-run review tabs are not touched. [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return
    setup_sheet(DEFAULT_SHEET_ID)


def main():
    parser = argparse.ArgumentParser(
        prog="leap-surveillance",
        description="LEAP Surveillance - LLM forecasts for expert panel questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run --limit 5        Run surveillance on 5 questions
  %(prog)s run --limit 5 -y     Same, skip confirmation
  %(prog)s sync                 Sync reviewed items from latest run tab
  %(prog)s sync --tab run_...   Sync reviewed items from a specific run tab
  %(prog)s sync --no-bq         Dry-run: print what would be synced, no write
  %(prog)s setup                Rebuild the Instructions tab
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    run_parser = subparsers.add_parser("run", help="Run surveillance on questions")
    run_parser.add_argument("--limit", "-n", type=int, help="Limit to N questions")
    run_parser.add_argument(
        "--questions",
        "-q",
        type=str,
        help="Comma-separated question IDs to run (or path to file with one ID per line)",
    )
    run_parser.add_argument("--no-sheet", action="store_true", help="Skip publishing to Sheet")
    run_parser.add_argument("--no-bq", action="store_true", help="Skip BigQuery writes")
    run_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Skip browser extraction (faster; may miss dashboard-only data)",
    )
    run_parser.add_argument("--test-mode", "-t", action="store_true", help="Use lower-cost test models")
    run_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    sync_parser = subparsers.add_parser("sync", help="Sync reviewed items from a run tab to BigQuery")
    sync_parser.add_argument("--no-bq", action="store_true", help="Dry-run: print what would be synced; no write")
    sync_parser.add_argument("--tab", type=str, default=None, help="Specific run_<run_id> tab to read (default: most recent)")

    setup_parser = subparsers.add_parser("setup", help="Rebuild the Instructions tab (does not touch run_* tabs)")
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
