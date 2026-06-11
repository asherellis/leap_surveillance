"""CLI for LEAP surveillance runs and review sync."""

import argparse
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace as _dc_replace
from datetime import datetime

from .common import (
    CLAUDE_EVALUATOR_MODEL,
    CLAUDE_RESEARCH_MODEL,
    DEFAULT_BROWSER_MODEL,
    DEFAULT_EVALUATOR_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHEET_ID,
    TEST_CLAUDE_EVALUATOR_MODEL,
    TEST_CLAUDE_MODEL,
    TEST_EVALUATOR_MODEL,
    TEST_MODEL,
    safe_str,
)
from .models import (
    EvidenceItem,
    QuestionSpec,
    ResearchQualityReport,
    RunCost,
    SurveillanceResponse,
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
    run_tab_name,
    get_reviewed_items,
    publish_to_sheet,
    setup_sheet,
)
from .questions import load_questions
from .browser import browser_extract
from .research import (
    judge_response,
    refine_with_browser,
    research_question,
)
from .consensus import compute_consensus


@dataclass(frozen=True)
class ModelStack:
    """Per-provider model selection. tag is the cost-bucket prefix ("gpt" -> "", "claude" -> "claude_")."""
    tag: str
    research_model: str
    eval_model: str
    browser_navigator: str

    def cost_bucket(self, stage: str) -> str:
        """stage in {'research', 'judge_stage1', 'judge_stage2', 'refinement'}."""
        prefix = "" if self.tag == "gpt" else f"{self.tag}_"
        return f"{prefix}{stage}"


def _build_stacks(test_mode: bool) -> tuple[ModelStack, ModelStack]:
    if test_mode:
        gpt = ModelStack("gpt", TEST_MODEL, TEST_EVALUATOR_MODEL, TEST_MODEL)
        # Haiku fails browser-use's AgentOutput schema at step 12+; use gpt-4o-mini for test browser too.
        claude = ModelStack("claude", TEST_CLAUDE_MODEL, TEST_CLAUDE_EVALUATOR_MODEL, TEST_MODEL)
    else:
        gpt = ModelStack("gpt", DEFAULT_MODEL, DEFAULT_EVALUATOR_MODEL, DEFAULT_BROWSER_MODEL)
        # gpt-4o drives browser for both paths — Haiku and Sonnet both have intermittent AgentOutput schema
        # failures in the full browser-use agent loop; gpt-4o is the proven reliable navigator.
        claude = ModelStack("claude", CLAUDE_RESEARCH_MODEL, CLAUDE_EVALUATOR_MODEL, DEFAULT_BROWSER_MODEL)
    return gpt, claude


@dataclass
class ModelRunResult:
    """One model's full pipeline outcome: research + judge + maybe browser refinement."""
    response: SurveillanceResponse | None
    evidence: list[EvidenceItem]
    validation: dict
    quality: ResearchQualityReport | None
    browser_used: bool = False
    browser_status: str = "not_proposed"
    error: dict | None = None  # {type, message, traceback[-2000:]} if this model errored


@dataclass
class QuestionRunResult:
    """Question-level outcome. Both models live as siblings in per_model.

    `per_model` keys are model tags ("gpt", "claude"); values are ModelRunResult.
    Single-model runs (e.g. --gpt) have only one key; --both has both.
    """
    per_model: dict[str, ModelRunResult]
    costs: RunCost
    consensus: dict | None = None


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
    # Reject only when the refinement is mechanically worse; the judge's verdict is noisy.
    if not refined_validation.get("usable_for_scoring", False):
        return False, "refined response is not usable for scoring"

    if original_validation.get("ok", False) and not refined_validation.get("ok", False):
        issues = ", ".join(refined_validation.get("issues", []) or [])
        return False, f"refinement introduced validation issues: {issues}"

    if _validation_score(refined_validation) > _validation_score(original_validation):
        return False, "refinement increased deterministic validation issues"

    return True, "refinement validation OK; quality warnings preserved"


def _try_browser_refinement_for_model(
    q: QuestionSpec,
    result: ModelRunResult,
    *,
    stack: ModelStack,
    costs: RunCost,
    test_mode: bool = False,
) -> ModelRunResult:
    print(f"  [{stack.tag}] Browser: {result.quality.browser_url}", flush=True)
    browser_result = browser_extract(
        result.quality.browser_url,
        result.quality.browser_objective or "Extract data",
        test_mode=test_mode,
        model_override=stack.browser_navigator,
    )
    if not browser_result.success:
        err = browser_result.error or "no error message returned"
        print(f"  [{stack.tag}] Browser failed: {err}")
        result.browser_status = "extract_failed"
        return result

    browser_evidence = EvidenceItem(
        source_type="browser",
        url=browser_result.url,
        title=f"Browser extraction: {browser_result.objective}",
        full_text=browser_result.extracted_text,
    )
    evidence = [*result.evidence, browser_evidence]

    refined_response = refine_with_browser(
        q,
        result.response,
        browser_result,
        evidence=evidence,
        test_mode=test_mode,
        costs=costs,
        model=stack.research_model,
        cost_bucket=stack.cost_bucket("refinement"),
    )
    refined_quality = judge_response(
        refined_response, q, evidence,
        test_mode=test_mode, costs=costs, propose_browser=False,
        eval_model=stack.eval_model,
        cost_bucket_stage1=stack.cost_bucket("judge_stage1"),
    )
    refined_validation = validate_response(refined_response, q.expected_forecasts)
    accept_refinement, accept_reason = _should_accept_browser_refinement(
        result.validation,
        refined_validation,
        result.quality,
        refined_quality,
    )
    if accept_refinement:
        status = "ok" if refined_quality.adequate else "flagged for review"
        print(
            f"  [{stack.tag}] Refined with browser data "
            f"({refined_quality.confidence}% confidence, {status}) — {accept_reason}"
        )
        return ModelRunResult(
            response=refined_response,
            evidence=evidence,
            validation=refined_validation,
            quality=refined_quality,
            browser_used=True,
            browser_status="accepted",
        )

    print(f"  [{stack.tag}] Browser data kept as evidence; original response retained ({accept_reason})")
    for issue in (refined_quality.missing_data or [])[:3]:
        print(f"     refined issue: {issue[:160]}")
    return ModelRunResult(
        response=result.response,
        evidence=evidence,
        validation=result.validation,
        quality=result.quality,
        browser_used=True,
        browser_status="refinement_rejected",
    )


def _run_single_model_pipeline(
    q: QuestionSpec,
    *,
    stack: ModelStack,
    costs: RunCost,
    use_browser: bool,
    test_mode: bool,
) -> ModelRunResult:
    """One model's full pipeline: research -> judge -> maybe browser -> maybe refine."""
    print(f"  [{stack.tag}] Researching with web search...", flush=True)
    response, evidence = research_question(
        q, model=stack.research_model, test_mode=test_mode, costs=costs,
        cost_bucket=stack.cost_bucket("research"),
    )
    print(f"  [{stack.tag}] Research: {len(evidence)} sources, {len(response.forecasts)} forecasts", flush=True)

    print(f"  [{stack.tag}] Evaluating quality...", flush=True)
    quality = judge_response(
        response, q, evidence, test_mode=test_mode, costs=costs,
        eval_model=stack.eval_model,
        cost_bucket_stage1=stack.cost_bucket("judge_stage1"),
        cost_bucket_stage2=stack.cost_bucket("judge_stage2"),
    )
    validation = validate_response(response, q.expected_forecasts)
    result = ModelRunResult(
        response=response,
        evidence=evidence,
        validation=validation,
        quality=quality,
    )

    if not quality.adequate:
        if not use_browser:
            print(f"  [{stack.tag}] Browser skipped: disabled ({quality.confidence}% confidence)")
        elif not quality.browser_would_help:
            print(f"  [{stack.tag}] Browser skipped: not useful ({quality.confidence}% confidence)")
        elif not quality.browser_url:
            print(f"  [{stack.tag}] Browser skipped: no URL proposed ({quality.confidence}% confidence)")
            result.browser_status = "proposed_no_url"
        else:
            result = _try_browser_refinement_for_model(
                q, result, stack=stack, costs=costs, test_mode=test_mode,
            )
    else:
        print(f"  [{stack.tag}] Quality: {quality.confidence}% confidence")

    color = result.response.forecasts[0].color_code.value if result.response.forecasts else "N/A"
    print(f"  [{stack.tag}] {color} | {'ok' if result.validation['ok'] else 'warning'}", flush=True)
    return result


def process_question(
    q: QuestionSpec,
    model: str | None = None,
    use_browser: bool = True,
    test_mode: bool = False,
    *,
    mode: str = "gpt",
) -> QuestionRunResult:
    """Run the surveillance pipeline for a question.

    mode: "gpt", "claude", or "both" (default from the CLI).
    `model` arg overrides the GPT research model when provided.
    """
    costs = RunCost()
    gpt_stack, claude_stack = _build_stacks(test_mode)
    if model is not None:
        gpt_stack = _dc_replace(gpt_stack, research_model=model)
    selected = {}
    if mode in ("gpt", "both"):
        selected["gpt"] = gpt_stack
    if mode in ("claude", "both"):
        selected["claude"] = claude_stack

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            tag: ex.submit(_run_single_model_pipeline, q,
                           stack=stack, costs=costs,
                           use_browser=use_browser, test_mode=test_mode)
            for tag, stack in selected.items()
        }
        per_model: dict[str, ModelRunResult] = {}
        for tag, fut in futures.items():
            try:
                per_model[tag] = fut.result()
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  [{tag}] pipeline error: {type(e).__name__}: {e}", flush=True)
                per_model[tag] = ModelRunResult(
                    response=None, evidence=[],
                    validation={"ok": False, "usable_for_scoring": False, "issues": ["model_errored"]},
                    quality=None,
                    error={"type": type(e).__name__, "message": str(e)[:500], "traceback": tb[-2000:]},
                )

    consensus = compute_consensus(per_model) if len(per_model) >= 2 else None
    return QuestionRunResult(per_model=per_model, costs=costs, consensus=consensus)


def _resolve_mode_flag(args) -> str:
    """Pick run mode from --gpt/--claude/--both. Default = both."""
    return getattr(args, "mode_flag", None) or "both"


def cmd_run(args):
    active_model = TEST_MODEL if args.test_mode else DEFAULT_MODEL
    active_claude = TEST_CLAUDE_MODEL if args.test_mode else CLAUDE_RESEARCH_MODEL
    mode = _resolve_mode_flag(args)
    # Run-level metadata reflects which models actually ran.
    run_models = {}
    if mode in ("gpt", "both"):
        run_models["gpt"] = active_model
    if mode in ("claude", "both"):
        run_models["claude"] = active_claude
    if mode in ("gpt", "both") and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set; required for --gpt and --both modes. Rerun with --claude to skip GPT.")
        return 2
    if mode in ("claude", "both") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set; required for --claude and --both modes. Rerun with --gpt to skip Claude.")
        return 2
    if not args.yes:
        active_models = {
            "gpt": active_model,
            "claude": TEST_CLAUDE_MODEL if args.test_mode else CLAUDE_RESEARCH_MODEL,
            "both": f"{active_model} + {TEST_CLAUDE_MODEL if args.test_mode else CLAUDE_RESEARCH_MODEL}",
        }[mode]
        print(f"Mode: {mode} ({active_models})")
        print(f"Limit: {args.limit or 'all'}")
        print(f"BigQuery writes: {'skip' if args.no_bq else 'yes'}")
        print(f"Sheet: {'skip' if args.no_sheet else 'yes'}")
        if input("Continue? [y/N] ").lower() != "y":
            return

    print("Loading questions from BigQuery...")
    questions = load_questions(None if args.questions else args.limit, dev=args.dev)

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
    n = len(questions)
    per_model_lists: list[dict | None] = [None] * n
    costs_list = [None] * n
    consensus_blocks = [None] * n
    errors_list = [None] * n

    workers = max(1, min(args.workers, n))
    print_lock = threading.Lock()

    def _run_one(i: int, q):
        with print_lock:
            print(f"[{i + 1}/{n}] {q.name} - starting", flush=True)
        try:
            result = process_question(
                q, use_browser=not args.no_browser, test_mode=args.test_mode,
                mode=mode,
            )
            with print_lock:
                print(f"[{i + 1}/{n}] {q.name} - done", flush=True)
            return i, result, None
        except Exception as e:
            tb = traceback.format_exc()
            with print_lock:
                print(f"[{i + 1}/{n}] {q.name} - error: {e}", flush=True)
            return i, None, {
                "type": type(e).__name__,
                "message": str(e)[:1000],
                "traceback": tb[-2000:],
            }

    print(f"Running with {workers} worker(s)")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_one, i, q) for i, q in enumerate(questions)]
        for fut in as_completed(futures):
            i, result, err = fut.result()
            if err:
                errors_list[i] = err
            else:
                per_model_lists[i] = result.per_model
                costs_list[i] = result.costs
                consensus_blocks[i] = result.consensus

    run_data = build_run_data(
        run_id,
        questions,
        mode=mode,
        models=run_models,
        per_model_lists=per_model_lists,
        costs_list=costs_list,
        consensus_blocks=consensus_blocks,
        errors_list=errors_list,
    )
    json_path = write_json_output(run_data, DEFAULT_OUTPUT_DIR)
    # CSV has model_id as the first column; both models go in as siblings.
    write_csv_output(run_id, questions, per_model_lists, DEFAULT_OUTPUT_DIR)
    s = run_data["summary"]
    print(f"\nSaved: {json_path}")
    print(
        f"Summary: {s['ok_count']}/{s['question_count']} validation OK, "
        f"{s['quality_issue_count']} quality issues, "
        f"{s['error_count']} errors, {s['browser_count']} browser extractions"
    )
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
            print(f"Published {n} rows to Sheet tab '{run_tab_name(run_id)}'")
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
        description="LEAP Surveillance - forecasts for expert panel questions",
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
    run_parser.add_argument(
        "--dev",
        action="store_true",
        help="Load only the canonical dev question set (LCB Pro, Labor Share, US-China Military)",
    )
    run_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Parallel question workers (default 1, recommended 5)",
    )
    run_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    model_group = run_parser.add_mutually_exclusive_group()
    model_group.add_argument("--gpt", dest="mode_flag", action="store_const", const="gpt",
                              help="Run GPT-5.5 only")
    model_group.add_argument("--claude", dest="mode_flag", action="store_const", const="claude",
                              help="Run Claude Sonnet only")
    model_group.add_argument("--both", dest="mode_flag", action="store_const", const="both",
                              help="Run both GPT and Claude in parallel (default)")

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
