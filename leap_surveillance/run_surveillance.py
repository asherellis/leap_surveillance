"""CLI for LEAP surveillance runs and review sync."""

import argparse
import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

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
)
from .models import (
    BrowserEvidence,
    EvidenceItem,
    QuestionSpec,
    ResearchQualityReport,
    RunCost,
    SurveillanceResponse,
    validate_response,
)
from .storage import (
    _serialize_model_result,
    build_run_data,
    enrich_with_run_stability,
    enrich_with_value_changes,
    write_csv_output,
    write_json_output,
)
from .sheets import (
    run_tab_name,
    publish_to_sheet,
    setup_sheet,
    write_review_csv,
)
from .sync import cmd_sync
from .questions import load_questions
from .browser import browser_extract
from .research import (
    annotate_browser_evidence,
    ensure_evidence_plan,
    ensure_rc_source,
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
        gpt = ModelStack("gpt", TEST_MODEL, TEST_EVALUATOR_MODEL, DEFAULT_BROWSER_MODEL)
        # Browser-use requires gpt-4o even in test mode — mini fails AgentOutput schema at step 12+.
        claude = ModelStack("claude", TEST_CLAUDE_MODEL, TEST_CLAUDE_EVALUATOR_MODEL, DEFAULT_BROWSER_MODEL)
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
    browser_url: str = ""
    browser_objective: str = ""
    browser_error: str = ""
    error: dict | None = None  # {type, message, traceback[-2000:]} if this model errored


@dataclass
class QuestionRunResult:
    """Question-level outcome: per_model dict keyed by tag ("gpt"/"claude"), plus costs and consensus."""
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
    refined_quality,
) -> tuple[bool, str]:
    # Mechanical floor: never accept a broken response.
    if not refined_validation.get("usable_for_scoring", False):
        return False, "refined response is not usable for scoring"
    if _validation_score(refined_validation) > _validation_score(original_validation):
        return False, "refinement increased deterministic validation issues"

    # Re-judge must pass. If it's still unhappy we can't tell whether the new data is right
    # or the judge caught a real defect (wrong column, wrong scope, wrong date).
    if refined_quality.adequate:
        return True, "re-judge passed"

    return False, "re-judge still inadequate; retaining original"


def _earliest_past_target(response) -> str | None:
    """Earliest past target date in a response — the row whose value needs an as-of-date (historical) lookup."""
    if response is None:
        return None
    # UTC to stay consistent with date_value_type/row_resolution_status (avoids near-midnight drift).
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    past = [str(f.forecast_date)[:10] for f in (response.forecasts or [])
            if getattr(f, "forecast_date", None) and str(f.forecast_date)[:10] < today]
    return min(past) if past else None


def _question_js_risk(q: QuestionSpec) -> bool:
    return bool((q.evidence_plan or {}).get("js_risk") or (q.rc_source or {}).get("js_risk"))


def _refine_model_with_browser_result(
    q: QuestionSpec,
    result: ModelRunResult,
    *,
    stack: ModelStack,
    costs: RunCost,
    browser_result: BrowserEvidence,
    test_mode: bool = False,
) -> ModelRunResult:
    if result.response is None or result.quality is None:
        return result

    browser_evidence = annotate_browser_evidence(browser_result, q.evidence_plan)
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
    refined_validation = validate_response(refined_response, q.expected_forecasts, unit_min=q.unit_min, unit_max=q.unit_max)
    accept_refinement, accept_reason = _should_accept_browser_refinement(
        result.validation, refined_validation, refined_quality,
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
            browser_url=browser_result.url,
            browser_objective=browser_result.objective,
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
        browser_url=browser_result.url,
        browser_objective=browser_result.objective,
        browser_error=accept_reason,
    )


def _browser_request_key(result: ModelRunResult) -> tuple[str, str] | None:
    if result.error or result.quality is None:
        return None
    if result.quality.adequate or not result.quality.browser_would_help or not result.quality.browser_url:
        return None
    return (result.quality.browser_url, result.quality.browser_objective or "Extract data")


def _apply_shared_browser_refinements(
    q: QuestionSpec,
    per_model: dict[str, ModelRunResult],
    stacks: dict[str, ModelStack],
    *,
    costs: RunCost,
    test_mode: bool = False,
) -> dict[str, ModelRunResult]:
    requests: dict[tuple[str, str], list[str]] = {}
    for tag, result in per_model.items():
        key = _browser_request_key(result)
        if key is not None:
            requests.setdefault(key, []).append(tag)

    if not requests:
        for result in per_model.values():
            if result.quality and not result.quality.adequate and not result.browser_status.startswith("proposed"):
                result.browser_status = "not_useful" if not result.quality.browser_would_help else "proposed_no_url"
        return per_model

    js_risk = _question_js_risk(q)
    as_of_dates = [_earliest_past_target(r.response) for r in per_model.values() if r.response is not None]
    as_of_dates = [d for d in as_of_dates if d]
    as_of_date = min(as_of_dates) if as_of_dates else None

    for (url, objective), requesters in requests.items():
        print(f"  [shared] Browser: {url}", flush=True)
        browser_result = browser_extract(
            url,
            objective,
            test_mode=test_mode,
            model_override=DEFAULT_BROWSER_MODEL,
            as_of_date=as_of_date,
            skip_jina=js_risk,
        )
        if not browser_result.success:
            err = browser_result.error or "no error message returned"
            print(f"  [shared] Browser failed: {err}")
            for tag in requesters:
                result = per_model[tag]
                result.browser_status = "extract_failed"
                result.browser_url = url
                result.browser_objective = objective
                result.browser_error = err
            continue

        for tag, result in list(per_model.items()):
            if result.response is None or result.quality is None:
                continue
            if tag not in stacks:
                continue
            result.browser_url = url
            result.browser_objective = objective
            per_model[tag] = _refine_model_with_browser_result(
                q,
                result,
                stack=stacks[tag],
                costs=costs,
                browser_result=browser_result,
                test_mode=test_mode,
            )

    return per_model


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
    validation = validate_response(response, q.expected_forecasts, unit_min=q.unit_min, unit_max=q.unit_max)
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
            result.browser_status = "not_useful"
        elif not quality.browser_url:
            print(f"  [{stack.tag}] Browser skipped: no URL proposed ({quality.confidence}% confidence)")
            result.browser_status = "proposed_no_url"
        else:
            # Browser extraction always runs deferred/shared at the question level (process_question).
            print(f"  [{stack.tag}] Browser proposed for shared extraction ({quality.confidence}% confidence)")
            result.browser_status = "proposed_shared"
            result.browser_url = quality.browser_url
            result.browser_objective = quality.browser_objective or ""
    else:
        print(f"  [{stack.tag}] Quality: {quality.confidence}% confidence")

    color = result.response.forecasts[0].color_code.value if result.response.forecasts else "N/A"
    print(f"  [{stack.tag}] {color} | {'ok' if result.validation['ok'] else 'warning'}", flush=True)
    return result


def process_question(
    q: QuestionSpec,
    use_browser: bool = True,
    test_mode: bool = False,
    *,
    mode: str = "gpt",
) -> QuestionRunResult:
    """Run the surveillance pipeline for one question across the selected model(s)."""
    costs = RunCost()
    gpt_stack, claude_stack = _build_stacks(test_mode)
    selected = {}
    if mode in ("gpt", "both"):
        selected["gpt"] = gpt_stack
    if mode in ("claude", "both"):
        selected["claude"] = claude_stack

    rc_stack = selected.get("gpt") or selected.get("claude")
    if rc_stack is not None:
        ensure_rc_source(q, rc_stack.eval_model)
        ensure_evidence_plan(q)

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            tag: ex.submit(_run_single_model_pipeline, q,
                           stack=stack, costs=costs,
                           use_browser=use_browser, test_mode=test_mode,
                           )
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

    if use_browser:
        per_model = _apply_shared_browser_refinements(
            q, per_model, selected, costs=costs, test_mode=test_mode,
        )

    if len(per_model) >= 2:
        consensus = compute_consensus(per_model, q.question_type)
    else:
        consensus = {"status": "single_model_only", "color_agreement": False, "q50_agreement": False, "row_diffs": []}
    return QuestionRunResult(per_model=per_model, costs=costs, consensus=consensus)


def _resolve_mode_flag(args) -> str:
    """Pick run mode from --gpt/--claude/--both. Default = both."""
    return getattr(args, "mode_flag", None) or "both"


def cmd_run(args):
    # Derive recorded models from the same stacks the pipeline actually runs with,
    # so run metadata can't drift from reality.
    gpt_stack, claude_stack = _build_stacks(args.test_mode)
    active_model = gpt_stack.research_model
    active_claude = claude_stack.research_model
    mode = _resolve_mode_flag(args)
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

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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
        # The final writers mkdir too, but the partial writer runs first — on a fresh
        # checkout `outputs/` doesn't exist yet and the first completion would crash.
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        partial_path = os.path.join(DEFAULT_OUTPUT_DIR, f"run_{run_id}_partial.json")
        completed_payloads: dict[int, dict] = {}  # serialize each finished question once, not O(n²)
        for fut in as_completed(futures):
            i, result, err = fut.result()
            if err:
                errors_list[i] = err
            else:
                per_model_lists[i] = result.per_model
                costs_list[i] = result.costs
                consensus_blocks[i] = result.consensus
            completed_payloads[i] = {
                "id": questions[i].id, "name": questions[i].name,
                "per_model": {m: _serialize_model_result(r) for m, r in per_model_lists[i].items()} if per_model_lists[i] else None,
                "consensus": consensus_blocks[i], "cost": costs_list[i].as_dict() if costs_list[i] else None,
                "error": errors_list[i],
            }
            completed = [completed_payloads[j] for j in sorted(completed_payloads)]
            tmp_path = partial_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump({"run_id": run_id, "questions": completed}, f, indent=2)
            os.replace(tmp_path, partial_path)

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
    run_data = enrich_with_run_stability(run_data, DEFAULT_OUTPUT_DIR)
    run_data = enrich_with_value_changes(run_data, DEFAULT_OUTPUT_DIR)
    json_path = write_json_output(run_data, DEFAULT_OUTPUT_DIR)
    try:
        os.remove(partial_path)
    except OSError:
        pass
    write_csv_output(run_id, questions, per_model_lists, DEFAULT_OUTPUT_DIR)
    s = run_data["summary"]
    print(f"\nSaved: {json_path}")
    model_errs = s.get("model_error_count", 0)
    model_err_str = f", {model_errs} model error(s)" if model_errs else ""
    qbm = s.get("quality_by_model") or {}
    qbm_str = ", ".join(f"{tag} {n}" for tag, n in sorted(qbm.items())) or "none"
    print(
        f"Summary: {s['ok_count']}/{s['question_count']} validation OK, "
        f"{s['quality_issue_count']} questions with quality issues ({qbm_str} inadequate), "
        f"{s['error_count']} errors{model_err_str}, {s['browser_count']} browser extractions"
    )
    print(f"Due/unresolved (past-dated, no authoritative value found): {s['due_unresolved_count']}")
    print(f"Estimated cost: ${s['total_cost']:.4f}")

    if not args.no_sheet:
        try:
            n = publish_to_sheet(run_data, DEFAULT_SHEET_ID)
            print(f"Published {n} rows to Sheet tab '{run_tab_name(run_id)}'")
        except Exception as e:
            print(f"Sheet publishing failed: {e}")
    else:
        review_path = write_review_csv(run_data, DEFAULT_OUTPUT_DIR)
        print(f"Sheet skipped; review rows written to {review_path}")


def cmd_setup(args):
    if not args.yes:
        confirm = input("Rebuild the Instructions tab from current code? Per-run review tabs are not touched. [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return
    setup_sheet(DEFAULT_SHEET_ID)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leap-surveillance",
        description="LEAP Surveillance - forecasts for expert panel questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run --limit 5        Run surveillance on 5 questions
  %(prog)s run --limit 5 -y     Same, skip confirmation
  %(prog)s sync                 Sync latest run tab to BigQuery
  %(prog)s sync --tab run_...   Sync a specific run tab to BigQuery
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
    run_parser.add_argument(
        "--no-sheet",
        action="store_true",
        help="Skip publishing to Sheet; write the review rows to run_<run_id>_review.csv instead",
    )
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
                              help="Run Claude Opus only")
    model_group.add_argument("--both", dest="mode_flag", action="store_const", const="both",
                              help="Run both GPT and Claude in parallel (default)")

    sync_parser = subparsers.add_parser("sync", help="Sync a run tab to BigQuery")
    sync_parser.add_argument("--no-bq", action="store_true", help="Dry-run: print what would be written; no write")
    sync_parser.add_argument("--tab", type=str, default=None, help="Specific run_<run_id> tab to read (default: most recent)")

    setup_parser = subparsers.add_parser("setup", help="Rebuild the Instructions tab (does not touch run_* tabs)")
    setup_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        raise SystemExit(cmd_run(args) or 0)
    elif args.command == "sync":
        raise SystemExit(cmd_sync(args) or 0)
    elif args.command == "setup":
        raise SystemExit(cmd_setup(args) or 0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
