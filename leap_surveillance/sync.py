"""Sheet review sync into BigQuery tables."""

import json
import os

from .common import DEFAULT_DEV_SHEET_ID, DEFAULT_OUTPUT_DIR, DEFAULT_SHEET_ID, safe_str
from .sheets import RUN_TAB_PREFIX, get_reviewed_items, promote_tab_to_prod, run_tab_name
from .storage import (
    write_to_fact_resolution,
    write_to_dim_baseline,
    write_to_surveillance_result,
)


def _print_sheet_rows(all_items: list[dict]) -> None:
    print(f"\n  {'#':>3}  {'question':40s} {'date':12s} {'dim':20s} {'verdict':18s} {'reviewed':8s} rlov  res")
    for i, item in enumerate(all_items, 1):
        name = (item.get("question_name") or "")[:38]
        tdate = (item.get("target_date") or "")[:10]
        dim = (item.get("dimension") or "")[:18]
        verdict = (item.get("review_verdict") or "")[:16]
        rval = safe_str(item.get("review_last_official_value") or "")
        resval = safe_str(item.get("reviewed_question_resolution_value") or item.get("question_resolution_value") or "")
        reviewed = "yes" if item.get("reviewed") else ""
        print(f"  {i:>3}  {name:40s} {tdate:12s} {dim:20s} {verdict:18s} {reviewed:8s} {rval}  {resval}")


def _load_run_data(run_id: str) -> dict | None:
    json_path = os.path.join(DEFAULT_OUTPUT_DIR, f"run_{run_id}.json")
    try:
        with open(json_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  warning: run_{run_id}.json not found; skipping BigQuery writes for this run")
        return None
    except Exception as e:
        print(f"  warning: could not load run_{run_id}.json: {e}")
        return None


def _write_run_items(run_id: str, items: list[dict]) -> int:
    """Write one run's reviewed rows to the three BQ tables. Returns the number of failed writes."""
    run_data = _load_run_data(run_id)
    if run_data is None:
        return 1

    failures = 0

    try:
        base_stats = write_to_dim_baseline(run_data, items)
        if base_stats is not None:
            print(f"BigQuery dim_baseline ({run_id}): {base_stats.inserted_row_count} inserted, {base_stats.updated_row_count} updated")
        else:
            print(f"BigQuery dim_baseline ({run_id}): no rows")
    except Exception as e:
        failures += 1
        print(f"BigQuery dim_baseline write failed ({run_id}): {e}")

    try:
        fr_stats = write_to_fact_resolution(run_data, items)
        if fr_stats is not None:
            print(f"BigQuery fact_resolution ({run_id}): {fr_stats.inserted_row_count} inserted, {fr_stats.updated_row_count} updated")
        else:
            print(f"BigQuery fact_resolution ({run_id}): no rows written")
    except Exception as e:
        failures += 1
        print(f"BigQuery fact_resolution write failed ({run_id}): {e}")

    try:
        sr_stats = write_to_surveillance_result(run_data, items)
        if sr_stats is not None:
            print(f"BigQuery surveillance_result ({run_id}): {sr_stats.inserted_row_count} inserted, {sr_stats.updated_row_count} updated")
        else:
            print(f"BigQuery surveillance_result ({run_id}): no rows")
    except Exception as e:
        failures += 1
        print(f"BigQuery surveillance_result write failed ({run_id}): {e}")

    return failures


def cmd_sync(args) -> int:
    if not DEFAULT_DEV_SHEET_ID:
        print("LEAP_DEV_SHEET_ID is not set (see .env.example) - sync reads from the Dev sheet, nothing to do without it.")
        return 1
    if not DEFAULT_SHEET_ID:
        print("LEAP_SHEET_ID is not set (see .env.example) - sync promotes to the Prod sheet, nothing to promote to without it.")
        return 1

    all_items, _ = get_reviewed_items(DEFAULT_DEV_SHEET_ID, tab_name=args.tab, reviewed_only=False)

    expected_run_id = args.tab.removeprefix(RUN_TAB_PREFIX)
    mismatched = [r for r in all_items if r.get("run_id") != expected_run_id]
    if mismatched:
        bad_ids = sorted({r.get("run_id") for r in mismatched})
        print(f"Aborting: {len(mismatched)} row(s) on tab {args.tab} carry a run_id that doesn't match the tab ({bad_ids}) - the tab's surveillance_timestamp cells may have been edited or copied from elsewhere.")
        return 1

    reviewed_count = sum(1 for r in all_items if r.get("reviewed"))
    rlov_count = sum(1 for r in all_items if r.get("review_last_official_value") not in (None, ""))
    rres_count = sum(1 for r in all_items if r.get("reviewed_question_resolution_value") not in (None, ""))
    print(f"Found {len(all_items)} total rows, {reviewed_count} reviewed, {rlov_count} with rlov, {rres_count} with reviewed resolution")

    if not all_items:
        print("Nothing to sync.")
        return 0

    _print_sheet_rows(all_items)

    by_run: dict[str, list[dict]] = {}
    for item in all_items:
        by_run.setdefault(item.get("run_id", ""), []).append(item)

    total_failures = 0
    for run_id, items in by_run.items():
        total_failures += _write_run_items(run_id, items)
        # promotion happens regardless of the BQ outcome above - a warehouse hiccup shouldn't block
        # preserving the reviewed record in Prod, which is the only path a tab ever reaches Prod.
        try:
            promote_tab_to_prod(DEFAULT_DEV_SHEET_ID, DEFAULT_SHEET_ID, run_id)
            print(f"Promoted {run_tab_name(run_id)} to Prod")
        except Exception as e:
            total_failures += 1
            print(f"Promote to Prod failed ({run_id}): {e}")

    if total_failures:
        print(f"\nsync finished with {total_failures} failed operation(s) — re-run after fixing; BQ writes are idempotent MERGEs, promotion overwrites cleanly.")
        return 1
    return 0
