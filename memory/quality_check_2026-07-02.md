# Code-quality audit — 2026-07-02

A thorough read of the `leap_surveillance/` package (RA-written, plus recent
resolution/conditional edits). Findings from four independent reviewers, deduped and
prioritized. `[verified]` = confirmed against the source during this pass; others are
reported by review (high confidence but not line-verified).

Each item has an ID (`ST`=storage, `QMC`=questions/models/consensus/common, `RE`=research,
`IO`=sheets/run_surveillance/sync/browser) so they can be turned into issues individually.

## Summary (HIGH first)

| ID | Sev | File:line | One-liner |
|----|-----|-----------|-----------|
| QMC-1 | HIGH | questions.py:197 | NaN dimension → `"nan"` key; breaks dim_question_id mapping for "Overall" rows |
| ST-1 | HIGH | storage.py:140,537 | CSV `fieldnames=rows[0].keys()` crashes on heterogeneous rows |
| ST-2 | HIGH | storage.py:764,929 | `key.split("|")` unpack crashes on a key without `"|"` |
| RE-1 | HIGH | research.py:463,539 | Unbounded rate-limit retry loop (`sleep(60); continue`, no ceiling) |
| RE-2 | HIGH | research.py:499,555,1076 | Parse-failure retry re-runs the whole paid web-search call |
| IO-1 | HIGH | run_surveillance.py:334 | Browser refinement applied to ALL models, not just requesters |
| IO-2 | HIGH | sheets.py:288 | "Latest run" = lexicographic max; a `run_backup` tab syncs to BQ |
| IO-4 | HIGH | run_surveillance.py:686 | CLI never `sys.exit`s → failures/`return 2` exit 0 |
| IO-8 | HIGH | browser.py:58 | SSRF guard never resolves hostname (DNS → metadata/internal) |
| QMC-2 | MED | models.py:283 + research.py:255 | `-999` forecast sentinel is illegal under `unit_min` bound → dead path |
| ST-3 | MED | storage.py:743,911 | Unguarded `float(v)` on model value crashes the whole writer |
| ST-4 | MED | storage.py:786,794,803 | Bare `date.fromisoformat(fdate)` can raise and abort writer |
| ST-5 | MED | storage.py:408 | value_changed compares GPT vs a prior run matched on a different model |
| ST-6 | MED | storage.py:363,402 | Prior-run history read + JSON-parsed from disk twice per run |
| ST-7 | MED | storage.py:718-755,906-919 | Repeated linear rescans of forecasts (~4 per key); should index |
| RE-3 | MED | research.py:1034 | `refine_with_browser` can switch provider (Claude→OpenAI) silently |
| RE-4 | MED | research.py:988 | Refine prompt drops LEAP anti-anchoring + RC-source + run_date guardrails |
| RE-5 | MED | research.py:458,534 | Timeout detected by substring match, not typed exception |
| RE-6 | MED | research.py:470 + common.py:71 | Cost silently $0 for unknown model; web-search/cache tokens uncounted |
| QMC-3 | MED | questions.py:26 | probability-unit + no-dates misclassified as `when` (message says quantile) |
| QMC-4 | MED | consensus.py:197 | quantile rows can auto_accept with wildly different q50 (by design?) |
| QMC-6 | MED | models.py:559 | `mixed_colors` false-positive on unresolved (nulled) resolution rows |
| IO-6 | MED | sync.py:41 | 3 BQ writes non-atomic; partial failure still exits 0 |
| IO-5 | MED | run_surveillance.py:553 | Partial-progress JSON rewritten in full each future → O(n²) |
| IO-7 | MED | run_surveillance.py:472 | Two sources of truth for "which models ran" can drift |
| ST-10 | MED | storage.py:781 | Reviewed value + blank status silently becomes "confirmed" |

Plus LOW / cleanup items listed at the end.

---

## HIGH

### QMC-1 — NaN dimension produces a `"nan"` map key `[verified]`
`questions.py:197` — `dim = str(r.get("question_dimension") or "Overall").strip() or "Overall"`.
A NULL BigQuery dimension arrives from pandas as float `NaN`, which is **truthy**, so
`NaN or "Overall"` → `NaN` → `"nan"`. The key becomes `"{fdate}|nan"` while expected
forecasts and the sheet lookup (`sheets.py:474`) use `"Overall"`. The sibling set-builder
(`questions.py:152-159`) correctly uses `is_empty`.
**Impact:** for single-dimension ("Overall") questions whose dim column comes back as NaN,
the `dim_question.question_id` never attaches to output rows (silently `""`). If the column
is a nullable dtype, `pd.NA or "Overall"` raises `TypeError`. Severity depends on whether BQ
returns `None` (fine) or `NaN` (broken) for that column — verify against real data.
**Fix:** `d = safe_str(r.get("question_dimension")).strip(); dim = d or "Overall"`.

### ST-1 — CSV writer crashes on heterogeneous rows `[verified]`
`storage.py:537` uses `fieldnames=rows[0].keys()`, but `_forecast_output_row`
(`storage.py:140-142`) only adds `validation_ok`/`usable_for_scoring` when `validation is
not None`. If `rows[0]` is an errored model (no validation) but a later row has it,
`DictWriter.writerows` raises `ValueError: dict contains fields not in fieldnames` and the
whole CSV write dies. Order-dependent → passes in tests, fails on mixed real data.
**Fix:** `fieldnames = list(dict.fromkeys(k for r in rows for k in r))`, or always emit the
two keys (default `""`/`False`).

### ST-2 — `key.split("|")` unpack can crash the writer
`storage.py:764` and `:929` — `fdate, dim = key.split("|", 1)` assumes every
`dim_question_map` key contains `"|"`. A malformed key → 1-element list → `ValueError`,
outside the `_try_merge_bigquery_rows` try/except, killing the writer.
**Fix:** `if "|" not in key: continue` (or `.partition("|")`) with a log.

### RE-1 — Unbounded rate-limit retry loop
`research.py:463-466` (`_claude_research`) and `:539-542` (`_gpt_research`) — on
`RateLimitError`: `time.sleep(60); continue` with **no counter/ceiling** (unlike the
timeout path bounded by `MAX_TIMEOUT_RETRIES`). A throttled key hangs one question forever
with no cost bound; this sits on top of the SDK's own 429 backoff.
**Fix:** add `MAX_RATE_LIMIT_RETRIES`; honor the `retry-after` header instead of fixed 60s.

### RE-2 — Parse-failure retry re-runs the entire paid search
`research.py:499`, `:555`, `:1076` — when `strict_to_regular_response` fails on malformed
JSON, the code recurses into the whole function (`attempt+1`), re-issuing the full
web-search research call (real $ per Claude call) up to 2 more times just to fix JSON, and
resets `timeout_attempt`/`search_round` each recursion.
**Fix:** on parse error send a cheap "re-emit valid JSON" follow-up rather than re-searching;
lift retry state out of the recursion. (Options: (a) follow-up message; (b) local reformat
via a cheap model.)

### IO-1 — Browser refinement applied to every model, not just requesters
`run_surveillance.py:334` — `requests` is keyed by `(url, objective)` with a `requesters`
list, but the refinement loop iterates `per_model.items()` (all models). An adequate model
that never asked for a browser lookup gets re-judged/re-refined against another model's URL
and can be overwritten; with two URLs the second pass overwrites the first; wasted
refine+judge cost.
**Fix:** iterate `for tag in requesters:` inside each `(url, objective)` block, keeping the
existing guards.

### IO-2 — "Latest run" chosen by lexicographic tab name
`sheets.py:288-293` (`_latest_run_tab`) uses `max(..., key=title)`. Any `run_*` tab whose
suffix isn't a zero-padded timestamp (`run_backup`, `run_final`) sorts after `run_2026…`
(`'b' > '2'`) and silently becomes "latest" → gets synced to BigQuery.
**Fix:** filter to `^run_\d{8}_\d{6}$` before `max`, or sort by parsed datetime; ignore
non-conforming tabs. (Related: ST-8 run_id format assumption.)

### IO-4 — CLI never surfaces a non-zero exit code `[verified]`
`run_surveillance.py:686-698` — `main()` calls `cmd_run/cmd_sync/cmd_setup` and discards
return values; no `sys.exit`. `cmd_run` `return 2` on missing API keys and all `sync` BQ
failures still exit 0 → cron/CI can't detect failure.
**Fix:** `raise SystemExit(cmd_run(args) or 0)`; have `cmd_sync` return non-zero on any BQ
write failure (ties to IO-6).

### IO-8 — SSRF guard never resolves the hostname
`browser.py:58-96` (`is_safe_url`) blocks only literal private IPs + a hardcoded domain
list; `ipaddress.ip_address(host)` raises `ValueError` for hostnames and is swallowed. A
hostname resolving to `127.0.0.1`/`169.254.169.254`/RFC1918 passes → the browser agent can
be pointed at internal services / cloud metadata.
**Fix:** resolve the host and re-check every resolved IP against private/reserved/link-local
ranges; block `*.internal`.

---

## MEDIUM

### QMC-2 — `-999` forecast sentinel is illegal under a unit bound (dead path) — ✅ FIXED 2026-07-02 (pending live API test)
`models.py:283-287` sets `forecast_value["minimum"]=unit_min`; the prompt
(`research.py:255,269`) tells the model to emit `-999` for "no estimate", and
`_convert_sentinel_value` maps `-999→None`. For any `unit_min>=0` question the strict schema
**forbids** `-999`, so the model can't legally emit it — the null path is dead and the model
is pushed to fabricate an in-range number.
**Resolution (chosen: full nullable, option b):** made `StrictOfficialValue.value`,
`StrictCurrentValue.value`, and `StrictForecastValue.forecast_value` `Optional[float]` (all
still required keys, now `anyOf:[number,null]`); `make_strict_schema` applies unit bounds to
the number branch via `_numeric_branch` (so null is allowed but non-null stays bounded);
prompts now say "set to null" instead of `-999`; `_convert_sentinel_value` and the
consensus `-999` skip-marker remain as defensive backstops for stray legacy values.
**Still to confirm:** run `check_nullable_schema.py --live` (repo root) with API keys — it
sends the real strict schema to GPT + Claude to verify both providers accept the
nullable+bounded field and can emit `null`. `9999`/`NEVER_YEAR` (timing "never") is
intentionally NOT changed — it is semantic, not a missing value.

### ST-3 — Unguarded `float(v)` on model-supplied values
`storage.py:743` (`black_q50_avg`), `:911` (`q50_for`) call `float(v)` directly (vs
`_safe_float_or_none` everywhere else); `"N/A"`/`"1,200"` raises `ValueError` and aborts the
writer (outside the try). **Fix:** use `_safe_float_or_none` and skip `None`.

### ST-4 — Bare `date.fromisoformat(fdate)` can raise
`storage.py:786,794,803` — `_parse_date_or_none(...) or date.fromisoformat(fdate)`; if the
first is `None` and `fdate` is non-ISO (any non-`TIMING_FORECAST_DATE` bad key), this raises
and crashes the writer. **Fix:** use `_parse_date_or_none(fdate)` then `continue`/log.

### ST-5 — value_changed compares GPT against a mismatched prior run
`storage.py:408` + `320-339` — `_read_production_history` admits a prior run if **any** tag
matched (`any(...)`), then `enrich_with_value_changes` takes `prior_runs[-1]` and compares
only GPT blocks. If the latest prior matched on Claude with a different/absent GPT model →
spurious `value_changed=True` or silent `False`. `enrich_with_run_stability` re-filters
per-tag (357) so the two enrichers disagree on matching semantics.
**Fix:** select the most recent prior whose `models["gpt"]` equals current GPT (per-tag).

### ST-6 — Prior-run history read twice
`storage.py:363,402` — both enrichers independently call `_read_production_history` (glob +
sort + up to 40 `json.loads`) → ~80 reads instead of 40. **Fix:** read once, pass the parsed
list to both.

### ST-7 — Repeated linear rescans of `forecasts`
`storage.py:718-755` and `906-919` — per (question × dim-date), `effective_color`,
`black_q50_avg`, `model_resolution_source`, and `q50_for`+`color_for` each re-walk the model
forecast list; ≈ O(Q·D·F·models) with ~4 redundant scans per key. Not a perf emergency (small
N) but it's the main efficiency + duplication smell (folds ST-11).
**Fix:** build one `dict[(quantile,fdate,dim)]→forecast` index per model, then O(1) lookups.

### RE-3 — `refine_with_browser` can switch provider silently
`research.py:1034` — refine defaults `model = TEST_MODEL if test_mode else DEFAULT_MODEL`
(both OpenAI), while `research_question` picks by provider. A Claude research run refined in
test mode switches to OpenAI; in prod it falls to `DEFAULT_MODEL` even if research ran on
Claude. **Fix:** thread the original research model into `refine_with_browser`.

### RE-4 — Refine prompt drops key guardrails
`research.py:988-1031` — the refine prompt omits `RESEARCH_PRINCIPLES`, the RC-named-source
directive, the JS-dashboard note, the **LEAP anti-anchoring** warning, and `run_date`. A
refinement pass can re-introduce LEAP anchoring / ignore the RC source / lose "today".
**Fix:** factor shared rule blocks into one builder used by both paths.

### RE-5 — Timeout detected by substring, not exception type
`research.py:458-459,534-535` — `"timeout" in error_str` misses real
`APITimeoutError`s and false-positives on unrelated messages. **Fix:** `except
(anthropic.APITimeoutError, openai.APITimeoutError)`.

### RE-6 — Cost accounting silently under-reports
`research.py:470,546,745,1054` + `common.py:71-78` — `cost_for_tokens` returns `0.0` for a
model not in the price table (substring match), and only counts input/output tokens (no
server-side web-search charges, no cache tokens). Claude research cost is understated; a
renamed model silently costs $0. **Fix:** warn/raise on missing price key; count web-search +
cache token fields.

### QMC-3 — probability-unit + no-dates misclassified as `when`
`questions.py:26-33` — when `unit_name=="probability"` but percentiles ≠ {50}, it prints
"handled as a quantile question" then falls through to `if not dates: return "when"`. A
no-dates probability-unit question becomes `when`, contradicting the message.
**Fix:** make it explicit: `return "when" if not dates else "quantile"` with an accurate log.

### QMC-4 — quantile rows auto-accept regardless of q50 (confirm intent)
`consensus.py:197` — for `quantile`, `all_rows_agree` excludes `q50_all_match` (only added
for when/probability). Two quantile models with very different medians can `auto_accepted` as
long as color + official/current agree. Looks intentional (official/current are the anchors)
but `q50_agreement` is still reported, so a reader assumes it gates acceptance. **Confirm; if
unintended, include q50 for quantile too.**

### QMC-6 — `mixed_colors` false-positive on unresolved resolution rows
`models.py:559-561` — flags any (date,dim) group with >1 color. For an *unresolved* past-date
group, `strict_to_regular_response` keeps the model's per-quantile colors (only `resolved`
forces uniform black), so differing colors on a nulled group raise a spurious
`mixed_colors_*`. (Touches the recent resolution rework.) **Fix:** skip color-consistency /
black checks for `value_type=="resolution"` rows whose status ≠ resolved.

### IO-6 — Non-atomic BQ sync, success-looking on partial failure
`sync.py:41-96` — three BQ writes each in their own print-only try/except; if dim_baseline
succeeds and fact_resolution fails, the warehouse is half-written, `cmd_sync` returns `None`,
exit 0. **Fix:** track failures, exit non-zero (ties IO-4); order/idempotency so re-run
reconciles.

### IO-5 — Partial-progress JSON rewritten in full each iteration (O(n²))
`run_surveillance.py:553-571` — inside `as_completed`, `completed` re-scans all indices and
re-serializes every finished question, then dumps the whole file, per completion. Quadratic
serialization + disk I/O for large batches. **Fix:** append the one finished question; rewrite
every k completions.

### IO-7 — Two sources of truth for "which models ran"
`run_surveillance.py:472-479` vs `_build_stacks` (`:75-85`) — recorded `run_models` metadata
is built separately from the models the pipeline actually uses; can drift (test-mode eval
models, browser navigator). **Fix:** derive `run_models` from `_build_stacks`.

### ST-10 — Reviewed value + blank status → silently "confirmed"
`storage.py:781` — a reviewer supplying `reviewed_question_resolution_value` with an empty
`reviewed_question_resolution_status` becomes a `confirmed` resolution stamped `resolved_at=
now`. **Fix:** require an explicit status before marking confirmed.

---

## LOW / cleanup

- **ST-8 / IO-11 / IO-12 — run_id format + timezone.** run ordering (`storage.py:322`),
  `_run_date_from_id` (`:824`), latest-tab (IO-2) all assume `YYYYMMDD_HHMMSS`; `run_id` and
  `_earliest_past_target` use **local** time while `resolution_status` uses **UTC**
  (near-midnight class-drift); second-resolution run_id can collide/overwrite. Fix: sort runs
  by `created_at` in the JSON; standardize on UTC; add a short uuid suffix to run_id.
- **ST-9 — storage.py:781-782** dead `"resolved"` branch (`resolution_status_value` is only
  projected/confirmed).
- **ST-11 — storage.py:906-919** `q50_for`/`color_for` are identical loops; fold into ST-7 index.
- **ST-12 — storage.py:109** uses `getattr(color,"value",color)` instead of `enum_value`.
- **ST-13 — storage.py:424** value-change `all_dims` from current only; misses dims dropped
  since prior run.
- **ST-14/15/16** — `_combine_stability` "one_stable" label edge; `run_data['run_id']` `[]` vs
  `.get`; MERGE strict `>` skips equal-clock re-syncs.
- **QMC-5 — models.py:253-269** `fix_schema` processes `$defs` twice (generic recursion + the
  explicit loop). Remove one.
- **QMC-8 — consensus.py:77-79** two models both failing to resolve count as agreement without
  comparing `resolution_status` (failed vs unresolved "agree"). (Touches recent rework.)
- **QMC-9 — models.py:403 vs 415-438** `response.sources` is URLs-only, but
  `_contains_leap_forecast_anchor` scans it for "leap wave"/domain text → sources half of the
  anchor check is near-dead; only the rationale scan works.
- **QMC-11 — common.py:141 `resolution_status()`** returns a *different* vocabulary
  (`resolved/resolved_early/due_unresolved/forecast`) than the `ResolutionStatus` enum
  (`resolved/failed/unresolved`). Both live → foot-gun. Consider renaming the row-status one to
  `display_row_status`.
- **QMC-10 / QMC-12** — `iterrows` in `dim_q_map` build could be a vectorized filter (small N);
  `_infer_surveillance_question_type` has unused `dimensions` param and `question_text`
  (warning-only).
- **RE-7 / RE-8 — research.py** three hand-rolled provider dispatches (`_claude_research`/
  `_gpt_research`/`_call_structured_judge`/`refine_with_browser`) and parallel FULL/BRIEF
  prompt constants — the main drift surface; consolidate into one `_call_model` + one canonical
  prompt block.
- **RE-9/10/11/12/13/14** — `_gpt_research` retry predicate omits `"expected ident"`;
  `web_search_20250305` (basic) hardcoded regardless of model; `max_tokens=64000`
  non-streaming relies on the explicit timeout to bypass the SDK long-request guard, and
  retry/timeout layers compound wall-clock; fresh SDK client per call; broad `except Exception`
  can mask programming errors; `decide_browser` returns a URL even on a `False` decision.
- **IO-3 — run_surveillance.py:177-207** `_try_browser_refinement_for_model` + its `else`
  branch (`:399`) are unreachable dead code (defer_browser is always set).
- **IO-9 — sheets.py:264-285** single-model runs flag every row `needs_review=TRUE`
  (`single_model_only != auto_accepted`). Confirm intended.
- **IO-10 / IO-16 — run_surveillance.py:141** `_should_accept_browser_refinement` has 3 unused
  params; `sheets.py:914` `row.get("group_id")` legacy column; `sync.py:75` discards
  `row_numbers`.
- **IO-13/14/15/17** — publish makes several sequential Sheets `batch_update`s (mergeable);
  `browser.py:209` `'"error":' in extracted` false-rejects pages containing that text;
  `browser.py:140` extension check ignores query strings (`report.csv?dl=1`); Jina fetch/
  validation duplicated between `wayback_snapshot` and `browser_extract`.

---

## Cross-cutting themes
1. **Crashes outside try/except in the BQ writers** (ST-1/2/3/4) — several unguarded
   conversions can abort a whole write; the writers need consistent `_safe_*` usage + skips.
2. **Silent failure / exit-0** (IO-4/6, RE-6) — the pipeline under-signals failure to
   cron/CI and under-reports cost.
3. **research_question vs refine_with_browser drift** (RE-3/4/7/8) — the two model-call paths
   diverge on model, guardrails, and prompt constants; consolidate.
4. **Unvoiced assumptions** — dimension defaults to "Overall" (NaN-unsafe in one spot),
   ISO dates, `run_id`=timestamp & lexically sortable, gpt-before-claude precedence,
   `TIMING_FORECAST_DATE` as the sole when-date. Worth a documented invariants list.
5. **Two "resolution status" vocabularies** (QMC-11) coexist — rename one.

## Suggested triage order
Fix now (cheap + real): QMC-1, ST-1, ST-2, ST-3, ST-4, IO-4, RE-1, RE-5, QMC-3, QMC-6.
Design decision needed: QMC-2, QMC-4, ST-5, IO-1, RE-2, RE-3/4.
Security: IO-8 (SSRF) — fix before any untrusted-URL browsing.
Cleanup/refactor when touching the area: ST-6/7, RE-7/8, IO-3/5, common.py vocab.
