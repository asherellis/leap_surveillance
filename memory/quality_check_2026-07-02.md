# Code-quality audit — 2026-07-02

A thorough read of the `leap_surveillance/` package (RA-written, plus recent
resolution/conditional edits). Findings from four independent reviewers, deduped and
prioritized. `[verified]` = confirmed against the source during this pass; others are
reported by review (high confidence but not line-verified).

Each item has an ID (`ST`=storage, `QMC`=questions/models/consensus/common, `RE`=research,
`IO`=sheets/run_surveillance/sync/browser) so they can be turned into issues individually.

## Summary (HIGH first) — Resolution column updated 2026-07-03

Legend: ✅ fixed · 🟡 partial / mitigated · ⏸ AWAITING DECISION (consult Nadja) · ❌ not a bug
(auditor claim rejected on verification) · 📋 deferred (documented, not fixed).

| ID | Sev | File:line | One-liner | Resolution |
|----|-----|-----------|-----------|------------|
| QMC-1 | HIGH | questions.py:197 | NaN dimension → `"nan"` key; breaks dim_question_id mapping | ✅ `is_empty`+`safe_str` guard, mirrors the sibling set-builder |
| ST-1 | HIGH | storage.py:140,537 | CSV `fieldnames=rows[0].keys()` crashes on heterogeneous rows | ✅ union of all row keys + `restval=""` |
| ST-2 | HIGH | storage.py:764,929 | `key.split("\|")` unpack crashes on a key without `"\|"` | ✅ guard + warn + skip in both writers |
| RE-1 | HIGH | research.py:463,539 | Unbounded rate-limit retry loop | ✅ `MAX_RATE_LIMIT_RETRIES` (env-tunable, default 5) on both providers |
| RE-2 | HIGH | research.py:499,555,1076 | Parse-failure retry re-runs the whole paid web-search call | ⏸ AWAITING DECISION — recommend leave-as-is+document (strict schemas make it rare; bounded at 3); alternative: cheap re-emit follow-up (needs live test) |
| IO-1 | HIGH | run_surveillance.py:334 | Browser refinement applied to ALL models, not just requesters | ⏸ AWAITING DECISION — recommend requesters-only; alternative: keep sharing but fix the 2-URL overwrite |
| IO-2 | HIGH | sheets.py:288 | "Latest run" = lexicographic max; `run_backup` tab could sync to BQ | ✅ strict `^run_\d{8}_\d{6}$` filter |
| IO-4 | HIGH | run_surveillance.py:686 | CLI never `sys.exit`s → failures exit 0 | ✅ `raise SystemExit(cmd(args) or 0)` for run/sync/setup |
| IO-8 | HIGH | browser.py:58 | SSRF guard never resolves hostname | ✅ resolves via `getaddrinfo`, checks every IP (private/reserved/loopback/link-local), blocks `*.internal`; unresolvable → blocked |
| QMC-2 | MED | models.py + research.py | `-999` forecast sentinel illegal under unit bound | ✅ (2026-07-02) fully nullable value fields; bounds on number branch; pending `check_nullable_schema.py --live` |
| ST-3 | MED | storage.py:743,911 | Unguarded `float(v)` crashes writer | ✅ `_safe_float_or_none` via shared `_q50_forecast` helper |
| ST-4 | MED | storage.py:786+ | Bare `date.fromisoformat(fdate)` can abort writer | ✅ `_parse_date_or_none(fdate)` fallback; row skipped+warned only if BOTH source date and fdate unparseable |
| ST-5 | MED | storage.py:408 | value_changed compares GPT vs prior run matched on a different model | ✅ filters prior runs to same-GPT-model (matches stability enricher semantics) |
| ST-6 | MED | storage.py:363,402 | Prior-run history read from disk twice | ✅ module-level cache keyed by (dir, run_id, models) |
| ST-7 | MED | storage.py writers | Repeated linear rescans of forecasts | 🟡 duplication removed via shared `_q50_forecast` (folds ST-11); full dict-index deferred — N is small |
| RE-3 | MED | research.py:1034 | refine can switch provider silently | ❌ NOT A BUG — call site already threads `model=stack.research_model`; `_build_stacks` keeps provider in test mode. The `None` fallback is a latent hazard for future callers only (noted) |
| RE-4 | MED | research.py:988 | Refine prompt drops guardrails | 🟡 added run_date + LEAP anti-anchoring to refine prompt; full shared-builder consolidation deferred (RE-7/8) |
| RE-5 | MED | research.py:458,534 | Timeout detected by substring only | ✅ typed `APITimeoutError` check first, substring as fallback (`_is_timeout_error`) |
| RE-6 | MED | common.py:71 | Cost silently $0 for unknown model | 🟡 warns once per unpriced model; web-search/cache token accounting deferred (needs price data) |
| QMC-3 | MED | questions.py:26 | probability-unit + no-dates misrouted to `when` | ✅ explicit `return "quantile"` after warning; no-dates case then skips via the empty-expected guard |
| QMC-4 | MED | consensus.py:197 | quantile auto-accepts regardless of q50 divergence | ⏸ AWAITING DECISION — recommend gating quantile auto-accept on q50 too (tolerance already exists) |
| QMC-6 | MED | models.py:559 | `mixed_colors` false-positive on failed/unresolved resolution rows | ✅ those groups exempted from mixed-colors check |
| IO-6 | MED | sync.py:41 | Partial BQ-sync failure still exits 0 | ✅ failures counted; `cmd_sync` returns 1; MERGEs are idempotent so re-run reconciles (atomicity itself unchanged) |
| IO-5 | MED | run_surveillance.py:553 | O(n²) partial-progress serialization | ✅ per-question payload cached once (`completed_payloads`) |
| IO-7 | MED | run_surveillance.py:472 | Two sources of truth for "which models ran" | ✅ `run_models` derived from `_build_stacks(test_mode)` |
| ST-10 | MED | storage.py:781 | Reviewed value + blank status → silently "confirmed" | ⏸ AWAITING DECISION — recommend keep-current (value implies confirm) + document in sheet instructions; alternatives: projected, or skip+warn |

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

## LOW / cleanup — with resolutions (2026-07-03)

- **ST-8 / IO-11 / IO-12 — run_id format + timezone.** 🟡 partial: `run_id` generation and
  `_earliest_past_target` switched to **UTC** (consistent with `date_value_type`/
  `row_resolution_status`). 📋 deferred: sorting history by `created_at` instead of filename,
  and a collision suffix on run_id (same-second double-run is unlikely; suffix would ripple
  through tab names / review ids — do deliberately if ever needed).
- **ST-9** ✅ dead `"resolved"` branch removed (`resolved_at = now if status == "confirmed"`).
- **ST-11** ✅ folded: `q50_for`/`color_for`/`effective_color`/`black_q50_avg` all share one
  `_q50_forecast()` helper.
- **ST-12** ✅ uses `enum_value` now.
- **ST-13** ✅ `all_dims` includes prior-run dims, so a disappeared value flags
  `value_changed=True`.
- **ST-14/15/16** — ST-15 ✅ (`.get("run_id", "")`). ST-14 📋 (`_combine_stability` "one_stable"
  label semantics — confirm intended). ST-16 📋 (MERGE strict `>` on equal clocks — fine for
  run-scoped PKs).
- **QMC-5** ✅ redundant explicit `$defs` loop removed (generic recursion covers it).
- **QMC-8** ❌ NOT A BUG in practice: within a question type the statuses that can co-occur
  make a status-mismatch-with-equal-values impossible (resolved-vs-failed already differ on
  value; failed and unresolved don't co-occur across quantile vs timing). No change.
- **QMC-9** ❌ MOSTLY NOT A BUG: the domain string `leap.forecastingresearch.org` DOES match
  inside bare URLs, so the sources half of the anchor check works; only the `"leap wave"`
  text match is dead there (still covered by the rationale scan). No change.
- **QMC-11** ✅ renamed `common.resolution_status` → `row_resolution_status` (+docstring
  contrasting it with the `ResolutionStatus` enum); call sites in sheets.py/storage.py updated.
- **QMC-10 / QMC-12** — QMC-12 ✅ unused `dimensions` param removed from
  `_infer_surveillance_question_type`. QMC-10 📋 disregarded (iterrows on tiny per-group data;
  vectorizing adds noise, not value).
- **RE-7 / RE-8** 📋 deferred: consolidating the three provider dispatches into one
  `_call_model` and unifying FULL/BRIEF prompt constants is the right refactor but too large
  to do blind (no live API tests available). Do when next touching research.py with keys.
- **RE-9/10/11/12/13/14** — RE-9 ✅ shared `_is_retryable_parse_error` (adds "expected ident"
  to the GPT path). RE-10/11/12/13 📋 deferred (web_search tool version, streaming for 64k,
  client reuse, broad-except narrowing — all behavior-preserving improvements needing live
  validation). RE-14 📋 harmless (downstream gates on `browser_would_help`).
- **IO-3** ✅ dead `_try_browser_refinement_for_model` + unreachable `else` branch + the
  always-true `defer_browser` param deleted; browser extraction is now explicitly
  shared/deferred-only.
- **IO-9** ⏸ AWAITING DECISION (minor): single-model runs mark every row
  `needs_review=TRUE` because `single_model_only != auto_accepted`. Likely fine (single-model
  output HAS no consensus check), but confirm.
- **IO-10 / IO-16** — IO-10 ✅ three unused params removed from
  `_should_accept_browser_refinement`. IO-16 📋 kept: `group_id` fallback reads legacy tabs;
  `row_numbers` return value harmless.
- **IO-13/14/15/17** — IO-14 ✅ `'"error":'` only treated as an error payload when the
  extraction is a JSON blob (`lstrip().startswith("{")`). IO-15 ✅ extension check parses the
  URL path (query strings can't slip through). IO-13 📋 (merge Sheets batch_updates —
  optimization). IO-17 📋 (dedupe Jina fetch helper — do when next touching browser.py).

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

## Status after fix pass (2026-07-03)

**Fixed & verified** (imports clean; retrieval + nullable-schema suites all-pass; per-fix
spot tests pass): QMC-1/3/5/6/11/12, ST-1/2/3/4/5/6/9/11/12/13/15, RE-1/5/9 (+RE-4/6
partial), IO-2/3/4/5/6/7/8/10/14/15, and QMC-2 from the previous pass.

**⏸ Awaiting Nadja's decision** (recommendations in the table above):
1. **QMC-4** — gate quantile auto-accept on q50 agreement? (recommend: yes)
2. **IO-1** — browser refinement: requesters-only vs shared-with-overwrite-fix?
   (recommend: requesters-only)
3. **ST-10** — reviewed value + blank status: keep "confirmed" / projected / skip+warn?
   (recommend: keep confirmed + document)
4. **RE-2** — parse-failure retry: leave-as-is+document vs cheap follow-up rewrite?
   (recommend: leave; rare under strict schemas, bounded at 3)
5. **IO-9** (minor) — single-model runs flag all rows needs_review; confirm intended.

**❌ Rejected auditor claims** (checked, not bugs): RE-3 (model IS threaded via
`stack.research_model`), QMC-8 (status-mismatch scenario can't occur), QMC-9 (domain check
works on URLs).

**📋 Deferred** (documented, do when touching the area / with live API keys): full forecast
index (ST-7 remainder), provider-dispatch + prompt-constant consolidation (RE-7/8),
web_search version / streaming / client reuse / broad-except narrowing (RE-10..13),
created_at-based history sort + run_id collision suffix (ST-8/IO-12), Sheets batch merge
(IO-13), Jina dedupe (IO-17), web-search cost accounting (RE-6 remainder).

**Needs live API test when keys available**: `python check_nullable_schema.py --live`
(QMC-2 nullable schema acceptance on both providers).

## Post-audit finds (from the first fresh-machine run, 2026-07-03)
- **IO-18** ✅ HIGH — partial-progress writer crashed with `FileNotFoundError` on a fresh
  checkout: it wrote `outputs/run_*_partial.json.tmp` before `outputs/` existed (only the
  END-of-run writers mkdir'd). Both questions' results were lost. Fixed:
  `os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)` before the loop. Classic
  worked-on-the-author's-machine bug (dir pre-existed there).
- **IO-19** ✅ — pyproject.toml missing runtime deps on a fresh venv:
  `google-cloud-bigquery` needs the `[pandas]` extra (`db-dtypes`/`pyarrow` for
  `to_dataframe`), and `requests` was only transitive. Both declared now.
- **ASSUMPTION (documented, not changed)** — `DEFAULT_OUTPUT_DIR="outputs"` and `.env`
  loading are cwd-relative: the CLI must be launched from the repo root.
- **RE-15** ✅ — Claude's `produce_forecast` tool calls could come back structurally
  malformed (observed: missing required `sources` on the Coffee Test in test mode),
  burning a parse-retry that re-runs the paid search (RE-2's cost). Root fix applied:
  **Anthropic strict tool use** (`strict: true` on the tool definition, GA, no beta header;
  supported on Opus 4.8 + Haiku 4.5) — the API now guarantees tool inputs validate against
  the schema, matching the guarantee the OpenAI path already had. Numeric bounds
  (`minimum`/`maximum`, unsupported under strict) are stripped from the Anthropic copy via
  `models.strict_tool_schema()`; bounds remain in the prompt and are enforced post-hoc by
  `validate_response`. Applied to both `_claude_research` and `refine_with_browser`.
  **Pending live test:** `check_nullable_schema.py --live` now exercises strict + nullable
  together (needs a valid ANTHROPIC_API_KEY).
