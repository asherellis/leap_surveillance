# Refactoring / Code Review Conversation — LEAP Surveillance

_Date: 2026-07-02_

A code-review walkthrough of the `leap_surveillance` codebase (written by a research
assistant), tracing data flow, resolution logic, and how LLM research is requested.

---

## Q1 — What's the source of `fdate`?

`fdate` is a **local variable**, not a stored field. Everywhere it appears it's shorthand
for a **forecast date**, obtained one of two ways:

- **Split from a `"fdate|dim"` composite key** — `fdate, dim = key.split("|", 1)`
  (storage.py:764/929)
- **Read off a forecast dict** — `forecast.get("forecast_date", "")` (sheets.py:341, storage.py)

Both trace back to the same underlying `forecast_date` field.

### Root source: the `question_resolution_date` input column

1. **`questions.py:206-207`** — `fdate` is first built from the input dataframe column
   `question_resolution_date`:
   ```python
   raw_date = r.get("question_resolution_date")
   fdate = str(raw_date).strip() if not is_empty(raw_date) else TIMING_FORECAST_DATE
   ```
   For timing/"when"-type questions the date is NULL, so it becomes the
   `TIMING_FORECAST_DATE` placeholder.

2. These dates populate `ExpectedForecast.forecast_date` (models.py:17) via
   `_expected_forecasts(...)`, and the sorted set becomes `allowed_forecast_dates`
   (research.py:391).

3. **`models.py:265-266`** — that list is injected as an `enum` constraint into the LLM's
   output schema:
   ```python
   properties["forecast_date"]["enum"] = allowed_forecast_dates
   ```

4. The LLM then emits `forecast_date` on each `ForecastValue`/`ResolutionValue`
   (models.py:156/146), echoed back downstream as `fdate`.

**Summary:** `fdate` originates from the `question_resolution_date` column of the source
question data (with `TIMING_FORECAST_DATE` substituted for null dates), is passed to the LLM
as an allowed-value enum, and is echoed back on each forecast the LLM produces.

---

## Q2 — What does `ExpectedForecast` do?

An `ExpectedForecast` is the **specification of one forecast cell the model is required to
produce** — a `(forecast_date, dimension, quantile, value_type)` tuple (models.py:15-20).
The list of them on each `QuestionSpec` is the "answer key shape."

Built once per question in **`_expected_forecasts()`** (questions.py:80), where the cartesian
product differs by question type:

- **`probability`** → one row per (date, dim) at quantile 50
- **`when`** (timing) → one row per (dim, quantile) using the `TIMING_FORECAST_DATE` placeholder
- **quantile (default)** → one row per (date, dim, quantile) across the full quantile set

It drives three things:

1. **Constrains the LLM's output schema** (research.py:389-391 → models.py:265-274) — distinct
   dimensions, quantiles, forecast_dates become `enum` constraints.
2. **Tells the LLM what to fill in** — rendered into research + judge prompts as a checklist
   (`_expected_forecast_lines`, research.py:326).
3. **Validates the response** (`validate_response`, models.py:421-463) — expected set compared
   against returned to flag missing / unexpected / incomplete rows:
   ```python
   expected_keys = {(e.forecast_date, e.dimension) for e in expected}
   missing_keys    = expected_keys - forecast_keys
   unexpected_keys = forecast_keys - expected_keys
   missing_q50s    = expected_keys - q50s
   ```

**Summary:** `ExpectedForecast` is the contract for a question's required output — defines the
grid of cells, shapes the LLM's allowed output space, prompts the model, and detects
missing/extra rows afterward.

---

## Q3 — Is `effective_color` (storage.py:769) inside a 4-nested for loop? Is the codebase the right complexity?

Yes — at the innermost line of `effective_color` there are **4 nested loops**:

```
for q in run_data["questions"]              # line 758   (outer)
  for key, dq_id in dim_q_map.items()       # line 763
    → effective_color(...)                  # line 769
        for tag in ("gpt", "claude")        # line 724
          for fc in ...forecasts            # line 725   (innermost)
```

Plus `black_q50_avg` (797) and `model_resolution_source` (800) re-walk the same
`per_model[...].forecasts` list in the same iteration.

**Performance: not a problem.** Two inner levels are tiny constants (2 models × a handful of
forecast rows). Worst case ≈ low tens of thousands of dict lookups per run — milliseconds.

**The real smell is duplication:** the same "scan forecasts, filter by
`quantile==50 && forecast_date==fdate && dimension==dim`" pattern is copy-pasted in
`effective_color`, `black_q50_avg`, `q50_for` (908), and `color_for` (916).

**Overall verdict:** the algorithm is fine; the domain (reconciling two models' forecasts +
resolution + review overrides into one row, via a clear precedence ladder) is irreducibly
fiddly and written clearly. Optional cleanup *if the file is touched again*: build a per-question
index `{(fdate, dim, quantile): fc}` once and do O(1) lookups — a DRY/readability win, **not**
a performance need. Don't do it speculatively.

---

## Q4 — Current resolution value / status / date logic (in `write_accepted_to_fact_resolution`, storage.py:708)

**Step 0 — candidate cells:** iterate each question's `dim_question_map`, split key into
`fdate, dim` (764); skip timing rows (765-766); look up review `item` from `item_map` (768).

**Write gate (773-774):** a row is written only if the cell is `black` OR status ∈
{resolved, projected}:
```python
if color != "black" and status not in ("resolved", "projected"):
    continue
```
- `color` ← `effective_color` (reviewer's `review_color` if set, else the gpt/claude q50
  `color_code`).
- `status` ← `reviewed_question_resolution_status` if present, else `question_resolution_status`
  (reviewed field always wins).

**resolution_value — 3-tier precedence (776-803):**

| Priority | Source | value | source | status | resolved_at |
|---|---|---|---|---|---|
| 1 | `reviewed_question_resolution_value` (human) | that value | `review_source` / `"surveillance_reviewed"` | `"projected"` if reviewer marked projected, else `"confirmed"` | `now` |
| 2 | `question_resolution_value` (system) | that value | `question_resolution_source` / `"surveillance_projected"` | `"projected"` | `None` |
| 3 | fallback `black_q50_avg` (avg of both models' black q50 forecasts) | avg | from `model_resolution_source`, else `"surveillance_projected"` | `"projected"` | `None` |

Tier 3 drops the row if `black_q50_avg` is `None` (798-799).

**resolution_status:** only a human-reviewed cell can become `confirmed`; everything
auto-derived stays `projected`.

**resolved_at:** `now` only when status resolves to `confirmed`; else `None`.

**resolution_date:** source date first, then fall back to the forecast date:
`_parse_date_or_none(item["question_resolution_source_date"]) or date.fromisoformat(fdate)`
(tiers 1/2); tier 3 uses `source_date or date.fromisoformat(fdate)`.

Each written row → `(question_id=dq_id, resolution_value, resolution_date, resolution_source,
resolution_status, resolved_at)` merged into `fact.fact_resolution` keyed by `question_id`.

**Governing principle:** precedence ladder **human review > system value > model black-q50
average**; only human review yields `confirmed`/`resolved_at`; dates fall back to the target
`fdate`.

_Flag:_ the gate (773) checks `color`/`status`, but a cell can pass on `color=="black"` alone
(blank status) and still write a system value as `"projected"` — worth confirming that's intended.

---

## Q5 — Which sheet-output columns could be influenced by a *previous* sheet? What's the data flow? (Context: sheet values feeding back into output was never intended.)

**Bottom line: within this codebase, no sheet-written column is fed from a previous sheet, and
no sheet value loops back into a later run's output.** Sheet data flows one direction only —
**sheet → BigQuery (3 tables)** — and those tables are a *terminal sink*, not read back by the
question loader or the research step.

### Data flow

```
INPUT (BigQuery dim dataset)
  dim.dim_question_group · dim.dim_question · dim.dim_unit
        │  load_questions()  [questions.py:114]
        ▼
  research → browser → consensus (WEB)  → per_model gpt/claude responses
        │
prior   ▼
run_*.json ── enrich_with_run_stability / value_changed  [storage.py:343/398]
 (disk)      (reads prior run_*.json, saved BEFORE review = model output only)
        │  = run_data
        ▼
  WRITE NEW run_<id> TAB to Google Sheet  [sheets.py:490-569]
   • every column ← run_data (this run)
   • review_* columns written BLANK (559-567)
   • does NOT read any previous tab
        ▼
  ╔═══ HUMAN fills review_* columns in sheet ═══╗
        │  cmd_sync → get_reviewed_items()  [sync.py:75, reads _latest_run_tab]
        ▼
  3 BQ WRITERS consume sheet rows (review_* values):
    write_to_dim_baseline            → dim.dim_baseline
    write_accepted_to_fact_resolution → fact.fact_resolution
    write_to_surveillance_result     → surveillance.surveillance_result
        ▼
  ✗ TERMINAL — none of these tables are read by load_questions or research
    (load_questions reads dim_question* / dim_unit only)
```

### Why there is no feedback loop

1. **Sheet-write path never reads a prior sheet.** New tab is fresh; every cell from `run_data`;
   review columns blanked (sheets.py:559-567). `get_reviewed_items` is only called in `sync.py:75`.
2. **Question loader ignores the sink tables.** `load_questions` reads `dim_question_group` /
   `dim_question` / `dim_unit`; review sync writes `dim_baseline` / `fact_resolution` /
   `surveillance_result` (different tables). `storage.py:554` is the only `query_bq`; `research.py`
   reads no BQ.
3. **Cross-run columns read disk, not the sheet.** `run_stability`, `runs_seen`, `value_changed`
   come from `_read_production_history` (reads `run_*.json`); `sync.py` opens those read-only, and
   they're written before any review.

### Columns to watch (theoretical exposure surfaces)

| Column(s) | Source today | Leak risk to watch |
|---|---|---|
| `run_stability`, `runs_seen`, `gpt/claude_run_stability` | prior `run_*.json` | leaks if review ever written back into `run_*.json` |
| `value_has_changed`, `consensus_*`, `confidence_tier` | prior `run_*.json` + this run | same |
| `question_resolution_value/status` (system, non-`reviewed_`) | this run's model output (`_question_resolution_fields`) | leaks if ever sourced from `fact.fact_resolution` |
| `gpt/claude_latest_official_value`, `current_estimate` | this run's web research | leaks if `research.py` ever seeds context from `dim.dim_baseline` |

**Answer:** currently **none** — but the four rows above are the surfaces to defend, because each
has a BQ/disk twin that review *does* write to.

---

## Q6 — Is any code in `storage.py` / `sync.py` used if we only write to sheets + do surveillance, but NOT write to BQ?

Key distinction: **reading from BQ vs. writing to BQ.** Loading questions still requires BQ *reads*.

### `storage.py` — used by the surveillance + sheet-write path (no BQ write)

`run_surveillance.py:34` uses: `build_run_data`, `_serialize_model_result`,
`enrich_with_run_stability` (→ `_read_production_history`, `_adequate_q50s`, `_stable_pair`,
`_classify_sequence`, `_worst_stability`, `_combine_stability`), `enrich_with_value_changes`
(→ `_values_differ`, `context_maps`, `pick_by_dimension`), `write_json_output`, `write_csv_output`
(→ `_summarize_run`). `sheets.py:16` uses `context_maps`, `pick_by_dimension`.
`questions.py` uses `query_bq` → `_get_client` (a **BQ read**, still happens).

→ Roughly the **top two-thirds of `storage.py` (lines 31–544) is core to surveillance** and
writes nothing to BQ.

### Dead-if-not-writing-to-BQ

Reachable only via the `sync` subcommand (`run_surveillance.py:693 → cmd_sync`):
- **All of `sync.py`** (`cmd_sync`, `_write_run_items`, `_load_run_data`, `_print_sheet_rows`)
- `write_accepted_to_fact_resolution` (708), `write_to_dim_baseline` (829),
  `write_to_surveillance_result` (895)
- Exclusive helpers: `_run_date_from_id` (822), `_merge_bq` (571), `_try_merge_bigquery_rows` (674),
  and the nested closures (`effective_color`, `black_q50_avg`, `model_resolution_source`).

Note: `run_surveillance.py:47` *imports* `cmd_sync`, so `sync.py` is parsed — but `cmd_sync` only
*executes* under the `sync` subcommand.

| | Used for sheets-only workflow? |
|---|---|
| `storage.py` lines 31–544 (run data, enrichment, JSON/CSV, `query_bq` read) | **Yes** |
| shared BQ client/merge helpers (`_get_client`, `_merge_bq`, `_try_merge_bigquery_rows`) | `_get_client` yes (reads); `_merge_bq`/`_try_merge` no |
| `storage.py` lines 708–end (3 writers) | **No** |
| `sync.py` | **No** (imported, not executed) |

---

## Q7 — Are models currently prompted to find resolution values?

**Yes.** Driven mainly by `_resolution_guidance()` (research.py:333), branching by question type.

**General task (research.py:404-405, 413):**
> "Determine whether any expected target row is already resolved. If an authoritative resolution
> exists, report it in `resolution_values` with source and source_date."
> "For past target dates, report a resolution value only when an authoritative source supports it.
> If no such value exists, leave it unresolved rather than guessing one."

**Type-specific (`_resolution_guidance`, 333-351):**
- No past-dated rows (334-335): return empty `resolution_values` unless a future target resolved early.
- probability/when (339-343): find whether resolved as of the date; if authoritative → report in
  `resolution_values`, collapse group to that value, `color_code="black"`.
- quantile with `value_type="resolution"` (346-351): find the metric's authoritative value as of that
  exact target date.

**Guardrails (conservative):**
- A **past date alone is not sufficient** for black/resolved (275, 288, 346) — requires a
  source-backed value.
- **Don't fabricate** (349): if none exists, give a best-estimate distribution + put the best guess
  in `current_estimates`; a non-black distribution for a past date is explicitly OK.
- `source_date` = date the value represents, not publication date (343, 350).
- `current_estimates` must never overwrite a past resolution value (351).

Returned in the `resolution_values` list (`ResolutionValue` model), flowing into `run_data` and onto
the sheet. The **model-found resolution value is the primary auto-source**; only a human's
`reviewed_question_resolution_value` overrides it during sync.

---

## Q8 — Current logic for requesting LLM research. Does it match this proposed structure?

**Proposed structure (user's mental model):**
> - If conditional (`question_condition` not NULL): forecasts only; no current/official; no resolution check.
> - If `unit == 'YEAR'`: forecasts; check resolution; update status/value; `needs_review=TRUE`.
> - elif `unit == probability`: if date < today → forecasts, status "open" deterministically,
>   `needs_review=FALSE`; else → status resolved/failed deterministically, value, `needs_review=TRUE`.
> - else (\$/%): if date < today → forecasts + last official + current; else → deterministic resolve
>   + value, `needs_review=TRUE`.

**Verdict: does NOT match.** The code is organized around a different axis.

### How research is actually requested

No per-branch tree. Each question is classified into **one of three types** by
`_infer_surveillance_question_type` (questions.py:18), and the *type* (not unit or date) decides the ask.

Classification (heuristic, not a unit switch):
- `"when"` ← `not dates and pct_set` (percentiles, no resolution dates). **Not** `unit=='YEAR'`.
- `"probability"` ← text matches `probability|will` **and** (scalar-prob or multi-dim distribution).
- `"quantile"` ← everything else.

What each type asks the LLM for (`_task_steps`, research.py:401):

| Type | LLM forecasts | Last official | Current value | Check resolution |
|---|---|---|---|---|
| probability / when | ✓ (all rows) | ✗ (absent from schema) | ✗ (absent from schema) | ✓ |
| quantile (else) | ✓ | ✓ | ✓ | ✓ (past dates) |

**Research is never gated on `resolution_date < today`.** The model is always asked for the full
type-appropriate set, for every expected row (past rows "collapse to the resolution value"). The
past/future decision is applied afterward, deterministically.

### Deterministic status (post-research)

`resolution_status(fdate, color)` (common.py:141) keys off **date AND the model's color**, not date alone:

| | past date | future date |
|---|---|---|
| black | `resolved` | `resolved_early` |
| not black | `due_unresolved` | `forecast` |

Then `_question_resolution_fields` (sheets.py:228) → `question_resolution_status`; `_needs_review`
(sheets.py:264) → TRUE if status ∈ {resolved, resolved_early, due_unresolved} OR value changed OR
consensus ≠ auto_accepted OR browser failed OR missing data/validation issues.

### Diff table

| Proposed | Reality |
|---|---|
| Conditional-question top branch (`question_condition`) | ❌ No such concept exists anywhere in the code. |
| Branch on `question_unit` (YEAR / probability / else) | ⚠️ Branches on inferred `question_type`, not unit. `when` = no-dates+percentiles; `probability` needs text patterns. |
| YEAR: forecasts + resolution + `needs_review=TRUE` | ⚠️ Partial. `when` uses `TIMING_FORECAST_DATE` placeholder → `resolution_status` can't parse a date → not unconditionally `needs_review=TRUE`. |
| probability: deterministic status by date; forecasts only if past; `needs_review=FALSE` when open | ❌ LLM still asked to check resolution; forecasts always produced; past unresolved → `failed_to_resolve`, not `open`. |
| else (\$/%): date-gate what's researched | ⚠️ `quantile` always does forecasts + official + current + resolution, regardless of past/future. Split applied only in the deterministic status step. |
| "resolved / failed" set deterministically by date | ❌ Depends on model's **black** color (found authoritative value), combined with is_past. |

### What lines up
- probability & when suppress last-official + current-estimate (schema-level).
- quantile does official + current + resolution.
- Resolved / unresolved-past rows flag `needs_review=TRUE`.

### Core structural gap
The proposed model is a **deterministic tree on `question_unit` + `question_condition` + date** that
sometimes skips the LLM. The implementation **always runs the LLM for the full type-appropriate task**
and derives status deterministically afterward from color + date — and has **no notion of conditional
questions at all.** A conditional question today would be classified `quantile`/`probability` and get
the full treatment.

**Open follow-up:** confirm whether `question_condition` even exists in the source BQ `dim_question`
table, to know whether conditional handling needs wiring in from scratch.
