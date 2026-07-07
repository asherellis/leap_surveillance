# Handoff notes — what changed and why (Nadja → Asher, July 2026)

I spent a couple of days reviewing and reworking the surveillance pipeline. This document walks through **everything that changed and the reasoning behind it**, so you can see how I was thinking about it and push back where you disagree. Nothing here is meant as criticism of the original code — most of it is genuinely good, and several "findings" below turned out to be my own bugs or false alarms. Treat this as a design conversation in written form.

Companion docs (in `memory/`):

- `memory/quality_check_2026-07-02.md` — a full code-quality audit (~50 findings, each with
location, impact, fix applied or decision pending). The per-item detail lives there; this
doc gives the narrative.
- `memory/conditional_questions.md` — why conditional (scenario) questions are excluded and
how to re-enable them later.

---



## The guiding ideas

Four principles drove most of the changes:

1. **Don't elicit forecasts for dates that have already passed.** A past target date needs a
  *resolution*, not a distribution. The old pipeline asked the LLM for the full quantile
   grid on every date and only softly suggested it look up resolutions.
2. **Make resolution checking mandatory and explicit** — especially for timing ("when")
  questions, where *only research can tell us whether the event has occurred*. (There was a
   real bug here: timing questions were being told **not** to check for resolution — see §2.)
3. **Deterministic classification over text heuristics.** Question types now derive from
  structural data (unit, dates, percentiles), not regexes on question text.
4. **Fail loudly, work on a fresh machine.** Several failure modes only appeared when I ran
  the pipeline on my laptop (missing deps, missing `outputs/` dir, exit code 0 on failure).
   These are fixed, and the pipeline should now be reproducible from a clean checkout.

---



## 1. Question-type inference is now structural (`questions.py`)

**Before:** `_infer_surveillance_question_type` used regexes on the question text
(`\bprobabilit(y|ies)\b|\bwill\b`, `startswith("will ")`, …) plus the unit.

**Now:** it derives from structure only:

- `unit_name == "probability"` **and** percentiles == {50} → `probability`
- no horizon dates → `when` (timing)
- everything else → `quantile`

**Why:** text heuristics misclassify quietly. The unit column is authoritative. Two
subtleties we fixed along the way: (a) `[50] == {50}` is always False in Python, so an early
version of my own rewrite never returned "probability" at all — the set conversion matters;
(b) a probability-unit question with non-{50} percentiles now **explicitly** returns
`quantile` with a warning, instead of falling through to `when` when it happens to have no
dates.

The three type strings (`quantile` / `probability` / `when`) are load-bearing across
research.py, models.py, and consensus.py — if you ever add a fourth, grep for
`("probability", "when")` first; that pair is the "event-like" axis used everywhere.

---



## 2. Resolution-aware retrieval — the big rework (`research.py`, `models.py`, `questions.py`, `consensus.py`)

This is the core behavioral change. The target matrix:


| Path                     | Forecast quantiles                               | LOV / current | Resolution check      |
| ------------------------ | ------------------------------------------------ | ------------- | --------------------- |
| quantile, future date    | full 7-quantile distribution                     | ✓             | —                     |
| quantile, past date      | **nulled**                                       | ✓             | **mandatory**         |
| probability, future date | q50                                              | ✗             | —                     |
| probability, past date   | q50 **nulled**                                   | ✗             | **mandatory**         |
| when (timing)            | 7 timing quantiles, **nulled if event occurred** | ✗             | **always, every run** |


Key pieces:

### 2a. Past dates: resolution instead of forecast

`date_value_type` already tagged past rows `value_type="resolution"` — but the old pipeline
still elicited the full distribution for them and only collapsed post-hoc *if* the model
volunteered a resolution value. Now:

- The prompt makes the resolution lookup **mandatory** for past dates ("check hard, consult
multiple sources"), judged **against the question's resolution criteria** (not just "did
something happen").
- `strict_to_regular_response` **nulls the forecast quantiles** for every resolution row —
the value lives in `resolution_values`, not smeared across 7 identical quantiles.
- Resolved rows are colored black; failed/unresolved rows keep the model's color.



### 2b. New fields: `resolution_status` and `best_guess_resolution`

Every resolution entry now carries an explicit status the LLM must set:

- `resolved` — authoritative value found (goes in `value`, with source + source_date)
- `failed` — the date passed but no authoritative value is findable. `value` is null and the
model puts its single best estimate in the new `best_guess_resolution` field.
- `unresolved` — not resolved and not failed (timing questions whose event hasn't occurred).

Two deliberate rules: **timing questions can never be** `failed` (they have no resolution
date to miss — the event has either happened or it hasn't), and `best_guess_resolution` **is
only for** `failed` **date-based rows** (for timing, the "best guess" is already the forecast
distribution itself — duplicating it into a resolution field would be redundant and
confusing).

### 2c. The timing-question bug (worth reading twice)

`_resolution_guidance` used to short-circuit: *if no expected row is tagged*
`value_type=="resolution"`*, tell the model to return an empty resolution_values list.*
Timing rows use the `TIMING_FORECAST_DATE` placeholder, which `date_value_type` can't parse
— so **every timing question was tagged "forecast" and told not to bother resolving.**
Exactly backwards: timing questions are the ones where only research reveals resolution.
They now get an always-on, every-run resolution check.

### 2d. Consensus now compares resolution values, not q50, for resolution rows

Nulling q50 on past rows would have broken the gpt-vs-claude agreement check (q50=None →
automatic mismatch). Rather than keeping the collapsed-value hack, consensus now indexes
each model's `resolution_values` by (date, dim) and compares **the resolved values** (±5%
tolerance; two `failed` rows count as agreement). Future/forecast rows still compare q50 as
before. There's a new `resolution_agreement` key in the consensus output.

### 2e. Validation updated to match

`validate_response` no longer punishes intentionally-nulled rows: resolution rows are
excluded from the missing-q50 check and the >50%-null heuristic; `failed`/`unresolved` rows
are exempt from `resolution_not_black` and `mixed_colors`; a resolution row counts as
"returned" if it has a value **or** a best guess.

---



## 3. Conditional (scenario) questions: explicitly excluded

`dim_question.scenario_id` marks conditional questions ("assuming scenario X holds, forecast
Y"). Conditionality is **per-question, not per-group** — a group can mix conditional and
unconditional rows.

**Decision: exclude them for now.** The query filters `AND q.scenario_id IS NULL`, applied
**before any GROUP BY** (in the `forecast_groups` CTE *and* the main join) so mixed groups
keep their unconditional rows and fully-conditional groups drop out cleanly.

**Why exclude rather than handle:** I actually built the full handling first (an
`is_conditional` flag orthogonal to question_type, a forecasts-only schema, prompt framing
for conditional probabilities) and then ripped it out — I wasn't confident in the design
(how do conditional forecasts get scored? where do they live downstream?), and I'd rather
ship the unconditional pipeline cleanly than carry a half-settled design. The removed
prototype, the open design questions, and step-by-step re-enable instructions are all in
`memory/conditional_questions.md`. The prototype itself is in the git history
(`b2f2d4c` added it, `e221206` removed it).

---



## 4. Real NULLs instead of the `-999` sentinel (`models.py`, `research.py`)

The old design used `-999` as "no value" because the strict output schemas required plain
floats. Two problems:

1. **The sentinel was broken for bounded questions**: `make_strict_schema` sets
  `minimum = unit_min` on `forecast_value`, so for any `unit_min >= 0` question the schema
   *forbade* `-999` — the model literally could not signal "no value" and was pushed to
   fabricate an in-range number instead.
2. Modern structured-output APIs support nullable fields, so the workaround was obsolete.

**Now:** `forecast_value`, official/current `value`, and resolution `value` /
`best_guess_resolution` are all `Optional[float]` (still *required* keys — nullable, not
omittable). Unit bounds attach to the **number branch** of the `anyOf: [number, null]` union
(a subtle but important detail — bounds on the wrapper would silently stop applying).
Prompts say "set to null" instead of "-999". `_convert_sentinel_value` stays as a harmless
backstop for stray legacy values. `9999`/`NEVER_YEAR` for timing is **not** touched — that's
a semantic "never" signal, not a missing value.

---



## 5. Anthropic strict tool use (`research.py`, `models.py`)

We hit a real instance of Claude returning a structurally malformed `produce_forecast` tool
call (missing the required `sources` field) — caught by your parse-retry machinery, but each
retry re-runs the whole paid web search. Root cause: OpenAI's strict `json_schema` mode
*guarantees* schema-valid output, but the Anthropic tool path had no equivalent guarantee.

**Fix:** `strict: True` on the Anthropic tool definition (GA, supported on both our prod and
test models). One catch: strict mode doesn't support numeric bounds (`minimum`/`maximum`), so
the new `strict_tool_schema()` strips them from the Anthropic copy — bounds are still stated
in the prompt and enforced post-hoc by `validate_response`. The OpenAI path is unchanged.
This should make malformed-payload retries essentially disappear on the Claude side.

---



## 6. The quality audit and fix pass

I ran a systematic audit of the whole package (four independent review passes, then I
verified each claim against the source before acting — several auditor claims were wrong and
are documented as rejected). Full detail with per-item resolutions:
`memory/quality_check_2026-07-02.md`. The highlights, grouped:

### Crashes waiting to happen (all fixed)

- **NaN dimension →** `"nan"` **key** (questions.py): pandas NaN is truthy, so
`str(x or "Overall")` produced the literal key `"2027-01-01|nan"` while everything else
looked up `"…|Overall"` — silently detaching `question_id` from output rows for
single-dimension questions. Now uses `is_empty` like the sibling code.
- **CSV writer** used `fieldnames=rows[0].keys()` on heterogeneous rows (validation fields
are conditional) → order-dependent crash. Now a union of all keys.
- **Unguarded** `float(v)` **/** `date.fromisoformat` **/** `key.split("|")` in the BQ writers could
each abort a whole write on one bad value. All guarded with the `_safe_*` helpers + skip
and warn.
- `outputs/` **didn't exist on a fresh checkout** — the partial-progress writer crashed
*after* both questions had finished (results lost, LLM money spent). The end-of-run
writers did `mkdir`; the mid-run one didn't. Classic works-on-the-author's-machine bug —
it worked for you because your `outputs/` already existed.



### Silent failures made loud (fixed)

- **CLI always exited 0** — `main()` discarded return codes, so missing API keys or failed
BQ syncs looked like success to cron/CI. Now `raise SystemExit(...)`, and `cmd_sync`
returns non-zero if any of the three BQ writes failed.
- **Unknown model → $0 cost, silently.** `cost_for_tokens` now warns once per unpriced model.
- **Unbounded rate-limit retry** (`sleep(60); continue` forever) now capped at
`MAX_RATE_LIMIT_RETRIES` (env-tunable, default 5).
- **"Latest run" tab was** `max()` **by lexicographic name** — a manual `run_backup` tab would
sort after every timestamped tab and get synced to BigQuery. Now a strict
`^run_\d{8}_\d{6}$` regex filter.



### Correctness / consistency (fixed)

- `value_changed` **compared GPT against a prior run matched on *any* model** — if the most
recent prior run only matched on Claude, GPT was compared against a different GPT version
(spurious change flags). Now filters to same-GPT-model priors, matching the stability
enricher's per-tag semantics.
- `all_dims` **for value-change detection** now includes prior-run dims, so a value that
*disappeared* since last run also flags.
- **Timeout detection** was substring matching on exception text; now typed
`APITimeoutError` checks first with substring fallback. Parse-retry predicates unified
across both providers.
- **UTC everywhere**: `run_id` and past-date detection used local time while
`date_value_type` used UTC — near-midnight runs could classify the same date differently
in two places.
- **Refinement prompt drift**: `refine_with_browser` was missing the LEAP anti-anchoring
guardrail and run date that the main research prompt has. Added.



### Dead code removed

- `_try_browser_refinement_for_model` + the `defer_browser` parameter (unreachable — browser
extraction always runs via the shared/deferred path), three unused params on
`_should_accept_browser_refinement`, the double `$defs` pass in `fix_schema`, a dead
`"resolved"` branch in the fact-resolution writer.
- The four near-identical q50-scan loops in storage.py are folded into one `_q50_forecast`
helper.



### Renames for clarity

- `common.resolution_status()` → `row_resolution_status()` — it returns the *row display
status* (resolved/resolved_early/due_unresolved/forecast), a different vocabulary from the
new LLM-reported `ResolutionStatus` enum (resolved/failed/unresolved). Two things named
"resolution status" with different value sets was a foot-gun.
- Sheet column `run_id` → `surveillance_timestamp` (the value is unchanged —
`YYYYMMDD_HHMMSS`, now UTC). The reader falls back to `run_id` so **existing tabs still
sync**. Internals (`sync.py` grouping, `run_<id>.json` lookup) are untouched. Note the raw
value is load-bearing: it's the JSON filename suffix, the tab-name suffix, and part of
`review_row_id` — that's why I renamed the column but not the format. (Converting the
value to a real ISO timestamp is a possible follow-up, but needs all three of those
consumers handled.)



### Security

- **SSRF guard in browser.py never resolved hostnames** — a DNS name pointing at
`169.254.169.254` or RFC1918 sailed through. Now resolves and checks every returned IP;
`*.internal` blocked; unresolvable hosts blocked.
- Download-extension check now parses the URL path (`report.csv?dl=1` no longer slips
through); the `'"error":'` extraction check only fires on JSON-shaped payloads so pages
that merely contain that substring aren't false-rejected.



### Rejected findings (checked, NOT bugs — for your peace of mind)

- "refine_with_browser silently switches provider" — false; the call site threads
`stack.research_model` and `_build_stacks` keeps the provider in test mode.
- "consensus both-failed rows should compare status" — the status-mismatch scenario can't
actually occur given which statuses co-occur per question type.
- "LEAP anchor check on sources is dead" — the domain match works fine on bare URLs.



### Packaging / environment

- `pyproject.toml`: `google-cloud-bigquery` → `google-cloud-bigquery[pandas]` (provides
`db-dtypes` + `pyarrow`, required at runtime by `to_dataframe`) and added `requests`
(directly imported in browser.py, previously only transitive). Both bit me on a fresh venv.
- Trivia for your amusement: I lost an hour to `ModuleNotFoundError` that turned out to be
**iCloud setting the macOS hidden flag on** `.venv` **files + Python 3.13 silently skipping
hidden** `.pth` **files**. If you ever develop from `~/Documents` on a Mac: don't, or keep the
venv outside the repo.

---

## 7. How to verify

- `python check_nullable_schema.py` — offline checks for the nullable-schema migration
(schema shape, bounds placement, round-trip parsing). All pass.
- `python check_nullable_schema.py --live` — **needs valid API keys**; sends the real strict
schema to both providers to confirm they accept nullable + strict + stripped bounds, and
that the model can emit `null`. This is the one thing I could not verify (my Anthropic key
was invalid) — please run it before the first production run.
- A dev run: `leap-surveillance run --dev --test-mode --no-sheet --no-browser -n 2 -y`
(from the repo root — `.env` loading and `outputs/` are cwd-relative). I ran this
successfully with GPT; inspect the `run_*.json` to see nulled past-date rows,
`resolution_status`, and `best_guess_resolution` in action.

**Explicitly out of scope / unchanged:** the downstream write semantics (sheets review flow,
the three BQ writers' precedence ladder, DBT treatment) — deliberately untouched apart from
the robustness fixes above. That's a separate conversation.

Looking forward to your take on all of this. If any change seems wrong for  
context I'm missing, the git history is granular and everything is documented.