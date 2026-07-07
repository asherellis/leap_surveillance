# Conditional (scenario) questions — how we think about them & why they're excluded

_Last updated: 2026-07-02_

## What a conditional question is

Some LEAP questions are **conditional**: they ask for a forecast *given that a specified
scenario holds* (e.g. "Conditional on a major-power war starting before 2030, what will X
be?"). In the data warehouse this is marked at the **`dim_question` (individual question)
level**, not the question-group level:

- `dim_question.scenario_id` — non-NULL ⇒ that individual question is conditional.
- `dim_scenario.scenario_name`, `dim_scenario.scenario_description` — the scenario text.

**Important:** conditionality is per-row, not per-group. A single `question_group` can
contain a mix of conditional and non-conditional `dim_question` rows (e.g. an unconditional
variant plus one or more scenario variants). So you cannot decide conditionality by looking
at the group.

## Current decision: EXCLUDE them (2026-07-02)

We are **not handling conditional questions for now.** The surveillance pipeline filters
them out at the query level and treats every remaining question as a normal
(unconditional) quantile / probability / timing question.

**How the exclusion is implemented** (`leap_surveillance/questions.py`, `load_questions`):
the BigQuery query filters `AND q.scenario_id IS NULL` on the `dim_question` join (and in
the `forecast_groups` CTE). Rows with a scenario are dropped; a group made up entirely of
conditional questions simply produces no rows and is skipped. No `scenario_*` columns are
selected and there is no `dim_scenario` join.

**Why we excluded them (rather than ship the handling we prototyped):**
- We were not yet confident in how conditional forecasts should be elicited, scored, or
  stored, and would rather ship the non-conditional pipeline cleanly than carry a
  half-settled design.
- Conditional handling is easy to re-enable later once the design is settled — nothing
  about excluding them now blocks that.

## How we *were* thinking about handling them (the prototype we removed)

Before deciding to exclude, we prototyped a conditional path. Recording it here so we don't
have to re-derive it when we revisit:

- **`is_conditional` flag, orthogonal to `question_type`.** A conditional question keeps its
  underlying distribution shape (probability vs quantile vs timing); conditionality was a
  separate axis, set from `scenario_id IS NOT NULL`.
- **Forecasts only.** Conditional questions would elicit *only* forecasts — no
  last-official-value, no current-day estimate, and **no resolution check** (a hypothetical
  scenario never resolves). All expected rows were forced to `value_type="forecast"` even
  for past dates.
- **Dedicated output schema** `StrictConditionalResponse` = `forecasts` + `rationale` +
  `sources` only (no `resolution_values` / `last_official_values` / `current_estimates`).
- **Prompt framing.** `_build_prompt_context` injected a "CONDITIONAL QUESTION" block telling
  the model every value is a *conditional* estimate that assumes the scenario holds, that it
  should NOT forecast whether the scenario itself occurs, followed by the scenario name +
  description.
- **Plumbing.** `_research_schema` / `_schema_name` branched on `is_conditional` first;
  `_task_steps` / `_resolution_guidance` / `_rationale_requirements` had conditional
  branches; `strict_to_regular_response` took an `is_conditional` arg that forced LOV /
  current / resolution empty.

## Open questions to resolve before re-including conditionals

- How should conditional forecasts be scored against a scenario that may or may not occur?
- Where do conditional forecasts live downstream (fact table / DBT), separate from
  unconditional ones?
- For a group with both conditional and unconditional variants, how do the two relate?

## How to re-enable later

1. In `load_questions`, drop the `q.scenario_id IS NULL` filter; re-select `scenario_id`,
   `scenario_name`, `scenario_description` and re-add the `dim_scenario` LEFT JOIN.
2. Re-introduce the `is_conditional` flag + `scenario_*` fields on `QuestionSpec`, the
   `StrictConditionalResponse` schema, the conditional branches in `research.py`, and the
   conditional framing block in `_build_prompt_context`.
3. See git history around 2026-07-02 for the exact prototype implementation.

## Related

- The resolution handling that shipped (null past-date forecasts; `resolution_status`
  resolved/failed/unresolved; `best_guess_resolution`; always-on `when` resolution check;
  consensus comparing resolution values) is **independent** of conditional questions and
  stays in place.
