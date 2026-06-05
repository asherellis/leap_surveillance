# LEAP Surveillance

Runs LLM surveillance on LEAP forecasting questions.

The pipeline loads questions from BigQuery, generates structured forecasts with web search,
optionally uses browser automation, and writes local JSON/CSV plus Google Sheet review tabs.
BigQuery writes are enabled by default and require write access.

## Setup

```bash
cp .env.example .env
python3 -m pip install -e .
```

Commands below use the installed `leap-surveillance` script. If your shell cannot find it,
use `python3 -m leap_surveillance.run_surveillance` in its place.

Required:

```text
OPENAI_API_KEY
OPENAI_SAFETY_IDENTIFIER
```

Optional:

```text
LEAP_SHEET_ID
LEAP_BQ_PROJECT
LEAP_SURVEILLANCE_DATASET
LEAP_MODEL
LEAP_EVALUATOR_MODEL
LEAP_BROWSER_MODEL
LEAP_TEST_MODEL
LEAP_TEST_EVALUATOR_MODEL
```

Defaults:

- `LEAP_SHEET_ID` uses the shared LEAP review sheet.
- `LEAP_BQ_PROJECT` uses the LEAP development warehouse.
- `LEAP_SURVEILLANCE_DATASET` uses `surveillance`.

## Run Surveillance

Start with a low-cost live run. This still calls the pipeline, but uses lower-cost models:

```bash
leap-surveillance run --test-mode --limit 3 --no-bq -y
```

Run a normal batch without BigQuery writes:

```bash
leap-surveillance run --limit 10 --no-bq -y
```

Run specific questions:

```bash
leap-surveillance run --questions <question_id_1>,<question_id_2> --no-bq -y
```

Drop `--no-bq` to write to BigQuery. Add `--no-browser` to skip browser automation.

## Review and Sync

Each run creates a Sheet tab named `run_<run_id>` (e.g. `run_20260602_143000`).
The **Instructions** tab explains the columns and status values:
`resolved`, `due_unresolved`, `forecast`, `resolved_early`.

Run tabs have one row per question / target date / dimension. Quantile questions produce
q0/q5/q25/q50/q75/q95/q100 in JSON/CSV/BigQuery; the Sheet shows q50 as `llm_answer`
plus `q25` and `q75`.

Review `status`, `question_type`, `unit`, `judge_confidence`, and `validation_issues`.
Fill the `review_*` columns, tick `reviewed`, then sync:

```bash
leap-surveillance sync          # writes reviewed rows to BigQuery
leap-surveillance sync --no-bq  # previews reviewed rows without writing
```

`sync` reads the most recent `run_*` tab by default. Use `--tab run_<run_id>` for a specific tab.

To rebuild only the **Instructions** tab:

```bash
leap-surveillance setup -y
```

## Offline Tests

These tests do not call OpenAI, Google Sheets, or BigQuery:

```bash
python3 -m pytest -q
```

## Repo Layout

Source lives in the root-level `leap_surveillance` package:

- `leap_surveillance.run_surveillance`: CLI for `run`, `sync`, and `setup`.
- `leap_surveillance.research`: LLM research, judge, browser/refinement, and prompts.
- `leap_surveillance.questions`: BigQuery question loading and expected-row shaping.
- `leap_surveillance.storage`: local JSON/CSV outputs and BigQuery writes/sync.
- `leap_surveillance.sheets`: Google Sheets review UI.
- `leap_surveillance.models`: dataclasses, Pydantic schemas, and deterministic validation.
- `leap_surveillance.common`: shared constants, env-configurable defaults, and small helpers.

Each run writes its artifacts to `outputs/` as `run_<run_id>.json` and `run_<run_id>.csv`.
