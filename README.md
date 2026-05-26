# LEAP Surveillance

Pipeline for running LLM surveillance on LEAP forecasting questions.

It pulls questions from BigQuery, generates structured forecasts with web search, optionally uses browser automation when needed, and writes results to local files and Google Sheets. BigQuery write/sync support is included but requires edit permissions.

## Setup

```bash
cp .env.example .env
```

Fill in:

```text
OPENAI_API_KEY
OPENAI_SAFETY_IDENTIFIER
LEAP_SHEET_ID
```

Install:

```bash
pip install -e .
```

## Run

Cheap test mode (uses gpt-4o-mini):

```bash
python run_surveillance.py run --test-mode --limit 3 --no-bq -y
```

Production:

```bash
python run_surveillance.py run --limit 10 --no-bq -y
```

Specific questions:

```bash
python run_surveillance.py run --questions recABC123,recDEF456 -y
```

Drop `--no-bq` once BigQuery write access is configured. Add `--no-browser` to skip browser-use (faster, more reliable but loses dashboard extraction).

## Review workflow

After a run, open the Google Sheet (`LEAP_SHEET_ID`). The **Review** tab has one row per question / target_date / dimension. The `type` column tells the reviewer what to do:

- **type=resolved** (color_code=black): the LLM found the actual answer.
  - `llm_value` is the LLM's answer. Read `rationale` and `data_source` to judge quality.
  - Fill `actual_value` from your own research, `verified_source`, and pick a `score` from the dropdown (correct / close / wrong / confidently_wrong).
- **type=forecast** (color_code ≠ black): the LLM is still forecasting.
  - `llm_value` is the LLM's q50. `q25` / `q75` are the uncertainty range.
  - If something seems wrong, fill `corrected_value` (fixes q50) or `corrected_color` (fixes classification).

Check `reviewed` when done. Then:

```bash
python run_surveillance.py sync
```

Syncs corrections back to BigQuery and stamps `reviewed_at`. For sheet-only (no BQ write):

```bash
python run_surveillance.py sync --no-bq
```

## Reset the sheet

```bash
python run_surveillance.py setup -y
```

Wipes the Review tab, deletes legacy tabs, and recreates Instructions.
