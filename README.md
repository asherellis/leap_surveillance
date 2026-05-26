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

After a run, open the Google Sheet (`LEAP_SHEET_ID`). The **Review** tab has one row per question / target_date / dimension.

Each row has two value columns:
- `target_date_value` — the value AT the question's target_date (the answer being verified).
- `current_value` — the LLM's live estimate AS OF TODAY (context only, not scored).

- **color_code = black** (resolved): the LLM found the actual answer.
  - `target_date_value` is the locked historical value. Read `rationale` and `data_source`.
  - Fill `review_value` from your own research, `review_source`, and pick a `review_verdict` (correct / close / wrong / confidently wrong).
- **color_code ≠ black** (forecast): the LLM is still forecasting.
  - `target_date_value` is the LLM's q50. `q25` / `q75` are the uncertainty range.
  - If something seems wrong, fill `review_value` (your corrected median) or `review_color` (fixes classification).

Check `reviewed` when done. Then:

```bash
python run_surveillance.py sync
```

Syncs corrections back to BigQuery. For sheet-only (no BQ write):

```bash
python run_surveillance.py sync --no-bq
```

## Reset the sheet

```bash
python run_surveillance.py setup -y
```

Wipes the Review tab and recreates it with current formatting and column structure.
