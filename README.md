# LEAP Surveillance

Pipeline for running LLM surveillance on LEAP forecasting questions.

It pulls questions from BigQuery, generates structured forecasts with web search, optionally uses browser automation when needed, and writes results to local files and Google Sheets. (BigQuery write/sync support is included but requires permissions.)

## Setup

Create a `.env` file:

```bash
cp .env.example .env
```

Fill in:

```text
OPENAI_API_KEY
OPENAI_SAFETY_IDENTIFIER
LEAP_SHEET_ID
```

Install dependencies:

```bash
pip install -e .
```

## Run

Cheap test mode:

```bash
python run_surveillance.py run --test-mode --limit 3 --no-bq -y
```

Production mode:

```bash
python run_surveillance.py run --limit 10 --no-bq -y
```

Omit `--no-bq` when BigQuery write access is configured.

Reset the review sheet:

```bash
python run_surveillance.py setup -y
```

Sync reviewed rows to BigQuery:

```bash
python run_surveillance.py sync
```

For Sheet-only review movement:

```bash
python run_surveillance.py sync --no-bq
```
