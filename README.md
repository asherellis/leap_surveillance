# LEAP Surveillance

Runs LLM surveillance on LEAP forecasting questions. Loads questions from BigQuery,
generates structured forecasts with web search (and optional browser automation), and
writes local JSON/CSV plus a per-run Google Sheet review tab.

## Setup

```bash
cp .env.example .env
python3 -m pip install -e .
```

Required in `.env`:

```text
OPENAI_API_KEY
```

Optional config (safety identifier, sheet ID, BQ project, model overrides) is documented in `.env.example`.

## Run

```bash
leap-surveillance run --test-mode --limit 3 --no-bq -y   # cheap live run
leap-surveillance run --limit 10 --no-bq -y              # normal batch, no BQ write
leap-surveillance run --questions <id1>,<id2> -y         # specific questions
```

Drop `--no-bq` once write access is configured. Add `--no-browser` to skip browser automation.

## Review and sync

Each run creates a Sheet tab named `run_<run_id>`. The **Instructions** tab explains the
columns and the `status` values (`resolved`, `due_unresolved`, `forecast`, `resolved_early`).

Fill the `review_*` columns, tick `reviewed`, then:

```bash
leap-surveillance sync          # writes reviewed rows to BigQuery
leap-surveillance sync --no-bq  # previews without writing
```

`sync` reads the most recent `run_*` tab; pass `--tab run_<run_id>` for a specific one.
To rebuild only the Instructions tab: `leap-surveillance setup -y`.

## Layout

`leap_surveillance/` is the package. Runs land in `outputs/` as `run_<run_id>.{json,csv}`.
