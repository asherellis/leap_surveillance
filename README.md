# LEAP Surveillance

Runs LLM surveillance on LEAP forecasting questions. Loads questions from BigQuery,
generates structured forecasts with web search (and optional browser automation), and
writes local JSON/CSV plus a per-run Google Sheet review tab. Runs land in `outputs/`
as `run_<run_id>.{json,csv}`.

This pipeline intentionally excludes conditional questions for now. To re-implement them, look at memory/conditional_questions.md

## Setup

```bash
cp .env.example .env
python3 -m pip install -e .
```

Required in `.env` for the default `--both` mode:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
```

Use `--gpt` for OpenAI only or `--claude` for Anthropic only.
Optional config (sheet ID, BQ project, model overrides) is in `.env.example`.

## Run

```bash
leap-surveillance run --both -y                            # dual-model run (default)
leap-surveillance run --gpt -y                             # GPT only
leap-surveillance run --claude -y                          # Claude only
leap-surveillance run --test-mode --limit 3 --no-sheet -y  # cheap smoke test
leap-surveillance run --questions <id1>,<id2> -y           # specific questions
```

## Review and sync

Each run creates a Sheet tab named `run_<run_id>`. The **Instructions** tab explains the
columns and the `status` values (`resolved`, `due_unresolved`, `forecast`, `resolved_early`).

Fill the `review_*` columns, tick `reviewed`, then:

```bash
leap-surveillance sync --tab run_<run_id>          # previews that tab's rows without writing
leap-surveillance sync --tab run_<run_id> --write   # writes dim_baseline + fact_resolution + surveillance_result
```

`sync` without `--tab` reads the most recent `run_*` tab, but `--tab` is required whenever `--write` is set — an actual write always names its tab explicitly.
All Sheet rows sync to `surveillance.surveillance_result`; reviewed rows include human
review fields, and unreviewed rows carry model projections. Historical/current baselines
sync to `dim.dim_baseline`. Resolved or projected target-date values sync to
`fact.fact_resolution`.
To rebuild only the Instructions tab: `leap-surveillance setup -y`.
