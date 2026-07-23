# LEAP Surveillance

Runs LLM surveillance on LEAP forecasting questions. Loads questions from BigQuery,
generates structured forecasts with web search (and optional browser automation), and
writes local JSON/CSV plus a per-run Google Sheet review tab. Runs land in `outputs/`
as `run_<run_id>.{json,csv}`.

This pipeline intentionally excludes conditional questions for now.

## Setup

macOS / Linux:

```bash
cp .env.example .env
python3 -m pip install -e .
```

Windows:

```bat
copy .env.example .env
python -m pip install -e .
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
leap-surveillance run                                   # dual-model run (default)
leap-surveillance run --gpt                             # GPT only
leap-surveillance run --claude                          # Claude only
leap-surveillance run --test-mode --limit 3 --no-sheet  # cheap smoke test
leap-surveillance run --questions <id1>,<id2>           # specific questions
```

## Review and sync

Every run publishes a `run_<run_id>` tab to the Dev sheet unless `--no-sheet` is passed.
Review the `review_*` fields there; partial review is fine. Prod receives promoted reference
copies and should not be edited directly.

- [Dev review Sheet](https://docs.google.com/spreadsheets/d/1lT7zVfKAsVZU7bKaEALq1AWApfFmWMisprTK42l7RDo)
- [Prod verified Sheet](https://docs.google.com/spreadsheets/d/1OsFxYG9ar5JtIn7kkYbPEwKJGGRfTsjBnzOtQfPTbAc)

```bash
leap-surveillance sync --tab run_<run_id>
```

`sync` writes the full run to BigQuery, with reviewed values taking priority, then promotes the
same tab to Prod. `--tab` is required so sync never guesses at the latest run. Use
`leap-surveillance setup` to rebuild the Instructions tab on both sheets.
