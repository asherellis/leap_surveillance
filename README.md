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
leap-surveillance run --both                            # dual-model run (default)
leap-surveillance run --gpt                             # GPT only
leap-surveillance run --claude                          # Claude only
leap-surveillance run --test-mode --limit 3 --no-sheet  # cheap smoke test
leap-surveillance run --questions <id1>,<id2>           # specific questions
```

## Review and sync

Every run publishes to the Dev sheet (`LEAP_DEV_SHEET_ID`), as a tab named `run_<run_id>`,
unless `--no-sheet` is passed. Review happens there. Prod run tabs only ever arrive through
promotion (below) and aren't meant to be edited directly - `setup` is the one exception, since
it rebuilds the Instructions tab on both sheets. The **Instructions**
tab on either sheet explains the columns and the `status` values (`resolved`, `due_unresolved`,
`forecast`, `resolved_early`).

Fill the `review_*` columns, tick `reviewed` on the rows you've settled - partial review is fine,
finalizing doesn't require every row to be reviewed - then:

```bash
leap-surveillance sync --tab run_<run_id>
```

`--tab` is required — sync always names its tab explicitly, never guesses at "the latest one."
It does two things, in order: writes all Sheet rows to BigQuery -
`surveillance.surveillance_result` (every row; reviewed rows carry human review fields,
unreviewed rows carry model projections), plus `dim.dim_baseline` and `fact.fact_resolution`
(baseline/resolution values, reviewed values taking priority where present) - then copies that
same Dev tab into the Prod sheet, overwriting any earlier promotion of the same tab. The
promotion step runs regardless of whether the BigQuery writes succeeded, so a warehouse hiccup
doesn't block preserving the reviewed record in Prod - but any failure (BigQuery or promotion)
is still reported and exits nonzero.
To rebuild the Instructions tab on both sheets: `leap-surveillance setup`.

## Troubleshooting

**BigQuery/Google API calls fail with an SSL error** (e.g. `SSLError`, `NO_CERTIFICATE_OR_CRL_FOUND`,
`Max retries exceeded ... oauth2.googleapis.com`), while `curl https://oauth2.googleapis.com` works
fine in the same terminal: your Python venv's bundled `certifi` cert store doesn't recognize a CA
your network injects (common on university/corporate networks with their own network-auth
profile). Point Python at your OS's trust store instead:

macOS:

```bash
export SSL_CERT_FILE=/etc/ssl/cert.pem
export REQUESTS_CA_BUNDLE=/etc/ssl/cert.pem
```

Linux (path varies by distro - Debian/Ubuntu shown; RHEL/Fedora use `/etc/pki/tls/certs/ca-bundle.crt`):

```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

On Windows there's no equivalent PEM file - Python doesn't read the Windows Certificate Store by
default. Install `pip-system-certs` instead (`python -m pip install pip-system-certs`), which
patches Python's SSL calls to use the OS trust store automatically.

This is a local network/environment issue, not a pipeline bug — setting these globally would hide
unrelated certificate errors, so don't add them to `.env` or hardcode them in code.
