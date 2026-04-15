# Trump Social Trading Research Workbench

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![UI: Streamlit](https://img.shields.io/badge/ui-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Storage: DuckDB + Parquet](https://img.shields.io/badge/storage-DuckDB%20%2B%20Parquet-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![CI](https://github.com/keithwegner/allcaps/actions/workflows/ci.yml/badge.svg)](https://github.com/keithwegner/allcaps/actions/workflows/ci.yml)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange)](https://github.com/keithwegner/allcaps)

This is a local Streamlit workbench for researching whether President Donald Trump's social posts, plus posts from influential X accounts that mention him, help predict the **next trading session's SPY return**.

In plain English, the app helps you:

- collect Trump-related social data
- decide which X accounts matter enough to track
- turn that data into trading features
- backtest a simple **long / flat SPY** strategy
- compare saved runs and inspect why a run did or did not work

## Table of Contents

- [What The App Is For](#what-the-app-is-for)
- [What You Need Before You Start](#what-you-need-before-you-start)
- [Quick Start](#quick-start)
- [Contributor Workflow](#contributor-workflow)
- [Hosting On Render](#hosting-on-render)
- [Hosted Environment Variables](#hosted-environment-variables)
- [Recommended First Run](#recommended-first-run)
- [How To Work With Each Page](#how-to-work-with-each-page)
  - [`Datasets`](#datasets)
- [`Discovery`](#discovery)
- [`Research View`](#research-view)
- [`Models & Backtests`](#models--backtests)
- [`Live Monitor`](#live-monitor)
- [Data Inputs](#data-inputs)
- [CSV Expectations](#csv-expectations)
- [What Gets Stored Locally](#what-gets-stored-locally)
- [Typical User Workflow](#typical-user-workflow)
- [Troubleshooting](#troubleshooting)
- [Current Limits](#current-limits)
- [Architecture](#architecture)
- [Testing](#testing)

## What The App Is For

Use this app when you want to answer questions like:

- "Did Trump-related posting activity cluster around certain market moves?"
- "Which X accounts consistently mention Trump and seem worth tracking?"
- "If I turn these posts into features, can I predict next-session SPY returns at all?"
- "How does this strategy compare with simple baselines like always-long SPY?"

## What You Need Before You Start

- Python and a virtual environment
- Internet access for the built-in Truth Social and market-data loaders
- Optional X CSVs if you want richer discovery and mention-account analysis

The app can start with just Truth Social plus market data, but the **Discovery** page is most useful when you also supply X/mention CSVs.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Contributor Workflow

Use Python `3.11` for local development and CI parity.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
bash scripts/ci.sh
```

`main` is intended to stay behind pull requests with a green `ci` check, so `bash scripts/ci.sh` is the local pre-push baseline.

On the first launch, the app may take a little longer because it can bootstrap local working datasets automatically.

The app opens as a multi-page Streamlit workbench with these sections:

- `Research View`
- `Datasets`
- `Discovery`
- `Models & Backtests`
- `Live Monitor`

## Hosting On Render

The repo includes a first-pass Render deployment blueprint in `render.yaml` plus a startup script at `scripts/start_render.sh`.

This deployment shape assumes:

- one Render web service
- one persistent disk mounted at `/var/data`
- public read-only browsing by default
- admin-only writes for dataset refreshes, watchlist edits, and run creation
- a background scheduler inside the same service for nightly full refreshes and 30-minute incremental refreshes

Basic deployment flow:

1. Create a new Render web service from this repo using `render.yaml`.
2. Add a persistent disk.
3. Set `ALLCAPS_ADMIN_PASSWORD` in Render.
4. Optionally set `ALPHA_VANTAGE_API_KEY` and `ALLCAPS_REMOTE_X_CSV_URL`.
5. Deploy and then bootstrap datasets from the app as an admin, or let the scheduler do the first bootstrap if the instance starts empty.

Important note:

- hosted state does **not** live in the repo checkout; it lives under `ALLCAPS_STATE_DIR`, which defaults to `/var/data` in Render

## Hosted Environment Variables

The hosted deployment uses these env vars:

- `ALLCAPS_STATE_DIR`: persistent writable state root, such as `/var/data`
- `ALLCAPS_PUBLIC_MODE`: when `true`, visitors are read-only until an admin unlocks the session
- `ALLCAPS_ADMIN_PASSWORD`: admin password for mutating actions
- `ALLCAPS_AUTO_BOOTSTRAP_ON_START`: keep `false` for Render so empty instances still start quickly
- `ALLCAPS_SCHEDULER_ENABLED`: enables the in-process scheduler
- `ALLCAPS_SCHEDULER_INCREMENTAL_MINUTES`: incremental refresh cadence, default `30`
- `ALLCAPS_SCHEDULER_FULL_HOUR`: nightly full-refresh hour in app time zone
- `ALLCAPS_SCHEDULER_FULL_MINUTE`: nightly full-refresh minute
- `ALLCAPS_REMOTE_X_CSV_URL`: optional remote X/mentions CSV URL for scheduled refreshes
- `ALPHA_VANTAGE_API_KEY`: optional intraday research key

## Recommended First Run

If you are new to the app, follow this order:

1. Open `Datasets`.
2. Click `Refresh full datasets`.
3. If you have X data, upload CSVs or point the app at a remote CSV URL.
4. Open `Discovery` and review the tracked-account universe.
5. Open `Research View` to inspect mapped posts and market context.
6. Open `Models & Backtests` and run a baseline walk-forward backtest.
7. Open `Live Monitor` only after you have at least one saved model run.

## How To Work With Each Page

### `Datasets`

This is the best place to start.

Use it to:

- refresh Truth Social, X CSV, and market datasets
- upload local CSVs for X posts or influential mentions
- set a remote CSV URL
- inspect the local dataset registry and source manifest
- preview the normalized post table the rest of the app uses

Buttons:

- `Refresh full datasets`: rebuilds the local working datasets from scratch
- `Incremental refresh`: polls for newer rows and appends them when possible

## `Discovery`

This page ranks X accounts that mention Trump and decides which ones belong in the active tracked universe.

Use it to:

- review the current active tracked accounts
- inspect the latest ranking snapshot
- manually `pin` an account so it stays included
- manually `suppress` an account so it stays excluded
- review and delete override history

Important note:

- If you do not provide X/mention data, the page may have little or nothing to rank
- The page uses historical effective dates, so overrides and account inclusion can be evaluated without lookahead leakage

## `Research View`

This is the descriptive analysis page.

Use it to:

- filter by date range, platform, keyword, and reshare behavior
- see S&P 500 price history with post-session markers
- inspect sentiment candlesticks built from mapped post sessions
- review session-level and post-level tables
- drill into an intraday SPY reaction window for a selected post

Important note:

- The research page is for exploration, not proof of causality
- Intraday drill-down depends on Alpha Vantage data and is optional

## `Models & Backtests`

This page turns the dataset into features, trains a model, and evaluates the trading idea.

Use it to:

- build the latest session-feature dataset
- train a next-session expected-return model
- run walk-forward optimization
- compare saved runs
- inspect strategy metrics, benchmark tables, leakage audits, and prediction misses

Inputs you can tune here include:

- run name
- whether semantic enrichment is enabled
- train / validation / test window sizes
- step size
- transaction costs
- ridge regularization
- threshold grid
- minimum-post-count grid
- tracked-account-weight grid

## `Live Monitor`

This page gives you a lightweight, polling-style live view after you already have a saved model.

Use it to:

- poll sources for new rows
- see the latest signal session and expected next-session return
- inspect the current suggested stance
- review prediction snapshot history over time

Important note:

- This page will not do much until you have already created at least one saved run in `Models & Backtests`

## Data Inputs

The app supports:

- built-in Truth Social historical archive loading
- local X CSV files
- local influential-mentions CSV files
- uploaded CSV files
- remote CSV URLs

Default local file locations:

- `data/realDonaldTrump_x_current_term.csv`
- `data/influential_x_mentions.csv`

Templates:

- `templates/x_posts_template.csv`
- `templates/x_mentions_template.csv`

## CSV Expectations

The parser is flexible, but the easiest path is to match the templates.

Typical X-post CSV columns:

- `timestamp`
- `text`
- `url`
- `is_retweet`
- `author_handle`
- `author_name`
- `author_id`
- `replies_count`
- `reblogs_count`
- `favourites_count`

Typical mention-account CSV columns:

- the same columns as above
- plus `mentions_trump`

## What Gets Stored Locally

The app stores working data in:

- `.workbench/workbench.duckdb`
- `.workbench/lake/*.parquet`
- `.workbench/artifacts/runs/*`

Truth Social raw archive caching uses:

- `.cache/truth_archive.csv`

For hosted deployments, the same paths are created under `ALLCAPS_STATE_DIR` instead of the repo root.

## Typical User Workflow

Here is the most common way to use the app from start to finish:

1. Refresh data in `Datasets`.
2. Add X or mention CSVs if you want account discovery.
3. Review the `Discovery` page and pin or suppress accounts you care about.
4. Explore `Research View` to sanity-check whether the mapped posts look reasonable.
5. Run a default experiment in `Models & Backtests`.
6. Compare the strategy against the built-in baselines.
7. If one run looks promising, use `Live Monitor` to watch the latest score from the most recent saved model.

## Troubleshooting

If `Discovery` is empty:

- make sure you loaded X mention data, not only Truth Social data
- refresh datasets again after adding CSVs

If `Research View` says it has no source data:

- open `Datasets` and run a full refresh

If `Models & Backtests` says there is no data:

- make sure both normalized posts and SPY daily data were loaded successfully

If `Live Monitor` says there is no saved model:

- create and save a run in `Models & Backtests` first

If the intraday drill-down fails:

- verify your Alpha Vantage setup or skip the intraday section

## Current Limits

- `SPY` is the only traded instrument in v1
- the strategy is `long / flat` only in v1
- semantic enrichment is optional and heuristic-backed by default
- the app is single-user and local-first
- the current research layer is exploratory, not production trading infrastructure

## Architecture

The code is organized as a modular monolith under `trump_workbench/`:

- `ingestion.py` for source adapters and post normalization
- `discovery.py` for influential-account ranking and tracked-universe history
- `enrichment.py` for optional semantic enrichment
- `features.py` for session mapping and feature engineering
- `modeling.py` for expected-return model training and prediction
- `backtesting.py` for walk-forward evaluation and strategy simulation
- `experiments.py` for saved runs and artifacts
- `research.py` for descriptive visualization helpers
- `ui.py` for the Streamlit app shell

## Testing

Install dev tooling:

```bash
pip install -r requirements-dev.txt
```

Run the full test suite:

```bash
python -m unittest discover -s tests -v
```

Run the configured coverage report:

```bash
python -m coverage run -m unittest discover -s tests
python -m coverage report -m
```
