# Trump Social Trading Research Workbench

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![API: FastAPI](https://img.shields.io/badge/api-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Primary UI: React](https://img.shields.io/badge/primary%20ui-React%20%2B%20Vite-61DAFB?logo=react&logoColor=black)](https://vite.dev/)
[![Fallback UI: Streamlit](https://img.shields.io/badge/fallback%20ui-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Storage: DuckDB + Parquet](https://img.shields.io/badge/storage-DuckDB%20%2B%20Parquet-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![CI](https://github.com/keithwegner/allcaps/actions/workflows/ci.yml/badge.svg)](https://github.com/keithwegner/allcaps/actions/workflows/ci.yml)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange)](https://github.com/keithwegner/allcaps)

This is a React + FastAPI workbench for researching whether President Donald Trump's social posts, plus posts from influential X accounts that mention him, help predict next-session market behavior across `SPY` and tracked assets. The Python analytics engine remains intact, and the Streamlit app is kept as a fallback shell while legacy-only surfaces are retired.

In plain English, the app helps you:

- collect Trump-related social data
- decide which X accounts matter enough to track
- turn that data into trading features
- backtest **long / flat** strategies for `SPY` and tracked assets
- compare saved runs, portfolio variants, and inspect why a run did or did not work
- monitor live portfolio decisions and a paper-trading audit trail

## Table of Contents

- [What The App Is For](#what-the-app-is-for)
- [What You Need Before You Start](#what-you-need-before-you-start)
- [Quick Start](#quick-start)
- [Contributor Workflow](#contributor-workflow)
- [React + FastAPI Primary App](#react--fastapi-primary-app)
- [Run With Docker](#run-with-docker)
- [Hosting On Render](#hosting-on-render)
- [Hosted Environment Variables](#hosted-environment-variables)
- [Recommended First Run](#recommended-first-run)
- [Trump Truth Social-Only Workflow](#trump-truth-social-only-workflow)
- [How To Work With Each Page](#how-to-work-with-each-page)
  - [`Data Admin`](#data-admin)
  - [`Discovery`](#discovery)
  - [`Research`](#research)
  - [`Model Training` and `Run Explorer`](#model-training-and-run-explorer)
  - [`Live Ops` and `Paper + Performance`](#live-ops-and-paper--performance)
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
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn trump_workbench.api:app --reload --host 127.0.0.1 --port 8000
```

Run the React frontend in a second terminal:

```bash
npm install --prefix frontend
npm run dev --prefix frontend
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

By default, Vite proxies `/api/*` requests to the local FastAPI server on `127.0.0.1:8000`. Leave `VITE_ALLCAPS_API_BASE_URL` unset unless your backend is running somewhere else.

Fallback Streamlit shell:

```bash
source .venv/bin/activate
python -m streamlit run app.py
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

## React + FastAPI Primary App

React + FastAPI is now the primary day-to-day application surface. The current web app includes:

- a FastAPI backend in `trump_workbench/api.py`
- a React + TypeScript + Vite frontend in `frontend/`
- API-backed views for status, research, Discovery, saved runs, replay, live decisions, paper portfolios, performance diagnostics, and dataset administration
- read-only research surfaces with sentiment filters, Asset Lab, interactive Plotly charts, Narrative Lab outputs, and research ZIP export
- admin-gated React controls for Discovery overrides, Data Admin, Model Training, Live Ops, and paper portfolio workflows

Run the API locally:

```bash
source .venv/bin/activate
python -m uvicorn trump_workbench.api:app --reload --host 127.0.0.1 --port 8000
```

Use `python -m uvicorn` from the active project virtual environment. Running bare `uvicorn` can accidentally use a global/Homebrew Python install that does not have this repository's dependencies installed.

Run the frontend locally in a second terminal:

```bash
npm install --prefix frontend
npm run dev --prefix frontend
```

Open:

- [http://127.0.0.1:5173](http://127.0.0.1:5173)

Optional frontend env var:

- `VITE_ALLCAPS_API_BASE_URL=http://127.0.0.1:8000`

Normally this is not required for local development because the Vite dev server proxies same-origin `/api/*` requests to FastAPI. If the React UI shows API load errors while the network tab shows HTTP 200 responses, confirm this variable is not pointing at the frontend dev server.

The React app is mixed read/write. Public users can browse research, saved runs, replay, live history, paper history, and performance pages. Mutating Discovery, Data Admin, Model Training, Live Ops, and paper controls require an admin token from `/api/admin/session`.

Run browser UI tests for the React app:

```bash
npm exec --prefix frontend playwright install chromium
npm run test:ui --prefix frontend
```

Run measured React unit/component coverage:

```bash
npm run test:coverage --prefix frontend
```

The frontend coverage gate requires at least 90% statements, functions, and lines. Branch coverage is tracked with a lower ratchet because V8 counts many JSX optional-render fallbacks as branches; raise it as branch-focused tests are added.

The React app opens as a tabbed workbench with these primary sections:

- `Overview`
- `Research`
- `Discovery`
- `Run Explorer`
- `Replay`
- `Model Training`
- `Data Admin`
- `Live Ops`
- `Paper + Performance`

Streamlit remains available for fallback validation and the legacy Alpha Vantage SPY intraday drill-down. New day-to-day workflows should use React first.

## Run With Docker

The repo includes local Docker packaging for a single-user browser workflow. The default Compose package now serves the **React + FastAPI web app** from one container.

Default behavior:

- builds the React frontend during the Docker image build
- serves the built frontend from FastAPI with `/api/*` handled by the backend
- opens on [http://127.0.0.1:8000](http://127.0.0.1:8000)
- keeps the app private and writable by default
- stores runtime state in a named Docker volume mounted at `/var/data`
- leaves the scheduler off by default

Start it:

```bash
docker compose up --build
```

Open:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

Stop it:

```bash
docker compose down
```

Remove the container and the named state volume:

```bash
docker compose down -v
```

Optional env vars:

1. Copy `.env.docker.example` to `.env`
2. Set any values you want, such as:
   - `ALPHA_VANTAGE_API_KEY`
   - `ALLCAPS_REMOTE_X_CSV_URL`

Optional local CSV mount:

If you want the container to read local CSV files from `./data`, start with the extra Compose file:

```bash
docker compose -f compose.yaml -f compose.data.yaml up --build
```

That mounts:

- `./data` -> `/app/data` (read-only)

Optional host bind mount for persistent state:

The default setup uses a named Docker volume. If you prefer to keep state in a visible host folder, replace the volume mapping in `compose.yaml` with:

```yaml
volumes:
  - ./docker-state:/var/data
```

That will store DuckDB, parquet data, cache files, and run artifacts in `./docker-state`.

Optional Streamlit fallback in Docker:

The image still includes the Streamlit shell. To run it instead of the web-first UI, override the runtime and port:

```bash
docker build -t allcaps .
docker run --rm -p 127.0.0.1:8501:8501 \
  -v allcaps_state:/var/data \
  -e ALLCAPS_RUNTIME=streamlit \
  -e PORT=8501 \
  allcaps
```

Then open [http://127.0.0.1:8501](http://127.0.0.1:8501).

## Hosting On Render

The repo includes a first-pass Render deployment blueprint in `render.yaml` plus a startup script at `scripts/start_render.sh`.

Current hosted packaging still launches the fallback Streamlit shell. A production React + FastAPI hosting cutover should update the Render start command to serve the FastAPI API plus the built React frontend, or split those into explicit web services. Until then, Render is useful for the existing single-service Streamlit deployment shape.

This deployment shape assumes:

- one Render web service
- one persistent disk mounted at `/var/data`
- public read-only browsing by default in the fallback shell
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

1. Open React `Data Admin`.
2. Run `Bootstrap`, `Full`, or `Incremental` refresh.
3. If you have X data, upload CSVs or point the app at a remote CSV URL.
4. Open `Discovery` and review the tracked-account universe if you loaded X/mention data.
5. Open `Research` to inspect mapped posts, narratives, Asset Lab charts, and market context.
6. Open `Model Training` and run a baseline or joint portfolio model.
7. Open `Run Explorer` to inspect metrics, variants, diagnostics, and comparisons.
8. Open `Live Ops` to pin a deployment run, inspect the live board, and optionally enable paper trading.

## Trump Truth Social-Only Workflow

Use this workflow when you want to review sentiment based only on Donald Trump's Truth Social posts.

1. Open `Data Admin`.
2. Click `Full` or `Bootstrap`.
3. Open `Research`.
4. Confirm `Platforms` is set to `Truth Social`.
5. Confirm `Trump-authored only` is enabled.
6. Ignore `Discovery` unless you also want to load X/mention CSVs and rank non-Trump X accounts.

When the stored dataset contains only Truth Social rows, the app auto-detects that mode and seeds the Research controls to the Truth-only scope.

## How To Work With Each Page

### `Data Admin`

This is the best place to start.

Use it to:

- refresh Truth Social, X CSV, and market datasets
- upload local CSVs for X posts or influential mentions
- set a remote CSV URL
- inspect the local dataset registry and source manifest
- preview the normalized post table the rest of the app uses

The React `Data Admin` tab covers day-to-day dataset operations through FastAPI: operating mode, scheduler status, watchlist save/reset, CSV URL/upload inputs, `bootstrap`/`full`/`incremental` refresh jobs, refresh-job history, data health, registry, and manifests. These writes require an admin unlock in public mode.

Buttons:

- `Full`: rebuilds the working datasets from stored/source inputs
- `Bootstrap`: initializes missing core datasets
- `Incremental`: polls for newer rows and appends them when possible

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

## `Research`

This is the descriptive analysis page.

Use it to:

- filter by date range, platform, keyword, and reshare behavior
- restrict the view to posts authored by Donald Trump's account
- see S&P 500 price history with post-session markers
- inspect sentiment candlesticks built from mapped post sessions
- review session-level and post-level tables
- download a ZIP research pack for the current filters
- compare selected assets, event-study windows, and stored intraday reaction windows

Important note:

- The research page is for exploration, not proof of causality
- The primary React intraday workflow uses stored `asset_intraday` rows
- The older Alpha Vantage SPY drill-down remains available only in the fallback Streamlit shell
- For a Truth Social-only review, set `Platforms` to `Truth Social` and enable `Trump-authored only`

## `Model Training` and `Run Explorer`

These tabs turn the dataset into features, train models, and inspect saved results.

Use it to:

- run single-asset, saved-run portfolio, or joint portfolio training jobs
- compare saved runs and portfolio variants
- inspect strategy metrics, benchmark tables, leakage audits, feature contributions, and prediction misses
- reconstruct historical asset-model decisions in `Replay`

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

## `Live Ops` and `Paper + Performance`

This page gives you a live decision console after you already have a saved joint portfolio run.

Use it to:

- pin a saved joint portfolio run for live monitoring
- inspect the ranked live asset board and current suggested stance
- review explanation details for the winner and runner-up
- review persisted live board and decision history over time
- enable paper trading, inspect the paper decision journal, and track realized equity vs `SPY`
- inspect the `Performance Observatory` for warn-only diagnostics on paper PnL, alpha, drawdown, fallback rate, score calibration, and live candidate-board drift

Important note:

- These pages will not do much until you have already created at least one saved joint portfolio run in `Model Training`
- The `Performance Observatory` is informational only. It does not retrain models, block live decisions, or change paper-trading behavior.

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

1. Refresh data in `Data Admin`.
2. Add X or mention CSVs if you want account discovery.
3. Review the `Discovery` page and pin or suppress accounts you care about.
4. Explore `Research` to sanity-check whether the mapped posts look reasonable.
5. Run a default job in `Model Training`.
6. Compare the strategy against the built-in baselines in `Run Explorer`.
7. Save a joint portfolio run if you want live portfolio monitoring.
8. Use `Live Ops` and `Paper + Performance` to watch the latest board, pin a deployment run, and optionally track paper PnL.

## Troubleshooting

If `Discovery` is empty:

- make sure you loaded X mention data, not only Truth Social data
- refresh datasets again after adding CSVs

If `Research` says it has no source data:

- open `Data Admin` and run a full refresh

If `Model Training` says there is no data:

- make sure both normalized posts and SPY daily data were loaded successfully

If `Live Ops` says there is no saved model:

- create and save a joint portfolio run in `Model Training` first

If `Paper + Performance` shows no paper portfolio history:

- save and pin a joint portfolio run first
- enable paper trading from the `Paper Portfolio` tab
- use polling or let the scheduler persist live decisions before the next session opens

If the `Performance Observatory` has limited diagnostics:

- wait for paper decisions to settle against next-session prices
- use polling or the scheduler to build more live snapshot history
- treat early insufficient-sample warnings as expected until several sessions have settled

If the intraday drill-down fails:

- use the React Asset Lab stored-intraday workflow first
- verify your Alpha Vantage setup only if you are using the fallback Streamlit SPY drill-down

## Current Limits

- the strategy is `long / flat` only
- the portfolio allocator holds at most one asset per session
- paper trading is simulated only; there is no broker integration or order routing
- semantic enrichment is optional and heuristic-backed by default
- hosted mode is still single-instance and admin-gated rather than full multi-user auth
- the research and live layers are decision-support tools, not production trading infrastructure
- Streamlit remains in the repo as a fallback shell; it is not the primary user interface

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
- `ui.py` for the fallback Streamlit app shell
- `api.py` for the primary FastAPI surface
- `frontend/` for the primary React + TypeScript web app

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

Run the React browser UI tests:

```bash
npm exec --prefix frontend playwright install chromium
npm run test:ui --prefix frontend
```

Run the React unit/component coverage gate:

```bash
npm run test:coverage --prefix frontend
```

Run the same checks as GitHub Actions:

```bash
bash scripts/ci.sh
```
