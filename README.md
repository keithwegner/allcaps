# Trump Social Trading Research Workbench

This project is now a modular single-user research workbench for evaluating and optimizing **SPY trading strategies** driven by:

- Donald Trump's Truth Social posts
- X accounts that mention him
- a dynamic universe of influential mention accounts

## What the app does

- ingests and normalizes Truth Social and X/mention CSV data
- stores datasets locally in **DuckDB + Parquet**
- discovers and auto-includes influential X accounts mentioning Trump
- supports manual `pin` / `suppress` overrides for the discovered account universe
- builds per-session feature datasets for **next-session SPY expected-return** modeling
- supports optional cached semantic enrichment on top of deterministic features
- runs walk-forward backtests for a **long / flat SPY** strategy
- compares the strategy against benchmark baselines such as always-long, Trump-only, and tracked-accounts-only
- runs leakage-oriented diagnostics around feature cutoffs and next-session target alignment
- saves experiment runs, trades, predictions, and model artifacts locally
- preserves the original descriptive research view with:
  - S&P 500 close overlay
  - sentiment candlesticks
  - post and session tables
  - optional Alpha Vantage intraday SPY drill-down

## Architecture

The old single-file app has been split into a modular monolith under `trump_workbench/`:

- `ingestion.py` for source adapters and post normalization
- `discovery.py` for influential-account ranking and tracked-universe history
- `enrichment.py` for optional cached semantic enrichment
- `features.py` for session mapping and feature engineering
- `modeling.py` for expected-return model training and prediction
- `backtesting.py` for walk-forward evaluation and strategy simulation
- `experiments.py` for saved runs and artifacts
- `research.py` for the preserved visualization workflow
- `ui.py` for the Streamlit workbench

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Local storage

The workbench stores data in:

- `.workbench/workbench.duckdb`
- `.workbench/lake/*.parquet`
- `.workbench/artifacts/runs/*`

Truth Social raw archive caching still uses:

- `.cache/truth_archive.csv`

## Data inputs

The app supports:

- Truth Social historical archive via built-in source adapter
- local X CSVs
- local influential-mentions CSVs
- uploaded CSVs
- remote CSV URLs

The default local files are:

- `data/realDonaldTrump_x_current_term.csv`
- `data/influential_x_mentions.csv`

Templates are included in:

- `templates/x_posts_template.csv`
- `templates/x_mentions_template.csv`

## Notes

- `SPY` is the only traded instrument in v1.
- The strategy is `long / flat` only in v1.
- Semantic enrichment is optional and cached; the full system works with it disabled.
- The current semantic enrichment layer is local and heuristic-backed, with a pluggable interface for richer providers later.
- The research view remains descriptive; it is not a claim of causality.
