from __future__ import annotations

from typing import Any

import pandas as pd

from .contracts import LiveMonitorConfig, LiveMonitorPinnedRun
from .portfolio import PORTFOLIO_CANDIDATE_COLUMNS, PORTFOLIO_DECISION_COLUMNS, VALID_FALLBACK_MODES, rank_portfolio_candidates

LIVE_MONITOR_CONFIG_PATH = "live_monitor/config.json"
LIVE_ASSET_SNAPSHOT_COLUMNS = [
    "generated_at",
    *PORTFOLIO_CANDIDATE_COLUMNS,
]
LIVE_DECISION_SNAPSHOT_COLUMNS = [
    "generated_at",
    *PORTFOLIO_DECISION_COLUMNS,
]


def seed_live_monitor_config(runs: pd.DataFrame) -> LiveMonitorConfig | None:
    if runs.empty or "run_id" not in runs.columns:
        return None
    normalized = runs.copy()
    if "target_asset" not in normalized.columns:
        normalized["target_asset"] = "SPY"
    normalized["target_asset"] = normalized["target_asset"].fillna("SPY").astype(str).str.upper()
    if "created_at" in normalized.columns:
        normalized = normalized.sort_values("created_at", ascending=False)

    pinned_runs: list[LiveMonitorPinnedRun] = []
    for asset_symbol, group in normalized.groupby("target_asset", sort=False):
        row = group.iloc[0]
        pinned_runs.append(
            LiveMonitorPinnedRun(
                asset_symbol=str(asset_symbol).upper(),
                run_id=str(row.get("run_id", "") or ""),
                run_name=str(row.get("run_name", "") or ""),
            ),
        )
    pinned_runs = sorted(pinned_runs, key=lambda item: (item.asset_symbol != "SPY", item.asset_symbol))
    return LiveMonitorConfig(fallback_mode="SPY", pinned_runs=pinned_runs)


def validate_live_monitor_config(config: LiveMonitorConfig | None, runs: pd.DataFrame) -> list[str]:
    if config is None:
        return ["Save a pinned live model set before using the decision console."]

    errors: list[str] = []
    fallback_mode = str(config.fallback_mode or "SPY").upper()
    if fallback_mode not in VALID_FALLBACK_MODES:
        errors.append("Fallback mode must be `SPY` or `FLAT`.")

    if not config.pinned_runs:
        errors.append("At least one pinned run is required.")
        return errors

    run_lookup = runs.copy()
    if not run_lookup.empty:
        if "target_asset" not in run_lookup.columns:
            run_lookup["target_asset"] = "SPY"
        run_lookup["target_asset"] = run_lookup["target_asset"].fillna("SPY").astype(str).str.upper()
        run_lookup = run_lookup.set_index("run_id", drop=False)

    seen_assets: set[str] = set()
    spy_count = 0
    for pinned in config.pinned_runs:
        asset_symbol = str(pinned.asset_symbol or "").upper()
        if not asset_symbol:
            errors.append("Pinned runs must include an asset symbol.")
            continue
        if asset_symbol in seen_assets:
            errors.append(f"Only one pinned run is allowed for `{asset_symbol}`.")
            continue
        seen_assets.add(asset_symbol)
        if asset_symbol == "SPY":
            spy_count += 1

        run_id = str(pinned.run_id or "")
        if not run_id:
            errors.append(f"`{asset_symbol}` must have a saved run selected.")
            continue
        if run_lookup.empty or run_id not in run_lookup.index:
            errors.append(f"Pinned run `{run_id}` for `{asset_symbol}` is not available anymore.")
            continue
        row_asset = str(run_lookup.loc[run_id, "target_asset"] or "SPY").upper()
        if row_asset != asset_symbol:
            errors.append(f"Pinned run `{run_id}` targets `{row_asset}`, not `{asset_symbol}`.")

    if spy_count != 1:
        errors.append("Exactly one pinned `SPY` run is required.")
    return errors


def rank_live_asset_snapshots(snapshots: pd.DataFrame, fallback_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if snapshots.empty:
        return (
            pd.DataFrame(columns=LIVE_ASSET_SNAPSHOT_COLUMNS),
            pd.DataFrame(columns=LIVE_DECISION_SNAPSHOT_COLUMNS),
        )

    board = snapshots.copy()
    board["generated_at"] = pd.to_datetime(board["generated_at"], errors="coerce")
    ranked_board, decision = rank_portfolio_candidates(board, fallback_mode=fallback_mode, require_tradeable=False)
    for column in LIVE_ASSET_SNAPSHOT_COLUMNS:
        if column not in ranked_board.columns:
            ranked_board[column] = pd.NA
    for column in LIVE_DECISION_SNAPSHOT_COLUMNS:
        if column not in decision.columns:
            decision[column] = pd.NA
    return ranked_board[LIVE_ASSET_SNAPSHOT_COLUMNS].copy(), decision[LIVE_DECISION_SNAPSHOT_COLUMNS].copy()
