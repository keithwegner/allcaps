from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .contracts import LiveMonitorConfig, LiveMonitorPinnedRun

LIVE_MONITOR_CONFIG_PATH = "live_monitor/config.json"
LIVE_ASSET_SNAPSHOT_COLUMNS = [
    "generated_at",
    "signal_session_date",
    "next_session_date",
    "asset_symbol",
    "run_id",
    "run_name",
    "feature_version",
    "model_version",
    "expected_return_score",
    "confidence",
    "threshold",
    "min_post_count",
    "post_count",
    "qualifies",
    "eligible_rank",
    "is_winner",
    "decision_source",
    "stance",
]
LIVE_DECISION_SNAPSHOT_COLUMNS = [
    "generated_at",
    "signal_session_date",
    "winning_asset",
    "winning_run_id",
    "decision_source",
    "fallback_mode",
    "stance",
    "eligible_asset_count",
    "runner_up_asset",
    "winner_score",
    "runner_up_score",
]
VALID_FALLBACK_MODES = {"SPY", "FLAT"}


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
    for column in LIVE_ASSET_SNAPSHOT_COLUMNS:
        if column not in board.columns:
            board[column] = pd.NA
    board["asset_symbol"] = board["asset_symbol"].fillna("").astype(str).str.upper()
    board["run_id"] = board["run_id"].fillna("").astype(str)
    board["run_name"] = board["run_name"].fillna("").astype(str)
    board["feature_version"] = board["feature_version"].fillna("").astype(str)
    board["model_version"] = board["model_version"].fillna("").astype(str)
    for column in ["expected_return_score", "confidence", "threshold", "post_count"]:
        board[column] = pd.to_numeric(board[column], errors="coerce").fillna(0.0)
    board["min_post_count"] = pd.to_numeric(board["min_post_count"], errors="coerce").fillna(1).astype(int)
    board["generated_at"] = pd.to_datetime(board["generated_at"], errors="coerce")
    board["signal_session_date"] = pd.to_datetime(board["signal_session_date"], errors="coerce")
    board["next_session_date"] = pd.to_datetime(board["next_session_date"], errors="coerce")
    board["qualifies"] = (board["expected_return_score"] > board["threshold"]) & (board["post_count"] >= board["min_post_count"])
    board["eligible_rank"] = pd.Series([pd.NA] * len(board), dtype="Int64")

    sort_columns = ["qualifies", "expected_return_score", "confidence", "asset_symbol", "run_id"]
    sort_ascending = [False, False, False, True, True]
    eligible = board.loc[board["qualifies"]].sort_values(sort_columns[1:], ascending=sort_ascending[1:]).reset_index(drop=True)
    if not eligible.empty:
        eligible["eligible_rank"] = pd.Series(range(1, len(eligible) + 1), dtype="Int64")
        for _, row in eligible.iterrows():
            mask = (board["asset_symbol"] == row["asset_symbol"]) & (board["run_id"] == row["run_id"])
            board.loc[mask, "eligible_rank"] = row["eligible_rank"]

    normalized_fallback = str(fallback_mode or "SPY").upper()
    if normalized_fallback not in VALID_FALLBACK_MODES:
        normalized_fallback = "SPY"

    winning_asset = ""
    winning_run_id = ""
    decision_source = "eligible" if not eligible.empty else "fallback"
    stance = "FLAT"
    winner_score = 0.0

    if not eligible.empty:
        winner = eligible.iloc[0]
        winning_asset = str(winner["asset_symbol"])
        winning_run_id = str(winner["run_id"])
        winner_score = float(winner["expected_return_score"])
        stance = f"LONG {winning_asset} NEXT SESSION"
    elif normalized_fallback == "SPY":
        spy_rows = board.loc[board["asset_symbol"] == "SPY"].sort_values(sort_columns[1:], ascending=sort_ascending[1:])
        if not spy_rows.empty:
            winner = spy_rows.iloc[0]
            winning_asset = "SPY"
            winning_run_id = str(winner["run_id"])
            winner_score = float(winner["expected_return_score"])
            stance = "LONG SPY NEXT SESSION"

    board["decision_source"] = decision_source
    board["is_winner"] = (board["asset_symbol"] == winning_asset) & (board["run_id"] == winning_run_id)
    board["stance"] = np.where(board["is_winner"], stance, "FLAT")
    board = board.sort_values(sort_columns, ascending=sort_ascending).reset_index(drop=True)

    runner_up_asset = ""
    runner_up_score = np.nan
    if winning_asset:
        runner_candidates = board.loc[~board["is_winner"]].sort_values(
            ["qualifies", "expected_return_score", "confidence", "asset_symbol"],
            ascending=[False, False, False, True],
        )
        if not runner_candidates.empty:
            runner = runner_candidates.iloc[0]
            runner_up_asset = str(runner["asset_symbol"])
            runner_up_score = float(runner["expected_return_score"])

    signal_session_date = (
        board.loc[board["is_winner"], "signal_session_date"].iloc[0]
        if winning_asset and board["is_winner"].any()
        else board["signal_session_date"].max()
    )
    generated_at = board["generated_at"].max()
    decision = pd.DataFrame(
        [
            {
                "generated_at": generated_at,
                "signal_session_date": signal_session_date,
                "winning_asset": winning_asset,
                "winning_run_id": winning_run_id,
                "decision_source": decision_source,
                "fallback_mode": normalized_fallback,
                "stance": stance,
                "eligible_asset_count": int(board["qualifies"].sum()),
                "runner_up_asset": runner_up_asset,
                "winner_score": float(winner_score),
                "runner_up_score": float(runner_up_score) if pd.notna(runner_up_score) else np.nan,
            },
        ],
        columns=LIVE_DECISION_SNAPSHOT_COLUMNS,
    )

    return board[LIVE_ASSET_SNAPSHOT_COLUMNS].copy(), decision

