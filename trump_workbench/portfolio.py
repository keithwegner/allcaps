from __future__ import annotations

import numpy as np
import pandas as pd

VALID_FALLBACK_MODES = {"SPY", "FLAT"}
PORTFOLIO_CANDIDATE_COLUMNS = [
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
    "target_available",
    "tradeable",
    "next_session_open",
    "next_session_close",
    "next_session_open_ts",
    "signal_qualifies",
    "qualifies",
    "eligible_rank",
    "is_winner",
    "decision_source",
    "stance",
]
PORTFOLIO_DECISION_COLUMNS = [
    "signal_session_date",
    "next_session_date",
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


def _normalize_portfolio_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    board = candidates.copy()
    for column in PORTFOLIO_CANDIDATE_COLUMNS:
        if column not in board.columns:
            board[column] = pd.NA
    board["signal_session_date"] = pd.to_datetime(board["signal_session_date"], errors="coerce")
    board["next_session_date"] = pd.to_datetime(board["next_session_date"], errors="coerce")
    board["next_session_open_ts"] = pd.to_datetime(board["next_session_open_ts"], errors="coerce", utc=True)
    board["asset_symbol"] = board["asset_symbol"].fillna("").astype(str).str.upper()
    board["run_id"] = board["run_id"].fillna("").astype(str)
    board["run_name"] = board["run_name"].fillna("").astype(str)
    board["feature_version"] = board["feature_version"].fillna("").astype(str)
    board["model_version"] = board["model_version"].fillna("").astype(str)
    for column in ["expected_return_score", "confidence", "threshold", "post_count", "next_session_open", "next_session_close"]:
        board[column] = pd.to_numeric(board[column], errors="coerce")
    board["min_post_count"] = pd.to_numeric(board["min_post_count"], errors="coerce").fillna(1).astype(int)
    board["target_available"] = pd.Series(board["target_available"], dtype="boolean").fillna(False).astype(bool)
    board["tradeable"] = (
        pd.Series(board["tradeable"], dtype="boolean")
        .fillna(pd.Series(board["target_available"], dtype="boolean"))
        .fillna(False)
        .astype(bool)
    )
    return board


def rank_portfolio_candidates(
    candidates: pd.DataFrame,
    fallback_mode: str,
    require_tradeable: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return (
            pd.DataFrame(columns=PORTFOLIO_CANDIDATE_COLUMNS),
            pd.DataFrame(columns=PORTFOLIO_DECISION_COLUMNS),
        )

    board = _normalize_portfolio_candidates(candidates)
    board["signal_qualifies"] = (
        (board["expected_return_score"].fillna(0.0) > board["threshold"].fillna(0.0))
        & (board["post_count"].fillna(0.0) >= board["min_post_count"].fillna(1))
    )
    board["qualifies"] = board["signal_qualifies"] & (board["tradeable"] if require_tradeable else True)
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
        spy_rows = board.loc[board["asset_symbol"] == "SPY"].copy()
        if require_tradeable:
            spy_rows = spy_rows.loc[spy_rows["tradeable"]]
        spy_rows = spy_rows.sort_values(sort_columns[1:], ascending=sort_ascending[1:])
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

    decision_payload = {
        "signal_session_date": board["signal_session_date"].max(),
        "next_session_date": (
            board.loc[board["is_winner"], "next_session_date"].iloc[0]
            if winning_asset and board["is_winner"].any()
            else board["next_session_date"].max()
        ),
        "winning_asset": winning_asset,
        "winning_run_id": winning_run_id,
        "decision_source": decision_source,
        "fallback_mode": normalized_fallback,
        "stance": stance,
        "eligible_asset_count": int(board["qualifies"].sum()),
        "runner_up_asset": runner_up_asset,
        "winner_score": float(winner_score),
        "runner_up_score": float(runner_up_score) if pd.notna(runner_up_score) else np.nan,
    }
    if "generated_at" in board.columns:
        decision_payload["generated_at"] = pd.to_datetime(board["generated_at"], errors="coerce").max()

    decision_columns = list(PORTFOLIO_DECISION_COLUMNS)
    if "generated_at" in decision_payload:
        decision_columns = ["generated_at", *decision_columns]
    decision = pd.DataFrame([decision_payload], columns=decision_columns)
    return board, decision


def build_portfolio_decision_history(
    candidates: pd.DataFrame,
    fallback_mode: str,
    require_tradeable: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return (
            pd.DataFrame(columns=PORTFOLIO_CANDIDATE_COLUMNS),
            pd.DataFrame(columns=PORTFOLIO_DECISION_COLUMNS),
        )

    normalized = _normalize_portfolio_candidates(candidates)
    normalized = normalized.dropna(subset=["signal_session_date"]).copy()
    if normalized.empty:
        return (
            pd.DataFrame(columns=PORTFOLIO_CANDIDATE_COLUMNS),
            pd.DataFrame(columns=PORTFOLIO_DECISION_COLUMNS),
        )

    all_board_rows: list[pd.DataFrame] = []
    all_decision_rows: list[pd.DataFrame] = []
    grouped = normalized.groupby(normalized["signal_session_date"].dt.normalize(), sort=True)
    for _, group in grouped:
        board, decision = rank_portfolio_candidates(
            group.reset_index(drop=True),
            fallback_mode=fallback_mode,
            require_tradeable=require_tradeable,
        )
        all_board_rows.append(board)
        all_decision_rows.append(decision)

    candidate_board = pd.concat(all_board_rows, ignore_index=True) if all_board_rows else pd.DataFrame(columns=PORTFOLIO_CANDIDATE_COLUMNS)
    decision_history = pd.concat(all_decision_rows, ignore_index=True) if all_decision_rows else pd.DataFrame(columns=PORTFOLIO_DECISION_COLUMNS)
    if not candidate_board.empty:
        candidate_board = candidate_board.sort_values(
            ["signal_session_date", "qualifies", "expected_return_score", "asset_symbol"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
    if not decision_history.empty:
        decision_history = decision_history.sort_values("signal_session_date").reset_index(drop=True)
    return candidate_board, decision_history
