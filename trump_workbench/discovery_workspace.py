from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .contracts import MANUAL_OVERRIDE_COLUMNS, RANKING_HISTORY_COLUMNS, TRACKED_ACCOUNT_COLUMNS
from .discovery import DiscoveryService
from .research_workspace import _figure_json, _frame_records, detect_source_mode
from .storage import DuckDBStore

TABLE_LIMIT = 250


def _series_or_default(frame: pd.DataFrame, column: str, default: Any = False) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def latest_ranking_snapshot(ranking_history: pd.DataFrame) -> pd.DataFrame:
    required = {"author_account_id", "ranked_at"}
    if ranking_history.empty or not required.issubset(ranking_history.columns):
        return pd.DataFrame(columns=RANKING_HISTORY_COLUMNS)

    snapshot = ranking_history.copy()
    for column in RANKING_HISTORY_COLUMNS:
        if column not in snapshot.columns:
            snapshot[column] = pd.NA
    snapshot["ranked_at"] = pd.to_datetime(snapshot["ranked_at"], errors="coerce")
    snapshot = snapshot.dropna(subset=["ranked_at"]).copy()
    if snapshot.empty:
        return pd.DataFrame(columns=RANKING_HISTORY_COLUMNS)
    return (
        snapshot.sort_values(["ranked_at", "discovery_score"], ascending=[False, False])
        .drop_duplicates("author_account_id")
        .sort_values("discovery_score", ascending=False)
        .reset_index(drop=True)[RANKING_HISTORY_COLUMNS]
    )


def _x_candidate_posts(posts: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        return posts.copy()
    platform = _series_or_default(posts, "source_platform", "").fillna("").astype(str)
    mentions = _series_or_default(posts, "mentions_trump", False).fillna(False).astype(bool)
    is_trump = _series_or_default(posts, "author_is_trump", False).fillna(False).astype(bool)
    return posts.loc[(platform == "X") & mentions & (~is_trump)].copy()


def _active_accounts(discovery_service: DiscoveryService, tracked_accounts: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    if tracked_accounts.empty or posts.empty or "post_timestamp" not in posts.columns:
        return pd.DataFrame(columns=TRACKED_ACCOUNT_COLUMNS)
    if not {"effective_from", "effective_to", "status"}.issubset(tracked_accounts.columns):
        return pd.DataFrame(columns=TRACKED_ACCOUNT_COLUMNS)
    timestamps = pd.to_datetime(posts["post_timestamp"], errors="coerce")
    timestamps = timestamps.dropna()
    if timestamps.empty:
        return pd.DataFrame(columns=TRACKED_ACCOUNT_COLUMNS)
    active = discovery_service.current_active_accounts(tracked_accounts, as_of=timestamps.max())
    for column in TRACKED_ACCOUNT_COLUMNS:
        if column not in active.columns:
            active[column] = pd.NA
    return active[TRACKED_ACCOUNT_COLUMNS].reset_index(drop=True)


def _normalize_recent_history(ranking_history: pd.DataFrame) -> pd.DataFrame:
    if ranking_history.empty or "ranked_at" not in ranking_history.columns:
        return pd.DataFrame(columns=RANKING_HISTORY_COLUMNS)
    history = ranking_history.copy()
    for column in RANKING_HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = pd.NA
    history["ranked_at"] = pd.to_datetime(history["ranked_at"], errors="coerce")
    history = history.dropna(subset=["ranked_at"]).copy()
    if history.empty:
        return pd.DataFrame(columns=RANKING_HISTORY_COLUMNS)
    return history.sort_values(["ranked_at", "discovery_score"], ascending=[False, False]).head(TABLE_LIMIT)[RANKING_HISTORY_COLUMNS]


def _top_accounts_chart(latest_rankings: pd.DataFrame) -> dict[str, Any]:
    if latest_rankings.empty:
        fig = go.Figure()
        fig.update_layout(title="Top discovered accounts", xaxis_title="Discovery score", yaxis_title="Account")
        return _figure_json(fig)

    chart_data = latest_rankings.head(15).sort_values("discovery_score")
    labels = chart_data["author_handle"].fillna("").astype(str).replace("", "[unknown]")
    fig = go.Figure(
        go.Bar(
            x=chart_data["discovery_score"],
            y=labels,
            orientation="h",
            marker_color="#14532d",
        ),
    )
    fig.update_layout(
        title="Top discovered accounts",
        xaxis_title="Discovery score",
        yaxis_title="Account",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return _figure_json(fig)


def _ranking_trend_chart(ranking_history: pd.DataFrame) -> dict[str, Any]:
    if ranking_history.empty or not {"ranked_at", "author_account_id"}.issubset(ranking_history.columns):
        fig = go.Figure()
        fig.update_layout(title="Discovery ranking history", xaxis_title="Ranked at", yaxis_title="Accounts")
        return _figure_json(fig)
    history = ranking_history.copy()
    history["ranked_at"] = pd.to_datetime(history["ranked_at"], errors="coerce").dt.normalize()
    history = history.dropna(subset=["ranked_at"]).copy()
    if history.empty:
        fig = go.Figure()
        fig.update_layout(title="Discovery ranking history", xaxis_title="Ranked at", yaxis_title="Accounts")
        return _figure_json(fig)
    if "final_selected" not in history.columns:
        history["final_selected"] = False
    trend = (
        history.groupby("ranked_at", as_index=False)
        .agg(
            ranked_accounts=("author_account_id", "nunique"),
            selected_accounts=("final_selected", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
        )
        .sort_values("ranked_at")
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["ranked_at"], y=trend["ranked_accounts"], mode="lines+markers", name="Ranked accounts"))
    fig.add_trace(go.Scatter(x=trend["ranked_at"], y=trend["selected_accounts"], mode="lines+markers", name="Selected accounts"))
    fig.update_layout(title="Discovery ranking history", xaxis_title="Ranked at", yaxis_title="Accounts")
    return _figure_json(fig)


def _message(posts: pd.DataFrame, source_mode: dict[str, Any], x_candidates: pd.DataFrame, latest_rankings: pd.DataFrame) -> str:
    if posts.empty:
        return "Refresh datasets first so Discovery can inspect X mention data."
    if source_mode.get("mode") == "truth_only":
        return (
            "This dataset is currently Truth Social-only. Discovery ranks non-Trump X accounts that mention Trump, "
            "so it is optional for reviewing sentiment based only on Donald Trump's Truth Social posts. "
            "Load X/mention CSVs in Datasets if you want account discovery."
        )
    if x_candidates.empty:
        return "No non-Trump X posts mentioning Trump are available for Discovery ranking."
    if latest_rankings.empty:
        return "Discovery rankings have not been built yet. Refresh datasets in the admin flow to populate account discovery."
    return ""


def build_discovery_workspace(store: DuckDBStore, discovery_service: DiscoveryService | None = None) -> dict[str, Any]:
    discovery_service = discovery_service or DiscoveryService()
    posts = store.read_frame("normalized_posts")
    tracked_accounts = store.read_frame("tracked_accounts")
    ranking_history = store.read_frame("account_rankings")
    overrides = discovery_service.normalize_manual_overrides(store.read_frame("manual_account_overrides"))

    source_mode = detect_source_mode(posts)
    x_candidates = _x_candidate_posts(posts)
    latest_rankings = latest_ranking_snapshot(ranking_history)
    active_accounts = _active_accounts(discovery_service, tracked_accounts, posts)
    recent_history = _normalize_recent_history(ranking_history)
    message = _message(posts, source_mode, x_candidates, latest_rankings)
    latest_ranked_at = (
        pd.to_datetime(latest_rankings["ranked_at"], errors="coerce").max()
        if not latest_rankings.empty and "ranked_at" in latest_rankings.columns
        else pd.NaT
    )
    selected_status = latest_rankings["selected_status"].fillna("").astype(str) if "selected_status" in latest_rankings.columns else pd.Series(dtype=str)
    override_actions = overrides["action"].fillna("").astype(str) if "action" in overrides.columns else pd.Series(dtype=str)

    summary = {
        "post_count": int(len(posts)),
        "x_candidate_post_count": int(len(x_candidates)),
        "active_account_count": int(len(active_accounts)),
        "latest_ranking_count": int(len(latest_rankings)),
        "selected_account_count": int((selected_status != "excluded").sum()) if not selected_status.empty else 0,
        "pinned_account_count": int((selected_status == "pinned").sum()) if not selected_status.empty else 0,
        "suppressed_latest_count": int(latest_rankings["suppressed_by_override"].fillna(False).astype(bool).sum()) if "suppressed_by_override" in latest_rankings.columns else 0,
        "override_count": int(len(overrides)),
        "pin_override_count": int((override_actions == "pin").sum()) if not override_actions.empty else 0,
        "suppress_override_count": int((override_actions == "suppress").sum()) if not override_actions.empty else 0,
    }

    return {
        "ready": bool(message == ""),
        "message": message,
        "source_mode": source_mode,
        "latest_ranked_at": latest_ranked_at,
        "summary": summary,
        "charts": {
            "top_discovered_accounts": _top_accounts_chart(latest_rankings),
            "ranking_history": _ranking_trend_chart(recent_history),
        },
        "active_accounts": _frame_records(active_accounts.head(TABLE_LIMIT)),
        "latest_rankings": _frame_records(latest_rankings.head(TABLE_LIMIT)),
        "override_history": _frame_records(overrides.head(TABLE_LIMIT) if not overrides.empty else pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)),
        "recent_ranking_history": _frame_records(recent_history.head(TABLE_LIMIT)),
    }
