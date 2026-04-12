from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import EASTERN
from .contracts import TRACKED_ACCOUNT_COLUMNS, TrackedAccount
from .utils import ensure_tz_naive_date, stable_text_id


class DiscoveryService:
    def rank_candidates(
        self,
        posts: pd.DataFrame,
        accounts: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if posts.empty:
            return pd.DataFrame()

        candidates = posts.loc[
            (posts["source_platform"] == "X")
            & posts["mentions_trump"]
            & (~posts["author_is_trump"])
        ].copy()
        if candidates.empty:
            return pd.DataFrame()

        candidates["active_date"] = candidates["post_timestamp"].dt.tz_convert(EASTERN).dt.normalize()
        grouped = (
            candidates.groupby(
                ["author_account_id", "author_handle", "author_display_name", "source_platform"],
                dropna=False,
            )
            .agg(
                mention_count=("post_id", "count"),
                active_days=("active_date", "nunique"),
                last_seen_at=("post_timestamp", "max"),
                first_seen_at=("post_timestamp", "min"),
                engagement_mean=("engagement_score", "mean"),
                engagement_sum=("engagement_score", "sum"),
                avg_sentiment=("sentiment_score", "mean"),
            )
            .reset_index()
        )

        span_days = max(
            int((candidates["active_date"].max() - candidates["active_date"].min()).days) + 1,
            1,
        )
        recency_days = (
            pd.Timestamp.now(tz=EASTERN).tz_localize(None)
            - grouped["last_seen_at"].dt.tz_convert(EASTERN).dt.tz_localize(None)
        ).dt.days.clip(lower=0)
        grouped["persistence_score"] = grouped["active_days"] / span_days
        grouped["recency_score"] = 1.0 / (1.0 + recency_days)
        grouped["discovery_score"] = (
            np.log1p(grouped["mention_count"]) * 4.0
            + np.log1p(grouped["engagement_mean"].clip(lower=0.0)) * 1.5
            + grouped["persistence_score"] * 5.0
            + grouped["recency_score"] * 3.0
        )
        grouped["discovery_rank"] = grouped["discovery_score"].rank(ascending=False, method="dense").astype(int)
        return grouped.sort_values(["discovery_score", "mention_count"], ascending=[False, False]).reset_index(drop=True)

    def refresh_accounts(
        self,
        posts: pd.DataFrame,
        existing_accounts: pd.DataFrame,
        as_of: pd.Timestamp,
        auto_include_top_n: int = 20,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ranked = self.rank_candidates(posts, existing_accounts)
        if ranked.empty and existing_accounts.empty:
            return pd.DataFrame(columns=TRACKED_ACCOUNT_COLUMNS), pd.DataFrame()

        tracked = existing_accounts.copy()
        if tracked.empty:
            tracked = pd.DataFrame(columns=TRACKED_ACCOUNT_COLUMNS)

        for column in TRACKED_ACCOUNT_COLUMNS:
            if column not in tracked.columns:
                tracked[column] = pd.NA

        as_of_et = as_of.tz_convert(EASTERN) if as_of.tzinfo is not None else as_of.tz_localize(EASTERN)
        as_of_date = ensure_tz_naive_date(as_of_et)
        active_ranked = ranked.head(auto_include_top_n).copy()
        active_ids = set(active_ranked["author_account_id"].astype(str))

        currently_active = tracked.loc[
            tracked["status"].fillna("") == "active",
        ].copy()
        if not currently_active.empty:
            active_mask = currently_active["effective_to"].isna()
            currently_active = currently_active.loc[active_mask].copy()

        still_active_ids = set()
        for idx, row in currently_active.iterrows():
            account_id = str(row["account_id"])
            if account_id in active_ids:
                still_active_ids.add(account_id)
                ranked_row = active_ranked.loc[active_ranked["author_account_id"].astype(str) == account_id].iloc[0]
                tracked.loc[idx, "discovery_score"] = float(ranked_row["discovery_score"])
                tracked.loc[idx, "last_seen_at"] = ranked_row["last_seen_at"]
                tracked.loc[idx, "mention_count"] = int(ranked_row["mention_count"])
                tracked.loc[idx, "engagement_mean"] = float(ranked_row["engagement_mean"])
                tracked.loc[idx, "active_days"] = int(ranked_row["active_days"])
            else:
                tracked.loc[idx, "status"] = "inactive"
                tracked.loc[idx, "effective_to"] = as_of_date

        new_rows: list[dict[str, Any]] = []
        for _, row in active_ranked.iterrows():
            account_id = str(row["author_account_id"])
            if account_id in still_active_ids:
                continue
            first_seen = row["first_seen_at"]
            version_id = stable_text_id(account_id, as_of_date.isoformat())
            record = TrackedAccount(
                version_id=version_id,
                account_id=account_id,
                handle=str(row["author_handle"]),
                display_name=str(row["author_display_name"]),
                source_platform=str(row["source_platform"]),
                discovery_score=float(row["discovery_score"]),
                status="active",
                first_seen_at=first_seen,
                last_seen_at=row["last_seen_at"],
                effective_from=as_of_date,
                effective_to=None,
                auto_included=True,
                provenance="discovery_auto_include",
                mention_count=int(row["mention_count"]),
                engagement_mean=float(row["engagement_mean"]),
                active_days=int(row["active_days"]),
            )
            new_rows.append(asdict(record))

        if new_rows:
            new_rows_df = pd.DataFrame(new_rows)
            tracked = new_rows_df if tracked.empty else pd.concat([tracked, new_rows_df], ignore_index=True)

        ranking_history = active_ranked.copy()
        ranking_history["ranked_at"] = as_of_date
        ranking_history["auto_included"] = ranking_history["author_account_id"].astype(str).isin(active_ids)
        tracked = tracked[TRACKED_ACCOUNT_COLUMNS].copy().reset_index(drop=True)
        return tracked, ranking_history.reset_index(drop=True)

    def current_active_accounts(self, tracked_accounts: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        if tracked_accounts.empty:
            return tracked_accounts
        as_of_date = ensure_tz_naive_date(as_of)
        start = pd.to_datetime(tracked_accounts["effective_from"], errors="coerce")
        end = pd.to_datetime(tracked_accounts["effective_to"], errors="coerce")
        mask = (start <= as_of_date) & (end.isna() | (end > as_of_date))
        return tracked_accounts.loc[mask].copy().sort_values("discovery_score", ascending=False)
