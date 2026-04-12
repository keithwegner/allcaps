from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import EASTERN
from .contracts import (
    MANUAL_OVERRIDE_COLUMNS,
    RANKING_HISTORY_COLUMNS,
    TRACKED_ACCOUNT_COLUMNS,
    TrackedAccount,
)
from .utils import ensure_tz_naive_date, stable_text_id

DISCOVERY_POST_COLUMNS = [
    "source_platform",
    "mentions_trump",
    "author_is_trump",
    "author_account_id",
    "author_handle",
    "author_display_name",
    "post_id",
    "post_timestamp",
    "engagement_score",
    "sentiment_score",
]


class DiscoveryService:
    def rank_candidates(
        self,
        posts: pd.DataFrame,
        accounts: pd.DataFrame | None = None,
        as_of: pd.Timestamp | None = None,
        lookback_days: int | None = 90,
    ) -> pd.DataFrame:
        del accounts
        if posts.empty:
            return pd.DataFrame()

        candidates = posts.loc[
            (posts["source_platform"] == "X")
            & posts["mentions_trump"]
            & (~posts["author_is_trump"])
        ].copy()
        if candidates.empty:
            return pd.DataFrame()

        if as_of is not None:
            as_of_ts = as_of.tz_convert(EASTERN) if as_of.tzinfo is not None else as_of.tz_localize(EASTERN)
            candidates = candidates.loc[candidates["post_timestamp"] <= as_of_ts].copy()
            if lookback_days is not None:
                lookback_start = as_of_ts - pd.Timedelta(days=lookback_days)
                candidates = candidates.loc[candidates["post_timestamp"] >= lookback_start].copy()
        else:
            as_of_ts = pd.Timestamp.now(tz=EASTERN)

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
            as_of_ts.tz_convert(EASTERN).tz_localize(None)
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
        grouped["ranked_at"] = ensure_tz_naive_date(as_of_ts)
        return grouped.sort_values(["discovery_score", "mention_count"], ascending=[False, False]).reset_index(drop=True)

    def refresh_accounts(
        self,
        posts: pd.DataFrame,
        existing_accounts: pd.DataFrame,
        as_of: pd.Timestamp,
        auto_include_top_n: int = 20,
        manual_overrides: pd.DataFrame | None = None,
        lookback_days: int = 90,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        del existing_accounts
        overrides = self.normalize_manual_overrides(manual_overrides)
        if posts.empty:
            posts = posts.reindex(columns=DISCOVERY_POST_COLUMNS)
        candidates = posts.loc[
            (posts["source_platform"] == "X")
            & posts["mentions_trump"]
            & (~posts["author_is_trump"])
        ].copy()
        as_of_et = as_of.tz_convert(EASTERN) if as_of.tzinfo is not None else as_of.tz_localize(EASTERN)
        as_of_date = ensure_tz_naive_date(as_of_et)

        if candidates.empty and overrides.empty:
            return (
                pd.DataFrame(columns=TRACKED_ACCOUNT_COLUMNS),
                pd.DataFrame(columns=RANKING_HISTORY_COLUMNS),
            )

        candidate_dates = (
            candidates.loc[candidates["post_timestamp"] <= as_of_et, "post_timestamp"]
            .dt.tz_convert(EASTERN)
            .dt.normalize()
            .dt.tz_localize(None)
            .drop_duplicates()
            .sort_values()
            .tolist()
        ) if not candidates.empty else []
        override_dates = []
        if not overrides.empty:
            override_dates.extend(pd.to_datetime(overrides["effective_from"], errors="coerce").dropna().dt.normalize().tolist())
            override_dates.extend(pd.to_datetime(overrides["effective_to"], errors="coerce").dropna().dt.normalize().tolist())
        rebalance_dates = sorted({date for date in candidate_dates + override_dates if pd.notna(date) and pd.Timestamp(date) <= as_of_date})
        if not rebalance_dates:
            rebalance_dates = [as_of_date]

        open_rows: dict[str, dict[str, Any]] = {}
        tracked_rows: list[dict[str, Any]] = []
        current_keys: dict[str, tuple[str, str]] = {}
        ranking_rows: list[dict[str, Any]] = []

        for rank_date in rebalance_dates:
            rank_ts = pd.Timestamp(f"{pd.Timestamp(rank_date).date()} 23:59", tz=EASTERN)
            ranked = self.rank_candidates(
                posts=posts,
                as_of=rank_ts,
                lookback_days=lookback_days,
            )
            active_overrides = self.overrides_in_effect(overrides, pd.Timestamp(rank_date))
            suppressed_ids = set(active_overrides.loc[active_overrides["action"] == "suppress", "account_id"].astype(str))
            pinned_rows = active_overrides.loc[active_overrides["action"] == "pin"].copy()

            selected_meta: dict[str, dict[str, Any]] = {}
            for _, row in ranked.head(auto_include_top_n).iterrows():
                account_id = str(row["author_account_id"])
                if account_id in suppressed_ids:
                    continue
                selected_meta[account_id] = self._make_interval_meta(
                    account_id=account_id,
                    handle=str(row["author_handle"]),
                    display_name=str(row["author_display_name"]),
                    source_platform=str(row["source_platform"]),
                    discovery_score=float(row["discovery_score"]),
                    status="active",
                    first_seen_at=row["first_seen_at"],
                    last_seen_at=row["last_seen_at"],
                    auto_included=True,
                    provenance="discovery_auto_include",
                    mention_count=int(row["mention_count"]),
                    engagement_mean=float(row["engagement_mean"]),
                    active_days=int(row["active_days"]),
                    effective_from=pd.Timestamp(rank_date),
                )

            for _, override_row in pinned_rows.iterrows():
                account_id = str(override_row["account_id"])
                pinned_meta = self._account_meta_for_override(
                    posts=posts,
                    account_id=account_id,
                    as_of=rank_ts,
                    override_row=override_row,
                )
                selected_meta[account_id] = pinned_meta

            next_keys = {
                account_id: (meta["status"], meta["provenance"])
                for account_id, meta in selected_meta.items()
            }

            for account_id, open_row in list(open_rows.items()):
                if account_id not in next_keys:
                    open_row["effective_to"] = pd.Timestamp(rank_date)
                    tracked_rows.append(open_row)
                    del open_rows[account_id]
                    del current_keys[account_id]

            for account_id, meta in selected_meta.items():
                current_key = current_keys.get(account_id)
                next_key = next_keys[account_id]
                if current_key == next_key:
                    open_row = open_rows[account_id]
                    for key, value in meta.items():
                        if key != "effective_from":
                            open_row[key] = value
                else:
                    if account_id in open_rows:
                        previous = open_rows.pop(account_id)
                        previous["effective_to"] = pd.Timestamp(rank_date)
                        tracked_rows.append(previous)
                    open_rows[account_id] = meta
                    current_keys[account_id] = next_key

            ranking_rows.extend(
                self._build_ranking_rows(
                    ranked=ranked,
                    selected_meta=selected_meta,
                    suppressed_ids=suppressed_ids,
                    rank_date=pd.Timestamp(rank_date),
                ),
            )

        for open_row in open_rows.values():
            tracked_rows.append(open_row)

        tracked = pd.DataFrame(tracked_rows, columns=TRACKED_ACCOUNT_COLUMNS).reset_index(drop=True)
        ranking_history = pd.DataFrame(ranking_rows).reset_index(drop=True)
        if tracked.empty:
            tracked = pd.DataFrame(columns=TRACKED_ACCOUNT_COLUMNS)
        if ranking_history.empty:
            ranking_history = pd.DataFrame(columns=RANKING_HISTORY_COLUMNS)
        return tracked, ranking_history

    def current_active_accounts(self, tracked_accounts: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        if tracked_accounts.empty:
            return tracked_accounts
        as_of_date = ensure_tz_naive_date(as_of)
        start = pd.to_datetime(tracked_accounts["effective_from"], errors="coerce")
        end = pd.to_datetime(tracked_accounts["effective_to"], errors="coerce")
        status = tracked_accounts["status"].fillna("")
        mask = (
            (start <= as_of_date)
            & (end.isna() | (end > as_of_date))
            & status.isin(["active", "pinned"])
        )
        return tracked_accounts.loc[mask].copy().sort_values(["status", "discovery_score"], ascending=[True, False])

    def normalize_manual_overrides(self, overrides: pd.DataFrame | None) -> pd.DataFrame:
        if overrides is None or overrides.empty:
            return pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)
        out = overrides.copy()
        for column in MANUAL_OVERRIDE_COLUMNS:
            if column not in out.columns:
                out[column] = pd.NA
        out["override_id"] = out["override_id"].fillna("").astype(str)
        blank_ids = out["override_id"].str.strip() == ""
        out.loc[blank_ids, "override_id"] = out.loc[blank_ids].apply(
            lambda row: stable_text_id(row["account_id"], row["action"], row["effective_from"], row["created_at"]),
            axis=1,
        )
        out["account_id"] = out["account_id"].fillna("").astype(str)
        out["handle"] = out["handle"].fillna("").astype(str)
        out["display_name"] = out["display_name"].fillna("").astype(str)
        out["source_platform"] = out["source_platform"].fillna("X").astype(str)
        out["action"] = out["action"].fillna("").astype(str).str.lower()
        out["note"] = out["note"].fillna("").astype(str)
        out["effective_from"] = pd.to_datetime(out["effective_from"], errors="coerce").dt.normalize()
        out["effective_to"] = pd.to_datetime(out["effective_to"], errors="coerce").dt.normalize()
        out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
        out = out.dropna(subset=["effective_from"]).copy()
        out = out.loc[out["action"].isin(["pin", "suppress"])].copy()
        return out[MANUAL_OVERRIDE_COLUMNS].sort_values(["effective_from", "created_at", "account_id"]).reset_index(drop=True)

    def add_manual_override(
        self,
        overrides: pd.DataFrame,
        account_id: str,
        handle: str,
        display_name: str,
        action: str,
        effective_from: pd.Timestamp,
        effective_to: pd.Timestamp | None = None,
        note: str = "",
        source_platform: str = "X",
    ) -> pd.DataFrame:
        normalized = self.normalize_manual_overrides(overrides)
        created_at = pd.Timestamp.utcnow()
        row = pd.DataFrame(
            [
                {
                    "override_id": stable_text_id(account_id, action, effective_from, created_at),
                    "account_id": account_id,
                    "handle": handle,
                    "display_name": display_name,
                    "source_platform": source_platform,
                    "action": action,
                    "effective_from": pd.Timestamp(effective_from).normalize(),
                    "effective_to": pd.Timestamp(effective_to).normalize() if effective_to is not None else pd.NaT,
                    "note": note,
                    "created_at": created_at,
                },
            ],
        )
        combined = row if normalized.empty else pd.concat([normalized, row], ignore_index=True)
        return self.normalize_manual_overrides(combined)

    def remove_manual_override(self, overrides: pd.DataFrame, override_id: str) -> pd.DataFrame:
        normalized = self.normalize_manual_overrides(overrides)
        if normalized.empty:
            return normalized
        return normalized.loc[normalized["override_id"].astype(str) != str(override_id)].reset_index(drop=True)

    def overrides_in_effect(self, overrides: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
        normalized = self.normalize_manual_overrides(overrides)
        if normalized.empty:
            return normalized
        as_of = pd.Timestamp(as_of_date).normalize()
        start = pd.to_datetime(normalized["effective_from"], errors="coerce")
        end = pd.to_datetime(normalized["effective_to"], errors="coerce")
        mask = (start <= as_of) & (end.isna() | (end > as_of))
        return normalized.loc[mask].copy()

    def _account_meta_for_override(
        self,
        posts: pd.DataFrame,
        account_id: str,
        as_of: pd.Timestamp,
        override_row: pd.Series,
    ) -> dict[str, Any]:
        history = posts.loc[
            (posts["author_account_id"].astype(str) == account_id)
            & (posts["post_timestamp"] <= as_of),
        ].sort_values("post_timestamp")
        if history.empty:
            handle = str(override_row.get("handle", ""))
            display_name = str(override_row.get("display_name", ""))
            first_seen = pd.Timestamp(override_row["effective_from"]).tz_localize(EASTERN)
            last_seen = first_seen
            mention_count = 0
            engagement_mean = 0.0
            active_days = 0
            discovery_score = 0.0
        else:
            ranked = self.rank_candidates(history, as_of=as_of, lookback_days=None)
            match = ranked.loc[ranked["author_account_id"].astype(str) == account_id]
            row = match.iloc[0] if not match.empty else history.iloc[-1]
            handle = str(row.get("author_handle", override_row.get("handle", "")))
            display_name = str(row.get("author_display_name", override_row.get("display_name", "")))
            first_seen = pd.to_datetime(row.get("first_seen_at", history["post_timestamp"].min()))
            last_seen = pd.to_datetime(row.get("last_seen_at", history["post_timestamp"].max()))
            mention_count = int(row.get("mention_count", len(history)))
            engagement_mean = float(row.get("engagement_mean", history["engagement_score"].mean()))
            active_days = int(row.get("active_days", history["post_timestamp"].dt.tz_convert(EASTERN).dt.normalize().nunique()))
            discovery_score = float(row.get("discovery_score", 0.0))

        return self._make_interval_meta(
            account_id=account_id,
            handle=handle,
            display_name=display_name,
            source_platform=str(override_row.get("source_platform", "X")),
            discovery_score=discovery_score,
            status="pinned",
            first_seen_at=first_seen,
            last_seen_at=last_seen,
            auto_included=False,
            provenance="manual_override:pin",
            mention_count=mention_count,
            engagement_mean=engagement_mean,
            active_days=active_days,
            effective_from=pd.Timestamp(override_row["effective_from"]).normalize(),
        )

    def _make_interval_meta(
        self,
        account_id: str,
        handle: str,
        display_name: str,
        source_platform: str,
        discovery_score: float,
        status: str,
        first_seen_at: pd.Timestamp,
        last_seen_at: pd.Timestamp,
        auto_included: bool,
        provenance: str,
        mention_count: int,
        engagement_mean: float,
        active_days: int,
        effective_from: pd.Timestamp,
    ) -> dict[str, Any]:
        version_id = stable_text_id(account_id, effective_from.isoformat(), status, provenance)
        record = TrackedAccount(
            version_id=version_id,
            account_id=account_id,
            handle=handle,
            display_name=display_name,
            source_platform=source_platform,
            discovery_score=float(discovery_score),
            status=status,
            first_seen_at=pd.to_datetime(first_seen_at),
            last_seen_at=pd.to_datetime(last_seen_at),
            effective_from=pd.Timestamp(effective_from).normalize(),
            effective_to=None,
            auto_included=auto_included,
            provenance=provenance,
            mention_count=int(mention_count),
            engagement_mean=float(engagement_mean),
            active_days=int(active_days),
        )
        return asdict(record)

    def _build_ranking_rows(
        self,
        ranked: pd.DataFrame,
        selected_meta: dict[str, dict[str, Any]],
        suppressed_ids: set[str],
        rank_date: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        if ranked.empty and not selected_meta:
            return []
        rows: list[dict[str, Any]] = []
        included_ids = set(selected_meta)
        for _, row in ranked.head(max(len(selected_meta), 20) or 20).iterrows():
            account_id = str(row["author_account_id"])
            selected = selected_meta.get(account_id)
            rows.append(
                {
                    "author_account_id": account_id,
                    "author_handle": str(row["author_handle"]),
                    "author_display_name": str(row["author_display_name"]),
                    "source_platform": str(row["source_platform"]),
                    "discovery_score": float(row["discovery_score"]),
                    "mention_count": int(row["mention_count"]),
                    "engagement_mean": float(row["engagement_mean"]),
                    "active_days": int(row["active_days"]),
                    "ranked_at": pd.Timestamp(rank_date).normalize(),
                    "discovery_rank": int(row["discovery_rank"]),
                    "final_selected": account_id in included_ids,
                    "selected_status": selected["status"] if selected else "excluded",
                    "suppressed_by_override": account_id in suppressed_ids,
                    "pinned_by_override": bool(selected and selected["status"] == "pinned"),
                },
            )

        ranked_ids = {str(row["author_account_id"]) for _, row in ranked.iterrows()}
        for account_id, selected in selected_meta.items():
            if account_id in ranked_ids:
                continue
            rows.append(
                {
                    "author_account_id": account_id,
                    "author_handle": selected["handle"],
                    "author_display_name": selected["display_name"],
                    "source_platform": selected["source_platform"],
                    "discovery_score": float(selected["discovery_score"]),
                    "mention_count": int(selected["mention_count"]),
                    "engagement_mean": float(selected["engagement_mean"]),
                    "active_days": int(selected["active_days"]),
                    "ranked_at": pd.Timestamp(rank_date).normalize(),
                    "discovery_rank": -1,
                    "final_selected": True,
                    "selected_status": selected["status"],
                    "suppressed_by_override": False,
                    "pinned_by_override": selected["status"] == "pinned",
                },
            )
        return rows
