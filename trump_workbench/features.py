from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import EASTERN
from .enrichment import LLMEnrichmentService
from .utils import business_minutes_until_close, ensure_tz_naive_date, truncate_text

REGULAR_OPEN_MINUTE = 9 * 60 + 30
REGULAR_CLOSE_MINUTE = 16 * 60


def map_posts_to_trade_sessions(posts: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        result = posts.copy()
        result["session_date"] = pd.NaT
        result["reaction_anchor_ts"] = pd.NaT
        result["mapping_reason"] = ""
        return result

    market_dates = pd.DatetimeIndex(market["trade_date"].dropna().sort_values().unique())
    market_date_set = set(pd.Timestamp(d) for d in market_dates)

    session_dates: list[pd.Timestamp] = []
    anchor_times: list[pd.Timestamp | pd.NaT] = []
    reasons: list[str] = []

    for ts in posts["post_timestamp"]:
        local_ts = ts.tz_convert(EASTERN)
        local_date = pd.Timestamp(local_ts.date())
        minute_of_day = local_ts.hour * 60 + local_ts.minute
        next_day_idx = market_dates.searchsorted(local_date, side="right")

        if local_date in market_date_set and minute_of_day < REGULAR_CLOSE_MINUTE:
            session_date = local_date
            if minute_of_day < REGULAR_OPEN_MINUTE:
                anchor = pd.Timestamp(f"{local_date.date()} 09:30", tz=EASTERN)
                reason = "pre-market -> same session open"
            else:
                anchor = local_ts.floor("min")
                reason = "during regular hours -> same session"
        else:
            if next_day_idx >= len(market_dates):
                session_dates.append(pd.NaT)
                anchor_times.append(pd.NaT)
                reasons.append("no later market session in loaded range")
                continue
            session_date = pd.Timestamp(market_dates[next_day_idx])
            anchor = pd.Timestamp(f"{session_date.date()} 09:30", tz=EASTERN)
            if local_date in market_date_set:
                reason = "after close -> next session open"
            else:
                reason = "weekend/holiday -> next session open"

        session_dates.append(session_date)
        anchor_times.append(anchor)
        reasons.append(reason)

    mapped = posts.copy()
    mapped["session_date"] = session_dates
    mapped["reaction_anchor_ts"] = anchor_times
    mapped["mapping_reason"] = reasons
    return mapped.dropna(subset=["session_date"]).reset_index(drop=True)


class FeatureService:
    def __init__(self, enrichment_service: LLMEnrichmentService) -> None:
        self.enrichment_service = enrichment_service

    def build_session_dataset(
        self,
        posts: pd.DataFrame,
        spy_market: pd.DataFrame,
        tracked_accounts: pd.DataFrame,
        feature_version: str,
        llm_enabled: bool,
    ) -> pd.DataFrame:
        if spy_market.empty:
            return pd.DataFrame()

        market = spy_market.sort_values("trade_date").reset_index(drop=True).copy()
        mapped = map_posts_to_trade_sessions(posts, market[["trade_date"]])
        mapped = self.enrichment_service.enrich_posts(mapped, enabled=llm_enabled)
        mapped = self._flag_tracked_posts(mapped, tracked_accounts)

        market["session_return"] = market["close"].pct_change().fillna(0.0)
        market["prev_return_1d"] = market["close"].pct_change(1).fillna(0.0)
        market["prev_return_3d"] = market["close"].pct_change(3).fillna(0.0)
        market["prev_return_5d"] = market["close"].pct_change(5).fillna(0.0)
        market["rolling_vol_5d"] = market["close"].pct_change().rolling(5).std().fillna(0.0)
        market["close_ma_5"] = market["close"].rolling(5).mean().bfill().fillna(market["close"])
        market["close_vs_ma_5"] = (market["close"] / market["close_ma_5"] - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        market["volume_ma_5"] = market["volume"].rolling(5).mean().bfill().fillna(market["volume"])
        market["volume_z_5"] = ((market["volume"] - market["volume_ma_5"]) / market["volume"].rolling(5).std(ddof=0).replace(0, np.nan)).fillna(0.0)
        market["next_session_date"] = market["trade_date"].shift(-1)
        market["next_session_open"] = market["open"].shift(-1)
        market["next_session_close"] = market["close"].shift(-1)
        market["target_next_session_return"] = (market["next_session_close"] / market["next_session_open"] - 1.0).replace([np.inf, -np.inf], np.nan)
        market["target_available"] = market["next_session_open"].notna() & market["next_session_close"].notna()

        rows: list[dict[str, Any]] = []
        for _, session in market.iterrows():
            session_date = pd.Timestamp(session["trade_date"])
            group = mapped.loc[mapped["session_date"] == session_date].copy()
            group = group.sort_values("post_timestamp").reset_index(drop=True)
            rows.append(self._build_session_row(group, session, feature_version, llm_enabled))

        dataset = pd.DataFrame(rows).sort_values("signal_session_date").reset_index(drop=True)
        dataset["tradeable"] = dataset["target_available"].fillna(False)
        return dataset

    def _flag_tracked_posts(self, mapped: pd.DataFrame, tracked_accounts: pd.DataFrame) -> pd.DataFrame:
        if mapped.empty:
            mapped["is_active_tracked_account"] = pd.Series(dtype=bool)
            mapped["tracked_discovery_score"] = pd.Series(dtype=float)
            mapped["tracked_account_status"] = pd.Series(dtype=str)
            mapped["tracked_account_provenance"] = pd.Series(dtype=str)
            return mapped

        if tracked_accounts.empty:
            mapped["is_active_tracked_account"] = False
            mapped["tracked_discovery_score"] = 0.0
            mapped["tracked_account_status"] = "none"
            mapped["tracked_account_provenance"] = ""
            return mapped

        intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None, float, str, str]]] = {}
        for _, row in tracked_accounts.iterrows():
            account_id = str(row.get("account_id", ""))
            if not account_id:
                continue
            start = pd.to_datetime(row.get("effective_from"), errors="coerce")
            end = pd.to_datetime(row.get("effective_to"), errors="coerce")
            score = float(row.get("discovery_score", 0.0) or 0.0)
            status = str(row.get("status", "") or "")
            provenance = str(row.get("provenance", "") or "")
            if pd.isna(start):
                continue
            intervals.setdefault(account_id, []).append(
                (
                    ensure_tz_naive_date(start),
                    ensure_tz_naive_date(end) if pd.notna(end) else None,
                    score,
                    status,
                    provenance,
                ),
            )

        active_flags: list[bool] = []
        active_scores: list[float] = []
        active_statuses: list[str] = []
        active_provenance: list[str] = []
        for _, row in mapped.iterrows():
            account_id = str(row["author_account_id"])
            session_date = ensure_tz_naive_date(row["session_date"])
            intervals_for_account = intervals.get(account_id, [])
            score = 0.0
            active = False
            selected_status = "none"
            selected_provenance = ""
            for start, end, interval_score, status, provenance in intervals_for_account:
                if start <= session_date and (end is None or session_date < end):
                    if status == "suppressed":
                        active = False
                        score = 0.0
                        selected_status = status
                        selected_provenance = provenance
                        break
                    active = status in {"active", "pinned"}
                    score = max(score, interval_score)
                    selected_status = status
                    selected_provenance = provenance
            active_flags.append(active)
            active_scores.append(score)
            active_statuses.append(selected_status)
            active_provenance.append(selected_provenance)

        mapped = mapped.copy()
        mapped["is_active_tracked_account"] = active_flags
        mapped["tracked_discovery_score"] = active_scores
        mapped["tracked_account_status"] = active_statuses
        mapped["tracked_account_provenance"] = active_provenance
        return mapped

    def _build_session_row(
        self,
        group: pd.DataFrame,
        market_row: pd.Series,
        feature_version: str,
        llm_enabled: bool,
    ) -> dict[str, Any]:
        if group.empty:
            base = {
                "signal_session_date": market_row["trade_date"],
                "next_session_date": market_row["next_session_date"],
                "next_session_open": market_row["next_session_open"],
                "next_session_close": market_row["next_session_close"],
                "feature_version": feature_version,
                "llm_enabled": llm_enabled,
                "feature_source_min_ts": pd.NaT,
                "feature_source_max_ts": pd.NaT,
                "next_session_open_ts": pd.Timestamp(f"{pd.Timestamp(market_row['next_session_date']).date()} 09:30", tz=EASTERN) if pd.notna(market_row["next_session_date"]) else pd.NaT,
                "feature_cutoff_before_next_open": True,
                "has_posts": False,
                "post_count": 0,
                "trump_post_count": 0,
                "x_post_count": 0,
                "tracked_account_post_count": 0,
                "mention_post_count": 0,
                "positive_posts": 0,
                "neutral_posts": 0,
                "negative_posts": 0,
                "unique_author_count": 0,
                "active_tracked_author_count": 0,
                "top_author_share": 0.0,
                "tracked_weighted_mentions": 0.0,
                "tracked_weighted_engagement": 0.0,
                "total_engagement": 0.0,
                "avg_engagement": 0.0,
                "minutes_to_close_min": 0.0,
                "sentiment_open": 0.0,
                "sentiment_high": 0.0,
                "sentiment_low": 0.0,
                "sentiment_close": 0.0,
                "sentiment_avg": 0.0,
                "sentiment_std": 0.0,
                "sentiment_range": 0.0,
                "semantic_market_relevance_avg": 0.0,
                "semantic_urgency_avg": 0.0,
                "semantic_topic_markets": 0.0,
                "semantic_topic_trade": 0.0,
                "semantic_topic_geopolitics": 0.0,
                "semantic_topic_immigration": 0.0,
                "semantic_topic_judiciary": 0.0,
                "semantic_topic_campaign": 0.0,
                "semantic_topic_other": 0.0,
                "policy_economy": 0.0,
                "policy_trade": 0.0,
                "policy_foreign_policy": 0.0,
                "policy_immigration": 0.0,
                "policy_legal": 0.0,
                "policy_other": 0.0,
                "session_return": market_row["session_return"],
                "prev_return_1d": market_row["prev_return_1d"],
                "prev_return_3d": market_row["prev_return_3d"],
                "prev_return_5d": market_row["prev_return_5d"],
                "rolling_vol_5d": market_row["rolling_vol_5d"],
                "close_vs_ma_5": market_row["close_vs_ma_5"],
                "volume_z_5": market_row["volume_z_5"],
                "target_next_session_return": market_row["target_next_session_return"],
                "target_available": market_row["target_available"],
            }
            return base

        first_row = group.iloc[0]
        last_row = group.iloc[-1]
        author_counts = group["author_account_id"].value_counts()
        topic_counts = group["semantic_topic"].value_counts(normalize=True)
        policy_counts = group["semantic_policy_bucket"].value_counts(normalize=True)

        row = {
            "signal_session_date": market_row["trade_date"],
            "next_session_date": market_row["next_session_date"],
            "next_session_open": market_row["next_session_open"],
            "next_session_close": market_row["next_session_close"],
            "feature_version": feature_version,
            "llm_enabled": llm_enabled,
            "feature_source_min_ts": group["post_timestamp"].min(),
            "feature_source_max_ts": group["post_timestamp"].max(),
            "next_session_open_ts": pd.Timestamp(f"{pd.Timestamp(market_row['next_session_date']).date()} 09:30", tz=EASTERN) if pd.notna(market_row["next_session_date"]) else pd.NaT,
            "feature_cutoff_before_next_open": True,
            "has_posts": True,
            "post_count": int(len(group)),
            "trump_post_count": int(group["author_is_trump"].sum()),
            "x_post_count": int((group["source_platform"] == "X").sum()),
            "tracked_account_post_count": int(group["is_active_tracked_account"].sum()),
            "mention_post_count": int(group["mentions_trump"].sum()),
            "positive_posts": int((group["sentiment_label"] == "positive").sum()),
            "neutral_posts": int((group["sentiment_label"] == "neutral").sum()),
            "negative_posts": int((group["sentiment_label"] == "negative").sum()),
            "unique_author_count": int(group["author_account_id"].nunique()),
            "active_tracked_author_count": int(group.loc[group["is_active_tracked_account"], "author_account_id"].nunique()),
            "top_author_share": float(author_counts.iloc[0] / len(group)) if not author_counts.empty else 0.0,
            "tracked_weighted_mentions": float((group["is_active_tracked_account"] * group["tracked_discovery_score"]).sum()),
            "tracked_weighted_engagement": float((group["engagement_score"] * group["tracked_discovery_score"]).sum()),
            "total_engagement": float(group["engagement_score"].sum()),
            "avg_engagement": float(group["engagement_score"].mean()),
            "minutes_to_close_min": float(group["post_timestamp"].map(business_minutes_until_close).min()),
            "sentiment_open": float(first_row["sentiment_score"]),
            "sentiment_high": float(group["sentiment_score"].max()),
            "sentiment_low": float(group["sentiment_score"].min()),
            "sentiment_close": float(last_row["sentiment_score"]),
            "sentiment_avg": float(group["sentiment_score"].mean()),
            "sentiment_std": float(group["sentiment_score"].std(ddof=0)) if len(group) > 1 else 0.0,
            "sentiment_range": float(group["sentiment_score"].max() - group["sentiment_score"].min()),
            "semantic_market_relevance_avg": float(group["semantic_market_relevance"].mean()),
            "semantic_urgency_avg": float(group["semantic_urgency"].mean()),
            "semantic_topic_markets": float(topic_counts.get("markets", 0.0)),
            "semantic_topic_trade": float(topic_counts.get("trade", 0.0)),
            "semantic_topic_geopolitics": float(topic_counts.get("geopolitics", 0.0)),
            "semantic_topic_immigration": float(topic_counts.get("immigration", 0.0)),
            "semantic_topic_judiciary": float(topic_counts.get("judiciary", 0.0)),
            "semantic_topic_campaign": float(topic_counts.get("campaign", 0.0)),
            "semantic_topic_other": float(topic_counts.get("other", 0.0)),
            "policy_economy": float(policy_counts.get("economy", 0.0)),
            "policy_trade": float(policy_counts.get("trade", 0.0)),
            "policy_foreign_policy": float(policy_counts.get("foreign_policy", 0.0)),
            "policy_immigration": float(policy_counts.get("immigration", 0.0)),
            "policy_legal": float(policy_counts.get("legal", 0.0)),
            "policy_other": float(policy_counts.get("other", 0.0)),
            "session_return": market_row["session_return"],
            "prev_return_1d": market_row["prev_return_1d"],
            "prev_return_3d": market_row["prev_return_3d"],
            "prev_return_5d": market_row["prev_return_5d"],
            "rolling_vol_5d": market_row["rolling_vol_5d"],
            "close_vs_ma_5": market_row["close_vs_ma_5"],
            "volume_z_5": market_row["volume_z_5"],
            "target_next_session_return": market_row["target_next_session_return"],
            "target_available": market_row["target_available"],
        }
        if pd.notna(row["next_session_open_ts"]) and pd.notna(row["feature_source_max_ts"]):
            row["feature_cutoff_before_next_open"] = bool(row["feature_source_max_ts"] < row["next_session_open_ts"])
        return row


def latest_feature_preview(feature_rows: pd.DataFrame) -> dict[str, Any]:
    if feature_rows.empty:
        return {}
    row = feature_rows.sort_values("signal_session_date").iloc[-1]
    return {
        "signal_session_date": row["signal_session_date"],
        "post_count": int(row["post_count"]),
        "trump_post_count": int(row["trump_post_count"]),
        "tracked_account_post_count": int(row["tracked_account_post_count"]),
        "sentiment_avg": float(row["sentiment_avg"]),
        "market_context": f"prev 1d {row['prev_return_1d']:+.2%}, prev 5d {row['prev_return_5d']:+.2%}",
    }


def preview_post_texts(group: pd.DataFrame, max_items: int = 3) -> str:
    if group.empty:
        return ""
    bits = []
    for _, row in group.sort_values("post_timestamp").head(max_items).iterrows():
        handle = row["author_handle"] or row["author_display_name"] or row["source_platform"]
        bits.append(f"@{handle}: {truncate_text(row['cleaned_text'], max_chars=120)}")
    return " | ".join(bits)
