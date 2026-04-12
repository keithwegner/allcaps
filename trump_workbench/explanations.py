from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import truncate_text

POST_ATTRIBUTION_COLUMNS = [
    "signal_session_date",
    "post_timestamp",
    "author_account_id",
    "author_handle",
    "author_display_name",
    "source_platform",
    "author_is_trump",
    "is_active_tracked_account",
    "tracked_account_status",
    "mentions_trump",
    "sentiment_score",
    "engagement_score",
    "author_weight",
    "engagement_boost",
    "post_signal_score",
    "abs_post_signal_score",
    "post_preview",
    "post_url",
]

ACCOUNT_ATTRIBUTION_COLUMNS = [
    "signal_session_date",
    "author_account_id",
    "author_handle",
    "author_display_name",
    "source_platform",
    "author_is_trump",
    "is_active_tracked_account",
    "tracked_account_status",
    "post_count",
    "avg_sentiment",
    "total_engagement",
    "net_post_signal",
    "abs_net_post_signal",
]


def build_post_attribution(mapped_posts: pd.DataFrame) -> pd.DataFrame:
    if mapped_posts.empty:
        return pd.DataFrame(columns=POST_ATTRIBUTION_COLUMNS)

    posts = mapped_posts.copy()
    posts["signal_session_date"] = pd.to_datetime(posts["session_date"], errors="coerce")
    posts["engagement_score"] = pd.to_numeric(posts.get("engagement_score", 0.0), errors="coerce").fillna(0.0)
    posts["sentiment_score"] = pd.to_numeric(posts.get("sentiment_score", 0.0), errors="coerce").fillna(0.0)
    posts["tracked_discovery_score"] = pd.to_numeric(posts.get("tracked_discovery_score", 0.0), errors="coerce").fillna(0.0)
    posts["is_active_tracked_account"] = posts.get("is_active_tracked_account", False).fillna(False).astype(bool)
    posts["author_is_trump"] = posts.get("author_is_trump", False).fillna(False).astype(bool)
    posts["mentions_trump"] = posts.get("mentions_trump", False).fillna(False).astype(bool)
    posts["tracked_account_status"] = posts.get("tracked_account_status", "none").fillna("none")

    posts["author_weight"] = (
        1.0
        + posts["author_is_trump"].astype(float) * 0.6
        + posts["is_active_tracked_account"].astype(float) * 0.35
        + posts["tracked_discovery_score"].clip(lower=0.0, upper=5.0) * 0.1
    )
    posts["engagement_boost"] = 1.0 + np.log1p(posts["engagement_score"].clip(lower=0.0)) / 5.0
    posts["post_signal_score"] = posts["sentiment_score"] * posts["author_weight"] * posts["engagement_boost"]
    posts["abs_post_signal_score"] = posts["post_signal_score"].abs()
    posts["post_preview"] = posts["cleaned_text"].fillna("").map(lambda value: truncate_text(str(value), max_chars=140))

    keep = [column for column in POST_ATTRIBUTION_COLUMNS if column in posts.columns]
    return posts[keep].sort_values(
        ["signal_session_date", "abs_post_signal_score", "post_timestamp"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_account_attribution(post_attribution: pd.DataFrame) -> pd.DataFrame:
    if post_attribution.empty:
        return pd.DataFrame(columns=ACCOUNT_ATTRIBUTION_COLUMNS)

    grouped = (
        post_attribution.groupby(
            [
                "signal_session_date",
                "author_account_id",
                "author_handle",
                "author_display_name",
                "source_platform",
                "author_is_trump",
                "is_active_tracked_account",
                "tracked_account_status",
            ],
            dropna=False,
        )
        .agg(
            post_count=("post_preview", "size"),
            avg_sentiment=("sentiment_score", "mean"),
            total_engagement=("engagement_score", "sum"),
            net_post_signal=("post_signal_score", "sum"),
        )
        .reset_index()
    )
    grouped["abs_net_post_signal"] = grouped["net_post_signal"].abs()
    return grouped[ACCOUNT_ATTRIBUTION_COLUMNS].sort_values(
        ["signal_session_date", "abs_net_post_signal", "post_count"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

