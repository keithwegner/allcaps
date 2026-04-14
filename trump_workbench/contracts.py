from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import pandas as pd

NORMALIZED_POST_COLUMNS = [
    "source_platform",
    "source_type",
    "author_account_id",
    "author_handle",
    "author_display_name",
    "author_is_trump",
    "post_id",
    "post_url",
    "post_timestamp",
    "raw_text",
    "cleaned_text",
    "is_reshare",
    "has_media",
    "replies_count",
    "reblogs_count",
    "favourites_count",
    "mentions_trump",
    "source_provenance",
    "engagement_score",
    "sentiment_score",
    "sentiment_label",
]

TRACKED_ACCOUNT_COLUMNS = [
    "version_id",
    "account_id",
    "handle",
    "display_name",
    "source_platform",
    "discovery_score",
    "status",
    "first_seen_at",
    "last_seen_at",
    "effective_from",
    "effective_to",
    "auto_included",
    "provenance",
    "mention_count",
    "engagement_mean",
    "active_days",
]

MANUAL_OVERRIDE_COLUMNS = [
    "override_id",
    "account_id",
    "handle",
    "display_name",
    "source_platform",
    "action",
    "effective_from",
    "effective_to",
    "note",
    "created_at",
]

RANKING_HISTORY_COLUMNS = [
    "author_account_id",
    "author_handle",
    "author_display_name",
    "source_platform",
    "discovery_score",
    "mention_count",
    "engagement_mean",
    "active_days",
    "ranked_at",
    "discovery_rank",
    "final_selected",
    "selected_status",
    "suppressed_by_override",
    "pinned_by_override",
]


class SourceAdapter(Protocol):
    name: str

    def fetch_history(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        ...

    def fetch_since(
        self,
        last_cursor: Optional[pd.Timestamp],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        ...


@dataclass(frozen=True)
class NormalizedPost:
    source_platform: str
    source_type: str
    author_account_id: str
    author_handle: str
    author_display_name: str
    author_is_trump: bool
    post_id: str
    post_url: str
    post_timestamp: pd.Timestamp
    raw_text: str
    cleaned_text: str
    is_reshare: bool
    has_media: bool
    replies_count: int
    reblogs_count: int
    favourites_count: int
    mentions_trump: bool
    source_provenance: str
    engagement_score: float
    sentiment_score: float = 0.0
    sentiment_label: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackedAccount:
    version_id: str
    account_id: str
    handle: str
    display_name: str
    source_platform: str
    discovery_score: float
    status: str
    first_seen_at: pd.Timestamp
    last_seen_at: pd.Timestamp
    effective_from: pd.Timestamp
    effective_to: Optional[pd.Timestamp]
    auto_included: bool
    provenance: str
    mention_count: int
    engagement_mean: float
    active_days: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionFeatureRow:
    trade_date: pd.Timestamp
    feature_version: str
    has_posts: bool
    post_count: int
    trump_post_count: int
    x_post_count: int
    tracked_account_post_count: int
    mention_post_count: int
    target_next_session_return: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelRunConfig:
    run_name: str
    target_asset: str = "SPY"
    feature_version: str = "v1"
    llm_enabled: bool = False
    train_window: int = 90
    validation_window: int = 30
    test_window: int = 30
    step_size: int = 30
    threshold_grid: tuple[float, ...] = (0.0, 0.001, 0.0025, 0.005)
    minimum_signal_grid: tuple[int, ...] = (1, 2, 3)
    account_weight_grid: tuple[float, ...] = (0.5, 1.0, 1.5)
    ridge_alpha: float = 1.0
    transaction_cost_bps: float = 2.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["threshold_grid"] = list(self.threshold_grid)
        payload["minimum_signal_grid"] = list(self.minimum_signal_grid)
        payload["account_weight_grid"] = list(self.account_weight_grid)
        return payload


@dataclass(frozen=True)
class PredictionSnapshot:
    signal_session_date: pd.Timestamp
    next_session_date: Optional[pd.Timestamp]
    target_asset: str
    expected_return_score: float
    feature_version: str
    model_version: str
    confidence: float
    generated_at: pd.Timestamp
    stance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestRun:
    run_id: str
    run_name: str
    target_asset: str
    config_hash: str
    train_window: int
    validation_window: int
    test_window: int
    metrics: dict[str, float]
    selected_params: dict[str, Any]
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LinearModelArtifact:
    model_version: str
    feature_names: list[str]
    intercept: float
    coefficients: list[float]
    means: list[float]
    stds: list[float]
    residual_std: float
    train_rows: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearModelArtifact":
        return cls(**payload)


@dataclass(frozen=True)
class SavedRunArtifacts:
    summary_path: Path
    trades_path: Path
    predictions_path: Path
    windows_path: Path
    importance_path: Path
    model_path: Path
    feature_contributions_path: Path
    post_attribution_path: Path
    account_attribution_path: Path
    benchmarks_path: Path
    diagnostics_path: Path
    benchmark_curves_path: Path
    leakage_audit_path: Path
