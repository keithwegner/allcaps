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
    run_type: str = "asset_model"
    allocator_mode: str = ""
    fallback_mode: str = ""
    deployment_variant: str = ""
    component_run_ids: list[str] = field(default_factory=list)
    universe_symbols: list[str] = field(default_factory=list)
    topology_variants: list[str] = field(default_factory=list)
    model_families: list[str] = field(default_factory=list)
    selected_symbols: list[str] = field(default_factory=list)
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioCandidatePrediction:
    signal_session_date: pd.Timestamp
    next_session_date: Optional[pd.Timestamp]
    asset_symbol: str
    run_id: str
    run_name: str
    expected_return_score: float
    confidence: float
    threshold: float
    min_post_count: int
    post_count: int
    tradeable: bool = False
    target_available: bool = False
    next_session_open_ts: Optional[pd.Timestamp] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioDecision:
    signal_session_date: pd.Timestamp
    next_session_date: Optional[pd.Timestamp]
    winning_asset: str
    winning_run_id: str
    decision_source: str
    fallback_mode: str
    stance: str
    eligible_asset_count: int
    runner_up_asset: str = ""
    winner_score: float = 0.0
    runner_up_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioRunConfig:
    run_name: str
    allocator_mode: str = "saved_runs"
    fallback_mode: str = "SPY"
    transaction_cost_bps: float = 2.0
    component_run_ids: tuple[str, ...] = ()
    universe_symbols: tuple[str, ...] = ()
    selected_symbols: tuple[str, ...] = ()
    llm_enabled: bool = False
    feature_version: str = "asset-v1"
    train_window: int = 90
    validation_window: int = 30
    test_window: int = 30
    step_size: int = 30
    threshold_grid: tuple[float, ...] = (0.0, 0.001, 0.0025, 0.005)
    minimum_signal_grid: tuple[int, ...] = (1, 2, 3)
    account_weight_grid: tuple[float, ...] = (0.5, 1.0, 1.5)
    model_families: tuple[str, ...] = ()
    topology_variants: tuple[str, ...] = ()
    deployment_variant: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["component_run_ids"] = list(self.component_run_ids)
        payload["universe_symbols"] = list(self.universe_symbols)
        payload["selected_symbols"] = list(self.selected_symbols)
        payload["threshold_grid"] = list(self.threshold_grid)
        payload["minimum_signal_grid"] = list(self.minimum_signal_grid)
        payload["account_weight_grid"] = list(self.account_weight_grid)
        payload["model_families"] = list(self.model_families)
        payload["topology_variants"] = list(self.topology_variants)
        return payload


@dataclass(frozen=True)
class LiveMonitorPinnedRun:
    asset_symbol: str
    run_id: str
    run_name: str = ""
    model_version: str = ""
    pinned_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiveMonitorPinnedRun":
        return cls(**payload)


@dataclass(frozen=True)
class LiveMonitorConfig:
    mode: str = "portfolio_run"
    fallback_mode: str = "SPY"
    portfolio_run_id: str = ""
    portfolio_run_name: str = ""
    deployment_variant: str = ""
    pinned_runs: list[LiveMonitorPinnedRun] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pinned_runs"] = [item.to_dict() for item in self.pinned_runs]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiveMonitorConfig":
        pinned_runs = [
            LiveMonitorPinnedRun.from_dict(item)
            for item in payload.get("pinned_runs", [])
            if isinstance(item, dict)
        ]
        if "mode" in payload:
            mode = str(payload.get("mode", "portfolio_run") or "portfolio_run")
        elif pinned_runs:
            mode = "asset_model_set"
        else:
            mode = "portfolio_run"
        return cls(
            mode=mode,
            fallback_mode=str(payload.get("fallback_mode", "SPY") or "SPY").upper(),
            portfolio_run_id=str(payload.get("portfolio_run_id", "") or ""),
            portfolio_run_name=str(payload.get("portfolio_run_name", "") or ""),
            deployment_variant=str(payload.get("deployment_variant", "") or ""),
            pinned_runs=pinned_runs,
        )


@dataclass(frozen=True)
class PaperPortfolioConfig:
    paper_portfolio_id: str
    portfolio_run_id: str
    portfolio_run_name: str = ""
    deployment_variant: str = ""
    fallback_mode: str = "SPY"
    transaction_cost_bps: float = 0.0
    starting_cash: float = 100000.0
    enabled: bool = False
    created_at: str = ""
    archived_at: str = ""

    @property
    def is_archived(self) -> bool:
        return bool(self.archived_at)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PaperPortfolioConfig":
        return cls(
            paper_portfolio_id=str(payload.get("paper_portfolio_id", "") or ""),
            portfolio_run_id=str(payload.get("portfolio_run_id", "") or ""),
            portfolio_run_name=str(payload.get("portfolio_run_name", "") or ""),
            deployment_variant=str(payload.get("deployment_variant", "") or ""),
            fallback_mode=str(payload.get("fallback_mode", "SPY") or "SPY").upper(),
            transaction_cost_bps=float(payload.get("transaction_cost_bps", 0.0) or 0.0),
            starting_cash=float(payload.get("starting_cash", 100000.0) or 100000.0),
            enabled=bool(payload.get("enabled", False)),
            created_at=str(payload.get("created_at", "") or ""),
            archived_at=str(payload.get("archived_at", "") or ""),
        )


@dataclass(frozen=True)
class PaperDecisionRecord:
    paper_portfolio_id: str
    generated_at: pd.Timestamp
    signal_session_date: pd.Timestamp
    next_session_date: Optional[pd.Timestamp]
    decision_cutoff_ts: Optional[pd.Timestamp]
    portfolio_run_id: str
    portfolio_run_name: str
    deployment_variant: str
    winning_asset: str
    winning_run_id: str
    decision_source: str
    fallback_mode: str
    stance: str
    winner_score: float
    runner_up_asset: str = ""
    runner_up_score: float = 0.0
    eligible_asset_count: int = 0
    settlement_status: str = "pending"
    settled_at: Optional[pd.Timestamp] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperTradeRecord:
    paper_portfolio_id: str
    signal_session_date: pd.Timestamp
    next_session_date: Optional[pd.Timestamp]
    asset_symbol: str
    run_id: str
    decision_source: str
    stance: str
    next_session_open: float
    next_session_close: float
    gross_return: float
    net_return: float
    benchmark_return: float
    transaction_cost_bps: float
    starting_equity: float
    ending_equity: float
    settled_at: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperEquityPoint:
    paper_portfolio_id: str
    signal_session_date: pd.Timestamp
    next_session_date: Optional[pd.Timestamp]
    equity: float
    return_pct: float
    settled_at: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LinearModelArtifact:
    model_version: str
    feature_names: list[str]
    intercept: float = 0.0
    coefficients: list[float] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    residual_std: float = 0.0
    train_rows: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    model_family: str = "custom_linear"
    feature_importances: list[float] = field(default_factory=list)
    serialized_estimator_b64: str = ""
    explanation_kind: str = "linear_exact"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearModelArtifact":
        normalized = dict(payload)
        normalized.setdefault("intercept", 0.0)
        normalized.setdefault("coefficients", [])
        normalized.setdefault("means", [])
        normalized.setdefault("stds", [])
        normalized.setdefault("residual_std", 0.0)
        normalized.setdefault("train_rows", 0)
        normalized.setdefault("metadata", {})
        normalized.setdefault("model_family", "custom_linear")
        normalized.setdefault("feature_importances", [])
        normalized.setdefault("serialized_estimator_b64", "")
        normalized.setdefault("explanation_kind", "linear_exact")
        return cls(**normalized)


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
    candidate_predictions_path: Optional[Path] = None
    variant_summary_path: Optional[Path] = None
    portfolio_model_bundle_path: Optional[Path] = None
