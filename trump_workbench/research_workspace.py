from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import AppSettings
from .enrichment import LLMEnrichmentService, parse_semantic_asset_targets
from .features import FeatureService, map_posts_to_trade_sessions
from .research import (
    aggregate_research_sessions,
    build_combined_chart,
    build_event_frame,
    build_narrative_asset_heatmap_chart,
    build_narrative_asset_heatmap_frame,
    build_narrative_frequency_chart,
    build_narrative_frequency_frame,
    build_narrative_return_chart,
    build_narrative_return_frame,
    filter_narrative_rows,
    filter_posts,
    make_narrative_event_table,
    make_narrative_post_table,
    make_post_table,
    make_session_table,
)
from .research_exports import (
    build_research_export_bundle,
    build_research_export_manifest,
    research_export_filename,
)
from .storage import DuckDBStore

PLATFORM_OPTIONS = ("Truth Social", "X")
NARRATIVE_BUCKET_FIELDS = {
    "semantic_topic": "Topic",
    "semantic_policy_bucket": "Policy bucket",
    "semantic_stance": "Stance",
    "semantic_primary_asset": "Primary asset",
}
TRACKED_SCOPE_OPTIONS = ("All posts", "Trump + tracked accounts", "Tracked accounts only")


@dataclass(frozen=True)
class ResearchWorkspaceResult:
    payload: dict[str, Any]
    export_bundle: bytes
    export_filename: str


def detect_source_mode(posts: pd.DataFrame) -> dict[str, Any]:
    if posts.empty or "source_platform" not in posts.columns:
        return {
            "mode": "unknown",
            "has_truth_posts": False,
            "has_x_posts": False,
            "truth_post_count": 0,
            "x_post_count": 0,
        }

    platforms = posts["source_platform"].fillna("").astype(str)
    truth_post_count = int((platforms == "Truth Social").sum())
    x_post_count = int((platforms == "X").sum())
    has_truth_posts = truth_post_count > 0
    has_x_posts = x_post_count > 0
    mode = "truth_plus_x" if has_x_posts else "truth_only" if has_truth_posts else "unknown"
    return {
        "mode": mode,
        "has_truth_posts": has_truth_posts,
        "has_x_posts": has_x_posts,
        "truth_post_count": truth_post_count,
        "x_post_count": x_post_count,
    }


def source_mode_label(source_mode: dict[str, Any]) -> str:
    mode = str(source_mode.get("mode", "unknown"))
    if mode == "truth_only":
        return "Truth Social-only"
    if mode == "truth_plus_x":
        return "Truth Social + X mentions"
    return "No source data"


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [_json_safe(record) for record in frame.to_dict(orient="records")]


def _figure_json(fig: Any) -> dict[str, Any]:
    return json.loads(fig.to_json())


def _coerce_date(value: str | None, default: pd.Timestamp) -> pd.Timestamp:
    if value is None or str(value).strip() == "":
        return pd.Timestamp(default).normalize()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp(default).normalize()
    return pd.Timestamp(parsed).normalize()


def _normalize_choices(values: Iterable[str] | None, allowed: Iterable[str]) -> list[str] | None:
    if values is None:
        return None
    allowed_lookup = {str(value).lower(): str(value) for value in allowed}
    normalized: list[str] = []
    for value in values:
        for part in str(value).split(","):
            key = part.strip().lower()
            if key and key in allowed_lookup and allowed_lookup[key] not in normalized:
                normalized.append(allowed_lookup[key])
    return normalized or None


def _string_options(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame.columns:
        return []
    return sorted(
        value
        for value in frame[column].fillna("").astype(str).unique().tolist()
        if value.strip()
    )


def _asset_target_options(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    targets = set()
    if "semantic_asset_targets" in frame.columns:
        for value in frame["semantic_asset_targets"].fillna("").astype(str):
            targets.update(parse_semantic_asset_targets(value))
    if "semantic_primary_asset" in frame.columns:
        targets.update(
            str(value).upper()
            for value in frame["semantic_primary_asset"].fillna("").astype(str)
            if value.strip()
        )
    return sorted(targets)


def _provider_summary(mapped_posts: pd.DataFrame) -> pd.DataFrame:
    if mapped_posts.empty or "semantic_provider" not in mapped_posts.columns:
        return pd.DataFrame(columns=["semantic_provider", "posts", "cache_hit_rate", "avg_market_relevance"])
    frame = mapped_posts.copy()
    if "semantic_cache_hit" not in frame.columns:
        frame["semantic_cache_hit"] = False
    if "semantic_market_relevance" not in frame.columns:
        frame["semantic_market_relevance"] = 0.0
    return (
        frame.groupby("semantic_provider", as_index=False)
        .agg(
            posts=("semantic_provider", "size"),
            cache_hit_rate=("semantic_cache_hit", "mean"),
            avg_market_relevance=("semantic_market_relevance", "mean"),
        )
        .sort_values("posts", ascending=False)
        .reset_index(drop=True)
    )


def _resolve_filters(
    settings: AppSettings,
    posts: pd.DataFrame,
    *,
    date_start: str | None = None,
    date_end: str | None = None,
    platforms: Iterable[str] | None = None,
    include_reshares: bool | None = None,
    tracked_only: bool | None = None,
    trump_authored_only: bool | None = None,
    keyword: str | None = None,
    scale_markers: bool | None = None,
    narrative_topic: str | None = None,
    narrative_policy: str | None = None,
    narrative_stance: str | None = None,
    narrative_urgency: str | None = None,
    narrative_asset: str | None = None,
    narrative_platforms: Iterable[str] | None = None,
    narrative_tracked_scope: str | None = None,
    narrative_bucket_field: str | None = None,
) -> dict[str, Any]:
    source_mode = detect_source_mode(posts)
    today_et = pd.Timestamp.now(tz=settings.timezone).normalize().tz_localize(None)
    start = _coerce_date(date_start, settings.term_start)
    end = _coerce_date(date_end, today_et)
    if end < start:
        start, end = end, start

    selected_platforms = _normalize_choices(platforms, PLATFORM_OPTIONS)
    if selected_platforms is None:
        selected_platforms = ["Truth Social"] if source_mode["mode"] == "truth_only" else list(PLATFORM_OPTIONS)

    bucket_field = str(narrative_bucket_field or "semantic_topic")
    if bucket_field not in NARRATIVE_BUCKET_FIELDS:
        bucket_field = "semantic_topic"

    tracked_scope = str(narrative_tracked_scope or "All posts")
    if tracked_scope not in TRACKED_SCOPE_OPTIONS:
        tracked_scope = "All posts"

    return {
        "date_start": start,
        "date_end": end,
        "platforms": selected_platforms,
        "include_reshares": bool(include_reshares) if include_reshares is not None else False,
        "tracked_only": bool(tracked_only) if tracked_only is not None else False,
        "trump_authored_only": bool(trump_authored_only) if trump_authored_only is not None else source_mode["mode"] == "truth_only",
        "keyword": str(keyword or ""),
        "scale_markers": bool(scale_markers) if scale_markers is not None else True,
        "narrative_topic": str(narrative_topic or "All"),
        "narrative_policy": str(narrative_policy or "All"),
        "narrative_stance": str(narrative_stance or "All"),
        "narrative_urgency": str(narrative_urgency or "All"),
        "narrative_asset": str(narrative_asset or "All"),
        "narrative_platforms": _normalize_choices(narrative_platforms, PLATFORM_OPTIONS),
        "narrative_tracked_scope": tracked_scope,
        "narrative_bucket_field": bucket_field,
    }


def _public_filter_payload(filters: dict[str, Any]) -> dict[str, Any]:
    payload = dict(filters)
    payload["date_start"] = pd.Timestamp(payload["date_start"]).date().isoformat()
    payload["date_end"] = pd.Timestamp(payload["date_end"]).date().isoformat()
    return payload


def _prepare_session_posts_readonly(
    posts: pd.DataFrame,
    market: pd.DataFrame,
    tracked_accounts: pd.DataFrame,
    enrichment_service: LLMEnrichmentService,
) -> pd.DataFrame:
    if posts.empty or market.empty:
        return posts.head(0).copy()
    mapped = map_posts_to_trade_sessions(posts, market[["trade_date"]])
    mapped = enrichment_service.enrich_posts_readonly(mapped, enabled=True)
    return FeatureService(enrichment_service)._flag_tracked_posts(mapped, tracked_accounts)


def build_research_workspace(
    *,
    settings: AppSettings,
    store: DuckDBStore,
    enrichment_service: LLMEnrichmentService,
    date_start: str | None = None,
    date_end: str | None = None,
    platforms: Iterable[str] | None = None,
    include_reshares: bool | None = None,
    tracked_only: bool | None = None,
    trump_authored_only: bool | None = None,
    keyword: str | None = None,
    scale_markers: bool | None = None,
    narrative_topic: str | None = None,
    narrative_policy: str | None = None,
    narrative_stance: str | None = None,
    narrative_urgency: str | None = None,
    narrative_asset: str | None = None,
    narrative_platforms: Iterable[str] | None = None,
    narrative_tracked_scope: str | None = None,
    narrative_bucket_field: str | None = None,
) -> ResearchWorkspaceResult:
    posts = store.read_frame("normalized_posts")
    sp500 = store.read_frame("sp500_daily")
    asset_universe = store.read_frame("asset_universe")
    asset_post_mappings = store.read_frame("asset_post_mappings")
    tracked_accounts = store.read_frame("tracked_accounts")
    source_mode = detect_source_mode(posts)
    filters = _resolve_filters(
        settings,
        posts,
        date_start=date_start,
        date_end=date_end,
        platforms=platforms,
        include_reshares=include_reshares,
        tracked_only=tracked_only,
        trump_authored_only=trump_authored_only,
        keyword=keyword,
        scale_markers=scale_markers,
        narrative_topic=narrative_topic,
        narrative_policy=narrative_policy,
        narrative_stance=narrative_stance,
        narrative_urgency=narrative_urgency,
        narrative_asset=narrative_asset,
        narrative_platforms=narrative_platforms,
        narrative_tracked_scope=narrative_tracked_scope,
        narrative_bucket_field=narrative_bucket_field,
    )
    public_filters = _public_filter_payload(filters)

    empty_chart = build_combined_chart(pd.DataFrame(columns=["trade_date", "close", "daily_return_pct", "has_posts", "post_count"]), scale_markers=True)
    if posts.empty or sp500.empty:
        empty_manifest = build_research_export_manifest(
            filters=public_filters,
            source_mode=source_mode,
            headline_metrics={
                "sessions_with_posts": 0,
                "posts_in_view": 0,
                "truth_posts": 0,
                "tracked_x_posts": 0,
                "mean_sentiment": 0.0,
                "sp500_change": None,
            },
        )
        empty_bundle = build_research_export_bundle(
            manifest=empty_manifest,
            chart=empty_chart,
            sessions=pd.DataFrame(),
            posts=pd.DataFrame(),
            narrative_frequency=pd.DataFrame(),
            narrative_returns=pd.DataFrame(),
            narrative_asset_heatmap=pd.DataFrame(),
            narrative_posts=pd.DataFrame(),
            narrative_events=pd.DataFrame(),
        )
        return ResearchWorkspaceResult(
            payload={
                "ready": False,
                "message": "Refresh datasets first so the research workspace has source posts and S&P 500 market data.",
                "source_mode": source_mode,
                "filters": public_filters,
                "headline_metrics": {},
                "charts": {"social_activity": _figure_json(empty_chart)},
                "session_rows": [],
                "post_rows": [],
                "narrative_filter_options": {},
                "narrative_metrics": {},
                "provider_summary": [],
                "narrative_frequency": [],
                "narrative_returns": [],
                "narrative_asset_heatmap": [],
                "narrative_posts": [],
                "narrative_events": [],
                "export_filename": research_export_filename(filters["date_start"], filters["date_end"]),
            },
            export_bundle=empty_bundle,
            export_filename=research_export_filename(filters["date_start"], filters["date_end"]),
        )

    market = sp500.copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce").dt.normalize()
    market = market.loc[
        (market["trade_date"] >= filters["date_start"])
        & (market["trade_date"] <= filters["date_end"])
    ].copy()

    filtered_posts = filter_posts(
        posts=posts,
        date_start=filters["date_start"],
        date_end=filters["date_end"],
        include_reshares=filters["include_reshares"],
        platforms=filters["platforms"],
        keyword=filters["keyword"],
        tracked_only=filters["tracked_only"],
        trump_authored_only=filters["trump_authored_only"],
    )
    mapped = _prepare_session_posts_readonly(filtered_posts, market, tracked_accounts, enrichment_service)
    sessions = aggregate_research_sessions(mapped)
    events = build_event_frame(market, sessions)

    feature_service = FeatureService(enrichment_service)
    narrative_asset_view = pd.DataFrame()
    if not mapped.empty and not asset_universe.empty:
        narrative_asset_view = feature_service.build_asset_post_mappings(
            prepared_posts=mapped,
            asset_universe=asset_universe,
            llm_enabled=True,
        )
    elif not asset_post_mappings.empty:
        narrative_asset_view = asset_post_mappings.copy()
    if not narrative_asset_view.empty and "session_date" in narrative_asset_view.columns:
        narrative_asset_view["session_date"] = pd.to_datetime(narrative_asset_view["session_date"], errors="coerce").dt.normalize()
        narrative_asset_view = narrative_asset_view.loc[
            (narrative_asset_view["session_date"] >= filters["date_start"])
            & (narrative_asset_view["session_date"] <= filters["date_end"])
        ].copy()

    sessions_with_posts = int((events["post_count"] > 0).sum()) if "post_count" in events.columns else 0
    posts_in_view = int(events["post_count"].sum()) if "post_count" in events.columns else 0
    truth_posts = int(mapped.loc[mapped.get("author_is_trump", pd.Series(False, index=mapped.index)).astype(bool)].shape[0]) if not mapped.empty else 0
    tracked_x_posts = int(mapped.loc[mapped.get("is_active_tracked_account", pd.Series(False, index=mapped.index)).astype(bool)].shape[0]) if not mapped.empty else 0
    mean_sentiment = float(mapped["sentiment_score"].mean()) if not mapped.empty and "sentiment_score" in mapped.columns else 0.0
    sp500_change = float((events["close"].iloc[-1] / events["close"].iloc[0]) - 1.0) if len(events) > 1 else None
    headline_metrics = {
        "sessions_with_posts": sessions_with_posts,
        "posts_in_view": posts_in_view,
        "truth_posts": truth_posts,
        "tracked_x_posts": tracked_x_posts,
        "mean_sentiment": mean_sentiment,
        "sp500_change": sp500_change,
    }

    platform_options = _string_options(mapped, "source_platform")
    selected_narrative_platforms = filters["narrative_platforms"] if filters["narrative_platforms"] is not None else platform_options
    narrative_filter_options = {
        "topics": ["All"] + _string_options(mapped, "semantic_topic"),
        "policy_buckets": ["All"] + _string_options(mapped, "semantic_policy_bucket"),
        "stances": ["All"] + _string_options(mapped, "semantic_stance"),
        "urgency_bands": ["All", "low", "medium", "high"],
        "assets": ["All"] + _asset_target_options(mapped),
        "platforms": platform_options,
        "tracked_scopes": list(TRACKED_SCOPE_OPTIONS),
        "bucket_fields": [{"value": value, "label": label} for value, label in NARRATIVE_BUCKET_FIELDS.items()],
    }
    public_filters["narrative_platforms"] = selected_narrative_platforms

    filtered_narratives = filter_narrative_rows(
        mapped,
        topic=filters["narrative_topic"],
        policy_bucket=filters["narrative_policy"],
        stance=filters["narrative_stance"],
        urgency_band=filters["narrative_urgency"],
        narrative_asset=filters["narrative_asset"],
        platforms=selected_narrative_platforms,
        tracked_scope=filters["narrative_tracked_scope"],
    )
    filtered_narrative_assets = filter_narrative_rows(
        narrative_asset_view,
        topic=filters["narrative_topic"],
        policy_bucket=filters["narrative_policy"],
        stance=filters["narrative_stance"],
        urgency_band=filters["narrative_urgency"],
        narrative_asset=filters["narrative_asset"],
        platforms=selected_narrative_platforms,
        tracked_scope=filters["narrative_tracked_scope"],
    )

    provider_summary = _provider_summary(mapped)
    frequency = build_narrative_frequency_frame(filtered_narratives)
    returns = build_narrative_return_frame(
        filtered_narratives,
        market,
        bucket_field=filters["narrative_bucket_field"],
    )
    heatmap = build_narrative_asset_heatmap_frame(filtered_narrative_assets)
    narrative_posts_table = make_narrative_post_table(filtered_narratives).head(25)
    narrative_events_table = make_narrative_event_table(filtered_narratives, market)
    session_table = make_session_table(events)
    post_table = make_post_table(mapped)
    social_chart = build_combined_chart(events, scale_markers=filters["scale_markers"])

    export_manifest = build_research_export_manifest(
        filters={**public_filters, "narrative_bucket_field": filters["narrative_bucket_field"]},
        source_mode=source_mode,
        headline_metrics=headline_metrics,
    )
    export_bundle = build_research_export_bundle(
        manifest=export_manifest,
        chart=social_chart,
        sessions=session_table,
        posts=post_table,
        narrative_frequency=build_narrative_frequency_frame(mapped),
        narrative_returns=build_narrative_return_frame(mapped, market, bucket_field="semantic_topic"),
        narrative_asset_heatmap=build_narrative_asset_heatmap_frame(narrative_asset_view),
        narrative_posts=make_narrative_post_table(mapped).head(25),
        narrative_events=make_narrative_event_table(mapped, market),
    )
    export_filename = research_export_filename(filters["date_start"], filters["date_end"])

    payload = {
        "ready": True,
        "message": "",
        "source_mode": source_mode,
        "filters": public_filters,
        "headline_metrics": headline_metrics,
        "charts": {
            "social_activity": _figure_json(social_chart),
            "narrative_frequency": _figure_json(build_narrative_frequency_chart(frequency)),
            "narrative_returns": _figure_json(build_narrative_return_chart(returns, filters["narrative_bucket_field"])),
            "narrative_asset_heatmap": _figure_json(build_narrative_asset_heatmap_chart(heatmap)),
        },
        "session_rows": _frame_records(session_table.sort_values(["post_count", "trade_date"], ascending=[False, False]) if not session_table.empty else session_table),
        "post_rows": _frame_records(post_table),
        "narrative_filter_options": narrative_filter_options,
        "narrative_metrics": {
            "narrative_tagged_posts": int(len(filtered_narratives)),
            "narrative_sessions": int(filtered_narratives["session_date"].nunique()) if "session_date" in filtered_narratives.columns else 0,
            "cache_hit_rate": float(mapped["semantic_cache_hit"].astype(bool).mean()) if "semantic_cache_hit" in mapped.columns and not mapped.empty else 0.0,
            "providers_used": int(provider_summary["semantic_provider"].nunique()) if "semantic_provider" in provider_summary.columns else 0,
        },
        "provider_summary": _frame_records(provider_summary),
        "narrative_frequency": _frame_records(frequency),
        "narrative_returns": _frame_records(returns),
        "narrative_asset_heatmap": _frame_records(heatmap),
        "narrative_posts": _frame_records(narrative_posts_table),
        "narrative_events": _frame_records(narrative_events_table),
        "export_filename": export_filename,
    }
    return ResearchWorkspaceResult(payload=payload, export_bundle=export_bundle, export_filename=export_filename)
