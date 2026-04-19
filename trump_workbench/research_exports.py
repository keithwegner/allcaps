from __future__ import annotations

import datetime as dt
import io
import json
from collections.abc import Mapping
from typing import Any
import zipfile

import pandas as pd
import plotly.graph_objects as go

EXPORT_SCHEMA_VERSION = "research-export-v1"

EXPORT_TABLE_COLUMNS: dict[str, list[str]] = {
    "sessions.csv": [
        "trade_date",
        "post_count",
        "truth_posts",
        "x_posts",
        "tracked_account_posts",
        "positive_posts",
        "neutral_posts",
        "negative_posts",
        "sp500_close",
        "session_return",
        "next_session_return",
        "sentiment_open",
        "sentiment_high",
        "sentiment_low",
        "sentiment_close",
        "sentiment_avg",
        "sentiment_range",
        "sample_posts",
    ],
    "posts.csv": [
        "source_platform",
        "author_handle",
        "author_display_name",
        "post_time_et",
        "session_date",
        "mapping_reason",
        "mentions_trump",
        "is_active_tracked_account",
        "sentiment_score",
        "sentiment_label",
        "post_text",
        "post_url",
    ],
    "narrative_frequency.csv": [
        "session_date",
        "semantic_topic",
        "post_count",
        "avg_market_relevance",
        "avg_urgency",
    ],
    "narrative_returns.csv": [
        "semantic_topic",
        "avg_next_session_return",
        "median_next_session_return",
        "session_count",
        "total_posts",
        "avg_market_relevance",
        "bucket_field",
    ],
    "narrative_asset_heatmap.csv": [
        "semantic_topic",
        "asset_symbol",
        "post_count",
        "avg_asset_relevance",
        "avg_market_relevance",
    ],
    "narrative_posts.csv": [
        "session_date",
        "post_time_et",
        "source_platform",
        "author_handle",
        "semantic_topic",
        "semantic_policy_bucket",
        "semantic_stance",
        "semantic_primary_asset",
        "semantic_asset_targets",
        "semantic_market_relevance",
        "semantic_urgency",
        "urgency_band",
        "semantic_provider",
        "semantic_cache_hit",
        "semantic_summary",
        "post_text",
        "post_url",
    ],
    "narrative_events.csv": [
        "trade_date",
        "post_count",
        "avg_sentiment",
        "avg_market_relevance",
        "avg_urgency",
        "primary_topics",
        "primary_assets",
        "sample_posts",
        "next_session_return",
    ],
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def build_research_export_manifest(
    *,
    filters: Mapping[str, Any],
    source_mode: Mapping[str, Any],
    headline_metrics: Mapping[str, Any],
    generated_at: pd.Timestamp | None = None,
) -> dict[str, Any]:
    generated = pd.Timestamp.now(tz="UTC").floor("s") if generated_at is None else pd.Timestamp(generated_at).floor("s")
    if generated.tzinfo is None:
        generated = generated.tz_localize("UTC")
    else:
        generated = generated.tz_convert("UTC")
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "generated_at": generated.isoformat(),
        "filters": _json_safe(dict(filters)),
        "source_mode": _json_safe(dict(source_mode)),
        "headline_metrics": _json_safe(dict(headline_metrics)),
    }


def build_research_export_summary(manifest: Mapping[str, Any]) -> str:
    filters = dict(manifest.get("filters", {}) or {})
    metrics = dict(manifest.get("headline_metrics", {}) or {})
    source_mode = dict(manifest.get("source_mode", {}) or {})
    platforms = ", ".join(str(item) for item in filters.get("platforms", []) or []) or "none"
    lines = [
        "# Research Export Pack",
        "",
        f"Generated: {manifest.get('generated_at', 'unknown')}",
        f"Source mode: {source_mode.get('mode', 'unknown')}",
        "",
        "## Scope",
        "",
        f"- Date range: {filters.get('date_start', 'n/a')} to {filters.get('date_end', 'n/a')}",
        f"- Platforms: {platforms}",
        f"- Keyword: {filters.get('keyword', '') or '(none)'}",
        f"- Include reshares: {bool(filters.get('include_reshares', False))}",
        f"- Tracked accounts only: {bool(filters.get('tracked_only', False))}",
        f"- Trump-authored only: {bool(filters.get('trump_authored_only', False))}",
        "",
        "## Headline Metrics",
        "",
        f"- Sessions with posts: {metrics.get('sessions_with_posts', 0)}",
        f"- Posts in view: {metrics.get('posts_in_view', 0)}",
        f"- Truth posts: {metrics.get('truth_posts', 0)}",
        f"- Tracked X posts: {metrics.get('tracked_x_posts', 0)}",
        f"- Mean sentiment: {metrics.get('mean_sentiment', 'n/a')}",
        f"- S&P 500 change: {metrics.get('sp500_change', 'n/a')}",
        "",
        "## Notes",
        "",
        "- This pack reflects the current Research View filters at download time.",
        "- Research outputs are descriptive and are not proof of causality.",
    ]
    return "\n".join(lines) + "\n"


def _csv_bytes(frame: pd.DataFrame, columns: list[str]) -> bytes:
    export_frame = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if export_frame.empty:
        export_frame = export_frame.reindex(columns=columns)
    return export_frame.to_csv(index=False).encode("utf-8")


def build_research_export_bundle(
    *,
    manifest: Mapping[str, Any],
    chart: go.Figure,
    sessions: pd.DataFrame,
    posts: pd.DataFrame,
    narrative_frequency: pd.DataFrame,
    narrative_returns: pd.DataFrame,
    narrative_asset_heatmap: pd.DataFrame,
    narrative_posts: pd.DataFrame,
    narrative_events: pd.DataFrame,
) -> bytes:
    table_frames = {
        "sessions.csv": sessions,
        "posts.csv": posts,
        "narrative_frequency.csv": narrative_frequency,
        "narrative_returns.csv": narrative_returns,
        "narrative_asset_heatmap.csv": narrative_asset_heatmap,
        "narrative_posts.csv": narrative_posts,
        "narrative_events.csv": narrative_events,
    }
    manifest_payload = dict(manifest)
    manifest_payload["files"] = {
        filename: {
            "rows": int(len(frame)) if isinstance(frame, pd.DataFrame) else 0,
            "columns": (
                list(frame.columns)
                if isinstance(frame, pd.DataFrame) and not frame.empty
                else EXPORT_TABLE_COLUMNS[filename]
            ),
        }
        for filename, frame in table_frames.items()
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(_json_safe(manifest_payload), indent=2, sort_keys=True))
        archive.writestr("summary.md", build_research_export_summary(manifest_payload))
        archive.writestr(
            "social_activity_chart.html",
            chart.to_html(full_html=True, include_plotlyjs=True),
        )
        for filename, frame in table_frames.items():
            archive.writestr(filename, _csv_bytes(frame, EXPORT_TABLE_COLUMNS[filename]))
    return buffer.getvalue()


def research_export_filename(date_start: pd.Timestamp, date_end: pd.Timestamp) -> str:
    start_label = pd.Timestamp(date_start).strftime("%Y%m%d")
    end_label = pd.Timestamp(date_end).strftime("%Y%m%d")
    return f"research-pack-{start_label}-{end_label}.zip"
