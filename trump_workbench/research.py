from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import EASTERN
from .enrichment import parse_semantic_asset_targets
from .features import map_posts_to_trade_sessions, preview_post_texts
from .utils import fmt_pct, fmt_score, truncate_text

NARRATIVE_COLOR_MAP = {
    "markets": "#60a5fa",
    "trade": "#f59e0b",
    "geopolitics": "#ef4444",
    "immigration": "#22c55e",
    "judiciary": "#a78bfa",
    "campaign": "#f472b6",
    "other": "#94a3b8",
}


def filter_posts(
    posts: pd.DataFrame,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    include_reshares: bool,
    platforms: list[str],
    keyword: str,
    tracked_only: bool,
    trump_authored_only: bool = False,
) -> pd.DataFrame:
    if posts.empty:
        return posts.copy()

    local_dates = posts["post_timestamp"].dt.tz_convert(EASTERN).dt.normalize().dt.tz_localize(None)
    mask = (local_dates >= date_start.normalize()) & (local_dates <= date_end.normalize())
    filtered = posts.loc[mask].copy()
    if platforms:
        filtered = filtered.loc[filtered["source_platform"].isin(platforms)].copy()
    if not include_reshares:
        filtered = filtered.loc[~filtered["is_reshare"]].copy()
    if tracked_only:
        tracked_mask = (
            filtered["is_active_tracked_account"]
            if "is_active_tracked_account" in filtered.columns
            else pd.Series(False, index=filtered.index)
        )
        filtered = filtered.loc[filtered["author_is_trump"] | tracked_mask].copy()
    if trump_authored_only:
        trump_mask = (
            filtered["author_is_trump"]
            if "author_is_trump" in filtered.columns
            else pd.Series(False, index=filtered.index)
        )
        filtered = filtered.loc[trump_mask.astype(bool)].copy()
    keyword = keyword.strip()
    if keyword:
        filtered = filtered.loc[
            filtered["cleaned_text"].str.contains(keyword, case=False, na=False, regex=False),
        ].copy()
    return filtered.reset_index(drop=True)


def narrative_urgency_band(value: Any) -> str:
    urgency = float(value or 0.0)
    if urgency < 0.34:
        return "low"
    if urgency < 0.67:
        return "medium"
    return "high"


def filter_narrative_rows(
    frame: pd.DataFrame,
    *,
    topic: str = "All",
    policy_bucket: str = "All",
    stance: str = "All",
    urgency_band: str = "All",
    narrative_asset: str = "All",
    platforms: list[str] | None = None,
    tracked_scope: str = "All posts",
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    filtered = frame.copy()
    for column in [
        "semantic_topic",
        "semantic_policy_bucket",
        "semantic_stance",
        "semantic_primary_asset",
        "semantic_asset_targets",
        "source_platform",
    ]:
        if column not in filtered.columns:
            filtered[column] = ""
    if "semantic_urgency" not in filtered.columns:
        filtered["semantic_urgency"] = 0.0
    if "author_is_trump" not in filtered.columns:
        filtered["author_is_trump"] = False
    if "is_active_tracked_account" not in filtered.columns:
        filtered["is_active_tracked_account"] = False

    if topic != "All":
        filtered = filtered.loc[filtered["semantic_topic"].astype(str) == topic].copy()
    if policy_bucket != "All":
        filtered = filtered.loc[filtered["semantic_policy_bucket"].astype(str) == policy_bucket].copy()
    if stance != "All":
        filtered = filtered.loc[filtered["semantic_stance"].astype(str) == stance].copy()
    if urgency_band != "All":
        filtered = filtered.loc[filtered["semantic_urgency"].map(narrative_urgency_band) == urgency_band].copy()
    if narrative_asset != "All":
        selected_asset = str(narrative_asset).upper()
        primary_mask = filtered["semantic_primary_asset"].astype(str).str.upper() == selected_asset
        target_mask = filtered["semantic_asset_targets"].map(
            lambda value: selected_asset in parse_semantic_asset_targets(value),
        )
        filtered = filtered.loc[primary_mask | target_mask].copy()
    if platforms:
        filtered = filtered.loc[filtered["source_platform"].isin(platforms)].copy()
    if tracked_scope == "Tracked accounts only":
        filtered = filtered.loc[filtered["is_active_tracked_account"].astype(bool)].copy()
    elif tracked_scope == "Trump + tracked accounts":
        filtered = filtered.loc[
            filtered["author_is_trump"].astype(bool) | filtered["is_active_tracked_account"].astype(bool)
        ].copy()
    return filtered.reset_index(drop=True)


def build_narrative_frequency_frame(mapped_posts: pd.DataFrame) -> pd.DataFrame:
    if mapped_posts.empty:
        return pd.DataFrame()
    frame = mapped_posts.copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["session_date"]).copy()
    summary = (
        frame.groupby(["session_date", "semantic_topic"], as_index=False)
        .agg(
            post_count=("semantic_topic", "size"),
            avg_market_relevance=("semantic_market_relevance", "mean"),
            avg_urgency=("semantic_urgency", "mean"),
        )
        .sort_values(["session_date", "semantic_topic"])
        .reset_index(drop=True)
    )
    return summary


def build_narrative_frequency_chart(frequency: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frequency.empty:
        fig.update_layout(
            title="Narrative frequency over time",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        return fig

    for topic, group in frequency.groupby("semantic_topic", sort=False):
        fig.add_trace(
            go.Bar(
                x=group["session_date"],
                y=group["post_count"],
                name=str(topic),
                marker={"color": NARRATIVE_COLOR_MAP.get(str(topic), "#94a3b8")},
                customdata=np.stack(
                    [
                        group["avg_market_relevance"].fillna(0.0),
                        group["avg_urgency"].fillna(0.0),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    "Topic: %{fullData.name}<br>"
                    "Posts: %{y}<br>"
                    "Avg relevance: %{customdata[0]:.2f}<br>"
                    "Avg urgency: %{customdata[1]:.2f}<extra></extra>"
                ),
            ),
        )
    fig.update_layout(
        title="Narrative frequency over time",
        xaxis_title="Trading session",
        yaxis_title="Posts",
        barmode="stack",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    return fig


def build_narrative_return_frame(
    mapped_posts: pd.DataFrame,
    market: pd.DataFrame,
    bucket_field: str = "semantic_topic",
) -> pd.DataFrame:
    if mapped_posts.empty or market.empty or bucket_field not in mapped_posts.columns:
        return pd.DataFrame()

    session_posts = mapped_posts.copy()
    session_posts["session_date"] = pd.to_datetime(session_posts["session_date"], errors="coerce").dt.normalize()
    session_posts = session_posts.dropna(subset=["session_date"]).copy()
    session_posts = session_posts.loc[session_posts[bucket_field].astype(str).str.len() > 0].copy()
    if session_posts.empty:
        return pd.DataFrame()

    events = market.copy()
    events["trade_date"] = pd.to_datetime(events["trade_date"], errors="coerce").dt.normalize()
    events["daily_return_pct"] = pd.to_numeric(events["close"], errors="coerce").pct_change()
    events["next_day_return_pct"] = events["daily_return_pct"].shift(-1)

    per_session = (
        session_posts.groupby(["session_date", bucket_field], as_index=False)
        .agg(
            post_count=(bucket_field, "size"),
            avg_market_relevance=("semantic_market_relevance", "mean"),
            avg_urgency=("semantic_urgency", "mean"),
        )
        .merge(events[["trade_date", "next_day_return_pct"]], left_on="session_date", right_on="trade_date", how="left")
    )
    if per_session.empty:
        return pd.DataFrame()
    summary = (
        per_session.groupby(bucket_field, as_index=False)
        .agg(
            avg_next_session_return=("next_day_return_pct", "mean"),
            median_next_session_return=("next_day_return_pct", "median"),
            session_count=("session_date", "nunique"),
            total_posts=("post_count", "sum"),
            avg_market_relevance=("avg_market_relevance", "mean"),
        )
        .sort_values("avg_next_session_return", ascending=False)
        .reset_index(drop=True)
    )
    summary["bucket_field"] = bucket_field
    return summary


def build_narrative_return_chart(returns: pd.DataFrame, bucket_field: str) -> go.Figure:
    fig = go.Figure()
    if returns.empty:
        fig.update_layout(
            title="Next-session return by narrative bucket",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        return fig

    labels = returns.iloc[:, 0].astype(str)
    colors = [NARRATIVE_COLOR_MAP.get(label, "#94a3b8") for label in labels]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=returns["avg_next_session_return"],
            marker={"color": colors},
            customdata=np.stack(
                [
                    returns["median_next_session_return"].fillna(0.0),
                    returns["session_count"].fillna(0).astype(int),
                    returns["total_posts"].fillna(0).astype(int),
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Avg next-session return: %{y:+.2%}<br>"
                "Median next-session return: %{customdata[0]:+.2%}<br>"
                "Sessions: %{customdata[1]}<br>"
                "Posts: %{customdata[2]}<extra></extra>"
            ),
        ),
    )
    fig.add_hline(y=0.0, line_dash="dot", line_width=1, line_color="rgba(148, 163, 184, 0.9)")
    fig.update_layout(
        title="Next-session return by narrative bucket",
        xaxis_title=bucket_field.replace("_", " ").title(),
        yaxis_title="Average next-session return",
        yaxis_tickformat=".0%",
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    return fig


def build_narrative_asset_heatmap_frame(asset_post_mappings: pd.DataFrame) -> pd.DataFrame:
    if asset_post_mappings.empty:
        return pd.DataFrame()
    mappings = asset_post_mappings.copy()
    mappings["asset_symbol"] = mappings["asset_symbol"].astype(str).str.upper()
    summary = (
        mappings.groupby(["semantic_topic", "asset_symbol"], as_index=False)
        .agg(
            post_count=("asset_symbol", "size"),
            avg_asset_relevance=("asset_relevance_score", "mean"),
            avg_market_relevance=("semantic_market_relevance", "mean"),
        )
        .sort_values(["semantic_topic", "asset_symbol"])
        .reset_index(drop=True)
    )
    return summary


def build_narrative_asset_heatmap_chart(heatmap: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if heatmap.empty:
        fig.update_layout(
            title="Asset-by-narrative heatmap",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        return fig

    pivot = heatmap.pivot(index="semantic_topic", columns="asset_symbol", values="post_count").fillna(0.0)
    relevance_lookup = heatmap.pivot(index="semantic_topic", columns="asset_symbol", values="avg_asset_relevance").reindex_like(pivot).fillna(0.0)
    customdata = np.dstack([relevance_lookup.to_numpy()])
    fig.add_trace(
        go.Heatmap(
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            z=pivot.to_numpy(),
            customdata=customdata,
            colorscale="Blues",
            hovertemplate=(
                "<b>%{y}</b> -> %{x}<br>"
                "Mapped posts: %{z}<br>"
                "Avg asset relevance: %{customdata[0]:.2f}<extra></extra>"
            ),
        ),
    )
    fig.update_layout(
        title="Asset-by-narrative heatmap",
        xaxis_title="Asset",
        yaxis_title="Narrative topic",
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    return fig


def make_narrative_post_table(mapped_posts: pd.DataFrame) -> pd.DataFrame:
    if mapped_posts.empty:
        return mapped_posts.copy()
    table = mapped_posts.copy()
    table["post_time_et"] = pd.to_datetime(table["post_timestamp"], errors="coerce").dt.tz_convert(EASTERN).dt.strftime("%Y-%m-%d %H:%M")
    table["session_date"] = pd.to_datetime(table["session_date"], errors="coerce").dt.date
    table["urgency_band"] = table["semantic_urgency"].map(narrative_urgency_band)
    table["preview"] = table["cleaned_text"].map(lambda value: truncate_text(str(value), max_chars=180))
    keep = [
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
        "preview",
        "post_url",
    ]
    return table.sort_values(
        ["semantic_market_relevance", "semantic_urgency", "post_time_et"],
        ascending=[False, False, False],
    )[keep].rename(columns={"preview": "post_text"})


def make_narrative_event_table(mapped_posts: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if mapped_posts.empty or market.empty:
        return pd.DataFrame()
    events = market.copy()
    events["trade_date"] = pd.to_datetime(events["trade_date"], errors="coerce").dt.normalize()
    events["daily_return_pct"] = pd.to_numeric(events["close"], errors="coerce").pct_change()
    events["next_day_return_pct"] = events["daily_return_pct"].shift(-1)
    grouped = mapped_posts.copy()
    grouped["session_date"] = pd.to_datetime(grouped["session_date"], errors="coerce").dt.normalize()
    summary = (
        grouped.groupby("session_date", as_index=False)
        .agg(
            post_count=("cleaned_text", "size"),
            avg_sentiment=("sentiment_score", "mean"),
            avg_market_relevance=("semantic_market_relevance", "mean"),
            avg_urgency=("semantic_urgency", "mean"),
            primary_topics=("semantic_topic", lambda values: ", ".join(pd.Series(values).value_counts().head(2).index.tolist())),
            primary_assets=("semantic_primary_asset", lambda values: ", ".join([value for value in pd.Series(values).astype(str).replace("", pd.NA).dropna().value_counts().head(2).index.tolist()])),
            sample_posts=("cleaned_text", lambda values: " | ".join([truncate_text(str(value), max_chars=90) for value in pd.Series(values).head(2)])),
        )
        .merge(events[["trade_date", "next_day_return_pct"]], left_on="session_date", right_on="trade_date", how="left")
        .drop(columns=["trade_date"])
        .rename(columns={"session_date": "trade_date", "next_day_return_pct": "next_session_return"})
        .sort_values("trade_date", ascending=False)
        .reset_index(drop=True)
    )
    summary["trade_date"] = pd.to_datetime(summary["trade_date"], errors="coerce").dt.date
    return summary


def _format_post_summary(row: pd.Series, max_chars: int = 140) -> str:
    ts = row["post_timestamp"].tz_convert(EASTERN)
    author = row["author_handle"] or row["author_display_name"] or row["source_platform"]
    preview = truncate_text(row["cleaned_text"], max_chars=max_chars)
    return f"{ts:%Y-%m-%d %H:%M} ET | @{author} | sentiment {fmt_score(row['sentiment_score'])} | {preview}"


def aggregate_research_sessions(mapped_posts: pd.DataFrame) -> pd.DataFrame:
    if mapped_posts.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for session_date, group in mapped_posts.groupby("session_date"):
        group = group.sort_values("post_timestamp").reset_index(drop=True)
        first_row = group.iloc[0]
        last_row = group.iloc[-1]
        strongest_positive = group.loc[group["sentiment_score"].idxmax()]
        strongest_negative = group.loc[group["sentiment_score"].idxmin()]
        rows.append(
            {
                "session_date": session_date,
                "post_count": int(len(group)),
                "truth_posts": int((group["source_platform"] == "Truth Social").sum()),
                "x_posts": int((group["source_platform"] == "X").sum()),
                "tracked_account_posts": int(group["is_active_tracked_account"].sum()) if "is_active_tracked_account" in group.columns else 0,
                "positive_posts": int((group["sentiment_label"] == "positive").sum()),
                "neutral_posts": int((group["sentiment_label"] == "neutral").sum()),
                "negative_posts": int((group["sentiment_label"] == "negative").sum()),
                "sample_posts": preview_post_texts(group, max_items=3),
                "sentiment_open": float(first_row["sentiment_score"]),
                "sentiment_high": float(group["sentiment_score"].max()),
                "sentiment_low": float(group["sentiment_score"].min()),
                "sentiment_close": float(last_row["sentiment_score"]),
                "sentiment_avg": float(group["sentiment_score"].mean()),
                "sentiment_range": float(group["sentiment_score"].max() - group["sentiment_score"].min()),
                "first_post_summary": _format_post_summary(first_row),
                "last_post_summary": _format_post_summary(last_row),
                "strongest_positive_summary": _format_post_summary(strongest_positive),
                "strongest_negative_summary": _format_post_summary(strongest_negative),
            },
        )
    return pd.DataFrame(rows).sort_values("session_date").reset_index(drop=True)


def build_event_frame(sp500_market: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    events = sp500_market.merge(sessions, left_on="trade_date", right_on="session_date", how="left")
    for column in ["post_count", "truth_posts", "x_posts", "tracked_account_posts", "positive_posts", "neutral_posts", "negative_posts"]:
        if column in events.columns:
            events[column] = events[column].fillna(0).astype(int)
    text_columns = [
        "sample_posts",
        "first_post_summary",
        "last_post_summary",
        "strongest_positive_summary",
        "strongest_negative_summary",
    ]
    for column in text_columns:
        if column in events.columns:
            events[column] = events[column].fillna("")
    events["daily_return_pct"] = events["close"].pct_change()
    events["next_day_return_pct"] = events["daily_return_pct"].shift(-1)
    events["has_posts"] = events["post_count"] > 0
    return events


def build_asset_comparison_frame(
    asset_market: pd.DataFrame,
    selected_symbol: str,
    benchmark_symbol: str | None = None,
    date_start: pd.Timestamp | None = None,
    date_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if asset_market.empty or not str(selected_symbol or "").strip():
        return pd.DataFrame()

    market = asset_market.copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce").dt.normalize()
    market = market.dropna(subset=["trade_date"]).copy()
    if date_start is not None:
        market = market.loc[market["trade_date"] >= pd.Timestamp(date_start).normalize()].copy()
    if date_end is not None:
        market = market.loc[market["trade_date"] <= pd.Timestamp(date_end).normalize()].copy()

    symbol = str(selected_symbol).upper()
    spy = market.loc[market["symbol"] == "SPY", ["trade_date", "close"]].rename(columns={"close": "spy_close"})
    asset = market.loc[market["symbol"] == symbol, ["trade_date", "close"]].rename(columns={"close": "asset_close"})
    comparison = spy.merge(asset, on="trade_date", how="inner").sort_values("trade_date").reset_index(drop=True)
    if comparison.empty:
        return comparison

    comparison["asset_symbol"] = symbol
    comparison["spy_daily_return"] = comparison["spy_close"].pct_change().fillna(0.0)
    comparison["asset_daily_return"] = comparison["asset_close"].pct_change().fillna(0.0)
    comparison["spy_normalized_return"] = comparison["spy_close"] / comparison["spy_close"].iloc[0] - 1.0
    comparison["asset_normalized_return"] = comparison["asset_close"] / comparison["asset_close"].iloc[0] - 1.0

    benchmark = str(benchmark_symbol or "").upper()
    if benchmark and benchmark not in {"SPY", symbol}:
        benchmark_frame = market.loc[market["symbol"] == benchmark, ["trade_date", "close"]].rename(columns={"close": "benchmark_close"})
        comparison = comparison.merge(benchmark_frame, on="trade_date", how="left")
        if "benchmark_close" in comparison.columns and comparison["benchmark_close"].notna().any():
            comparison["benchmark_symbol"] = benchmark
            comparison["benchmark_daily_return"] = comparison["benchmark_close"].pct_change().fillna(0.0)
            base_value = comparison["benchmark_close"].dropna().iloc[0]
            comparison["benchmark_normalized_return"] = comparison["benchmark_close"] / base_value - 1.0
    return comparison


def build_asset_comparison_chart(
    comparison: pd.DataFrame,
    selected_symbol: str,
    mode: str,
) -> go.Figure:
    symbol = str(selected_symbol).upper()
    fig = make_subplots(specs=[[{"secondary_y": mode == "price"}]])
    if comparison.empty:
        fig.update_layout(
            title=f"SPY vs. {symbol} comparison",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        return fig

    if mode == "normalized":
        fig.add_trace(
            go.Scatter(
                x=comparison["trade_date"],
                y=comparison["spy_normalized_return"],
                mode="lines",
                name="SPY normalized return",
                line={"color": "#8ecae6", "width": 3},
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>SPY: %{y:+.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison["trade_date"],
                y=comparison["asset_normalized_return"],
                mode="lines",
                name=f"{symbol} normalized return",
                line={"color": "#f59e0b", "width": 2.5},
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{symbol}: %{{y:+.2%}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        if "benchmark_normalized_return" in comparison.columns and comparison["benchmark_normalized_return"].notna().any():
            benchmark_symbol = str(comparison.get("benchmark_symbol", pd.Series([""])).iloc[0] or "Benchmark")
            fig.add_trace(
                go.Scatter(
                    x=comparison["trade_date"],
                    y=comparison["benchmark_normalized_return"],
                    mode="lines",
                    name=f"{benchmark_symbol} normalized return",
                    line={"color": "#c084fc", "width": 2, "dash": "dash"},
                    hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{benchmark_symbol}: %{{y:+.2%}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        fig.update_yaxes(title_text="Return since range start", tickformat=".0%", row=1, col=1)
        title = f"SPY vs. {symbol} normalized returns"
    else:
        fig.add_trace(
            go.Scatter(
                x=comparison["trade_date"],
                y=comparison["spy_close"],
                mode="lines",
                name="SPY close",
                line={"color": "#8ecae6", "width": 3},
                customdata=comparison["spy_daily_return"],
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>SPY close: %{y:,.2f}<br>Daily return: %{customdata:+.2%}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison["trade_date"],
                y=comparison["asset_close"],
                mode="lines",
                name=f"{symbol} close",
                line={"color": "#f59e0b", "width": 2.5},
                customdata=comparison["asset_daily_return"],
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{symbol} close: %{{y:,.2f}}<br>Daily return: %{{customdata:+.2%}}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
        if "benchmark_close" in comparison.columns and comparison["benchmark_close"].notna().any():
            benchmark_symbol = str(comparison.get("benchmark_symbol", pd.Series([""])).iloc[0] or "Benchmark")
            fig.add_trace(
                go.Scatter(
                    x=comparison["trade_date"],
                    y=comparison["benchmark_close"],
                    mode="lines",
                    name=f"{benchmark_symbol} close",
                    line={"color": "#c084fc", "width": 2, "dash": "dash"},
                    hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{benchmark_symbol} close: %{{y:,.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
                secondary_y=True,
            )
        fig.update_yaxes(title_text="SPY close", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text=f"{symbol} close", row=1, col=1, secondary_y=True)
        title = f"SPY vs. {symbol} price overlay"

    fig.update_xaxes(title_text="Trading session", dtick="M3", tickformat="%b %Y", row=1, col=1)
    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    return fig


def build_combined_chart(events: pd.DataFrame, scale_markers: bool) -> go.Figure:
    plot_events = events.sort_values("trade_date").reset_index(drop=True).copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.62, 0.38],
        specs=[[{"secondary_y": True}], [{}]],
        subplot_titles=(
            "S&P 500 close with post activity overlay",
            "Session sentiment range and average",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=plot_events["trade_date"],
            y=plot_events["close"],
            mode="lines",
            name="S&P 500 close",
            line={"color": "#8ecae6", "width": 3},
            customdata=plot_events["daily_return_pct"],
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Close: %{y:,.2f}<br>Daily return: %{customdata:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    activity_events = plot_events.copy()
    if not activity_events.empty:
        raw_activity = (
            activity_events["post_count"].clip(lower=0).astype(float)
            if scale_markers
            else activity_events["has_posts"].astype(int).astype(float)
        )
        activity_events["activity_signal"] = raw_activity.rolling(7, min_periods=1).mean()
        activity_customdata = np.stack(
            [
                activity_events["post_count"].astype(float).to_numpy(),
                activity_events["truth_posts"].astype(float).to_numpy(),
                activity_events["x_posts"].astype(float).to_numpy(),
                activity_events.get("tracked_account_posts", pd.Series(0, index=activity_events.index)).astype(float).to_numpy(),
                activity_events["sentiment_avg"].astype(float).to_numpy(),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=activity_events["trade_date"],
                y=activity_events["activity_signal"],
                mode="lines",
                name="Post activity",
                line={"color": "rgba(59, 130, 246, 0.8)", "width": 1.4},
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.14)",
                customdata=activity_customdata,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    "Posts: %{customdata[0]:.0f}<br>"
                    "Truth: %{customdata[1]:.0f} | X: %{customdata[2]:.0f}<br>"
                    "Tracked: %{customdata[3]:.0f}<br>"
                    "Avg sentiment: %{customdata[4]:+.3f}<br>"
                    "7-session activity: %{y:.1f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    sentiment_events = events.loc[events["has_posts"]].copy()
    if not sentiment_events.empty:
        sentiment_events["marker_color"] = np.where(
            sentiment_events["sentiment_avg"] >= 0.0,
            "#86efac",
            "#fda4af",
        )
        sentiment_customdata = np.stack(
            [
                sentiment_events["post_count"].astype(float).to_numpy(),
                sentiment_events["sentiment_open"].astype(float).to_numpy(),
                sentiment_events["sentiment_high"].astype(float).to_numpy(),
                sentiment_events["sentiment_low"].astype(float).to_numpy(),
                sentiment_events["sentiment_close"].astype(float).to_numpy(),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=sentiment_events["trade_date"],
                y=sentiment_events["sentiment_high"],
                mode="lines",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sentiment_events["trade_date"],
                y=sentiment_events["sentiment_low"],
                mode="lines",
                name="Sentiment range",
                line={"color": "rgba(148, 163, 184, 0.2)", "width": 1},
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.18)",
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sentiment_events["trade_date"],
                y=sentiment_events["sentiment_avg"],
                mode="lines+markers",
                name="Avg sentiment",
                line={"color": "rgba(226, 232, 240, 0.95)", "width": 1.9},
                marker={
                    "size": 6,
                    "color": sentiment_events["marker_color"],
                    "line": {"color": "rgba(15, 23, 42, 0.8)", "width": 0.6},
                },
                customdata=sentiment_customdata,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    "Posts: %{customdata[0]:.0f}<br>"
                    "Open: %{customdata[1]:+.3f}<br>"
                    "High: %{customdata[2]:+.3f}<br>"
                    "Low: %{customdata[3]:+.3f}<br>"
                    "Close: %{customdata[4]:+.3f}<br>"
                    "Average: %{y:+.3f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0.0, line_dash="dot", line_width=1, line_color="rgba(148, 163, 184, 0.9)", row=2, col=1)

    fig.update_yaxes(title_text="S&P 500 close", row=1, col=1)
    fig.update_yaxes(
        title_text="Posts mapped",
        showgrid=False,
        rangemode="tozero",
        secondary_y=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Sentiment score", range=[-1.05, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="Trading session", dtick="M3", tickformat="%b %Y", row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
    fig.update_layout(
        title="Research View: social activity vs. market baseline",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
            "xanchor": "left",
            "x": 0.0,
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 100, "b": 20},
    )
    return fig


def make_session_table(events: pd.DataFrame) -> pd.DataFrame:
    session_days = events.loc[events["post_count"] > 0].copy()
    if session_days.empty:
        return session_days
    session_days["trade_date"] = pd.to_datetime(session_days["trade_date"]).dt.date
    keep = [
        "trade_date",
        "post_count",
        "truth_posts",
        "x_posts",
        "tracked_account_posts",
        "positive_posts",
        "neutral_posts",
        "negative_posts",
        "close",
        "daily_return_pct",
        "next_day_return_pct",
        "sentiment_open",
        "sentiment_high",
        "sentiment_low",
        "sentiment_close",
        "sentiment_avg",
        "sentiment_range",
        "sample_posts",
    ]
    return session_days[keep].rename(
        columns={
            "close": "sp500_close",
            "daily_return_pct": "session_return",
            "next_day_return_pct": "next_session_return",
        },
    )


def make_asset_session_table(asset_session_features: pd.DataFrame, selected_symbol: str) -> pd.DataFrame:
    if asset_session_features.empty:
        return asset_session_features.copy()

    symbol = str(selected_symbol).upper()
    table = asset_session_features.copy()
    table = table.loc[table["asset_symbol"].astype(str).str.upper() == symbol].copy()
    if table.empty:
        return table
    table["signal_session_date"] = pd.to_datetime(table["signal_session_date"]).dt.date
    keep = [
        "signal_session_date",
        "post_count",
        "rule_matched_post_count",
        "semantic_matched_post_count",
        "primary_match_post_count",
        "asset_relevance_score_avg",
        "sentiment_avg",
        "target_next_session_return",
        "target_available",
    ]
    return table[keep].rename(
        columns={
            "signal_session_date": "trade_date",
            "target_next_session_return": "next_session_return",
        },
    )


def make_post_table(mapped_posts: pd.DataFrame) -> pd.DataFrame:
    if mapped_posts.empty:
        return mapped_posts
    table = mapped_posts.copy()
    table["post_time_et"] = table["post_timestamp"].dt.tz_convert(EASTERN).dt.strftime("%Y-%m-%d %H:%M")
    table["session_date"] = pd.to_datetime(table["session_date"]).dt.date
    table["preview"] = table["cleaned_text"].map(lambda x: x if len(x) <= 220 else x[:219].rstrip() + "…")
    keep = [
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
        "preview",
        "post_url",
    ]
    return table[keep].rename(columns={"preview": "post_text"})


def make_asset_mapping_table(asset_post_mappings: pd.DataFrame, selected_symbol: str) -> pd.DataFrame:
    if asset_post_mappings.empty:
        return asset_post_mappings.copy()

    symbol = str(selected_symbol).upper()
    table = asset_post_mappings.copy()
    table = table.loc[table["asset_symbol"].astype(str).str.upper() == symbol].copy()
    if table.empty:
        return table
    table["post_time_et"] = pd.to_datetime(table["post_timestamp"], errors="coerce").dt.tz_convert(EASTERN).dt.strftime("%Y-%m-%d %H:%M")
    if "reaction_anchor_ts" in table.columns:
        anchor_series = pd.to_datetime(table["reaction_anchor_ts"], errors="coerce")
        if getattr(anchor_series.dt, "tz", None) is None:
            anchor_series = anchor_series.dt.tz_localize(EASTERN, nonexistent="NaT", ambiguous="NaT")
        else:
            anchor_series = anchor_series.dt.tz_convert(EASTERN)
        table["reaction_anchor_et"] = anchor_series.dt.strftime("%Y-%m-%d %H:%M")
    else:
        table["reaction_anchor_et"] = table["post_time_et"]
    table["session_date"] = pd.to_datetime(table["session_date"], errors="coerce").dt.date
    table["preview"] = table["cleaned_text"].map(lambda x: truncate_text(str(x), max_chars=180))
    keep = [
        "asset_symbol",
        "session_date",
        "post_time_et",
        "reaction_anchor_et",
        "author_handle",
        "author_display_name",
        "asset_relevance_score",
        "rule_match_score",
        "semantic_match_score",
        "match_rank",
        "is_primary_asset",
        "match_reasons",
        "preview",
    ]
    return table.sort_values(["session_date", "match_rank", "post_time_et"], ascending=[False, True, True])[keep].rename(
        columns={"preview": "post_text"},
    )


def build_event_study_frame(
    asset_market: pd.DataFrame,
    asset_session_features: pd.DataFrame,
    selected_symbol: str,
    benchmark_symbol: str | None = None,
    pre_sessions: int = 3,
    post_sessions: int = 5,
) -> pd.DataFrame:
    if asset_market.empty or asset_session_features.empty:
        return pd.DataFrame()

    symbol = str(selected_symbol).upper()
    required_symbols = ["SPY", symbol]
    benchmark = str(benchmark_symbol or "").upper()
    if benchmark and benchmark not in required_symbols:
        required_symbols.append(benchmark)

    session_rows = asset_session_features.copy()
    session_rows["signal_session_date"] = pd.to_datetime(session_rows["signal_session_date"], errors="coerce").dt.normalize()
    event_dates = (
        session_rows.loc[
            (session_rows["asset_symbol"].astype(str).str.upper() == symbol)
            & (pd.to_numeric(session_rows["post_count"], errors="coerce").fillna(0) > 0),
            "signal_session_date",
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not event_dates:
        return pd.DataFrame()

    market = asset_market.copy()
    market["symbol"] = market["symbol"].astype(str).str.upper()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce").dt.normalize()
    market = market.loc[market["symbol"].isin(required_symbols)].dropna(subset=["trade_date", "close"]).copy()
    if market.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for current_symbol, group in market.groupby("symbol", sort=False):
        ordered = group.sort_values("trade_date").reset_index(drop=True)
        date_lookup = {pd.Timestamp(value): idx for idx, value in enumerate(ordered["trade_date"])}
        closes = ordered["close"].astype(float).tolist()
        dates = ordered["trade_date"].tolist()
        for event_date in event_dates:
            event_idx = date_lookup.get(pd.Timestamp(event_date))
            if event_idx is None:
                continue
            base_close = closes[event_idx]
            if pd.isna(base_close) or float(base_close) == 0.0:
                continue
            for offset in range(-int(pre_sessions), int(post_sessions) + 1):
                target_idx = event_idx + offset
                if target_idx < 0 or target_idx >= len(ordered):
                    continue
                target_close = closes[target_idx]
                rows.append(
                    {
                        "symbol": current_symbol,
                        "event_date": pd.Timestamp(event_date),
                        "relative_session": offset,
                        "trade_date": pd.Timestamp(dates[target_idx]),
                        "relative_return": float(target_close / base_close - 1.0),
                    },
                )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["symbol", "relative_session"], as_index=False)
        .agg(
            avg_relative_return=("relative_return", "mean"),
            median_relative_return=("relative_return", "median"),
            event_count=("event_date", "nunique"),
        )
        .sort_values(["symbol", "relative_session"])
        .reset_index(drop=True)
    )
    summary["selected_symbol"] = symbol
    return summary


def build_event_study_chart(event_study: pd.DataFrame, selected_symbol: str) -> go.Figure:
    symbol = str(selected_symbol).upper()
    fig = go.Figure()
    if event_study.empty:
        fig.update_layout(
            title=f"SPY vs. {symbol} event study",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        return fig

    color_map = {
        "SPY": "#8ecae6",
        symbol: "#f59e0b",
    }
    for current_symbol, group in event_study.groupby("symbol", sort=False):
        color = color_map.get(current_symbol, "#c084fc")
        dash = "solid" if current_symbol in color_map else "dash"
        fig.add_trace(
            go.Scatter(
                x=group["relative_session"],
                y=group["avg_relative_return"],
                mode="lines+markers",
                name=current_symbol,
                line={"color": color, "width": 2.5, "dash": dash},
                marker={"size": 7},
                customdata=group["event_count"],
                hovertemplate="<b>Session %{x:+d}</b><br>Avg return: %{y:+.2%}<br>Events: %{customdata}<extra></extra>",
            ),
        )
    fig.add_hline(y=0.0, line_dash="dot", line_width=1, line_color="rgba(148, 163, 184, 0.9)")
    fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="rgba(148, 163, 184, 0.9)")
    fig.update_layout(
        title=f"SPY vs. {symbol} event study",
        xaxis_title="Relative trading session",
        yaxis_title="Average return vs. event session close",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    return fig


def build_intraday_chart(intraday: pd.DataFrame, anchor_ts: pd.Timestamp, title: str) -> go.Figure:
    anchor_dt = pd.Timestamp(anchor_ts).to_pydatetime()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=intraday["timestamp"],
            y=intraday["close"],
            mode="lines",
            name="SPY close",
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Close: %{y:,.2f}<extra></extra>",
        ),
    )
    fig.add_shape(
        type="line",
        x0=anchor_dt,
        x1=anchor_dt,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"dash": "dash"},
    )
    fig.add_annotation(
        x=anchor_dt,
        y=1,
        xref="x",
        yref="paper",
        text="reaction anchor",
        showarrow=False,
        yanchor="bottom",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (ET)",
        yaxis_title="SPY price",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def get_intraday_window(
    intraday_month: pd.DataFrame,
    anchor_ts: pd.Timestamp,
    before_minutes: int,
    after_minutes: int,
) -> pd.DataFrame:
    start_ts = anchor_ts - pd.Timedelta(minutes=before_minutes)
    end_ts = anchor_ts + pd.Timedelta(minutes=after_minutes)
    return intraday_month.loc[
        (intraday_month["timestamp"] >= start_ts) & (intraday_month["timestamp"] <= end_ts)
    ].reset_index(drop=True)


def build_intraday_comparison_frame(
    intraday_frame: pd.DataFrame,
    selected_symbol: str,
    anchor_ts: pd.Timestamp,
    before_minutes: int,
    after_minutes: int,
    benchmark_symbol: str | None = None,
) -> pd.DataFrame:
    if intraday_frame.empty:
        return pd.DataFrame()

    symbol = str(selected_symbol).upper()
    benchmark = str(benchmark_symbol or "").upper()
    symbols = ["SPY", symbol]
    if benchmark and benchmark not in symbols:
        symbols.append(benchmark)

    frame = intraday_frame.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "close"]).copy()
    frame = frame.loc[frame["symbol"].isin(symbols)].copy()
    if frame.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for current_symbol, group in frame.groupby("symbol", sort=False):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        window = get_intraday_window(ordered, anchor_ts, before_minutes, after_minutes)
        if window.empty:
            continue
        reference = window.loc[window["timestamp"] <= anchor_ts]
        if reference.empty:
            reference = window.iloc[[0]]
        else:
            reference = reference.tail(1)
        base_close = float(reference["close"].iloc[0])
        if base_close == 0.0:
            continue
        window = window.copy()
        window["symbol"] = current_symbol
        window["anchor_close"] = base_close
        window["normalized_return"] = window["close"].astype(float) / base_close - 1.0
        window["minutes_from_anchor"] = (window["timestamp"] - anchor_ts) / pd.Timedelta(minutes=1)
        rows.append(window)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_intraday_comparison_chart(
    intraday_comparison: pd.DataFrame,
    selected_symbol: str,
    anchor_ts: pd.Timestamp,
) -> go.Figure:
    symbol = str(selected_symbol).upper()
    fig = go.Figure()
    if intraday_comparison.empty:
        fig.update_layout(
            title=f"SPY vs. {symbol} intraday reaction",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        return fig

    color_map = {
        "SPY": "#8ecae6",
        symbol: "#f59e0b",
    }
    for current_symbol, group in intraday_comparison.groupby("symbol", sort=False):
        color = color_map.get(current_symbol, "#c084fc")
        dash = "solid" if current_symbol in color_map else "dash"
        fig.add_trace(
            go.Scatter(
                x=group["timestamp"],
                y=group["normalized_return"],
                mode="lines",
                name=current_symbol,
                line={"color": color, "width": 2.4, "dash": dash},
                hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Return vs anchor: %{y:+.2%}<extra></extra>",
            ),
        )
    anchor_dt = pd.Timestamp(anchor_ts).to_pydatetime()
    fig.add_shape(
        type="line",
        x0=anchor_dt,
        x1=anchor_dt,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"dash": "dash"},
    )
    fig.add_annotation(
        x=anchor_dt,
        y=1,
        xref="x",
        yref="paper",
        text="reaction anchor",
        showarrow=False,
        yanchor="bottom",
    )
    fig.add_hline(y=0.0, line_dash="dot", line_width=1, line_color="rgba(148, 163, 184, 0.9)")
    fig.update_layout(
        title=f"SPY vs. {symbol} intraday reaction",
        xaxis_title="Time (ET)",
        yaxis_title="Return vs. anchor",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    return fig
