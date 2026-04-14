from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import EASTERN
from .features import map_posts_to_trade_sessions, preview_post_texts
from .utils import fmt_pct, fmt_score, truncate_text


def filter_posts(
    posts: pd.DataFrame,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    include_reshares: bool,
    platforms: list[str],
    keyword: str,
    tracked_only: bool,
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
    keyword = keyword.strip()
    if keyword:
        filtered = filtered.loc[
            filtered["cleaned_text"].str.contains(keyword, case=False, na=False, regex=False),
        ].copy()
    return filtered.reset_index(drop=True)


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
        activity_events["hover_text"] = activity_events.apply(
            lambda row: (
                f"<b>{pd.Timestamp(row['trade_date']):%Y-%m-%d}</b><br>"
                f"Posts mapped to session: {int(row['post_count'])}<br>"
                f"Truth Social: {int(row['truth_posts'])} | X: {int(row['x_posts'])}<br>"
                f"Tracked accounts: {int(row.get('tracked_account_posts', 0))}<br>"
                f"Avg sentiment: {fmt_score(row.get('sentiment_avg', np.nan))}<br>"
                f"Smoothed activity: {float(row['activity_signal']):.1f}<br>"
                f"Sample posts: {html.escape(str(row.get('sample_posts', '')))}"
            ),
            axis=1,
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
                text=activity_events["hover_text"],
                hovertemplate="%{text}<extra></extra>",
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
        sentiment_events["hover_text"] = sentiment_events.apply(
            lambda row: (
                f"<b>{pd.Timestamp(row['trade_date']):%Y-%m-%d}</b><br>"
                f"Posts mapped: {int(row['post_count'])}<br>"
                f"Open: {fmt_score(row['sentiment_open'])}<br>"
                f"High: {fmt_score(row['sentiment_high'])}<br>"
                f"Low: {fmt_score(row['sentiment_low'])}<br>"
                f"Close: {fmt_score(row['sentiment_close'])}<br>"
                f"Average: {fmt_score(row['sentiment_avg'])}<br>"
                f"First post: {html.escape(str(row.get('first_post_summary', '')))}<br>"
                f"Last post: {html.escape(str(row.get('last_post_summary', '')))}"
            ),
            axis=1,
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
                text=sentiment_events["hover_text"],
                hovertemplate="%{text}<extra></extra>",
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
                text=sentiment_events["hover_text"],
                hovertemplate="%{text}<extra></extra>",
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
    table["session_date"] = pd.to_datetime(table["session_date"], errors="coerce").dt.date
    table["preview"] = table["cleaned_text"].map(lambda x: truncate_text(str(x), max_chars=180))
    keep = [
        "asset_symbol",
        "session_date",
        "post_time_et",
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
