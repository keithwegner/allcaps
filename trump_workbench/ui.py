from __future__ import annotations

from collections import Counter
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from .backtesting import BacktestService
from .config import AppSettings, DEFAULT_ETF_SYMBOLS
from .contracts import MANUAL_OVERRIDE_COLUMNS, ModelRunConfig, RANKING_HISTORY_COLUMNS
from .discovery import DiscoveryService
from .enrichment import LLMEnrichmentService
from .explanations import build_account_attribution, build_post_attribution
from .experiments import ExperimentStore
from .features import FeatureService, latest_feature_preview, map_posts_to_trade_sessions
from .ingestion import IngestionService, TruthSocialArchiveAdapter, XCsvAdapter
from .market import MarketDataService, build_asset_universe, build_watchlist_frame, normalize_symbols
from .modeling import ModelService, classify_feature_family
from .research import (
    aggregate_research_sessions,
    build_combined_chart,
    build_event_frame,
    build_intraday_chart,
    filter_posts,
    get_intraday_window,
    make_post_table,
    make_session_table,
)
from .storage import DuckDBStore
from .utils import fmt_score


def _parse_grid(value: str, cast: type[float] | type[int]) -> tuple[float, ...] | tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if cast is int:
        return tuple(int(part) for part in parts)
    return tuple(float(part) for part in parts)


def _watchlist_symbols(store: DuckDBStore) -> list[str]:
    watchlist = store.read_frame("asset_watchlist")
    if watchlist.empty or "symbol" not in watchlist.columns:
        return []
    return normalize_symbols(watchlist["symbol"].astype(str).tolist())


def _save_watchlist(store: DuckDBStore, symbols: list[str] | tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    watchlist = build_watchlist_frame(symbols)
    asset_universe = build_asset_universe(watchlist["symbol"].tolist() if not watchlist.empty else [])
    store.save_frame("asset_watchlist", watchlist, metadata={"row_count": int(len(watchlist))})
    store.save_frame(
        "asset_universe",
        asset_universe,
        metadata={
            "row_count": int(len(asset_universe)),
            "default_etfs": list(DEFAULT_ETF_SYMBOLS),
        },
    )
    return watchlist, asset_universe


def _watchlist_text_value(store: DuckDBStore) -> str:
    symbols = _watchlist_symbols(store)
    return ", ".join(symbols)


def _build_adapters(
    settings: AppSettings,
    remote_url: str,
    uploaded_files: list[Any],
) -> list[Any]:
    adapters: list[Any] = [TruthSocialArchiveAdapter(settings=settings)]
    if settings.local_x_path.exists():
        adapters.append(
            XCsvAdapter(
                settings=settings,
                name="Local X posts",
                provenance=f"file:{settings.local_x_path}",
                raw_bytes=settings.local_x_path.read_bytes(),
            ),
        )
    if settings.local_mentions_path.exists():
        adapters.append(
            XCsvAdapter(
                settings=settings,
                name="Local influential mentions",
                provenance=f"file:{settings.local_mentions_path}",
                raw_bytes=settings.local_mentions_path.read_bytes(),
            ),
        )
    if remote_url.strip():
        adapters.append(XCsvAdapter.from_remote_url(settings, remote_url.strip(), "Remote X / mention CSV"))
    for uploaded_file in uploaded_files:
        adapters.append(
            XCsvAdapter(
                settings=settings,
                name=f"Uploaded CSV: {uploaded_file.name}",
                provenance=f"upload:{uploaded_file.name}",
                raw_bytes=uploaded_file.getvalue(),
            ),
        )
    return adapters


def _refresh_datasets(
    settings: AppSettings,
    store: DuckDBStore,
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
    remote_url: str,
    uploaded_files: list[Any],
    incremental: bool = False,
) -> dict[str, Any]:
    adapters = _build_adapters(settings, remote_url, uploaded_files)
    existing_posts = store.read_frame("normalized_posts")
    last_cursor = pd.to_datetime(existing_posts["post_timestamp"], errors="coerce").max() if not existing_posts.empty else None
    if incremental and last_cursor is not None:
        new_posts, source_manifest = ingestion_service.run_incremental_refresh(adapters, last_cursor=last_cursor)
        posts = pd.concat([existing_posts, new_posts], ignore_index=True) if not existing_posts.empty else new_posts
        posts = posts.drop_duplicates(subset=["post_id"], keep="last").sort_values("post_timestamp").reset_index(drop=True)
    else:
        posts, source_manifest = ingestion_service.run_refresh(adapters)
    store.save_frame("normalized_posts", posts, metadata={"row_count": int(len(posts))})
    store.save_frame("source_manifests", source_manifest, metadata={"row_count": int(len(source_manifest))})

    start = settings.term_start.strftime("%Y-%m-%d")
    end = pd.Timestamp.now(tz=settings.timezone).strftime("%Y-%m-%d")
    sp500 = market_service.load_sp500_daily(start, end)
    spy = market_service.load_spy_daily(start, end)
    store.save_frame("sp500_daily", sp500, metadata={"row_count": int(len(sp500))})
    store.save_frame("spy_daily", spy, metadata={"row_count": int(len(spy))})

    watchlist_symbols = _watchlist_symbols(store)
    watchlist, asset_universe = _save_watchlist(store, watchlist_symbols)
    asset_symbols = asset_universe["symbol"].astype(str).tolist() if not asset_universe.empty else list(DEFAULT_ETF_SYMBOLS)
    asset_daily, daily_manifest = market_service.load_assets_daily(asset_symbols, start, end)
    asset_intraday, intraday_manifest = market_service.load_assets_intraday(asset_symbols, interval="5m", lookback_days=30)
    asset_market_manifest = pd.concat([daily_manifest, intraday_manifest], ignore_index=True)
    store.save_frame("asset_daily", asset_daily, metadata={"row_count": int(len(asset_daily)), "symbols": asset_symbols})
    store.save_frame(
        "asset_intraday",
        asset_intraday,
        metadata={"row_count": int(len(asset_intraday)), "symbols": asset_symbols, "interval": "5m", "lookback_days": 30},
    )
    store.save_frame("asset_market_manifest", asset_market_manifest, metadata={"row_count": int(len(asset_market_manifest))})

    as_of = posts["post_timestamp"].max() if not posts.empty else pd.Timestamp.now(tz=settings.timezone)
    tracked_accounts, ranking_history = _rebuild_discovery_state(store, discovery_service, posts, as_of)
    return {
        "posts": posts,
        "source_manifest": source_manifest,
        "sp500": sp500,
        "spy": spy,
        "asset_watchlist": watchlist,
        "asset_universe": asset_universe,
        "asset_daily": asset_daily,
        "asset_intraday": asset_intraday,
        "asset_market_manifest": asset_market_manifest,
        "tracked_accounts": tracked_accounts,
    }


def _ensure_bootstrap(
    settings: AppSettings,
    store: DuckDBStore,
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
) -> None:
    if store.read_frame("asset_watchlist").empty and store.read_frame("asset_universe").empty:
        _save_watchlist(store, [])
    if (
        store.read_frame("normalized_posts").empty
        or store.read_frame("sp500_daily").empty
        or store.read_frame("spy_daily").empty
        or store.read_frame("asset_daily").empty
        or store.read_frame("asset_intraday").empty
    ):
        _refresh_datasets(
            settings=settings,
            store=store,
            ingestion_service=ingestion_service,
            market_service=market_service,
            discovery_service=discovery_service,
            remote_url=st.session_state.get("remote_x_url", ""),
            uploaded_files=[],
            incremental=False,
        )


def _rebuild_discovery_state(
    store: DuckDBStore,
    discovery_service: DiscoveryService,
    posts: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overrides = store.read_frame("manual_account_overrides")
    normalized_overrides = discovery_service.normalize_manual_overrides(overrides)
    if normalized_overrides.empty:
        normalized_overrides = pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)
    tracked_accounts, ranking_history = discovery_service.refresh_accounts(
        posts=posts,
        existing_accounts=pd.DataFrame(),
        as_of=as_of,
        manual_overrides=normalized_overrides,
    )
    store.save_frame("manual_account_overrides", normalized_overrides, metadata={"row_count": int(len(normalized_overrides))})
    store.save_frame("tracked_accounts", tracked_accounts, metadata={"row_count": int(len(tracked_accounts))})
    store.save_frame("account_rankings", ranking_history, metadata={"row_count": int(len(ranking_history))})
    return tracked_accounts, ranking_history


def _metric_row(metrics: dict[str, Any]) -> None:
    cols = st.columns(6)
    cols[0].metric("Total return", f"{metrics.get('total_return', 0.0):+.2%}")
    cols[1].metric("Sharpe", f"{metrics.get('sharpe', 0.0):.2f}")
    cols[2].metric("Sortino", f"{metrics.get('sortino', 0.0):.2f}")
    cols[3].metric("Max drawdown", f"{metrics.get('max_drawdown', 0.0):+.2%}")
    cols[4].metric("Win rate", f"{metrics.get('win_rate', 0.0):.1%}")
    cols[5].metric("Exposure", f"{metrics.get('exposure', 0.0):.1%}")


def _series_or_false(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column] if column in df.columns else pd.Series(False, index=df.index)


def _latest_ranking_snapshot(ranking_history: pd.DataFrame) -> pd.DataFrame:
    required = {"author_account_id", "ranked_at"}
    if ranking_history.empty or not required.issubset(ranking_history.columns):
        return pd.DataFrame(columns=RANKING_HISTORY_COLUMNS)

    snapshot = ranking_history.copy()
    snapshot["ranked_at"] = pd.to_datetime(snapshot["ranked_at"], errors="coerce")
    snapshot = snapshot.dropna(subset=["ranked_at"]).copy()
    if snapshot.empty:
        return pd.DataFrame(columns=RANKING_HISTORY_COLUMNS)
    return (
        snapshot.sort_values("ranked_at", ascending=False)
        .drop_duplicates("author_account_id")
        .reset_index(drop=True)
    )


def _render_equity_curve(trades: pd.DataFrame, title: str) -> None:
    if trades.empty:
        st.info("No trade data is available.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trades["next_session_date"],
            y=trades["equity_curve"],
            mode="lines",
            name="Strategy equity",
        ),
    )
    fig.update_layout(title=title, xaxis_title="Trade date", yaxis_title="Equity", margin={"l": 20, "r": 20, "t": 60, "b": 20})
    st.plotly_chart(fig, use_container_width=True)


def _render_equity_curve_comparison(curves_by_run: dict[str, pd.DataFrame], title: str) -> None:
    fig = go.Figure()
    for run_id, curve in curves_by_run.items():
        if curve.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=curve["next_session_date"],
                y=curve["equity_curve"],
                mode="lines",
                name=run_id,
            ),
        )
    fig.update_layout(title=title, xaxis_title="Trade date", yaxis_title="Equity", margin={"l": 20, "r": 20, "t": 60, "b": 20})
    st.plotly_chart(fig, use_container_width=True)


def render_datasets_view(
    settings: AppSettings,
    store: DuckDBStore,
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
) -> None:
    st.subheader("Datasets")
    st.caption("Refresh historical sources, store normalized datasets, and inspect the local DuckDB + Parquet catalog.")

    st.markdown("**Asset universe**")
    st.caption("Manage a small stock watchlist here. The ETF starter set is always included: `SPY`, `QQQ`, `XLK`, `XLF`, `XLE`, `SMH`.")
    default_text = _watchlist_text_value(store)
    watchlist_input = st.text_area(
        "Watchlist symbols",
        value=st.session_state.get("watchlist_symbols", default_text),
        help="Enter a comma-separated list of stock tickers such as `AAPL, TSLA, NVDA`.",
    )
    st.session_state["watchlist_symbols"] = watchlist_input
    watchlist_cols = st.columns(2)
    if watchlist_cols[0].button("Save watchlist", use_container_width=True):
        symbols = normalize_symbols([part.strip() for part in watchlist_input.replace("\n", ",").split(",") if part.strip()])
        watchlist, asset_universe = _save_watchlist(store, symbols)
        st.session_state["watchlist_symbols"] = ", ".join(watchlist["symbol"].tolist()) if not watchlist.empty else ""
        st.success(f"Saved {len(watchlist):,} watchlist symbols and {len(asset_universe):,} total tracked assets.")
        st.rerun()
    if watchlist_cols[1].button("Reset watchlist", use_container_width=True):
        _save_watchlist(store, [])
        st.session_state["watchlist_symbols"] = ""
        st.success("Reset the manual watchlist to the ETF starter set only.")
        st.rerun()

    remote_url = st.text_input(
        "Remote X / mentions CSV URL",
        key="remote_x_url",
        value=st.session_state.get("remote_x_url", os.getenv("TRUMP_X_CSV_URL", "")),
    )
    uploaded_files = st.file_uploader(
        "Upload X or mention CSVs",
        type=["csv"],
        accept_multiple_files=True,
    )
    action_cols = st.columns(2)
    if action_cols[0].button("Refresh full datasets", use_container_width=True):
        with st.spinner("Refreshing source data, market data, and discovery state..."):
            summary = _refresh_datasets(
                settings=settings,
                store=store,
                ingestion_service=ingestion_service,
                market_service=market_service,
                discovery_service=discovery_service,
                remote_url=remote_url,
                uploaded_files=uploaded_files or [],
            )
        st.success(
            f"Refreshed {len(summary['posts']):,} normalized posts, {len(summary['asset_daily']):,} daily market rows, "
            f"and {len(summary['tracked_accounts']):,} tracked account versions.",
        )
    if action_cols[1].button("Incremental refresh", use_container_width=True):
        with st.spinner("Polling sources for new data..."):
            summary = _refresh_datasets(
                settings=settings,
                store=store,
                ingestion_service=ingestion_service,
                market_service=market_service,
                discovery_service=discovery_service,
                remote_url=remote_url,
                uploaded_files=uploaded_files or [],
                incremental=True,
            )
        st.success(
            f"Incremental refresh complete. Total posts stored: {len(summary['posts']):,}. "
            f"Asset intraday rows stored: {len(summary['asset_intraday']):,}.",
        )

    registry = store.dataset_registry()
    if not registry.empty:
        registry = registry.copy()
        registry["metadata_json"] = registry["metadata_json"].astype(str)
        st.markdown("**Dataset Registry**")
        st.dataframe(registry, use_container_width=True, hide_index=True)

    source_manifest = store.read_frame("source_manifests")
    if not source_manifest.empty:
        st.markdown("**Source Manifest**")
        st.dataframe(source_manifest, use_container_width=True, hide_index=True)

    asset_universe = store.read_frame("asset_universe")
    if not asset_universe.empty:
        st.markdown("**Tracked asset universe**")
        st.dataframe(asset_universe, use_container_width=True, hide_index=True)

    asset_market_manifest = store.read_frame("asset_market_manifest")
    if not asset_market_manifest.empty:
        st.markdown("**Asset market manifest**")
        st.dataframe(asset_market_manifest, use_container_width=True, hide_index=True)

    asset_daily = store.read_frame("asset_daily")
    if not asset_daily.empty:
        st.markdown("**Daily asset market sample**")
        st.dataframe(asset_daily.tail(20), use_container_width=True, hide_index=True)

    asset_intraday = store.read_frame("asset_intraday")
    if not asset_intraday.empty:
        st.markdown("**Intraday asset market sample**")
        st.dataframe(asset_intraday.tail(20), use_container_width=True, hide_index=True)

    posts = store.read_frame("normalized_posts")
    if not posts.empty:
        st.markdown("**Normalized Post Sample**")
        sample_columns = [
            "source_platform",
            "author_handle",
            "post_timestamp",
            "mentions_trump",
            "sentiment_score",
            "cleaned_text",
        ]
        st.dataframe(posts[sample_columns].tail(20), use_container_width=True, hide_index=True)
    st.caption(f"Templates: `{settings.x_template_path.name}` and `{settings.mention_template_path.name}`")


def render_discovery_view(
    settings: AppSettings,
    store: DuckDBStore,
    discovery_service: DiscoveryService,
) -> None:
    st.subheader("Discovery")
    st.caption("Dynamic discovery ranks X accounts mentioning Trump, auto-includes the strongest candidates, and lets you pin or suppress accounts manually.")
    posts = store.read_frame("normalized_posts")
    tracked_accounts = store.read_frame("tracked_accounts")
    ranking_history = store.read_frame("account_rankings")
    overrides = discovery_service.normalize_manual_overrides(store.read_frame("manual_account_overrides"))

    if posts.empty:
        st.info("Refresh datasets first so the workbench has posts to analyze.")
        return

    candidate_posts = posts.loc[
        (posts["source_platform"] == "X")
        & posts["mentions_trump"]
        & (~posts["author_is_trump"])
    ]
    ranking_columns_present = {"author_account_id", "ranked_at"}.issubset(ranking_history.columns)
    if (not candidate_posts.empty or not overrides.empty) and not ranking_columns_present:
        tracked_accounts, ranking_history = _rebuild_discovery_state(
            store,
            discovery_service,
            posts,
            posts["post_timestamp"].max(),
        )
        overrides = discovery_service.normalize_manual_overrides(store.read_frame("manual_account_overrides"))

    active_accounts = discovery_service.current_active_accounts(
        tracked_accounts,
        as_of=posts["post_timestamp"].max(),
    )
    if not active_accounts.empty:
        st.markdown("**Active tracked universe**")
        keep = [
            "handle",
            "display_name",
            "status",
            "discovery_score",
            "mention_count",
            "engagement_mean",
            "effective_from",
            "provenance",
        ]
        st.dataframe(
            active_accounts[keep].sort_values("discovery_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    latest_rankings = _latest_ranking_snapshot(ranking_history)
    if not latest_rankings.empty:
        st.markdown("**Latest ranking snapshot**")
        keep = [
            "author_handle",
            "author_display_name",
            "discovery_score",
            "mention_count",
            "engagement_mean",
            "active_days",
            "ranked_at",
            "selected_status",
            "suppressed_by_override",
            "pinned_by_override",
        ]
        st.dataframe(
            latest_rankings[keep].sort_values("discovery_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Manual overrides**")
        override_cols = st.columns([3, 2, 2, 2])
        option_rows = latest_rankings[["author_account_id", "author_handle", "author_display_name"]].drop_duplicates().copy()
        option_rows["label"] = option_rows.apply(
            lambda row: f"@{row['author_handle'] or '[unknown]'} | {row['author_display_name'] or row['author_account_id']}",
            axis=1,
        )
        selected_label = override_cols[0].selectbox("Account", options=option_rows["label"].tolist())
        selected_row = option_rows.loc[option_rows["label"] == selected_label].iloc[0]
        action = override_cols[1].selectbox("Action", options=["pin", "suppress"])
        effective_from = override_cols[2].date_input(
            "Effective from",
            value=posts["post_timestamp"].max().tz_convert(settings.timezone).date(),
            min_value=settings.term_start.date(),
            max_value=posts["post_timestamp"].max().tz_convert(settings.timezone).date(),
            key="override_effective_from",
        )
        note = override_cols[3].text_input("Note", value="", key="override_note")
        no_end_date = st.checkbox("No end date", value=True)
        effective_to = None
        if not no_end_date:
            effective_to = st.date_input(
                "Effective to",
                value=posts["post_timestamp"].max().tz_convert(settings.timezone).date(),
                min_value=effective_from,
                key="override_effective_to",
            )
        if st.button("Save override", use_container_width=True):
            updated = discovery_service.add_manual_override(
                overrides=overrides,
                account_id=str(selected_row["author_account_id"]),
                handle=str(selected_row["author_handle"]),
                display_name=str(selected_row["author_display_name"]),
                action=action,
                effective_from=pd.Timestamp(effective_from),
                effective_to=pd.Timestamp(effective_to) if effective_to is not None else None,
                note=note,
            )
            store.save_frame("manual_account_overrides", updated, metadata={"row_count": int(len(updated))})
            _rebuild_discovery_state(store, discovery_service, posts, posts["post_timestamp"].max())
            st.success(f"Saved {action} override for @{selected_row['author_handle']}.")
            st.rerun()
    else:
        st.info("No discovery ranking snapshot is available yet. Add or refresh X mention data to populate this view.")

    if not overrides.empty:
        st.markdown("**Override history**")
        st.dataframe(overrides, use_container_width=True, hide_index=True)
        remove_options = overrides.apply(
            lambda row: f"{row['override_id']} | {row['action']} | @{row['handle'] or row['account_id']} | from {pd.Timestamp(row['effective_from']).date()}",
            axis=1,
        ).tolist()
        remove_choice = st.selectbox("Remove override", options=remove_options)
        remove_id = remove_choice.split(" | ", 1)[0]
        if st.button("Delete selected override", use_container_width=True):
            updated = discovery_service.remove_manual_override(overrides, remove_id)
            store.save_frame("manual_account_overrides", updated, metadata={"row_count": int(len(updated))})
            _rebuild_discovery_state(store, discovery_service, posts, posts["post_timestamp"].max())
            st.success("Removed override.")
            st.rerun()

    if not latest_rankings.empty:
        chart_data = latest_rankings.head(15).sort_values("discovery_score")
        fig = go.Figure(
            go.Bar(
                x=chart_data["discovery_score"],
                y=chart_data["author_handle"].replace("", "[unknown]"),
                orientation="h",
                marker_color="#14532d",
            ),
        )
        fig.update_layout(
            title="Top discovered accounts",
            xaxis_title="Discovery score",
            yaxis_title="Account",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        st.plotly_chart(fig, use_container_width=True)


def render_research_view(
    settings: AppSettings,
    store: DuckDBStore,
    market_service: MarketDataService,
    feature_service: FeatureService,
) -> None:
    st.subheader("Research View")
    st.caption("Preserved descriptive market overlay plus cleaned-up post/session tables and intraday drill-down.")
    posts = store.read_frame("normalized_posts")
    sp500 = store.read_frame("sp500_daily")
    tracked_accounts = store.read_frame("tracked_accounts")
    if posts.empty or sp500.empty:
        st.info("Refresh datasets first so the research view has source data.")
        return

    today_et = pd.Timestamp.now(tz=settings.timezone).normalize().tz_localize(None)
    controls = st.columns(5)
    date_range = controls[0].date_input(
        "Date range",
        value=(settings.term_start.date(), today_et.date()),
        min_value=settings.term_start.date(),
        max_value=today_et.date(),
    )
    selected_platforms = controls[1].multiselect(
        "Platforms",
        options=["Truth Social", "X"],
        default=["Truth Social", "X"],
    )
    include_reshares = controls[2].checkbox("Include reshares", value=False)
    tracked_only = controls[3].checkbox("Tracked accounts only", value=False)
    scale_markers = controls[4].checkbox("Scale markers", value=True)
    keyword = st.text_input("Keyword filter", value="")

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])
    else:
        start_date = settings.term_start
        end_date = today_et

    market = sp500.loc[(sp500["trade_date"] >= start_date) & (sp500["trade_date"] <= end_date)].copy()
    filtered_posts = filter_posts(
        posts=posts,
        date_start=start_date,
        date_end=end_date,
        include_reshares=include_reshares,
        platforms=selected_platforms,
        keyword=keyword,
        tracked_only=tracked_only,
    )
    mapped = map_posts_to_trade_sessions(filtered_posts, market[["trade_date"]])
    mapped = feature_service._flag_tracked_posts(mapped, tracked_accounts)
    sessions = aggregate_research_sessions(mapped)
    events = build_event_frame(market, sessions)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Sessions with posts", f"{int((events['post_count'] > 0).sum()):,}")
    metric_cols[1].metric("Posts in view", f"{int(events['post_count'].sum()):,}")
    metric_cols[2].metric("Truth posts", f"{int(mapped.loc[mapped['author_is_trump']].shape[0]):,}")
    metric_cols[3].metric("Tracked X posts", f"{int(mapped.loc[_series_or_false(mapped, 'is_active_tracked_account')].shape[0]):,}")
    metric_cols[4].metric("Mean sentiment", fmt_score(mapped["sentiment_score"].mean() if not mapped.empty else 0.0))
    metric_cols[5].metric("S&P 500 change", f"{((events['close'].iloc[-1] / events['close'].iloc[0]) - 1.0):+.2%}" if len(events) > 1 else "n/a")

    st.plotly_chart(build_combined_chart(events, scale_markers=scale_markers), use_container_width=True)

    session_table = make_session_table(events)
    if not session_table.empty:
        st.markdown("**Session table**")
        st.dataframe(
            session_table.sort_values(["post_count", "trade_date"], ascending=[False, False]),
            use_container_width=True,
            hide_index=True,
        )

    post_table = make_post_table(mapped)
    if not post_table.empty:
        with st.expander("Underlying posts", expanded=False):
            st.dataframe(post_table, use_container_width=True, hide_index=True)

    with st.expander("Intraday SPY drill-down", expanded=False):
        alpha_vantage_key = st.text_input(
            "Alpha Vantage API key",
            value=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
            type="password",
        )
        intraday_interval = st.selectbox("Intraday bar size", options=["1min", "5min", "15min"], index=1, key="research_intraday_interval")
        before_minutes = st.slider("Minutes before anchor", min_value=30, max_value=390, value=120, step=30)
        after_minutes = st.slider("Minutes after anchor", min_value=30, max_value=780, value=390, step=30)
        if mapped.empty:
            st.info("No mapped posts are available in the current view.")
        elif not alpha_vantage_key.strip():
            st.info("Add an Alpha Vantage API key to enable intraday drill-down.")
        else:
            session_options = pd.to_datetime(mapped["session_date"]).dt.date.drop_duplicates().sort_values(ascending=False).tolist()
            selected_session_date = st.selectbox("Trading session", options=session_options)
            session_posts = mapped.loc[pd.to_datetime(mapped["session_date"]).dt.date == selected_session_date].sort_values("post_timestamp")
            post_labels = [
                f"{row['post_timestamp'].tz_convert(settings.timezone):%Y-%m-%d %H:%M} ET | @{row['author_handle'] or row['author_display_name']} | {truncate(row['cleaned_text'])}"
                for _, row in session_posts.iterrows()
            ]
            selected_label = st.selectbox("Post", options=post_labels)
            selected_idx = post_labels.index(selected_label)
            selected_post = session_posts.iloc[selected_idx]
            month_str = selected_post["reaction_anchor_ts"].strftime("%Y-%m")
            intraday_month = market_service.load_spy_intraday_month(month_str, intraday_interval, alpha_vantage_key.strip())
            window = get_intraday_window(intraday_month, selected_post["reaction_anchor_ts"], before_minutes, after_minutes)
            if window.empty:
                st.warning("No intraday data was returned for this window.")
            else:
                st.plotly_chart(
                    build_intraday_chart(
                        window,
                        selected_post["reaction_anchor_ts"],
                        f"SPY intraday around {selected_post['reaction_anchor_ts']:%Y-%m-%d %H:%M ET}",
                    ),
                    use_container_width=True,
                )


def truncate(text: str, max_chars: int = 90) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _normalize_session_date(value: Any) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.tz_localize(None)
    return pd.Timestamp(timestamp).normalize()


def _filter_for_session(frame: pd.DataFrame, session_date: pd.Timestamp | None, column: str = "signal_session_date") -> pd.DataFrame:
    if frame.empty or session_date is None or column not in frame.columns:
        return frame.head(0).copy()
    values = pd.to_datetime(frame[column], errors="coerce")
    if getattr(values.dt, "tz", None) is not None:
        values = values.dt.tz_localize(None)
    mask = values.dt.normalize() == session_date
    return frame.loc[mask].copy()


def _prediction_option_label(row: pd.Series) -> str:
    session_date = _normalize_session_date(row.get("signal_session_date"))
    date_label = f"{session_date:%Y-%m-%d}" if session_date is not None else "unknown session"
    score = float(row.get("expected_return_score", 0.0) or 0.0)
    post_count = int(row.get("post_count", 0) or 0)
    return f"{date_label} | score {score:+.3%} | posts {post_count}"


def _render_signal_explanation_panel(
    prediction_row: pd.Series,
    feature_contributions: pd.DataFrame,
    post_attribution: pd.DataFrame,
    account_attribution: pd.DataFrame,
    heading: str = "Why This Signal?",
) -> None:
    session_date = _normalize_session_date(prediction_row.get("signal_session_date"))
    expected_return = float(prediction_row.get("expected_return_score", 0.0) or 0.0)
    confidence = float(prediction_row.get("prediction_confidence", 0.0) or 0.0)
    actual_return = prediction_row.get("target_next_session_return")
    session_contributions = _filter_for_session(feature_contributions, session_date)
    session_posts = _filter_for_session(post_attribution, session_date)
    session_accounts = _filter_for_session(account_attribution, session_date)

    st.markdown(f"**{heading}**")
    st.caption(
        "Feature contributions are exact for the linear model. Post and account attribution is heuristic, "
        "using session sentiment, engagement, and tracked-account weighting.",
    )

    dominant_driver = "n/a"
    if not session_contributions.empty:
        family_mix = (
            session_contributions.groupby("feature_family", as_index=False)["abs_contribution"]
            .sum()
            .sort_values("abs_contribution", ascending=False)
            .reset_index(drop=True)
        )
        if not family_mix.empty:
            lead = family_mix.iloc[0]
            share = float(lead["abs_contribution"] / family_mix["abs_contribution"].sum()) if family_mix["abs_contribution"].sum() else 0.0
            dominant_driver = f"{lead['feature_family']} ({share:.0%})"
    else:
        family_mix = pd.DataFrame(columns=["feature_family", "abs_contribution"])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Signal session", f"{session_date:%Y-%m-%d}" if session_date is not None else "n/a")
    metric_cols[1].metric("Expected return", f"{expected_return:+.3%}")
    metric_cols[2].metric("Confidence", f"{confidence:.2f}")
    metric_cols[3].metric(
        "Dominant driver",
        dominant_driver,
        delta=f"Actual {float(actual_return):+.3%}" if pd.notna(actual_return) else None,
    )

    if not family_mix.empty:
        mix_display = family_mix.copy()
        mix_display["share"] = (
            mix_display["abs_contribution"] / mix_display["abs_contribution"].sum()
        ).fillna(0.0)
        st.dataframe(
            mix_display.rename(
                columns={
                    "feature_family": "Driver family",
                    "abs_contribution": "Absolute contribution",
                    "share": "Share of model move",
                },
            ),
            use_container_width=True,
            hide_index=True,
        )

    driver_cols = st.columns(2)
    positive = (
        session_contributions.loc[session_contributions["contribution"] > 0.0, ["feature_name", "raw_value", "coefficient", "contribution", "contribution_share"]]
        .sort_values("contribution", ascending=False)
        .head(6)
        .rename(
            columns={
                "feature_name": "Feature",
                "raw_value": "Value",
                "coefficient": "Coef",
                "contribution": "Contribution",
                "contribution_share": "Share",
            },
        )
    )
    negative = (
        session_contributions.loc[session_contributions["contribution"] < 0.0, ["feature_name", "raw_value", "coefficient", "contribution", "contribution_share"]]
        .sort_values("contribution", ascending=True)
        .head(6)
        .rename(
            columns={
                "feature_name": "Feature",
                "raw_value": "Value",
                "coefficient": "Coef",
                "contribution": "Contribution",
                "contribution_share": "Share",
            },
        )
    )
    with driver_cols[0]:
        st.markdown("**Positive feature drivers**")
        if positive.empty:
            st.info("No positive feature contributions for this session.")
        else:
            st.dataframe(positive, use_container_width=True, hide_index=True)
    with driver_cols[1]:
        st.markdown("**Negative feature drivers**")
        if negative.empty:
            st.info("No negative feature contributions for this session.")
        else:
            st.dataframe(negative, use_container_width=True, hide_index=True)

    direction = 1.0 if expected_return >= 0.0 else -1.0
    if not session_accounts.empty:
        account_view = session_accounts.copy()
        account_view["role"] = np.where(
            account_view["author_is_trump"],
            "Trump",
            np.where(account_view["is_active_tracked_account"], "Tracked", "Mention account"),
        )
        account_view["alignment_score"] = account_view["net_post_signal"] * direction
        account_view = account_view.sort_values(["alignment_score", "abs_net_post_signal"], ascending=[False, False]).head(8)
        st.markdown("**Most aligned accounts**")
        st.dataframe(
            account_view[
                [
                    "author_handle",
                    "role",
                    "post_count",
                    "avg_sentiment",
                    "net_post_signal",
                    "total_engagement",
                ]
            ].rename(
                columns={
                    "author_handle": "Handle",
                    "post_count": "Posts",
                    "avg_sentiment": "Avg sentiment",
                    "net_post_signal": "Signal score",
                    "total_engagement": "Engagement",
                },
            ),
            use_container_width=True,
            hide_index=True,
        )

    if not session_posts.empty:
        post_view = session_posts.copy()
        post_view["role"] = np.where(
            post_view["author_is_trump"],
            "Trump",
            np.where(post_view["is_active_tracked_account"], "Tracked", "Mention account"),
        )
        post_view["alignment_score"] = post_view["post_signal_score"] * direction
        post_view = post_view.sort_values(["alignment_score", "abs_post_signal_score"], ascending=[False, False]).head(8)
        st.markdown("**Most aligned posts**")
        st.dataframe(
            post_view[
                [
                    "post_timestamp",
                    "author_handle",
                    "role",
                    "sentiment_score",
                    "engagement_score",
                    "post_signal_score",
                    "post_preview",
                ]
            ].rename(
                columns={
                    "post_timestamp": "Timestamp",
                    "author_handle": "Handle",
                    "sentiment_score": "Sentiment",
                    "engagement_score": "Engagement",
                    "post_signal_score": "Signal score",
                    "post_preview": "Post",
                },
            ),
            use_container_width=True,
            hide_index=True,
        )


def _format_compare_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, float):
        return round(value, 6)
    return value


def _comparison_settings(run_bundle: dict[str, Any]) -> dict[str, Any]:
    config = run_bundle.get("config", {}) or {}
    selected = run_bundle.get("selected_params", {}) or {}
    artifact = run_bundle.get("model_artifact")
    return {
        "feature_version": config.get("feature_version", "v1"),
        "llm_enabled": bool(config.get("llm_enabled", getattr(artifact, "metadata", {}).get("llm_enabled", False))),
        "train_window": config.get("train_window"),
        "validation_window": config.get("validation_window"),
        "test_window": config.get("test_window"),
        "step_size": config.get("step_size"),
        "ridge_alpha": config.get("ridge_alpha"),
        "transaction_cost_bps": config.get("transaction_cost_bps"),
        "threshold_grid": tuple(config.get("threshold_grid", [])),
        "minimum_signal_grid": tuple(config.get("minimum_signal_grid", [])),
        "account_weight_grid": tuple(config.get("account_weight_grid", [])),
        "deploy_threshold": selected.get("threshold"),
        "deploy_min_post_count": selected.get("min_post_count"),
        "deploy_account_weight": selected.get("account_weight"),
    }


def _build_metric_comparison_table(base_run_id: str, run_bundles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if base_run_id not in run_bundles:
        return pd.DataFrame()
    base_metrics = run_bundles[base_run_id].get("metrics", {}) or {}
    rows: list[dict[str, Any]] = []
    for run_id, bundle in run_bundles.items():
        metrics = bundle.get("metrics", {}) or {}
        artifact = bundle.get("model_artifact")
        rows.append(
            {
                "run_id": run_id,
                "run_name": bundle.get("run", {}).get("run_name", run_id),
                "total_return": metrics.get("total_return", 0.0),
                "sharpe": metrics.get("sharpe", 0.0),
                "sortino": metrics.get("sortino", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "robust_score": metrics.get("robust_score", 0.0),
                "trade_count": metrics.get("trade_count", 0.0),
                "feature_count": len(getattr(artifact, "feature_names", [])),
                "delta_total_return_vs_base": metrics.get("total_return", 0.0) - base_metrics.get("total_return", 0.0),
                "delta_sharpe_vs_base": metrics.get("sharpe", 0.0) - base_metrics.get("sharpe", 0.0),
                "delta_robust_score_vs_base": metrics.get("robust_score", 0.0) - base_metrics.get("robust_score", 0.0),
            },
        )
    return pd.DataFrame(rows).sort_values("delta_robust_score_vs_base", ascending=False).reset_index(drop=True)


def _build_setting_diff_table(base_run_id: str, run_bundles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if base_run_id not in run_bundles:
        return pd.DataFrame()
    settings_by_run = {run_id: _comparison_settings(bundle) for run_id, bundle in run_bundles.items()}
    base_settings = settings_by_run[base_run_id]
    diff_rows: list[dict[str, Any]] = []
    for key in sorted({setting for settings in settings_by_run.values() for setting in settings}):
        row = {"setting": key}
        base_value = base_settings.get(key)
        is_different = False
        for run_id, settings in settings_by_run.items():
            value = settings.get(key)
            row[run_id] = _format_compare_value(value)
            if value != base_value:
                is_different = True
        if is_different:
            diff_rows.append(row)
    return pd.DataFrame(diff_rows)


def _build_feature_diff_table(base_run_id: str, run_bundles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if base_run_id not in run_bundles:
        return pd.DataFrame()
    base_features = set(run_bundles[base_run_id]["model_artifact"].feature_names)
    rows: list[dict[str, Any]] = []
    for run_id, bundle in run_bundles.items():
        feature_names = list(bundle["model_artifact"].feature_names)
        families = Counter(classify_feature_family(feature_name) for feature_name in feature_names)
        features = set(feature_names)
        unique_vs_base = sorted(features - base_features)
        omitted_vs_base = sorted(base_features - features)
        rows.append(
            {
                "run_id": run_id,
                "run_name": bundle.get("run", {}).get("run_name", run_id),
                "feature_count": len(feature_names),
                "semantic_features": families.get("semantic", 0),
                "policy_features": families.get("policy", 0),
                "market_context_features": families.get("market_context", 0),
                "social_sentiment_features": families.get("social_sentiment", 0),
                "activity_features": families.get("activity", 0),
                "account_structure_features": families.get("account_structure", 0),
                "unique_vs_base_count": len(unique_vs_base),
                "omitted_vs_base_count": len(omitted_vs_base),
                "unique_vs_base": ", ".join(unique_vs_base[:6]),
                "omitted_vs_base": ", ".join(omitted_vs_base[:6]),
            },
        )
    return pd.DataFrame(rows).sort_values(["unique_vs_base_count", "feature_count"], ascending=[False, False]).reset_index(drop=True)


def _build_benchmark_delta_table(base_run_id: str, run_bundles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if base_run_id not in run_bundles:
        return pd.DataFrame()
    base_benchmarks = run_bundles[base_run_id].get("benchmarks", pd.DataFrame())
    if base_benchmarks.empty or "benchmark_name" not in base_benchmarks.columns:
        return pd.DataFrame()
    base_lookup = base_benchmarks.set_index("benchmark_name")
    base_strategy = float(base_lookup.loc["strategy", "total_return"]) if "strategy" in base_lookup.index else np.nan
    rows: list[dict[str, Any]] = []
    for run_id, bundle in run_bundles.items():
        benchmarks = bundle.get("benchmarks", pd.DataFrame())
        if benchmarks.empty or "benchmark_name" not in benchmarks.columns:
            continue
        lookup = benchmarks.set_index("benchmark_name")
        strategy_return = float(lookup.loc["strategy", "total_return"]) if "strategy" in lookup.index else np.nan
        for benchmark_name in sorted(set(base_lookup.index).union(set(lookup.index))):
            current_total = float(lookup.loc[benchmark_name, "total_return"]) if benchmark_name in lookup.index else np.nan
            base_total = float(base_lookup.loc[benchmark_name, "total_return"]) if benchmark_name in base_lookup.index else np.nan
            current_edge = strategy_return - current_total if pd.notna(strategy_return) and pd.notna(current_total) else np.nan
            base_edge = base_strategy - base_total if pd.notna(base_strategy) and pd.notna(base_total) else np.nan
            rows.append(
                {
                    "run_id": run_id,
                    "benchmark_name": benchmark_name,
                    "total_return": current_total,
                    "delta_total_return_vs_base": current_total - base_total if pd.notna(current_total) and pd.notna(base_total) else np.nan,
                    "edge_vs_strategy": current_edge,
                    "delta_edge_vs_base": current_edge - base_edge if pd.notna(current_edge) and pd.notna(base_edge) else np.nan,
                },
            )
    return pd.DataFrame(rows).sort_values(["benchmark_name", "delta_edge_vs_base"], ascending=[True, False]).reset_index(drop=True)


def _summarize_run_changes(base_run_id: str, run_bundles: dict[str, dict[str, Any]]) -> list[str]:
    if base_run_id not in run_bundles:
        return []
    base_metrics = run_bundles[base_run_id].get("metrics", {}) or {}
    base_settings = _comparison_settings(run_bundles[base_run_id])
    base_features = set(run_bundles[base_run_id]["model_artifact"].feature_names)
    notes: list[str] = []
    for run_id, bundle in run_bundles.items():
        if run_id == base_run_id:
            continue
        metrics = bundle.get("metrics", {}) or {}
        settings = _comparison_settings(bundle)
        features = set(bundle["model_artifact"].feature_names)
        parts: list[str] = [
            f"robust score {metrics.get('robust_score', 0.0) - base_metrics.get('robust_score', 0.0):+.3f}",
            f"total return {metrics.get('total_return', 0.0) - base_metrics.get('total_return', 0.0):+.2%}",
        ]
        if settings.get("llm_enabled") != base_settings.get("llm_enabled"):
            parts.append(f"LLM {'on' if settings.get('llm_enabled') else 'off'} vs {'on' if base_settings.get('llm_enabled') else 'off'}")
        if settings.get("deploy_threshold") != base_settings.get("deploy_threshold"):
            parts.append(f"threshold {base_settings.get('deploy_threshold')} -> {settings.get('deploy_threshold')}")
        if settings.get("deploy_min_post_count") != base_settings.get("deploy_min_post_count"):
            parts.append(f"min posts {base_settings.get('deploy_min_post_count')} -> {settings.get('deploy_min_post_count')}")
        if settings.get("deploy_account_weight") != base_settings.get("deploy_account_weight"):
            parts.append(f"account weight {base_settings.get('deploy_account_weight')} -> {settings.get('deploy_account_weight')}")
        unique_vs_base = sorted(features - base_features)
        omitted_vs_base = sorted(base_features - features)
        if unique_vs_base:
            parts.append(f"{len(unique_vs_base)} added features ({', '.join(unique_vs_base[:3])})")
        if omitted_vs_base:
            parts.append(f"{len(omitted_vs_base)} removed features ({', '.join(omitted_vs_base[:3])})")
        notes.append(f"`{run_id}`: " + "; ".join(parts) + ".")
    return notes


def _bundle_to_run_config(run_bundle: dict[str, Any]) -> ModelRunConfig:
    config = run_bundle.get("config", {}) or {}
    run_meta = run_bundle.get("run", {}) or {}
    return ModelRunConfig(
        run_name=str(config.get("run_name") or run_meta.get("run_name") or "historical-replay"),
        feature_version=str(config.get("feature_version", "v1")),
        llm_enabled=bool(config.get("llm_enabled", False)),
        train_window=int(config.get("train_window", 90) or 90),
        validation_window=int(config.get("validation_window", 30) or 30),
        test_window=int(config.get("test_window", 30) or 30),
        step_size=int(config.get("step_size", 30) or 30),
        threshold_grid=tuple(float(value) for value in config.get("threshold_grid", [0.0, 0.001, 0.0025, 0.005])),
        minimum_signal_grid=tuple(int(value) for value in config.get("minimum_signal_grid", [1, 2, 3])),
        account_weight_grid=tuple(float(value) for value in config.get("account_weight_grid", [0.5, 1.0, 1.5])),
        ridge_alpha=float(config.get("ridge_alpha", 1.0) or 1.0),
        transaction_cost_bps=float(config.get("transaction_cost_bps", 2.0) or 2.0),
        start_date=config.get("start_date"),
        end_date=config.get("end_date"),
        notes=str(config.get("notes", "")),
    )


def _eligible_replay_sessions(feature_rows: pd.DataFrame, min_history_rows: int = 20) -> pd.DataFrame:
    if feature_rows.empty:
        return pd.DataFrame()
    eligible = feature_rows.sort_values("signal_session_date").reset_index(drop=True).copy()
    eligible["history_rows_available"] = eligible["target_available"].fillna(False).astype(int).cumsum().shift(fill_value=0)
    eligible = eligible.loc[eligible["history_rows_available"] >= min_history_rows].copy()
    return eligible.reset_index(drop=True)


def _replay_option_label(row: pd.Series) -> str:
    session_date = pd.Timestamp(row["signal_session_date"])
    return (
        f"{session_date:%Y-%m-%d} | posts {int(row.get('post_count', 0))} | "
        f"prior train rows {int(row.get('history_rows_available', 0))}"
    )


def _build_replay_comparison_frame(replay_row: pd.Series, full_history_row: pd.Series | None) -> pd.DataFrame:
    rows = [
        {"metric": "Replay score", "value": float(replay_row.get("expected_return_score", 0.0))},
        {"metric": "Replay confidence", "value": float(replay_row.get("prediction_confidence", 0.0))},
        {"metric": "Replay threshold", "value": float(replay_row.get("deployment_threshold", 0.0))},
        {"metric": "Replay min post count", "value": int(replay_row.get("deployment_min_post_count", 1))},
        {"metric": "Training rows used", "value": int(replay_row.get("training_rows_used", 0))},
    ]
    actual = replay_row.get("target_next_session_return")
    if pd.notna(actual):
        rows.append({"metric": "Actual next-session return", "value": float(actual)})
    if full_history_row is not None:
        full_score = float(full_history_row.get("expected_return_score", 0.0) or 0.0)
        replay_score = float(replay_row.get("expected_return_score", 0.0) or 0.0)
        rows.extend(
            [
                {"metric": "Full-history score", "value": full_score},
                {"metric": "Replay vs full-history drift", "value": replay_score - full_score},
                {"metric": "Full-history confidence", "value": float(full_history_row.get("prediction_confidence", 0.0) or 0.0)},
            ],
        )
    return pd.DataFrame(rows)


def render_models_view(
    settings: AppSettings,
    store: DuckDBStore,
    feature_service: FeatureService,
    model_service: ModelService,
    backtest_service: BacktestService,
    experiment_store: ExperimentStore,
) -> None:
    st.subheader("Models & Backtests")
    st.caption("Build the session dataset, train a next-session SPY expected-return model, compare saved runs, and inspect benchmark plus leakage diagnostics.")
    posts = store.read_frame("normalized_posts")
    spy = store.read_frame("spy_daily")
    tracked_accounts = store.read_frame("tracked_accounts")
    if posts.empty or spy.empty:
        st.info("Refresh datasets first so the modeling pipeline has normalized posts and SPY market data.")
        return

    control_cols = st.columns(4)
    run_name = control_cols[0].text_input("Run name", value="baseline-research-run")
    llm_enabled = control_cols[1].checkbox("Enable semantic enrichment", value=False)
    train_window = control_cols[2].number_input("Train window", min_value=20, max_value=252, value=90, step=5)
    validation_window = control_cols[3].number_input("Validation window", min_value=10, max_value=126, value=30, step=5)
    control_cols2 = st.columns(4)
    test_window = control_cols2[0].number_input("Test window", min_value=10, max_value=126, value=30, step=5)
    step_size = control_cols2[1].number_input("Step size", min_value=5, max_value=126, value=30, step=5)
    transaction_cost_bps = control_cols2[2].number_input("Round-trip cost (bps per side)", min_value=0.0, max_value=25.0, value=2.0, step=0.5)
    ridge_alpha = control_cols2[3].number_input("Ridge alpha", min_value=0.0, max_value=25.0, value=1.0, step=0.5)
    threshold_text = st.text_input("Threshold grid", value="0,0.001,0.0025,0.005")
    min_posts_text = st.text_input("Minimum signal post-count grid", value="1,2,3")
    account_weight_text = st.text_input("Tracked-account weight grid", value="0.5,1.0,1.5")

    if st.button("Build dataset and run walk-forward backtest", use_container_width=True):
        with st.spinner("Building session features and running walk-forward optimization..."):
            prepared_posts = feature_service.prepare_session_posts(
                posts=posts,
                market_calendar=spy,
                tracked_accounts=tracked_accounts,
                llm_enabled=llm_enabled,
            )
            feature_rows = feature_service.build_session_dataset(
                posts=posts,
                spy_market=spy,
                tracked_accounts=tracked_accounts,
                feature_version="v1",
                llm_enabled=llm_enabled,
                prepared_posts=prepared_posts,
            )
            store.save_frame("session_features_latest", feature_rows, metadata={"llm_enabled": llm_enabled, "row_count": int(len(feature_rows))})
            config = ModelRunConfig(
                run_name=run_name,
                llm_enabled=llm_enabled,
                train_window=int(train_window),
                validation_window=int(validation_window),
                test_window=int(test_window),
                step_size=int(step_size),
                threshold_grid=_parse_grid(threshold_text, float),
                minimum_signal_grid=_parse_grid(min_posts_text, int),
                account_weight_grid=_parse_grid(account_weight_text, float),
                ridge_alpha=float(ridge_alpha),
                transaction_cost_bps=float(transaction_cost_bps),
            )
            run, artifacts = backtest_service.run_walk_forward(config, feature_rows)
            post_attribution = build_post_attribution(prepared_posts)
            account_attribution = build_account_attribution(post_attribution)
            predicted_sessions = {
                session_date
                for session_date in pd.to_datetime(artifacts["predictions"]["signal_session_date"], errors="coerce").dropna().dt.normalize().tolist()
            }
            post_attribution = post_attribution.loc[
                pd.to_datetime(post_attribution["signal_session_date"], errors="coerce").dt.normalize().isin(predicted_sessions)
            ].reset_index(drop=True)
            account_attribution = account_attribution.loc[
                pd.to_datetime(account_attribution["signal_session_date"], errors="coerce").dt.normalize().isin(predicted_sessions)
            ].reset_index(drop=True)
            experiment_store.save_run(
                run=run,
                config=artifacts["config"],
                trades=artifacts["trades"],
                predictions=artifacts["predictions"],
                windows=artifacts["windows"],
                importance=artifacts["importance"],
                model_artifact=artifacts["model_artifact"],
                feature_contributions=artifacts["feature_contributions"],
                post_attribution=post_attribution,
                account_attribution=account_attribution,
                benchmarks=artifacts["benchmarks"],
                diagnostics=artifacts["diagnostics"],
                benchmark_curves=artifacts["benchmark_curves"],
                leakage_audit=artifacts["leakage_audit"],
            )
        st.success(f"Saved run `{run.run_id}`.")

    latest_features = store.read_frame("session_features_latest")
    if not latest_features.empty:
        preview = latest_feature_preview(latest_features)
        st.markdown("**Latest feature snapshot**")
        st.json(preview)

    runs = experiment_store.list_runs()
    if runs.empty:
        st.info("No experiment runs have been saved yet.")
        return

    leaderboard = runs.copy()
    leaderboard["total_return"] = leaderboard["metrics_json"].map(lambda metrics: metrics.get("total_return", 0.0))
    leaderboard["sharpe"] = leaderboard["metrics_json"].map(lambda metrics: metrics.get("sharpe", 0.0))
    leaderboard["sortino"] = leaderboard["metrics_json"].map(lambda metrics: metrics.get("sortino", 0.0))
    leaderboard["max_drawdown"] = leaderboard["metrics_json"].map(lambda metrics: metrics.get("max_drawdown", 0.0))
    leaderboard["robust_score"] = leaderboard["metrics_json"].map(lambda metrics: metrics.get("robust_score", 0.0))
    keep = ["run_id", "run_name", "created_at", "total_return", "sharpe", "sortino", "max_drawdown", "robust_score"]
    st.markdown("**Run leaderboard**")
    st.dataframe(
        leaderboard[keep].sort_values("robust_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    selected_run_id = st.selectbox("Saved runs", options=runs["run_id"].tolist())
    loaded = experiment_store.load_run(selected_run_id)
    if loaded is None:
        st.warning("The selected run could not be loaded.")
        return

    feature_contributions = loaded["feature_contributions"]
    if feature_contributions.empty:
        feature_contributions = model_service.explain_predictions(loaded["model_artifact"], loaded["predictions"])
    post_attribution = loaded["post_attribution"]
    account_attribution = loaded["account_attribution"]
    if post_attribution.empty or account_attribution.empty:
        fallback_posts = feature_service.prepare_session_posts(
            posts=posts,
            market_calendar=spy,
            tracked_accounts=tracked_accounts,
            llm_enabled=bool(loaded["model_artifact"].metadata.get("llm_enabled", False)),
        )
        post_attribution = build_post_attribution(fallback_posts)
        account_attribution = build_account_attribution(post_attribution)

    _metric_row(loaded["metrics"])
    _render_equity_curve(loaded["trades"], title="Walk-forward out-of-sample equity curve")

    compare_ids = st.multiselect(
        "Compare runs",
        options=runs["run_id"].tolist(),
        default=[selected_run_id],
    )
    if len(compare_ids) > 1:
        curves: dict[str, pd.DataFrame] = {}
        compare_bundles: dict[str, dict[str, Any]] = {}
        for run_id in compare_ids:
            run_bundle = experiment_store.load_run(run_id)
            if run_bundle is None:
                continue
            compare_bundles[run_id] = run_bundle
            curves[run_id] = run_bundle["trades"][["next_session_date", "equity_curve"]].copy()
        if compare_bundles:
            base_run_id = st.selectbox(
                "Base run for diffs",
                options=list(compare_bundles.keys()),
                index=list(compare_bundles.keys()).index(selected_run_id) if selected_run_id in compare_bundles else 0,
            )
            st.markdown("**Run Comparison Workspace**")
            tabs = st.tabs(["Scorecard", "Settings", "Features", "Benchmarks", "What changed"])
            metric_table = _build_metric_comparison_table(base_run_id, compare_bundles)
            setting_diff = _build_setting_diff_table(base_run_id, compare_bundles)
            feature_diff = _build_feature_diff_table(base_run_id, compare_bundles)
            benchmark_diff = _build_benchmark_delta_table(base_run_id, compare_bundles)
            with tabs[0]:
                st.dataframe(metric_table, use_container_width=True, hide_index=True)
                _render_equity_curve_comparison(curves, title="Selected run equity curves")
            with tabs[1]:
                if setting_diff.empty:
                    st.info("These runs are using the same saved configuration and deployment parameters.")
                else:
                    st.dataframe(setting_diff, use_container_width=True, hide_index=True)
            with tabs[2]:
                st.dataframe(feature_diff, use_container_width=True, hide_index=True)
            with tabs[3]:
                if benchmark_diff.empty:
                    st.info("Benchmark deltas are not available for one or more selected runs.")
                else:
                    st.dataframe(benchmark_diff, use_container_width=True, hide_index=True)
            with tabs[4]:
                notes = _summarize_run_changes(base_run_id, compare_bundles)
                if not notes:
                    st.info("Choose at least one non-base run to summarize what changed.")
                else:
                    for note in notes:
                        st.markdown(f"- {note}")

    if not loaded["benchmarks"].empty:
        st.markdown("**Benchmark suite**")
        st.dataframe(loaded["benchmarks"], use_container_width=True, hide_index=True)
    if not loaded["benchmark_curves"].empty:
        curve_fig = go.Figure()
        curves = loaded["benchmark_curves"].copy()
        for column in curves.columns:
            if column == "next_session_date":
                continue
            curve_fig.add_trace(go.Scatter(x=curves["next_session_date"], y=curves[column], mode="lines", name=column))
        curve_fig.update_layout(title="Strategy vs. benchmark equity curves", xaxis_title="Trade date", yaxis_title="Equity")
        st.plotly_chart(curve_fig, use_container_width=True)

    if loaded["leakage_audit"]:
        st.markdown("**Leakage audit**")
        st.json(loaded["leakage_audit"])

    st.markdown("**Window summary**")
    st.dataframe(loaded["windows"], use_container_width=True, hide_index=True)
    st.markdown("**Feature importance**")
    st.dataframe(loaded["importance"].head(25), use_container_width=True, hide_index=True)
    prediction_rows = loaded["predictions"].sort_values("signal_session_date", ascending=False).reset_index(drop=True)
    if not prediction_rows.empty:
        labels = prediction_rows.apply(_prediction_option_label, axis=1).tolist()
        selected_label = st.selectbox(
            "Explain session",
            options=labels,
            index=0,
            key=f"explain-session-{selected_run_id}",
        )
        selected_prediction = prediction_rows.iloc[labels.index(selected_label)]
        _render_signal_explanation_panel(
            prediction_row=selected_prediction,
            feature_contributions=feature_contributions,
            post_attribution=post_attribution,
            account_attribution=account_attribution,
        )
    if not loaded["diagnostics"].empty:
        diagnostics = loaded["diagnostics"].copy()
        diag_fig = go.Figure()
        diag_fig.add_trace(
            go.Scatter(
                x=diagnostics["expected_return_score"],
                y=diagnostics["actual_next_session_return"],
                mode="markers",
                marker={"size": 8, "opacity": 0.65},
                name="Predictions",
            ),
        )
        diag_fig.update_layout(
            title="Prediction diagnostics: expected vs actual next-session return",
            xaxis_title="Expected return score",
            yaxis_title="Actual next-session return",
        )
        st.plotly_chart(diag_fig, use_container_width=True)
        st.markdown("**Largest prediction misses**")
        st.dataframe(
            diagnostics.sort_values("absolute_error", ascending=False).head(20),
            use_container_width=True,
            hide_index=True,
        )
    with st.expander("Trades", expanded=False):
        st.dataframe(loaded["trades"], use_container_width=True, hide_index=True)


def render_live_monitor(
    settings: AppSettings,
    store: DuckDBStore,
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
    feature_service: FeatureService,
    model_service: ModelService,
    experiment_store: ExperimentStore,
) -> None:
    st.subheader("Live Monitor")
    st.caption("Polling-style refresh for new posts plus the latest next-session SPY score from the newest saved model.")
    remote_url = st.text_input(
        "Remote X / mentions CSV URL for polling",
        key="live_remote_x_url",
        value=st.session_state.get("remote_x_url", ""),
    )
    poll_enabled = st.checkbox("Enable browser polling refresh", value=False)
    poll_seconds = st.slider("Polling interval (seconds)", min_value=30, max_value=900, value=settings.default_poll_seconds, step=30)
    if poll_enabled:
        components.html(
            f"""
            <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {int(poll_seconds * 1000)});
            </script>
            """,
            height=0,
        )

    if st.button("Poll sources now", use_container_width=True):
        with st.spinner("Refreshing sources and market data..."):
            _refresh_datasets(
                settings=settings,
                store=store,
                ingestion_service=ingestion_service,
                market_service=market_service,
                discovery_service=discovery_service,
                remote_url=remote_url,
                uploaded_files=[],
                incremental=True,
            )
        st.success("Polling refresh complete.")

    model_bundle = experiment_store.load_latest_model_artifact()
    if model_bundle is None:
        st.info("Run a model in the Models & Backtests page to enable live predictions.")
        return

    artifact, deployment_params = model_bundle
    posts = store.read_frame("normalized_posts")
    spy = store.read_frame("spy_daily")
    tracked_accounts = store.read_frame("tracked_accounts")
    prepared_posts = feature_service.prepare_session_posts(
        posts=posts,
        market_calendar=spy,
        tracked_accounts=tracked_accounts,
        llm_enabled=bool(artifact.metadata.get("llm_enabled", False)),
    )
    feature_rows = feature_service.build_session_dataset(
        posts=posts,
        spy_market=spy,
        tracked_accounts=tracked_accounts,
        feature_version="v1",
        llm_enabled=bool(artifact.metadata.get("llm_enabled", False)),
        prepared_posts=prepared_posts,
    )
    predictions = model_service.predict(artifact, feature_rows)
    feature_contributions = model_service.explain_predictions(artifact, predictions)
    post_attribution = build_post_attribution(prepared_posts)
    account_attribution = build_account_attribution(post_attribution)
    latest = predictions.sort_values("signal_session_date").iloc[-1]
    threshold = float(deployment_params.get("threshold", 0.0))
    min_post_count = int(deployment_params.get("min_post_count", 1))
    stance = "LONG SPY NEXT SESSION" if latest["expected_return_score"] > threshold and latest["post_count"] >= min_post_count else "FLAT"
    snapshot = pd.DataFrame(
        [
            {
                "signal_session_date": latest["signal_session_date"],
                "next_session_date": latest["next_session_date"],
                "expected_return_score": latest["expected_return_score"],
                "feature_version": latest["feature_version"],
                "model_version": latest["model_version"],
                "confidence": latest["prediction_confidence"],
                "generated_at": pd.Timestamp.utcnow(),
                "stance": stance,
            },
        ],
    )
    experiment_store.save_prediction_snapshots(snapshot)

    cols = st.columns(4)
    cols[0].metric("Signal session", f"{pd.Timestamp(latest['signal_session_date']):%Y-%m-%d}")
    cols[1].metric("Expected next-session return", f"{latest['expected_return_score']:+.3%}")
    cols[2].metric("Confidence", f"{latest['prediction_confidence']:.2f}")
    cols[3].metric("Suggested stance", stance)

    st.json(
        {
            "threshold": threshold,
            "minimum_post_count": min_post_count,
            "latest_feature_preview": latest_feature_preview(feature_rows),
        },
    )

    history = store.read_frame("prediction_snapshots")
    if not history.empty:
        history = history.sort_values("generated_at")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=history["generated_at"],
                y=history["expected_return_score"],
                mode="lines+markers",
                name="Expected return score",
            ),
        )
        fig.update_layout(title="Prediction snapshot history", xaxis_title="Generated at", yaxis_title="Expected next-session return")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(history.tail(25), use_container_width=True, hide_index=True)

    _render_signal_explanation_panel(
        prediction_row=latest,
        feature_contributions=feature_contributions,
        post_attribution=post_attribution,
        account_attribution=account_attribution,
        heading="Why This Live Signal?",
    )


def render_historical_replay_view(
    settings: AppSettings,
    store: DuckDBStore,
    feature_service: FeatureService,
    model_service: ModelService,
    backtest_service: BacktestService,
    experiment_store: ExperimentStore,
) -> None:
    st.subheader("Historical Replay")
    st.caption(
        "Rebuild a signal as of a historical session using only earlier rows for training, then compare it to the saved full-history view.",
    )
    runs = experiment_store.list_runs()
    if runs.empty:
        st.info("Save at least one experiment run first so Historical Replay has a template configuration to follow.")
        return

    posts = store.read_frame("normalized_posts")
    spy = store.read_frame("spy_daily")
    tracked_accounts = store.read_frame("tracked_accounts")
    if posts.empty or spy.empty:
        st.info("Refresh datasets first so replay can rebuild historical features.")
        return

    selected_run_id = st.selectbox("Replay template run", options=runs["run_id"].tolist(), key="replay-run-id")
    loaded = experiment_store.load_run(selected_run_id)
    if loaded is None:
        st.warning("The selected replay template could not be loaded.")
        return

    run_config = _bundle_to_run_config(loaded)
    prepared_posts = feature_service.prepare_session_posts(
        posts=posts,
        market_calendar=spy,
        tracked_accounts=tracked_accounts,
        llm_enabled=run_config.llm_enabled,
    )
    feature_rows = feature_service.build_session_dataset(
        posts=posts,
        spy_market=spy,
        tracked_accounts=tracked_accounts,
        feature_version=run_config.feature_version,
        llm_enabled=run_config.llm_enabled,
        prepared_posts=prepared_posts,
    )
    if run_config.start_date:
        feature_rows = feature_rows.loc[feature_rows["signal_session_date"] >= pd.Timestamp(run_config.start_date)].copy()
    if run_config.end_date:
        feature_rows = feature_rows.loc[feature_rows["signal_session_date"] <= pd.Timestamp(run_config.end_date)].copy()

    eligible_sessions = _eligible_replay_sessions(feature_rows)
    if eligible_sessions.empty:
        st.info("No replay sessions are eligible yet. Historical replay needs at least 20 earlier target-available sessions.")
        return

    replay_choices = eligible_sessions.sort_values("signal_session_date", ascending=False).reset_index(drop=True)
    replay_labels = replay_choices.apply(_replay_option_label, axis=1).tolist()
    selected_label = st.selectbox("Historical signal session", options=replay_labels, index=0, key="replay-session-label")
    replay_row = replay_choices.iloc[replay_labels.index(selected_label)]

    with st.spinner("Rebuilding the historical signal without future-data leakage..."):
        replay = backtest_service.build_historical_replay(
            run_config=run_config,
            feature_rows=feature_rows,
            replay_session_date=pd.Timestamp(replay_row["signal_session_date"]),
            deployment_params=loaded["selected_params"],
        )

    replay_prediction = replay["prediction"].iloc[0]
    replay_feature_contributions = replay["feature_contributions"]
    session_post_attribution = build_post_attribution(
        prepared_posts.loc[
            pd.to_datetime(prepared_posts["session_date"], errors="coerce").dt.normalize()
            == pd.Timestamp(replay_prediction["signal_session_date"]).normalize()
        ].copy(),
    )
    session_account_attribution = build_account_attribution(session_post_attribution)

    full_history_match = _filter_for_session(loaded["predictions"], _normalize_session_date(replay_prediction["signal_session_date"]))
    full_history_row = full_history_match.iloc[0] if not full_history_match.empty else None

    metric_cols = st.columns(4)
    metric_cols[0].metric("Replay session", f"{pd.Timestamp(replay_prediction['signal_session_date']):%Y-%m-%d}")
    metric_cols[1].metric("Replay score", f"{float(replay_prediction['expected_return_score']):+.3%}")
    metric_cols[2].metric("Replay confidence", f"{float(replay_prediction['prediction_confidence']):.2f}")
    metric_cols[3].metric("Suggested stance", str(replay_prediction["suggested_stance"]))

    if full_history_row is not None:
        delta = float(replay_prediction["expected_return_score"] - full_history_row["expected_return_score"])
        st.info(
            "The full-history comparison below is shown for drift analysis only. "
            "It uses a model fit with future data that would not have been available on the replay date.",
        )
        drift_cols = st.columns(3)
        drift_cols[0].metric("Full-history score", f"{float(full_history_row['expected_return_score']):+.3%}")
        drift_cols[1].metric("Replay vs full-history", f"{delta:+.3%}")
        drift_cols[2].metric("Actual next-session return", f"{float(replay_prediction['target_next_session_return']):+.3%}" if pd.notna(replay_prediction["target_next_session_return"]) else "n/a")

    st.json(
        {
            "template_run_id": selected_run_id,
            "history_start": str(pd.Timestamp(replay["history_start"]).date()),
            "history_end": str(pd.Timestamp(replay["history_end"]).date()),
            "training_rows_used": replay["training_rows_used"],
            "deployment_params": replay["deployment_params"],
            "future_training_leakage": False,
        },
    )

    comparison_frame = _build_replay_comparison_frame(replay_prediction, full_history_row)
    st.markdown("**Replay summary**")
    st.dataframe(comparison_frame, use_container_width=True, hide_index=True)

    st.markdown("**Replay feature importance**")
    st.dataframe(replay["importance"].head(25), use_container_width=True, hide_index=True)

    _render_signal_explanation_panel(
        prediction_row=replay_prediction,
        feature_contributions=replay_feature_contributions,
        post_attribution=session_post_attribution,
        account_attribution=session_account_attribution,
        heading="Why This Historical Signal?",
    )


def main() -> None:
    settings = AppSettings()
    st.set_page_config(page_title=settings.title, layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        div[data-testid="stMetricValue"] {font-size: 1.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(settings.title)
    st.caption(
        "Single-user web workbench for ingesting Trump-related social data, discovering influential mention accounts, "
        "building next-session SPY features, and evaluating long/flat strategies with walk-forward testing.",
    )

    store = DuckDBStore(settings)
    ingestion_service = IngestionService()
    market_service = MarketDataService()
    discovery_service = DiscoveryService()
    enrichment_service = LLMEnrichmentService(store)
    feature_service = FeatureService(enrichment_service)
    model_service = ModelService()
    backtest_service = BacktestService(model_service)
    experiment_store = ExperimentStore(store)

    _ensure_bootstrap(settings, store, ingestion_service, market_service, discovery_service)

    page = st.sidebar.radio(
        "Workbench",
        options=["Research View", "Datasets", "Discovery", "Models & Backtests", "Historical Replay", "Live Monitor"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Storage: DuckDB + Parquet")
    st.sidebar.code(str(settings.db_path))

    if page == "Research View":
        render_research_view(settings, store, market_service, feature_service)
    elif page == "Datasets":
        render_datasets_view(settings, store, ingestion_service, market_service, discovery_service)
    elif page == "Discovery":
        render_discovery_view(settings, store, discovery_service)
    elif page == "Models & Backtests":
        render_models_view(settings, store, feature_service, model_service, backtest_service, experiment_store)
    elif page == "Historical Replay":
        render_historical_replay_view(settings, store, feature_service, model_service, backtest_service, experiment_store)
    else:
        render_live_monitor(
            settings,
            store,
            ingestion_service,
            market_service,
            discovery_service,
            feature_service,
            model_service,
            experiment_store,
        )
