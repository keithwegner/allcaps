from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from .backtesting import BacktestService
from .config import AppSettings
from .contracts import ModelRunConfig
from .discovery import DiscoveryService
from .enrichment import LLMEnrichmentService
from .experiments import ExperimentStore
from .features import FeatureService, latest_feature_preview, map_posts_to_trade_sessions
from .ingestion import IngestionService, TruthSocialArchiveAdapter, XCsvAdapter
from .market import MarketDataService
from .modeling import ModelService
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

    tracked_existing = store.read_frame("tracked_accounts")
    as_of = posts["post_timestamp"].max() if not posts.empty else pd.Timestamp.now(tz=settings.timezone)
    tracked_accounts, ranking_history = discovery_service.refresh_accounts(posts, tracked_existing, as_of=as_of)
    store.save_frame("tracked_accounts", tracked_accounts, metadata={"row_count": int(len(tracked_accounts))})
    if not ranking_history.empty:
        store.append_frame(
            "account_rankings",
            ranking_history,
            dedupe_on=["author_account_id", "ranked_at"],
            metadata={"row_count": int(len(ranking_history))},
        )
    return {
        "posts": posts,
        "source_manifest": source_manifest,
        "sp500": sp500,
        "spy": spy,
        "tracked_accounts": tracked_accounts,
    }


def _ensure_bootstrap(
    settings: AppSettings,
    store: DuckDBStore,
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
) -> None:
    if store.read_frame("normalized_posts").empty or store.read_frame("sp500_daily").empty or store.read_frame("spy_daily").empty:
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


def render_datasets_view(
    settings: AppSettings,
    store: DuckDBStore,
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
) -> None:
    st.subheader("Datasets")
    st.caption("Refresh historical sources, store normalized datasets, and inspect the local DuckDB + Parquet catalog.")

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
            f"Refreshed {len(summary['posts']):,} normalized posts and {len(summary['tracked_accounts']):,} tracked account versions.",
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
        st.success(f"Incremental refresh complete. Total posts stored: {len(summary['posts']):,}.")

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
    store: DuckDBStore,
    discovery_service: DiscoveryService,
) -> None:
    st.subheader("Discovery")
    st.caption("Dynamic discovery ranks X accounts mentioning Trump and auto-includes the top candidates into the tracked universe.")
    posts = store.read_frame("normalized_posts")
    tracked_accounts = store.read_frame("tracked_accounts")
    ranking_history = store.read_frame("account_rankings")

    if posts.empty:
        st.info("Refresh datasets first so the workbench has posts to analyze.")
        return

    active_accounts = discovery_service.current_active_accounts(
        tracked_accounts,
        as_of=posts["post_timestamp"].max(),
    )
    if not active_accounts.empty:
        st.markdown("**Active tracked universe**")
        keep = [
            "handle",
            "display_name",
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

    latest_rankings = ranking_history.sort_values("ranked_at", ascending=False).drop_duplicates("author_account_id")
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
        ]
        st.dataframe(
            latest_rankings[keep].sort_values("discovery_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    if not ranking_history.empty:
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


def render_models_view(
    settings: AppSettings,
    store: DuckDBStore,
    feature_service: FeatureService,
    backtest_service: BacktestService,
    experiment_store: ExperimentStore,
) -> None:
    st.subheader("Models & Backtests")
    st.caption("Build the session dataset, train a next-session SPY expected-return model, and evaluate it with walk-forward backtests.")
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
            feature_rows = feature_service.build_session_dataset(
                posts=posts,
                spy_market=spy,
                tracked_accounts=tracked_accounts,
                feature_version="v1",
                llm_enabled=llm_enabled,
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
            experiment_store.save_run(
                run=run,
                config=artifacts["config"],
                trades=artifacts["trades"],
                predictions=artifacts["predictions"],
                windows=artifacts["windows"],
                importance=artifacts["importance"],
                model_artifact=artifacts["model_artifact"],
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

    selected_run_id = st.selectbox("Saved runs", options=runs["run_id"].tolist())
    loaded = experiment_store.load_run(selected_run_id)
    if loaded is None:
        st.warning("The selected run could not be loaded.")
        return

    _metric_row(loaded["metrics"])
    _render_equity_curve(loaded["trades"], title="Walk-forward out-of-sample equity curve")

    st.markdown("**Window summary**")
    st.dataframe(loaded["windows"], use_container_width=True, hide_index=True)
    st.markdown("**Feature importance**")
    st.dataframe(loaded["importance"].head(25), use_container_width=True, hide_index=True)
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
    feature_rows = feature_service.build_session_dataset(
        posts=posts,
        spy_market=spy,
        tracked_accounts=tracked_accounts,
        feature_version="v1",
        llm_enabled=bool(artifact.metadata.get("llm_enabled", False)),
    )
    predictions = model_service.predict(artifact, feature_rows)
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
        options=["Research View", "Datasets", "Discovery", "Models & Backtests", "Live Monitor"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Storage: DuckDB + Parquet")
    st.sidebar.code(str(settings.db_path))

    if page == "Research View":
        render_research_view(settings, store, market_service, feature_service)
    elif page == "Datasets":
        render_datasets_view(settings, store, ingestion_service, market_service, discovery_service)
    elif page == "Discovery":
        render_discovery_view(store, discovery_service)
    elif page == "Models & Backtests":
        render_models_view(settings, store, feature_service, backtest_service, experiment_store)
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
