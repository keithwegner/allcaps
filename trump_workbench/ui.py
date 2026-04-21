from __future__ import annotations

from collections import Counter
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from .access import ADMIN_SESSION_KEY, app_mode_label, is_public_mode, verify_admin_password, writes_enabled
from .backtesting import BacktestService
from .config import AppSettings, DEFAULT_ETF_SYMBOLS, EASTERN
from .contracts import (
    LiveMonitorConfig,
    LiveMonitorPinnedRun,
    LinearModelArtifact,
    MANUAL_OVERRIDE_COLUMNS,
    ModelRunConfig,
    PortfolioRunConfig,
    RANKING_HISTORY_COLUMNS,
)
from .discovery import DiscoveryService
from .enrichment import LLMEnrichmentService
from .explanations import build_account_attribution, build_post_attribution
from .experiments import ExperimentStore
from .features import FeatureService, latest_feature_preview, map_posts_to_trade_sessions
from .health import (
    HEALTH_CHECK_COLUMNS,
    REFRESH_HISTORY_COLUMNS,
    DataHealthService,
    build_health_summary,
    build_health_trend_frame,
    create_refresh_id,
    ensure_refresh_history_frame,
    make_refresh_history_frame,
)
from .historical_replay import (
    _build_replay_comparison_frame,
    _bundle_to_run_config,
    _eligible_replay_sessions,
    _replay_option_label,
)
from .ingestion import IngestionService, TruthSocialArchiveAdapter, XCsvAdapter
from .market import MarketDataService, build_asset_universe, build_watchlist_frame, normalize_symbols
from .modeling import (
    ASSET_INDICATOR_PREFIX,
    NARRATIVE_FEATURE_MODES,
    ModelService,
    SUPPORTED_PORTFOLIO_MODEL_FAMILIES,
    add_asset_indicator_columns,
    classify_feature_family,
)
from .live_monitor import (
    LIVE_ASSET_SNAPSHOT_COLUMNS,
    LIVE_DECISION_SNAPSHOT_COLUMNS,
    build_live_portfolio_run_state,
    seed_live_monitor_config,
    rank_live_asset_snapshots,
    validate_live_monitor_config,
)
from .paper_trading import (
    PAPER_BENCHMARK_CURVE_COLUMNS,
    PAPER_DECISION_JOURNAL_COLUMNS,
    PAPER_EQUITY_CURVE_COLUMNS,
    PAPER_PORTFOLIO_REGISTRY_COLUMNS,
    PAPER_TRADE_LEDGER_COLUMNS,
    PaperTradingService,
    ensure_paper_benchmark_curve_frame,
    ensure_paper_decision_journal_frame,
    ensure_paper_equity_curve_frame,
    ensure_paper_portfolio_registry_frame,
    ensure_paper_trade_ledger_frame,
    paper_config_matches_live,
)
from .performance import (
    PerformanceObservatoryService,
    build_equity_comparison_frame,
    build_live_score_drift_frame,
    build_performance_summary,
    build_rolling_return_frame,
    build_score_bucket_outcome_frame,
    build_score_outcome_frame,
    build_winner_distribution_frame,
    ensure_performance_diagnostic_frame,
)
from .runtime import (
    append_refresh_history as runtime_append_refresh_history,
    build_source_adapters as runtime_build_source_adapters,
    ensure_bootstrap as runtime_ensure_bootstrap,
    missing_core_datasets,
    rebuild_discovery_state as runtime_rebuild_discovery_state,
    refresh_datasets as runtime_refresh_datasets,
    save_watchlist as runtime_save_watchlist,
    watchlist_symbols as runtime_watchlist_symbols,
)
from .research import (
    aggregate_research_sessions,
    build_asset_comparison_chart,
    build_asset_comparison_frame,
    build_combined_chart,
    build_event_frame,
    build_event_study_chart,
    build_event_study_frame,
    build_intraday_comparison_chart,
    build_intraday_comparison_frame,
    build_intraday_chart,
    build_narrative_asset_heatmap_chart,
    build_narrative_asset_heatmap_frame,
    build_narrative_frequency_chart,
    build_narrative_frequency_frame,
    build_narrative_return_chart,
    build_narrative_return_frame,
    filter_posts,
    filter_narrative_rows,
    get_intraday_window,
    make_asset_mapping_table,
    make_asset_session_table,
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
from .research_workspace import detect_source_mode, source_mode_label
from .storage import DuckDBStore
from .utils import fmt_score


def _writes_enabled(settings: AppSettings) -> bool:
    return writes_enabled(settings, st.session_state)


def _app_mode_label(settings: AppSettings) -> str:
    return app_mode_label(settings, st.session_state)


def _refresh_required_message(settings: AppSettings, base_message: str) -> str:
    if is_public_mode(settings) and not _writes_enabled(settings):
        return f"{base_message} An admin needs to bootstrap or refresh the shared datasets."
    return base_message


def _source_mode(posts: pd.DataFrame) -> dict[str, Any]:
    return detect_source_mode(posts)


def _source_mode_label(source_mode: dict[str, Any]) -> str:
    return source_mode_label(source_mode)


def _render_sidebar_access_panel(settings: AppSettings) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Access**")
    mode_label = _app_mode_label(settings)
    friendly_label = mode_label.replace("_", " ").title()
    if not is_public_mode(settings):
        st.sidebar.success(f"Mode: {friendly_label}")
        return

    if st.session_state.get(ADMIN_SESSION_KEY, False):
        st.sidebar.success("Mode: Admin")
        if st.sidebar.button("End admin session", use_container_width=True):
            st.session_state[ADMIN_SESSION_KEY] = False
            st.rerun()
        return

    st.sidebar.warning("Mode: Public read-only")
    if not settings.admin_password:
        st.sidebar.caption("Admin password is not configured for this deployment.")
        return
    password = st.sidebar.text_input("Admin password", type="password", key="allcaps_admin_password_input")
    if st.sidebar.button("Unlock admin access", use_container_width=True):
        if verify_admin_password(settings, password):
            st.session_state[ADMIN_SESSION_KEY] = True
            st.session_state.pop("allcaps_admin_password_input", None)
            st.rerun()
        st.sidebar.error("Invalid admin password.")


def _parse_grid(value: str, cast: type[float] | type[int]) -> tuple[float, ...] | tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if cast is int:
        return tuple(int(part) for part in parts)
    return tuple(float(part) for part in parts)


def _target_asset_options(store: DuckDBStore) -> list[str]:
    asset_universe = store.read_frame("asset_universe")
    if asset_universe.empty or "symbol" not in asset_universe.columns:
        return ["SPY"]
    symbols = normalize_symbols(asset_universe["symbol"].astype(str).tolist())
    return ["SPY"] + [symbol for symbol in symbols if symbol != "SPY"]


def _target_asset_label(store: DuckDBStore, symbol: str) -> str:
    normalized = str(symbol).upper()
    asset_universe = store.read_frame("asset_universe")
    if asset_universe.empty:
        return normalized
    rows = asset_universe.loc[asset_universe["symbol"].astype(str).str.upper() == normalized]
    if rows.empty:
        return normalized
    display_name = str(rows.iloc[0].get("display_name", normalized) or normalized)
    return f"{normalized} - {display_name}"


def _normalize_runs_frame(runs: pd.DataFrame) -> pd.DataFrame:
    normalized = runs.copy()
    if normalized.empty:
        return normalized
    if "target_asset" not in normalized.columns:
        normalized["target_asset"] = "SPY"
    normalized["target_asset"] = normalized["target_asset"].fillna("SPY").replace("", "SPY").astype(str).str.upper()
    if "run_type" not in normalized.columns:
        normalized["run_type"] = "asset_model"
    normalized["run_type"] = normalized["run_type"].fillna("asset_model").replace("", "asset_model").astype(str)
    if "allocator_mode" not in normalized.columns:
        normalized["allocator_mode"] = ""
    normalized["allocator_mode"] = normalized["allocator_mode"].fillna("").astype(str)
    return normalized


def _build_model_target_bundle(
    store: DuckDBStore,
    feature_service: FeatureService,
    posts: pd.DataFrame,
    spy_market: pd.DataFrame,
    tracked_accounts: pd.DataFrame,
    llm_enabled: bool,
    target_asset: str,
    feature_version: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_target = str(target_asset).upper()
    resolved_feature_version = feature_version or ("v1" if normalized_target == "SPY" else "asset-v1")
    asset_universe = store.read_frame("asset_universe")
    asset_daily = store.read_frame("asset_daily")
    prepared_posts = feature_service.prepare_session_posts(
        posts=posts,
        market_calendar=spy_market,
        tracked_accounts=tracked_accounts,
        llm_enabled=llm_enabled,
    )

    if normalized_target == "SPY":
        feature_rows = feature_service.build_session_dataset(
            posts=posts,
            spy_market=spy_market,
            tracked_accounts=tracked_accounts,
            feature_version=resolved_feature_version,
            llm_enabled=llm_enabled,
            prepared_posts=prepared_posts,
        )
        feature_rows["target_asset"] = "SPY"
        feature_rows["target_asset_display_name"] = "SPDR S&P 500 ETF Trust"
        return feature_rows.reset_index(drop=True), prepared_posts.reset_index(drop=True)

    if asset_universe.empty or asset_daily.empty:
        raise RuntimeError("Refresh datasets first so non-SPY target assets have market and universe data available.")

    asset_post_mappings = feature_service.build_asset_post_mappings(
        prepared_posts=prepared_posts,
        asset_universe=asset_universe,
        llm_enabled=llm_enabled,
    )
    asset_feature_rows = feature_service.build_asset_session_dataset(
        asset_post_mappings=asset_post_mappings,
        asset_market=asset_daily,
        feature_version=resolved_feature_version,
        llm_enabled=llm_enabled,
        asset_universe=asset_universe,
    )
    feature_rows = asset_feature_rows.loc[
        asset_feature_rows["asset_symbol"].astype(str).str.upper() == normalized_target
    ].copy()
    if feature_rows.empty:
        raise RuntimeError(f"No session feature rows were available for target asset `{normalized_target}`.")
    asset_rows = asset_universe.loc[asset_universe["symbol"].astype(str).str.upper() == normalized_target]
    display_name = str(asset_rows.iloc[0].get("display_name", normalized_target)) if not asset_rows.empty else normalized_target
    feature_rows["target_asset"] = normalized_target
    feature_rows["target_asset_display_name"] = display_name
    attribution_posts = asset_post_mappings.loc[
        asset_post_mappings["asset_symbol"].astype(str).str.upper() == normalized_target
    ].copy()
    attribution_posts["target_asset"] = normalized_target
    return feature_rows.reset_index(drop=True), attribution_posts.reset_index(drop=True)


def _live_run_option_label(run_row: pd.Series) -> str:
    created_at = pd.to_datetime(run_row.get("created_at"), errors="coerce")
    created_label = f"{created_at:%Y-%m-%d %H:%M}" if pd.notna(created_at) else "unknown time"
    run_name = str(run_row.get("run_name", run_row.get("run_id", "")) or run_row.get("run_id", ""))
    robust_score = float((run_row.get("metrics_json") or {}).get("robust_score", 0.0) or 0.0)
    return f"{run_name} | {created_label} | robust {robust_score:+.3f}"


def _runs_by_asset(runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if runs.empty:
        return {}
    normalized = runs.copy()
    if "target_asset" not in normalized.columns:
        normalized["target_asset"] = "SPY"
    normalized["target_asset"] = normalized["target_asset"].fillna("SPY").astype(str).str.upper()
    return {
        asset_symbol: group.reset_index(drop=True)
        for asset_symbol, group in normalized.groupby("target_asset", sort=False)
    }


def _build_live_monitor_state(
    store: DuckDBStore,
    feature_service: FeatureService,
    model_service: ModelService,
    experiment_store: ExperimentStore,
    posts: pd.DataFrame,
    spy_market: pd.DataFrame,
    tracked_accounts: pd.DataFrame,
    config: LiveMonitorConfig,
    generated_at: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], list[str]]:
    if str(config.mode or "portfolio_run") == "asset_model_set":
        return _build_live_monitor_state_from_asset_model_set(
            store=store,
            feature_service=feature_service,
            model_service=model_service,
            experiment_store=experiment_store,
            posts=posts,
            spy_market=spy_market,
            tracked_accounts=tracked_accounts,
            config=config,
            generated_at=generated_at,
        )
    return _build_live_monitor_state_from_portfolio_run(
        store=store,
        model_service=model_service,
        experiment_store=experiment_store,
        config=config,
        generated_at=generated_at,
    )


def _build_live_monitor_state_from_asset_model_set(
    store: DuckDBStore,
    feature_service: FeatureService,
    model_service: ModelService,
    experiment_store: ExperimentStore,
    posts: pd.DataFrame,
    spy_market: pd.DataFrame,
    tracked_accounts: pd.DataFrame,
    config: LiveMonitorConfig,
    generated_at: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], list[str]]:
    snapshot_rows: list[dict[str, Any]] = []
    explanation_lookup: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    snapshot_time = pd.Timestamp.utcnow().floor("s") if generated_at is None else pd.Timestamp(generated_at)

    for pinned in config.pinned_runs:
        asset_symbol = str(pinned.asset_symbol or "").upper()
        run_id = str(pinned.run_id or "")
        if not asset_symbol or not run_id:
            continue
        run_bundle = experiment_store.load_run(run_id)
        if run_bundle is None:
            warnings.append(f"Pinned run `{run_id}` for `{asset_symbol}` could not be loaded.")
            continue

        run_config = _bundle_to_run_config(run_bundle)
        try:
            feature_rows, attribution_posts = _build_model_target_bundle(
                store=store,
                feature_service=feature_service,
                posts=posts,
                spy_market=spy_market,
                tracked_accounts=tracked_accounts,
                llm_enabled=run_config.llm_enabled,
                target_asset=asset_symbol,
                feature_version=run_config.feature_version,
            )
        except RuntimeError as exc:
            warnings.append(f"`{asset_symbol}` live features could not be built: {exc}")
            continue
        if feature_rows.empty:
            warnings.append(f"`{asset_symbol}` does not have any current live feature rows.")
            continue

        predictions = model_service.predict(run_bundle["model_artifact"], feature_rows)
        feature_contributions = model_service.explain_predictions(run_bundle["model_artifact"], predictions)
        post_attribution = build_post_attribution(attribution_posts)
        account_attribution = build_account_attribution(post_attribution)
        latest_prediction = predictions.sort_values("signal_session_date").iloc[-1].copy()
        latest_prediction["prediction_confidence"] = float(latest_prediction.get("prediction_confidence", 0.0) or 0.0)
        latest_prediction["target_asset"] = asset_symbol
        latest_prediction["run_id"] = run_id
        latest_prediction["run_name"] = str(run_bundle.get("run", {}).get("run_name", run_id) or run_id)
        selected_params = run_bundle.get("selected_params", {}) or {}
        threshold = selected_params.get("threshold", 0.0)
        min_post_count = selected_params.get("min_post_count", 1)
        snapshot_rows.append(
            {
                "generated_at": snapshot_time,
                "signal_session_date": latest_prediction.get("signal_session_date"),
                "next_session_date": latest_prediction.get("next_session_date"),
                "asset_symbol": asset_symbol,
                "run_id": run_id,
                "run_name": latest_prediction["run_name"],
                "feature_version": str(latest_prediction.get("feature_version", run_config.feature_version) or run_config.feature_version),
                "model_version": str(latest_prediction.get("model_version", run_bundle["model_artifact"].model_version) or run_bundle["model_artifact"].model_version),
                "expected_return_score": float(latest_prediction.get("expected_return_score", 0.0) or 0.0),
                "confidence": float(latest_prediction.get("prediction_confidence", 0.0) or 0.0),
                "threshold": float(threshold if threshold is not None else 0.0),
                "min_post_count": int(min_post_count if min_post_count is not None else 1),
                "post_count": int(latest_prediction.get("post_count", 0) or 0),
                "next_session_open_ts": latest_prediction.get("next_session_open_ts"),
            },
        )
        explanation_lookup[asset_symbol] = {
            "prediction_row": latest_prediction,
            "feature_contributions": feature_contributions,
            "post_attribution": post_attribution,
            "account_attribution": account_attribution,
        }

    snapshots = pd.DataFrame(snapshot_rows, columns=LIVE_ASSET_SNAPSHOT_COLUMNS)
    if snapshots.empty:
        return snapshots, pd.DataFrame(columns=LIVE_DECISION_SNAPSHOT_COLUMNS), explanation_lookup, warnings

    ranked_board, decision = rank_live_asset_snapshots(snapshots, config.fallback_mode)
    for asset_symbol, payload in explanation_lookup.items():
        board_match = ranked_board.loc[ranked_board["asset_symbol"] == asset_symbol]
        if board_match.empty:
            continue
        payload["board_row"] = board_match.iloc[0]
    return ranked_board, decision, explanation_lookup, warnings


def _apply_live_account_weight(feature_rows: pd.DataFrame, account_weight: float) -> pd.DataFrame:
    adjusted = feature_rows.copy()
    for column in ["tracked_weighted_mentions", "tracked_weighted_engagement", "tracked_account_post_count"]:
        if column in adjusted.columns:
            adjusted[column] = adjusted[column] * float(account_weight)
    return adjusted


def _portfolio_live_model_artifact(payload: dict[str, Any]) -> LinearModelArtifact:
    return LinearModelArtifact.from_dict(payload)


def _predict_portfolio_live_variant(
    model_service: ModelService,
    feature_rows: pd.DataFrame,
    variant_payload: dict[str, Any],
    selected_symbols: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    topology = str(variant_payload.get("topology", "per_asset") or "per_asset")
    models_payload = variant_payload.get("models", {}) or {}
    predictions: list[pd.DataFrame] = []
    explanations: dict[str, dict[str, Any]] = {}
    if topology == "pooled":
        pooled_model = models_payload.get("pooled")
        if not isinstance(pooled_model, dict):
            return pd.DataFrame(), {}
        artifact = _portfolio_live_model_artifact(pooled_model)
        pooled_rows = add_asset_indicator_columns(feature_rows, selected_symbols)
        scored = model_service.predict(artifact, pooled_rows)
        predictions.append(scored)
        for asset_symbol in selected_symbols:
            asset_rows = scored.loc[scored["asset_symbol"].astype(str).str.upper() == asset_symbol].copy()
            if asset_rows.empty:
                continue
            explanations[asset_symbol] = {
                "artifact": artifact,
                "prediction_frame": asset_rows,
                "feature_contributions": model_service.explain_predictions(artifact, asset_rows),
            }
    else:
        for asset_symbol in selected_symbols:
            model_payload = models_payload.get(asset_symbol)
            if not isinstance(model_payload, dict):
                continue
            artifact = _portfolio_live_model_artifact(model_payload)
            asset_rows = feature_rows.loc[feature_rows["asset_symbol"].astype(str).str.upper() == asset_symbol].copy()
            if asset_rows.empty:
                continue
            scored = model_service.predict(artifact, asset_rows)
            predictions.append(scored)
            explanations[asset_symbol] = {
                "artifact": artifact,
                "prediction_frame": scored,
                "feature_contributions": model_service.explain_predictions(artifact, scored),
            }
    if not predictions:
        return pd.DataFrame(), {}
    combined = pd.concat(predictions, ignore_index=True).sort_values(["signal_session_date", "asset_symbol"]).reset_index(drop=True)
    return combined, explanations


def _build_live_monitor_state_from_portfolio_run(
    store: DuckDBStore,
    model_service: ModelService,
    experiment_store: ExperimentStore,
    config: LiveMonitorConfig,
    generated_at: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], list[str]]:
    return build_live_portfolio_run_state(
        store=store,
        model_service=model_service,
        experiment_store=experiment_store,
        config=config,
        generated_at=generated_at,
    )


def _build_live_runner_up_frame(board: pd.DataFrame, decision_row: pd.Series) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()
    winner_asset = str(decision_row.get("winning_asset", "") or "")
    ordered = board.sort_values(["qualifies", "expected_return_score", "confidence"], ascending=[False, False, False]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    winner = ordered.loc[ordered["asset_symbol"] == winner_asset]
    if not winner.empty:
        winner_row = winner.iloc[0]
        rows.append(
            {
                "asset_symbol": str(winner_row["asset_symbol"]),
                "run_name": str(winner_row["run_name"]),
                "score": float(winner_row["expected_return_score"]),
                "confidence": float(winner_row["confidence"]),
                "threshold_gap": float(winner_row["expected_return_score"] - winner_row["threshold"]),
                "post_count": int(winner_row["post_count"]),
                "qualifies": bool(winner_row["qualifies"]),
                "winner": True,
            },
        )
    runner_up_asset = str(decision_row.get("runner_up_asset", "") or "")
    if runner_up_asset:
        runner = ordered.loc[ordered["asset_symbol"] == runner_up_asset]
        if not runner.empty:
            runner_row = runner.iloc[0]
            rows.append(
                {
                    "asset_symbol": str(runner_row["asset_symbol"]),
                    "run_name": str(runner_row["run_name"]),
                    "score": float(runner_row["expected_return_score"]),
                    "confidence": float(runner_row["confidence"]),
                    "threshold_gap": float(runner_row["expected_return_score"] - runner_row["threshold"]),
                    "post_count": int(runner_row["post_count"]),
                    "qualifies": bool(runner_row["qualifies"]),
                    "winner": False,
                },
            )
    return pd.DataFrame(rows)


def _watchlist_symbols(store: DuckDBStore) -> list[str]:
    return runtime_watchlist_symbols(store)


def _save_watchlist(store: DuckDBStore, symbols: list[str] | tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return runtime_save_watchlist(store, symbols)


def _watchlist_text_value(store: DuckDBStore) -> str:
    symbols = _watchlist_symbols(store)
    return ", ".join(symbols)


def _build_adapters(
    settings: AppSettings,
    remote_url: str,
    uploaded_files: list[Any],
) -> list[Any]:
    return runtime_build_source_adapters(settings, remote_url, uploaded_files)


def _refresh_datasets(
    settings: AppSettings,
    store: DuckDBStore,
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
    feature_service: FeatureService,
    health_service: DataHealthService,
    remote_url: str,
    uploaded_files: list[Any],
    incremental: bool = False,
    refresh_mode: str = "full",
) -> dict[str, Any]:
    return runtime_refresh_datasets(
        settings=settings,
        store=store,
        ingestion_service=ingestion_service,
        market_service=market_service,
        discovery_service=discovery_service,
        feature_service=feature_service,
        health_service=health_service,
        remote_url=remote_url,
        uploaded_files=uploaded_files,
        incremental=incremental,
        refresh_mode=refresh_mode,
    )


def _ensure_bootstrap(
    settings: AppSettings,
    store: DuckDBStore,
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
    feature_service: FeatureService,
    health_service: DataHealthService,
) -> dict[str, Any] | None:
    return runtime_ensure_bootstrap(
        settings=settings,
        store=store,
        ingestion_service=ingestion_service,
        market_service=market_service,
        discovery_service=discovery_service,
        feature_service=feature_service,
        health_service=health_service,
    )


def _rebuild_discovery_state(
    store: DuckDBStore,
    discovery_service: DiscoveryService,
    posts: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return runtime_rebuild_discovery_state(store, discovery_service, posts, as_of)


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


def _append_refresh_history(
    store: DuckDBStore,
    refresh_id: str,
    refresh_mode: str,
    status: str,
    started_at: pd.Timestamp,
    completed_at: pd.Timestamp,
    error_message: str = "",
) -> pd.DataFrame:
    return runtime_append_refresh_history(
        store=store,
        refresh_id=refresh_id,
        refresh_mode=refresh_mode,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        error_message=error_message,
    )


def _load_data_health_view_state(
    store: DuckDBStore,
    health_service: DataHealthService,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    latest = store.read_frame("data_health_latest")
    if latest.empty:
        latest = health_service.evaluate_store(store)
    history = store.read_frame("data_health_history")
    refresh_history = ensure_refresh_history_frame(store.read_frame("refresh_history"))
    trend_source = history if not history.empty else latest
    return {
        "latest": latest if not latest.empty else pd.DataFrame(columns=HEALTH_CHECK_COLUMNS),
        "history": history if not history.empty else pd.DataFrame(columns=HEALTH_CHECK_COLUMNS),
        "refresh_history": refresh_history,
        "summary": build_health_summary(latest, refresh_history),
        "trend": build_health_trend_frame(trend_source),
    }


def _filter_health_rows(
    health_rows: pd.DataFrame,
    severities: list[str] | tuple[str, ...],
    scope_kind: str,
) -> pd.DataFrame:
    if health_rows.empty:
        return pd.DataFrame(columns=HEALTH_CHECK_COLUMNS)
    filtered = health_rows.copy()
    if severities:
        filtered = filtered.loc[filtered["severity"].astype(str).isin([str(item).lower() for item in severities])].copy()
    if scope_kind and scope_kind != "All":
        filtered = filtered.loc[filtered["scope_kind"].astype(str) == str(scope_kind)].copy()
    return filtered.reset_index(drop=True)


def render_datasets_view(
    settings: AppSettings,
    store: DuckDBStore,
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
    feature_service: FeatureService,
    health_service: DataHealthService,
) -> None:
    st.subheader("Datasets")
    st.caption("Refresh historical sources, store normalized datasets, and inspect the local DuckDB + Parquet catalog.")
    can_write = _writes_enabled(settings)
    missing_datasets = missing_core_datasets(store)
    posts = store.read_frame("normalized_posts")
    source_mode = _source_mode(posts)
    if is_public_mode(settings) and not can_write:
        st.info("This hosted deployment is in public read-only mode. Unlock admin access in the sidebar to refresh data, edit the watchlist, or change live configuration.")

    health_state = _load_data_health_view_state(store, health_service)
    latest_health = health_state["latest"]
    refresh_history = health_state["refresh_history"]
    trend = health_state["trend"]
    summary = health_state["summary"]

    st.markdown("**Hosted Operations**")
    ops_cols = st.columns(6)
    ops_cols[0].metric("Operating mode", _source_mode_label(source_mode))
    ops_cols[1].metric("App mode", _app_mode_label(settings).replace("_", " ").title())
    ops_cols[2].metric("Scheduler", "Enabled" if settings.scheduler_enabled else "Disabled")
    ops_cols[3].metric("Last refresh mode", str(summary.get("last_refresh_mode", "n/a")).upper())
    last_refresh_at = summary.get("last_refresh_at", pd.NaT)
    ops_cols[4].metric(
        "Last refresh at",
        pd.Timestamp(last_refresh_at).tz_convert(EASTERN).strftime("%Y-%m-%d %H:%M") if pd.notna(last_refresh_at) else "n/a",
    )
    ops_cols[5].metric("Missing core datasets", f"{len(missing_datasets):,}")
    st.caption(f"State root: `{settings.state_root}` | DuckDB: `{settings.db_path}`")

    if missing_datasets:
        if can_write:
            st.warning(
                "Core datasets are missing for this deployment. Use `Bootstrap datasets` to populate the shared state volume.",
            )
        else:
            st.warning(
                "This hosted instance has not been bootstrapped yet. An admin needs to populate the shared datasets before public pages will show data.",
            )

    st.markdown("**Asset universe**")
    st.caption("Manage a small stock watchlist here. The ETF starter set is always included: `SPY`, `QQQ`, `XLK`, `XLF`, `XLE`, `SMH`.")
    default_text = _watchlist_text_value(store)
    watchlist_input = st.text_area(
        "Watchlist symbols",
        value=st.session_state.get("watchlist_symbols", default_text),
        help="Enter a comma-separated list of stock tickers such as `AAPL, TSLA, NVDA`.",
        disabled=not can_write,
    )
    st.session_state["watchlist_symbols"] = watchlist_input
    watchlist_cols = st.columns(2)
    if watchlist_cols[0].button("Save watchlist", use_container_width=True, disabled=not can_write):
        symbols = normalize_symbols([part.strip() for part in watchlist_input.replace("\n", ",").split(",") if part.strip()])
        watchlist, asset_universe = _save_watchlist(store, symbols)
        st.session_state["watchlist_symbols"] = ", ".join(watchlist["symbol"].tolist()) if not watchlist.empty else ""
        st.success(f"Saved {len(watchlist):,} watchlist symbols and {len(asset_universe):,} total tracked assets.")
        st.rerun()
    if watchlist_cols[1].button("Reset watchlist", use_container_width=True, disabled=not can_write):
        _save_watchlist(store, [])
        st.session_state["watchlist_symbols"] = ""
        st.success("Reset the manual watchlist to the ETF starter set only.")
        st.rerun()

    remote_url = st.text_input(
        "Remote X / mentions CSV URL",
        key="remote_x_url",
        value=st.session_state.get("remote_x_url", settings.remote_x_csv_url),
        disabled=not can_write,
    )
    uploaded_files = st.file_uploader(
        "Upload X or mention CSVs",
        type=["csv"],
        accept_multiple_files=True,
        disabled=not can_write,
    )
    action_cols = st.columns(2)
    primary_refresh_label = "Bootstrap datasets" if missing_datasets else "Refresh full datasets"
    if action_cols[0].button(primary_refresh_label, use_container_width=True, disabled=not can_write):
        with st.spinner("Refreshing source data, market data, and discovery state..."):
            summary = _refresh_datasets(
                settings=settings,
                store=store,
                ingestion_service=ingestion_service,
                market_service=market_service,
                discovery_service=discovery_service,
                feature_service=feature_service,
                health_service=health_service,
                remote_url=remote_url,
                uploaded_files=uploaded_files or [],
                refresh_mode="bootstrap" if missing_datasets else "full",
            )
        st.success(
            f"Refreshed {len(summary['posts']):,} normalized posts, {len(summary['asset_daily']):,} daily market rows, "
            f"and {len(summary['tracked_accounts']):,} tracked account versions.",
        )
    if action_cols[1].button("Incremental refresh", use_container_width=True, disabled=not can_write):
        with st.spinner("Polling sources for new data..."):
            summary = _refresh_datasets(
                settings=settings,
                store=store,
                ingestion_service=ingestion_service,
                market_service=market_service,
                discovery_service=discovery_service,
                feature_service=feature_service,
                health_service=health_service,
                remote_url=remote_url,
                uploaded_files=uploaded_files or [],
                incremental=True,
                refresh_mode="incremental",
            )
        st.success(
            f"Incremental refresh complete. Total posts stored: {len(summary['posts']):,}. "
            f"Asset intraday rows stored: {len(summary['asset_intraday']):,}.",
        )

    st.markdown("**Data Health**")
    st.caption("Warn-only checks for freshness, completeness, manifest errors, duplicates, and anomaly drift across refresh history.")
    overall_severity = str(summary.get("overall_severity", "ok"))
    if overall_severity == "severe":
        st.error("Severe data health issues are present. Workflows remain available, but the stored datasets need inspection.")
    elif overall_severity == "warn":
        st.warning("Dataset health warnings are present. Workflows remain available, but review the checks below.")
    else:
        st.success("No current warning or severe data health checks are active.")

    last_refresh_at = summary.get("last_refresh_at", pd.NaT)
    if pd.notna(last_refresh_at):
        last_refresh_label = pd.Timestamp(last_refresh_at).tz_convert(EASTERN).strftime("%Y-%m-%d %H:%M")
    else:
        last_refresh_label = "n/a"
    metric_cols = st.columns(6)
    metric_cols[0].metric("Overall", str(summary.get("overall_severity", "ok")).upper())
    metric_cols[1].metric("Severe", f"{int(summary.get('severe_count', 0))}")
    metric_cols[2].metric("Warn", f"{int(summary.get('warn_count', 0))}")
    metric_cols[3].metric("Anomalies", f"{int(summary.get('anomaly_count', 0))}")
    metric_cols[4].metric("Last refresh", last_refresh_label)
    metric_cols[5].metric(
        "Refresh status",
        str(summary.get("last_refresh_status", "n/a")).upper(),
        str(summary.get("last_refresh_mode", "")).upper(),
    )

    latest_health = latest_health.copy() if isinstance(latest_health, pd.DataFrame) else pd.DataFrame(columns=HEALTH_CHECK_COLUMNS)
    if not latest_health.empty:
        latest_health["generated_at"] = pd.to_datetime(latest_health["generated_at"], errors="coerce", utc=True)
        latest_health["observed_value"] = pd.to_numeric(latest_health["observed_value"], errors="coerce")
        latest_health["baseline_value"] = pd.to_numeric(latest_health["baseline_value"], errors="coerce")
        latest_health["severity"] = latest_health["severity"].astype(str)
        latest_health["generated_at"] = latest_health["generated_at"].dt.tz_convert(EASTERN)

    filter_cols = st.columns([2, 2, 3])
    default_severities = [severity for severity in ["warn", "severe"] if latest_health.empty or severity in latest_health["severity"].astype(str).unique()]
    selected_severities = filter_cols[0].multiselect(
        "Health severities",
        options=["warn", "severe", "ok"],
        default=default_severities or ["warn", "severe"],
        key="datasets_health_severities",
    )
    scope_options = ["All"]
    if not latest_health.empty and "scope_kind" in latest_health.columns:
        scope_options.extend(sorted(latest_health["scope_kind"].dropna().astype(str).unique().tolist()))
    selected_scope = filter_cols[1].selectbox("Scope kind", options=scope_options, index=0, key="datasets_health_scope")
    filter_cols[2].caption("The table defaults to current warnings and severe checks. Add `ok` to inspect the full snapshot.")
    filtered_health = _filter_health_rows(latest_health, selected_severities, selected_scope)
    if filtered_health.empty:
        st.info("No health rows match the current filters.")
    else:
        st.dataframe(filtered_health, use_container_width=True, hide_index=True)

    st.markdown("**Health history**")
    trend_frame = trend.copy() if isinstance(trend, pd.DataFrame) else pd.DataFrame()
    if not trend_frame.empty:
        trend_frame["generated_at"] = pd.to_datetime(trend_frame["generated_at"], errors="coerce", utc=True)
        trend_frame["generated_at"] = trend_frame["generated_at"].dt.tz_convert(EASTERN)
        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                x=trend_frame["generated_at"],
                y=trend_frame["warn_count"],
                mode="lines+markers",
                name="Warn checks",
            ),
        )
        trend_fig.add_trace(
            go.Scatter(
                x=trend_frame["generated_at"],
                y=trend_frame["severe_count"],
                mode="lines+markers",
                name="Severe checks",
            ),
        )
        trend_fig.update_layout(
            title="Health severity counts by snapshot",
            xaxis_title="Snapshot time",
            yaxis_title="Check count",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("Health trend history will appear after at least one persisted refresh snapshot is available.")

    if not refresh_history.empty:
        refresh_history = refresh_history.copy()
        for column in ["started_at", "completed_at"]:
            refresh_history[column] = pd.to_datetime(refresh_history[column], errors="coerce", utc=True).dt.tz_convert(EASTERN)
        st.markdown("**Refresh history**")
        st.dataframe(
            refresh_history.sort_values("completed_at", ascending=False),
            use_container_width=True,
            hide_index=True,
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

    asset_post_mappings = store.read_frame("asset_post_mappings")
    if not asset_post_mappings.empty:
        st.markdown("**Asset post mappings sample**")
        keep = [
            "asset_symbol",
            "session_date",
            "author_handle",
            "asset_relevance_score",
            "rule_match_score",
            "semantic_match_score",
            "match_reasons",
            "cleaned_text",
        ]
        st.dataframe(asset_post_mappings[keep].tail(20), use_container_width=True, hide_index=True)

    asset_session_features = store.read_frame("asset_session_features")
    if not asset_session_features.empty:
        st.markdown("**Per-asset session feature sample**")
        keep = [
            "asset_symbol",
            "signal_session_date",
            "post_count",
            "asset_relevance_score_avg",
            "rule_matched_post_count",
            "semantic_matched_post_count",
            "target_next_session_return",
        ]
        st.dataframe(asset_session_features[keep].tail(20), use_container_width=True, hide_index=True)

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
    can_write = _writes_enabled(settings)
    if is_public_mode(settings) and not can_write:
        st.info("Discovery overrides are disabled in public read-only mode.")
    posts = store.read_frame("normalized_posts")
    tracked_accounts = store.read_frame("tracked_accounts")
    ranking_history = store.read_frame("account_rankings")
    overrides = discovery_service.normalize_manual_overrides(store.read_frame("manual_account_overrides"))
    source_mode = _source_mode(posts)

    if posts.empty:
        st.info(_refresh_required_message(settings, "Refresh datasets first so the workbench has posts to analyze."))
        return

    candidate_posts = posts.loc[
        (posts["source_platform"] == "X")
        & posts["mentions_trump"]
        & (~posts["author_is_trump"])
    ]
    ranking_columns_present = {"author_account_id", "ranked_at"}.issubset(ranking_history.columns)
    if (not candidate_posts.empty or not overrides.empty) and not ranking_columns_present and can_write:
        tracked_accounts, ranking_history = _rebuild_discovery_state(
            store,
            discovery_service,
            posts,
            posts["post_timestamp"].max(),
        )
        overrides = discovery_service.normalize_manual_overrides(store.read_frame("manual_account_overrides"))
    elif (not candidate_posts.empty or not overrides.empty) and not ranking_columns_present:
        st.info("Discovery rankings have not been built for this hosted instance yet. An admin needs to refresh datasets first.")

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
        if st.button("Save override", use_container_width=True, disabled=not can_write):
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
    elif source_mode["mode"] == "truth_only":
        st.info(
            "This dataset is currently Truth Social-only. Discovery ranks non-Trump X accounts that mention Trump, "
            "so it is optional for reviewing sentiment based only on Donald Trump's Truth Social posts. "
            "Load X/mention CSVs in `Datasets` if you want to populate account discovery.",
        )
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
        if st.button("Delete selected override", use_container_width=True, disabled=not can_write):
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
    asset_universe = store.read_frame("asset_universe")
    asset_daily = store.read_frame("asset_daily")
    asset_intraday = store.read_frame("asset_intraday")
    asset_post_mappings = store.read_frame("asset_post_mappings")
    asset_session_features = store.read_frame("asset_session_features")
    tracked_accounts = store.read_frame("tracked_accounts")
    source_mode = _source_mode(posts)
    if posts.empty or sp500.empty:
        st.info(_refresh_required_message(settings, "Refresh datasets first so the research view has source data."))
        return

    today_et = pd.Timestamp.now(tz=settings.timezone).normalize().tz_localize(None)
    controls = st.columns(6)
    date_range = controls[0].date_input(
        "Date range",
        value=(settings.term_start.date(), today_et.date()),
        min_value=settings.term_start.date(),
        max_value=today_et.date(),
    )
    source_mode_name = str(source_mode["mode"])
    if st.session_state.get("research_source_mode_seeded") != source_mode_name:
        st.session_state["research_platforms"] = ["Truth Social"] if source_mode_name == "truth_only" else ["Truth Social", "X"]
        st.session_state["research_trump_authored_only"] = source_mode_name == "truth_only"
        st.session_state["research_source_mode_seeded"] = source_mode_name
    selected_platforms = controls[1].multiselect(
        "Platforms",
        options=["Truth Social", "X"],
        key="research_platforms",
    )
    include_reshares = controls[2].checkbox("Include reshares", value=False)
    tracked_only = controls[3].checkbox("Tracked accounts only", value=False)
    trump_authored_only = controls[4].checkbox(
        "Trump-authored only",
        help="Restrict research inputs to posts where the normalized author is Donald Trump's account.",
        key="research_trump_authored_only",
    )
    scale_markers = controls[5].checkbox("Scale markers", value=True)
    keyword = st.text_input("Keyword filter", value="")

    if source_mode["mode"] == "truth_only":
        if selected_platforms == ["Truth Social"] and trump_authored_only:
            st.info(
                f"Truth Social-only mode detected: the current analysis is scoped to Donald Trump's Truth Social posts "
                f"from {int(source_mode['truth_post_count']):,} stored Truth Social rows. Use `Platforms` and "
                "`Trump-authored only` above to verify or change the research scope.",
            )
        else:
            st.info(
                f"Truth Social-only mode detected with {int(source_mode['truth_post_count']):,} stored Truth Social rows. "
                "The default scope is `Platforms = Truth Social` and `Trump-authored only`; adjust the controls above "
                "if you intentionally want a broader or different view.",
            )

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
        trump_authored_only=trump_authored_only,
    )
    mapped = feature_service.prepare_session_posts(
        posts=filtered_posts,
        market_calendar=market,
        tracked_accounts=tracked_accounts,
        llm_enabled=True,
    )
    sessions = aggregate_research_sessions(mapped)
    events = build_event_frame(market, sessions)
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
            (narrative_asset_view["session_date"] >= start_date.normalize())
            & (narrative_asset_view["session_date"] <= end_date.normalize())
        ].copy()

    sessions_with_posts = int((events["post_count"] > 0).sum()) if "post_count" in events.columns else 0
    posts_in_view = int(events["post_count"].sum()) if "post_count" in events.columns else 0
    truth_posts = int(mapped.loc[_series_or_false(mapped, "author_is_trump").astype(bool)].shape[0]) if not mapped.empty else 0
    tracked_x_posts = int(mapped.loc[_series_or_false(mapped, "is_active_tracked_account").astype(bool)].shape[0]) if not mapped.empty else 0
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

    metric_cols = st.columns(6)
    metric_cols[0].metric("Sessions with posts", f"{sessions_with_posts:,}")
    metric_cols[1].metric("Posts in view", f"{posts_in_view:,}")
    metric_cols[2].metric("Truth posts", f"{truth_posts:,}")
    metric_cols[3].metric("Tracked X posts", f"{tracked_x_posts:,}")
    metric_cols[4].metric("Mean sentiment", fmt_score(mean_sentiment))
    metric_cols[5].metric("S&P 500 change", f"{sp500_change:+.2%}" if sp500_change is not None else "n/a")

    combined_chart = build_combined_chart(events, scale_markers=scale_markers)
    st.plotly_chart(combined_chart, use_container_width=True)

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

    export_narrative_frequency = build_narrative_frequency_frame(mapped)
    export_narrative_returns = build_narrative_return_frame(mapped, market, bucket_field="semantic_topic")
    export_narrative_asset_heatmap = build_narrative_asset_heatmap_frame(narrative_asset_view)
    export_narrative_posts = make_narrative_post_table(mapped).head(25)
    export_narrative_events = make_narrative_event_table(mapped, market)
    export_manifest = build_research_export_manifest(
        filters={
            "date_start": start_date.date().isoformat(),
            "date_end": end_date.date().isoformat(),
            "platforms": list(selected_platforms),
            "keyword": keyword,
            "include_reshares": include_reshares,
            "tracked_only": tracked_only,
            "trump_authored_only": trump_authored_only,
            "narrative_bucket_field": "semantic_topic",
        },
        source_mode=source_mode,
        headline_metrics=headline_metrics,
    )
    export_bundle = build_research_export_bundle(
        manifest=export_manifest,
        chart=combined_chart,
        sessions=session_table,
        posts=post_table,
        narrative_frequency=export_narrative_frequency,
        narrative_returns=export_narrative_returns,
        narrative_asset_heatmap=export_narrative_asset_heatmap,
        narrative_posts=export_narrative_posts,
        narrative_events=export_narrative_events,
    )
    st.markdown("**Export current research pack**")
    st.caption(
        "Download a ZIP bundle for the current Research View filters. Includes a manifest, Markdown summary, "
        "social activity chart HTML, session/post CSVs, and all-narrative CSV outputs for this filtered slice.",
    )
    export_cols = st.columns([1.2, 1, 1])
    export_cols[0].metric("Files", "10")
    export_cols[1].metric("Session rows", f"{len(session_table):,}")
    export_cols[2].metric("Post rows", f"{len(post_table):,}")
    st.download_button(
        "Download research pack (.zip)",
        data=export_bundle,
        file_name=research_export_filename(start_date, end_date),
        mime="application/zip",
        use_container_width=True,
    )

    narratives_tab, asset_tab = st.tabs(["Narratives", "Multi-Asset Comparison"])

    with narratives_tab:
        st.caption(
            "Narrative Lab: inspect structured topics, policy buckets, urgency, and asset targets. "
            "Heuristic fallback remains available even when no hosted provider is configured.",
        )
        if mapped.empty:
            st.info("No enriched post sessions are available for the current date range and filters.")
        else:
            topic_options = ["All"] + sorted(mapped["semantic_topic"].fillna("other").astype(str).unique().tolist())
            policy_options = ["All"] + sorted(mapped["semantic_policy_bucket"].fillna("other").astype(str).unique().tolist())
            stance_options = ["All"] + sorted(mapped["semantic_stance"].fillna("unknown").astype(str).unique().tolist())
            asset_targets = sorted(
                {
                    asset
                    for value in mapped["semantic_asset_targets"].fillna("")
                    for asset in value.split(",")
                    if asset
                }
                | {
                    str(value).upper()
                    for value in mapped["semantic_primary_asset"].fillna("")
                    if str(value).strip()
                },
            )
            platform_options = sorted(mapped["source_platform"].dropna().astype(str).unique().tolist())
            bucket_field_options = {
                "Topic": "semantic_topic",
                "Policy bucket": "semantic_policy_bucket",
                "Stance": "semantic_stance",
                "Primary asset": "semantic_primary_asset",
            }

            narrative_controls_top = st.columns(4)
            selected_topic = narrative_controls_top[0].selectbox("Topic", options=topic_options, key="narrative_topic")
            selected_policy = narrative_controls_top[1].selectbox("Policy bucket", options=policy_options, key="narrative_policy")
            selected_stance = narrative_controls_top[2].selectbox("Stance", options=stance_options, key="narrative_stance")
            selected_bucket_label = narrative_controls_top[3].selectbox(
                "Return bucket",
                options=list(bucket_field_options.keys()),
                key="narrative_bucket_field",
            )

            narrative_controls_bottom = st.columns(4)
            selected_urgency = narrative_controls_bottom[0].selectbox(
                "Urgency band",
                options=["All", "low", "medium", "high"],
                key="narrative_urgency",
            )
            selected_narrative_asset = narrative_controls_bottom[1].selectbox(
                "Primary asset / target",
                options=["All"] + asset_targets,
                key="narrative_asset_target",
            )
            selected_narrative_platforms = narrative_controls_bottom[2].multiselect(
                "Platforms",
                options=platform_options,
                default=platform_options,
                key="narrative_platforms",
            )
            tracked_scope = narrative_controls_bottom[3].selectbox(
                "Tracked scope",
                options=["All posts", "Trump + tracked accounts", "Tracked accounts only"],
                key="narrative_tracked_scope",
            )

            filtered_narratives = filter_narrative_rows(
                mapped,
                topic=selected_topic,
                policy_bucket=selected_policy,
                stance=selected_stance,
                urgency_band=selected_urgency,
                narrative_asset=selected_narrative_asset,
                platforms=selected_narrative_platforms,
                tracked_scope=tracked_scope,
            )
            filtered_narrative_assets = filter_narrative_rows(
                narrative_asset_view,
                topic=selected_topic,
                policy_bucket=selected_policy,
                stance=selected_stance,
                urgency_band=selected_urgency,
                narrative_asset=selected_narrative_asset,
                platforms=selected_narrative_platforms,
                tracked_scope=tracked_scope,
            )

            provider_summary = (
                mapped.groupby("semantic_provider", as_index=False)
                .agg(
                    posts=("post_id", "size"),
                    cache_hit_rate=("semantic_cache_hit", "mean"),
                    avg_market_relevance=("semantic_market_relevance", "mean"),
                )
                .sort_values("posts", ascending=False)
            )
            narrative_metrics = st.columns(4)
            narrative_metrics[0].metric("Narrative-tagged posts", f"{len(filtered_narratives):,}")
            narrative_metrics[1].metric(
                "Narrative sessions",
                f"{filtered_narratives['session_date'].nunique():,}" if "session_date" in filtered_narratives.columns else "0",
            )
            narrative_metrics[2].metric(
                "Cache hit rate",
                f"{mapped['semantic_cache_hit'].astype(bool).mean():.0%}" if "semantic_cache_hit" in mapped.columns and not mapped.empty else "0%",
            )
            narrative_metrics[3].metric("Providers used", f"{provider_summary['semantic_provider'].nunique():,}")

            with st.expander("Provider, fallback, and cache indicators", expanded=False):
                st.dataframe(provider_summary, use_container_width=True, hide_index=True)

            if filtered_narratives.empty:
                st.info("No narrative rows match the current Narrative Lab filters.")
            else:
                frequency = build_narrative_frequency_frame(filtered_narratives)
                returns = build_narrative_return_frame(
                    filtered_narratives,
                    market,
                    bucket_field=bucket_field_options[selected_bucket_label],
                )
                heatmap = build_narrative_asset_heatmap_frame(filtered_narrative_assets)
                narrative_chart_left, narrative_chart_right = st.columns(2)
                with narrative_chart_left:
                    st.plotly_chart(build_narrative_frequency_chart(frequency), use_container_width=True)
                with narrative_chart_right:
                    st.plotly_chart(
                        build_narrative_return_chart(returns, bucket_field_options[selected_bucket_label]),
                        use_container_width=True,
                    )

                if heatmap.empty:
                    st.info("No asset-specific narrative mappings are available for the current filters.")
                else:
                    st.plotly_chart(build_narrative_asset_heatmap_chart(heatmap), use_container_width=True)

                narrative_posts_table = make_narrative_post_table(filtered_narratives)
                narrative_events_table = make_narrative_event_table(filtered_narratives, market)
                detail_left, detail_right = st.columns(2)
                with detail_left:
                    st.markdown("**Top posts for the selected narrative slice**")
                    st.dataframe(narrative_posts_table.head(25), use_container_width=True, hide_index=True)
                with detail_right:
                    st.markdown("**Selected-narrative event table**")
                    st.dataframe(narrative_events_table, use_container_width=True, hide_index=True)

    with asset_tab:
        st.caption("Compare `SPY` with one tracked stock or ETF across overlays, event studies, and intraday reaction windows.")
        if asset_universe.empty or asset_daily.empty:
            st.info(_refresh_required_message(settings, "Refresh datasets first to populate the tracked asset universe and daily asset market data."))
        else:
            comparison_candidates = asset_universe.loc[asset_universe["symbol"].astype(str).str.upper() != "SPY"].copy()
            if comparison_candidates.empty:
                st.info("Add at least one non-`SPY` asset to the tracked universe to enable asset comparison.")
            else:
                comparison_candidates["symbol"] = comparison_candidates["symbol"].astype(str).str.upper()
                comparison_candidates["display_name"] = comparison_candidates["display_name"].fillna(comparison_candidates["symbol"]).astype(str)
                candidate_symbols = comparison_candidates["symbol"].tolist()
                candidate_labels = {
                    row["symbol"]: f"{row['symbol']} - {row['display_name']}"
                    for _, row in comparison_candidates.iterrows()
                }
                watchlist_symbols = comparison_candidates.loc[
                    comparison_candidates["source"].astype(str) == "watchlist",
                    "symbol",
                ].tolist()
                default_asset = watchlist_symbols[0] if watchlist_symbols else candidate_symbols[0]

                asset_controls = st.columns([1.5, 1.1, 1.2])
                selected_asset = asset_controls[0].selectbox(
                    "Selected asset",
                    options=candidate_symbols,
                    index=candidate_symbols.index(default_asset),
                    format_func=lambda symbol: candidate_labels.get(symbol, symbol),
                    key="research_selected_asset",
                )
                comparison_mode = asset_controls[1].radio(
                    "View",
                    options=["price", "normalized"],
                    format_func=lambda value: "Price overlay" if value == "price" else "Normalized returns",
                    horizontal=True,
                    key="research_asset_comparison_mode",
                )
                benchmark_options = ["None"] + [symbol for symbol in DEFAULT_ETF_SYMBOLS if symbol not in {"SPY", selected_asset}]
                benchmark_choice = asset_controls[2].selectbox(
                    "ETF baseline",
                    options=benchmark_options,
                    key="research_asset_benchmark",
                )
                selected_benchmark = None if benchmark_choice == "None" else benchmark_choice

                asset_feature_view = asset_session_features.copy()
                if not asset_feature_view.empty and "signal_session_date" in asset_feature_view.columns:
                    asset_feature_view["signal_session_date"] = pd.to_datetime(asset_feature_view["signal_session_date"], errors="coerce").dt.normalize()
                    asset_feature_view = asset_feature_view.loc[
                        (asset_feature_view["signal_session_date"] >= start_date.normalize())
                        & (asset_feature_view["signal_session_date"] <= end_date.normalize())
                    ].copy()
                asset_mapping_view = asset_post_mappings.copy()
                if not asset_mapping_view.empty and "session_date" in asset_mapping_view.columns:
                    asset_mapping_view["session_date"] = pd.to_datetime(asset_mapping_view["session_date"], errors="coerce").dt.normalize()
                    asset_mapping_view = asset_mapping_view.loc[
                        (asset_mapping_view["session_date"] >= start_date.normalize())
                        & (asset_mapping_view["session_date"] <= end_date.normalize())
                    ].copy()
                selected_asset_raw_mappings = asset_mapping_view.loc[
                    asset_mapping_view["asset_symbol"].astype(str).str.upper() == selected_asset
                ].copy() if not asset_mapping_view.empty and "asset_symbol" in asset_mapping_view.columns else pd.DataFrame()
                if not selected_asset_raw_mappings.empty and "reaction_anchor_ts" not in selected_asset_raw_mappings.columns:
                    selected_asset_raw_mappings["reaction_anchor_ts"] = selected_asset_raw_mappings["post_timestamp"]

                overview_tab, event_study_tab, intraday_tab = st.tabs(["Overview", "Event Study", "Intraday Reaction"])

                with overview_tab:
                    comparison = build_asset_comparison_frame(
                        asset_market=asset_daily,
                        selected_symbol=selected_asset,
                        benchmark_symbol=selected_benchmark,
                        date_start=start_date,
                        date_end=end_date,
                    )
                    if comparison.empty:
                        st.info(f"No overlapping daily market data was found for `SPY` and `{selected_asset}` in the current date range.")
                    else:
                        asset_metric_cols = st.columns(4)
                        asset_metric_cols[0].metric("Sessions in range", f"{len(comparison):,}")
                        asset_metric_cols[1].metric("SPY move", f"{comparison['spy_normalized_return'].iloc[-1]:+.2%}")
                        asset_metric_cols[2].metric(f"{selected_asset} move", f"{comparison['asset_normalized_return'].iloc[-1]:+.2%}")
                        asset_metric_cols[3].metric(
                            f"{selected_asset} vs SPY spread",
                            f"{(comparison['asset_normalized_return'].iloc[-1] - comparison['spy_normalized_return'].iloc[-1]):+.2%}",
                        )
                        st.plotly_chart(
                            build_asset_comparison_chart(comparison, selected_asset, comparison_mode),
                            use_container_width=True,
                        )

                    asset_session_table = make_asset_session_table(asset_feature_view, selected_asset)
                    if not asset_session_table.empty:
                        st.markdown("**Per-asset session summary**")
                        st.dataframe(
                            asset_session_table.sort_values(["post_count", "trade_date"], ascending=[False, False]),
                            use_container_width=True,
                            hide_index=True,
                        )

                    asset_mapping_table = make_asset_mapping_table(asset_mapping_view, selected_asset)
                    if asset_mapping_table.empty:
                        st.info(f"No mapped posts for `{selected_asset}` were found in the current date range.")
                    else:
                        session_options = sorted(asset_mapping_table["session_date"].dropna().unique().tolist(), reverse=True)
                        selected_asset_session = st.selectbox(
                            "Matched session",
                            options=session_options,
                            format_func=lambda value: value.isoformat(),
                            key="research_asset_mapping_session",
                        )
                        session_mapping_table = asset_mapping_table.loc[
                            asset_mapping_table["session_date"] == selected_asset_session
                        ].copy()
                        st.markdown(f"**Matched posts for {selected_asset} on {selected_asset_session.isoformat()}**")
                        st.dataframe(session_mapping_table, use_container_width=True, hide_index=True)

                with event_study_tab:
                    study_controls = st.columns(2)
                    pre_sessions = study_controls[0].slider(
                        "Sessions before event",
                        min_value=1,
                        max_value=10,
                        value=3,
                        step=1,
                        key="research_event_study_pre",
                    )
                    post_sessions = study_controls[1].slider(
                        "Sessions after event",
                        min_value=1,
                        max_value=10,
                        value=5,
                        step=1,
                        key="research_event_study_post",
                    )
                    event_study = build_event_study_frame(
                        asset_market=asset_daily,
                        asset_session_features=asset_feature_view,
                        selected_symbol=selected_asset,
                        benchmark_symbol=selected_benchmark,
                        pre_sessions=pre_sessions,
                        post_sessions=post_sessions,
                    )
                    if event_study.empty:
                        st.info(f"No asset-specific event-study rows are available for `{selected_asset}` in the current range.")
                    else:
                        st.plotly_chart(
                            build_event_study_chart(event_study, selected_asset),
                            use_container_width=True,
                        )
                        pivot = (
                            event_study.pivot(index="relative_session", columns="symbol", values="avg_relative_return")
                            .reset_index()
                            .sort_values("relative_session")
                        )
                        st.markdown("**Average return by relative session**")
                        st.dataframe(pivot, use_container_width=True, hide_index=True)

                with intraday_tab:
                    st.caption("Uses the stored recent `asset_intraday` dataset, so older sessions may not have intraday coverage.")
                    if asset_intraday.empty:
                        st.info(_refresh_required_message(settings, "Refresh datasets first to populate recent multi-asset intraday data."))
                    elif selected_asset_raw_mappings.empty:
                        st.info(f"No mapped posts for `{selected_asset}` are available to anchor an intraday comparison.")
                    else:
                        intraday_required_symbols = {"SPY", selected_asset}
                        if selected_benchmark:
                            intraday_required_symbols.add(selected_benchmark)
                        intraday_view = asset_intraday.copy()
                        intraday_view["symbol"] = intraday_view["symbol"].astype(str).str.upper()
                        intraday_view["timestamp"] = pd.to_datetime(intraday_view["timestamp"], errors="coerce")
                        intraday_dates = {
                            symbol: set(
                                intraday_view.loc[intraday_view["symbol"] == symbol, "timestamp"]
                                .dropna()
                                .dt.tz_convert(settings.timezone)
                                .dt.date
                                .tolist()
                            )
                            for symbol in intraday_required_symbols
                        }
                        intraday_mapping_candidates = selected_asset_raw_mappings.copy()
                        intraday_mapping_candidates["session_date"] = pd.to_datetime(
                            intraday_mapping_candidates["session_date"],
                            errors="coerce",
                        ).dt.date
                        eligible_intraday_sessions = [
                            session_date
                            for session_date in sorted(intraday_mapping_candidates["session_date"].dropna().unique().tolist(), reverse=True)
                            if all(session_date in intraday_dates.get(symbol, set()) for symbol in intraday_required_symbols)
                        ]
                        if not eligible_intraday_sessions:
                            st.info(
                                f"No recent intraday coverage overlaps the mapped `{selected_asset}` sessions for the required symbols "
                                f"({', '.join(sorted(intraday_required_symbols))}).",
                            )
                        else:
                            intraday_controls = st.columns(4)
                            selected_intraday_session = intraday_controls[0].selectbox(
                                "Intraday session",
                                options=eligible_intraday_sessions,
                                format_func=lambda value: value.isoformat(),
                                key="research_intraday_asset_session",
                            )
                            session_intraday_posts = intraday_mapping_candidates.loc[
                                intraday_mapping_candidates["session_date"] == selected_intraday_session
                            ].sort_values("post_timestamp")
                            post_labels = [
                                f"{row['post_timestamp'].tz_convert(settings.timezone):%Y-%m-%d %H:%M} ET | @{row['author_handle'] or row['author_display_name']} | {truncate(row['cleaned_text'])}"
                                for _, row in session_intraday_posts.iterrows()
                            ]
                            selected_intraday_label = intraday_controls[1].selectbox(
                                "Anchor post",
                                options=post_labels,
                                key="research_intraday_asset_post",
                            )
                            before_minutes = intraday_controls[2].selectbox(
                                "Minutes before",
                                options=[30, 60, 120, 180, 240],
                                index=2,
                                key="research_intraday_asset_before",
                            )
                            after_minutes = intraday_controls[3].selectbox(
                                "Minutes after",
                                options=[60, 120, 240, 390],
                                index=2,
                                key="research_intraday_asset_after",
                            )
                            selected_intraday_post = session_intraday_posts.iloc[post_labels.index(selected_intraday_label)]
                            anchor_ts = pd.to_datetime(
                                selected_intraday_post.get("reaction_anchor_ts", selected_intraday_post["post_timestamp"]),
                                errors="coerce",
                            )
                            if pd.isna(anchor_ts):
                                anchor_ts = selected_intraday_post["post_timestamp"]
                            if getattr(anchor_ts, "tzinfo", None) is None:
                                anchor_ts = anchor_ts.tz_localize(settings.timezone)
                            else:
                                anchor_ts = anchor_ts.tz_convert(settings.timezone)
                            intraday_comparison = build_intraday_comparison_frame(
                                intraday_frame=intraday_view,
                                selected_symbol=selected_asset,
                                anchor_ts=anchor_ts,
                                before_minutes=int(before_minutes),
                                after_minutes=int(after_minutes),
                                benchmark_symbol=selected_benchmark,
                            )
                            if intraday_comparison.empty:
                                st.warning("No intraday comparison rows were returned for the selected window.")
                            else:
                                st.plotly_chart(
                                    build_intraday_comparison_chart(intraday_comparison, selected_asset, anchor_ts),
                                    use_container_width=True,
                                )
                                coverage = (
                                    intraday_comparison.groupby("symbol", as_index=False)
                                    .agg(
                                        bars=("timestamp", "size"),
                                        first_timestamp=("timestamp", "min"),
                                        last_timestamp=("timestamp", "max"),
                                    )
                                )
                                st.markdown("**Intraday coverage**")
                                st.dataframe(coverage, use_container_width=True, hide_index=True)

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


def _portfolio_decision_option_label(row: pd.Series) -> str:
    session_date = _normalize_session_date(row.get("signal_session_date"))
    date_label = f"{session_date:%Y-%m-%d}" if session_date is not None else "unknown session"
    winning_asset = str(row.get("winning_asset", "") or "FLAT")
    winner_score = float(row.get("winner_score", 0.0) or 0.0)
    runner_up = str(row.get("runner_up_asset", "") or "n/a")
    return f"{date_label} | winner {winning_asset} | score {winner_score:+.3%} | runner-up {runner_up}"


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


def _render_portfolio_session_panel(
    decision_row: pd.Series,
    session_candidates: pd.DataFrame,
) -> None:
    session_date = _normalize_session_date(decision_row.get("signal_session_date"))
    winning_asset = str(decision_row.get("winning_asset", "") or "FLAT")
    winner_score = float(decision_row.get("winner_score", 0.0) or 0.0)
    runner_up = str(decision_row.get("runner_up_asset", "") or "n/a")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Signal session", f"{session_date:%Y-%m-%d}" if session_date is not None else "n/a")
    metric_cols[1].metric("Winner", winning_asset)
    metric_cols[2].metric("Decision source", str(decision_row.get("decision_source", "n/a")).upper())
    metric_cols[3].metric("Winner score", f"{winner_score:+.3%}")
    metric_cols[4].metric("Runner-up", runner_up)

    if session_candidates.empty:
        st.info("No candidate rows were available for this portfolio session.")
        return

    display = session_candidates.copy()
    display = display.sort_values(["is_winner", "qualifies", "expected_return_score"], ascending=[False, False, False])
    display["threshold_gap"] = pd.to_numeric(display["expected_return_score"], errors="coerce") - pd.to_numeric(display["threshold"], errors="coerce")
    keep = [
        "asset_symbol",
        "run_name",
        "expected_return_score",
        "confidence",
        "threshold",
        "threshold_gap",
        "post_count",
        "tradeable",
        "signal_qualifies",
        "qualifies",
        "eligible_rank",
        "is_winner",
        "decision_source",
        "stance",
    ]
    available = [column for column in keep if column in display.columns]
    st.dataframe(
        display[available].rename(
            columns={
                "asset_symbol": "Asset",
                "run_name": "Run",
                "expected_return_score": "Score",
                "confidence": "Confidence",
                "threshold": "Threshold",
                "threshold_gap": "Threshold gap",
                "post_count": "Posts",
                "tradeable": "Tradeable",
                "signal_qualifies": "Signal rules",
                "qualifies": "Eligible",
                "eligible_rank": "Eligible rank",
                "is_winner": "Winner",
                "decision_source": "Decision source",
                "stance": "Stance",
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
    run_meta = run_bundle.get("run", {}) or {}
    portfolio_bundle = run_bundle.get("portfolio_model_bundle", {}) or {}
    deployment_variant = str(
        selected.get("deployment_variant")
        or config.get("deployment_variant")
        or run_meta.get("deployment_variant")
        or portfolio_bundle.get("deployment_variant")
        or "",
    )
    variant_payload = (portfolio_bundle.get("variants", {}) or {}).get(deployment_variant, {})
    return {
        "run_type": str(run_meta.get("run_type") or getattr(artifact, "metadata", {}).get("run_type", "asset_model")),
        "allocator_mode": str(
            config.get("allocator_mode")
            or run_meta.get("allocator_mode")
            or getattr(artifact, "metadata", {}).get("allocator_mode", "")
        ),
        "target_asset": str(config.get("target_asset") or run_meta.get("target_asset") or getattr(artifact, "metadata", {}).get("target_asset", "SPY")),
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
        "fallback_mode": str(
            config.get("fallback_mode")
            or run_meta.get("fallback_mode")
            or selected.get("fallback_mode")
            or "",
        ).upper(),
        "component_run_ids": tuple(selected.get("component_run_ids", config.get("component_run_ids", run_meta.get("component_run_ids", [])))),
        "universe_symbols": tuple(selected.get("universe_symbols", config.get("universe_symbols", run_meta.get("universe_symbols", [])))),
        "selected_symbols": tuple(
            selected.get(
                "selected_symbols",
                config.get("selected_symbols", run_meta.get("selected_symbols", [])),
            ),
        ),
        "deployment_variant": deployment_variant,
        "deployment_topology": str(
            selected.get("deployment_topology")
            or variant_payload.get("topology")
            or deployment_variant
            or "",
        ),
        "deployment_narrative_feature_mode": str(
            selected.get("deployment_narrative_feature_mode")
            or variant_payload.get("narrative_feature_mode")
            or "unspecified",
        ),
        "topology_variants": tuple(config.get("topology_variants", run_meta.get("topology_variants", []))),
        "narrative_feature_modes": tuple(
            config.get(
                "narrative_feature_modes",
                run_meta.get("narrative_feature_modes", portfolio_bundle.get("narrative_feature_modes", [])),
            ),
        ),
        "model_families": tuple(config.get("model_families", run_meta.get("model_families", []))),
        "deploy_threshold": selected.get("threshold"),
        "deploy_min_post_count": selected.get("min_post_count"),
        "deploy_account_weight": selected.get("account_weight"),
    }


def _bundle_feature_names(run_bundle: dict[str, Any], variant_name: str | None = None) -> list[str]:
    run_type = _comparison_settings(run_bundle).get("run_type", "asset_model")
    if run_type == "asset_model":
        return list(run_bundle.get("model_artifact").feature_names)

    portfolio_bundle = run_bundle.get("portfolio_model_bundle", {}) or {}
    deployment_variant = str(
        variant_name
        or portfolio_bundle.get("deployment_variant")
        or _comparison_settings(run_bundle).get("deployment_variant")
        or "",
    )
    variant_payload = (portfolio_bundle.get("variants", {}) or {}).get(deployment_variant, {})
    models_payload = variant_payload.get("models", {}) or {}
    feature_names: set[str] = set()
    for artifact_payload in models_payload.values():
        if not isinstance(artifact_payload, dict):
            continue
        feature_names.update(str(name) for name in artifact_payload.get("feature_names", []) if name)
    return sorted(feature_names)


def _build_metric_comparison_table(base_run_id: str, run_bundles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if base_run_id not in run_bundles:
        return pd.DataFrame()
    base_metrics = run_bundles[base_run_id].get("metrics", {}) or {}
    rows: list[dict[str, Any]] = []
    for run_id, bundle in run_bundles.items():
        metrics = bundle.get("metrics", {}) or {}
        feature_names = _bundle_feature_names(bundle)
        rows.append(
            {
                "run_id": run_id,
                "run_name": bundle.get("run", {}).get("run_name", run_id),
                "run_type": _comparison_settings(bundle).get("run_type", "asset_model"),
                "allocator_mode": _comparison_settings(bundle).get("allocator_mode", ""),
                "target_asset": _comparison_settings(bundle).get("target_asset", "SPY"),
                "deployment_variant": _comparison_settings(bundle).get("deployment_variant", ""),
                "deployment_narrative_feature_mode": _comparison_settings(bundle).get("deployment_narrative_feature_mode", ""),
                "total_return": metrics.get("total_return", 0.0),
                "sharpe": metrics.get("sharpe", 0.0),
                "sortino": metrics.get("sortino", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "robust_score": metrics.get("robust_score", 0.0),
                "trade_count": metrics.get("trade_count", 0.0),
                "feature_count": len(feature_names),
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
    base_features = set(_bundle_feature_names(run_bundles[base_run_id]))
    rows: list[dict[str, Any]] = []
    for run_id, bundle in run_bundles.items():
        feature_names = _bundle_feature_names(bundle)
        families = Counter(classify_feature_family(feature_name) for feature_name in feature_names)
        features = set(feature_names)
        unique_vs_base = sorted(features - base_features)
        omitted_vs_base = sorted(base_features - features)
        rows.append(
            {
                "run_id": run_id,
                "run_name": bundle.get("run", {}).get("run_name", run_id),
                "target_asset": _comparison_settings(bundle).get("target_asset", "SPY"),
                "deployment_variant": _comparison_settings(bundle).get("deployment_variant", ""),
                "deployment_narrative_feature_mode": _comparison_settings(bundle).get("deployment_narrative_feature_mode", ""),
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


def _variant_summary_with_narrative_defaults(variant_summary: pd.DataFrame) -> pd.DataFrame:
    normalized = variant_summary.copy()
    if normalized.empty:
        return normalized
    if "variant_name" not in normalized.columns:
        normalized["variant_name"] = ""
    if "topology" not in normalized.columns:
        normalized["topology"] = normalized["variant_name"].astype(str)
    if "narrative_feature_mode" not in normalized.columns:
        normalized["narrative_feature_mode"] = "unspecified"
    normalized["topology"] = normalized["topology"].replace("", pd.NA).fillna(normalized["variant_name"].astype(str))
    normalized["narrative_feature_mode"] = normalized["narrative_feature_mode"].replace("", "unspecified").fillna("unspecified")
    return normalized


def _build_narrative_lift_table(variant_summary: pd.DataFrame) -> pd.DataFrame:
    summary = _variant_summary_with_narrative_defaults(variant_summary)
    if summary.empty or "baseline" not in set(summary["narrative_feature_mode"].astype(str)):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    metric_pairs = [
        ("validation_robust_score", "validation_robust_lift"),
        ("validation_total_return", "validation_return_lift"),
        ("test_robust_score", "test_robust_lift"),
        ("test_total_return", "test_return_lift"),
    ]
    baseline_rows = summary.loc[summary["narrative_feature_mode"].astype(str) == "baseline"].copy()
    for _, row in summary.iterrows():
        mode = str(row.get("narrative_feature_mode", "unspecified") or "unspecified")
        if mode in {"baseline", "unspecified"}:
            continue
        topology = str(row.get("topology", "") or "")
        baseline = baseline_rows.loc[baseline_rows["topology"].astype(str) == topology]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        lift_row = {
            "variant_name": row.get("variant_name", ""),
            "topology": topology,
            "narrative_feature_mode": mode,
            "baseline_variant": baseline_row.get("variant_name", ""),
        }
        for metric_column, output_column in metric_pairs:
            current_value = pd.to_numeric(pd.Series([row.get(metric_column)]), errors="coerce").iloc[0]
            baseline_value = pd.to_numeric(pd.Series([baseline_row.get(metric_column)]), errors="coerce").iloc[0]
            lift_row[output_column] = (
                float(current_value - baseline_value)
                if pd.notna(current_value) and pd.notna(baseline_value)
                else np.nan
            )
        rows.append(lift_row)
    return pd.DataFrame(rows).sort_values("validation_robust_lift", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def _build_feature_family_summary(
    run_bundle: dict[str, Any],
    variant_name: str | None = None,
    importance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    importance_frame = pd.DataFrame() if importance is None else importance.copy()
    if not importance_frame.empty and "feature_name" in importance_frame.columns:
        if variant_name and "variant_name" in importance_frame.columns:
            filtered = importance_frame.loc[importance_frame["variant_name"].astype(str) == variant_name].copy()
            if not filtered.empty:
                importance_frame = filtered
        feature_names = importance_frame["feature_name"].dropna().astype(str).tolist()
        importance_frame["feature_family"] = importance_frame["feature_name"].astype(str).map(classify_feature_family)
        value_column = "importance" if "importance" in importance_frame.columns else "abs_coefficient"
        if value_column not in importance_frame.columns:
            value_column = None
        rows: list[dict[str, Any]] = []
        for family, family_rows in importance_frame.groupby("feature_family", sort=True):
            top_features = family_rows.copy()
            if value_column is not None:
                top_features[value_column] = pd.to_numeric(top_features[value_column], errors="coerce").fillna(0.0)
                top_features = top_features.sort_values(value_column, ascending=False)
            rows.append(
                {
                    "feature_family": family,
                    "feature_count": int(family_rows["feature_name"].nunique()),
                    "total_importance": float(pd.to_numeric(family_rows[value_column], errors="coerce").fillna(0.0).sum()) if value_column else np.nan,
                    "top_features": ", ".join(top_features["feature_name"].dropna().astype(str).head(5).tolist()),
                },
            )
        return pd.DataFrame(rows).sort_values(["total_importance", "feature_count"], ascending=[False, False]).reset_index(drop=True)

    feature_names = _bundle_feature_names(run_bundle, variant_name=variant_name)
    families = Counter(classify_feature_family(feature_name) for feature_name in feature_names)
    rows = [
        {
            "feature_family": family,
            "feature_count": int(count),
            "total_importance": np.nan,
            "top_features": ", ".join(
                feature_name for feature_name in feature_names if classify_feature_family(feature_name) == family
            ),
        }
        for family, count in families.items()
    ]
    return pd.DataFrame(rows).sort_values("feature_count", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


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
        target_asset = _comparison_settings(bundle).get("target_asset", "SPY")
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
                    "target_asset": target_asset,
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
    base_features = set(_bundle_feature_names(run_bundles[base_run_id]))
    notes: list[str] = []
    for run_id, bundle in run_bundles.items():
        if run_id == base_run_id:
            continue
        metrics = bundle.get("metrics", {}) or {}
        settings = _comparison_settings(bundle)
        features = set(_bundle_feature_names(bundle))
        parts: list[str] = [
            f"robust score {metrics.get('robust_score', 0.0) - base_metrics.get('robust_score', 0.0):+.3f}",
            f"total return {metrics.get('total_return', 0.0) - base_metrics.get('total_return', 0.0):+.2%}",
        ]
        if settings.get("llm_enabled") != base_settings.get("llm_enabled"):
            parts.append(f"LLM {'on' if settings.get('llm_enabled') else 'off'} vs {'on' if base_settings.get('llm_enabled') else 'off'}")
        if settings.get("run_type") != base_settings.get("run_type"):
            parts.append(f"run type {base_settings.get('run_type')} -> {settings.get('run_type')}")
        if settings.get("allocator_mode") != base_settings.get("allocator_mode"):
            parts.append(f"allocator {base_settings.get('allocator_mode') or 'n/a'} -> {settings.get('allocator_mode') or 'n/a'}")
        if settings.get("target_asset") != base_settings.get("target_asset"):
            parts.append(f"target asset {base_settings.get('target_asset')} -> {settings.get('target_asset')}")
        if settings.get("fallback_mode") != base_settings.get("fallback_mode"):
            parts.append(f"fallback {base_settings.get('fallback_mode') or 'n/a'} -> {settings.get('fallback_mode') or 'n/a'}")
        if settings.get("deploy_threshold") != base_settings.get("deploy_threshold"):
            parts.append(f"threshold {base_settings.get('deploy_threshold')} -> {settings.get('deploy_threshold')}")
        if settings.get("deploy_min_post_count") != base_settings.get("deploy_min_post_count"):
            parts.append(f"min posts {base_settings.get('deploy_min_post_count')} -> {settings.get('deploy_min_post_count')}")
        if settings.get("deploy_account_weight") != base_settings.get("deploy_account_weight"):
            parts.append(f"account weight {base_settings.get('deploy_account_weight')} -> {settings.get('deploy_account_weight')}")
        if settings.get("deployment_variant") != base_settings.get("deployment_variant"):
            parts.append(f"deployment variant {base_settings.get('deployment_variant') or 'n/a'} -> {settings.get('deployment_variant') or 'n/a'}")
        if settings.get("deployment_narrative_feature_mode") != base_settings.get("deployment_narrative_feature_mode"):
            parts.append(
                "narrative mode "
                f"{base_settings.get('deployment_narrative_feature_mode') or 'n/a'} -> "
                f"{settings.get('deployment_narrative_feature_mode') or 'n/a'}",
            )
        unique_vs_base = sorted(features - base_features)
        omitted_vs_base = sorted(base_features - features)
        if unique_vs_base:
            parts.append(f"{len(unique_vs_base)} added features ({', '.join(unique_vs_base[:3])})")
        if omitted_vs_base:
            parts.append(f"{len(omitted_vs_base)} removed features ({', '.join(omitted_vs_base[:3])})")
        notes.append(f"`{run_id}`: " + "; ".join(parts) + ".")
    return notes


def render_models_view(
    settings: AppSettings,
    store: DuckDBStore,
    feature_service: FeatureService,
    model_service: ModelService,
    backtest_service: BacktestService,
    experiment_store: ExperimentStore,
) -> None:
    st.subheader("Models & Backtests")
    st.caption(
        "Build the session dataset, train a next-session expected-return model for SPY or a selected asset, "
        "compare saved runs, and inspect benchmark plus leakage diagnostics.",
    )
    can_write = _writes_enabled(settings)
    if is_public_mode(settings) and not can_write:
        st.info("Run creation is disabled in public read-only mode. Saved runs remain viewable.")
    posts = store.read_frame("normalized_posts")
    spy = store.read_frame("spy_daily")
    tracked_accounts = store.read_frame("tracked_accounts")
    if posts.empty or spy.empty:
        st.info(_refresh_required_message(settings, "Refresh datasets first so the modeling pipeline has normalized posts and SPY market data."))
        return

    target_asset_options = _target_asset_options(store)
    control_cols = st.columns(5)
    run_name = control_cols[0].text_input("Run name", value="baseline-research-run")
    selected_target_asset = control_cols[1].selectbox(
        "Target asset",
        options=target_asset_options,
        index=0,
        format_func=lambda symbol: _target_asset_label(store, symbol),
    )
    llm_enabled = control_cols[2].checkbox("Enable semantic enrichment", value=False)
    train_window = control_cols[3].number_input("Train window", min_value=20, max_value=252, value=90, step=5)
    validation_window = control_cols[4].number_input("Validation window", min_value=10, max_value=126, value=30, step=5)
    control_cols2 = st.columns(4)
    test_window = control_cols2[0].number_input("Test window", min_value=10, max_value=126, value=30, step=5)
    step_size = control_cols2[1].number_input("Step size", min_value=5, max_value=126, value=30, step=5)
    transaction_cost_bps = control_cols2[2].number_input("Round-trip cost (bps per side)", min_value=0.0, max_value=25.0, value=2.0, step=0.5)
    ridge_alpha = control_cols2[3].number_input("Ridge alpha", min_value=0.0, max_value=25.0, value=1.0, step=0.5)
    threshold_text = st.text_input("Threshold grid", value="0,0.001,0.0025,0.005")
    min_posts_text = st.text_input("Minimum signal post-count grid", value="1,2,3")
    account_weight_text = st.text_input("Tracked-account weight grid", value="0.5,1.0,1.5")

    if st.button("Build dataset and run walk-forward backtest", use_container_width=True, disabled=not can_write):
        with st.spinner("Building session features and running walk-forward optimization..."):
            normalized_target_asset = str(selected_target_asset).upper()
            target_feature_version = "v1" if normalized_target_asset == "SPY" else "asset-v1"
            try:
                feature_rows, attribution_posts = _build_model_target_bundle(
                    store=store,
                    feature_service=feature_service,
                    posts=posts,
                    spy_market=spy,
                    tracked_accounts=tracked_accounts,
                    llm_enabled=llm_enabled,
                    target_asset=normalized_target_asset,
                    feature_version=target_feature_version,
                )
                store.save_frame(
                    "session_features_latest",
                    feature_rows,
                    metadata={
                        "llm_enabled": llm_enabled,
                        "row_count": int(len(feature_rows)),
                        "target_asset": normalized_target_asset,
                        "feature_version": target_feature_version,
                    },
                )
                config = ModelRunConfig(
                    run_name=run_name,
                    target_asset=normalized_target_asset,
                    feature_version=target_feature_version,
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
                post_attribution = build_post_attribution(attribution_posts)
                account_attribution = build_account_attribution(post_attribution)
                predicted_sessions = {
                    session_date
                    for session_date in pd.to_datetime(
                        artifacts["predictions"]["signal_session_date"],
                        errors="coerce",
                    ).dropna().dt.normalize().tolist()
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
            except RuntimeError as exc:
                st.error(str(exc))
                return
        st.success(f"Saved run `{run.run_id}` for `{normalized_target_asset}`.")

    latest_features = store.read_frame("session_features_latest")
    if not latest_features.empty:
        preview = latest_feature_preview(latest_features)
        st.markdown("**Latest feature snapshot**")
        st.json(preview)

    runs = _normalize_runs_frame(experiment_store.list_runs())
    asset_model_runs = runs.loc[runs["run_type"].astype(str) == "asset_model"].copy() if not runs.empty else pd.DataFrame()
    st.markdown("**Portfolio Allocator**")
    portfolio_mode = st.radio(
        "Portfolio workflow",
        options=["Saved Runs", "Joint Portfolio Model"],
        horizontal=True,
        key="portfolio-workflow-mode",
    )
    if portfolio_mode == "Saved Runs":
        st.caption(
            "Build a one-asset-per-session historical allocator from saved SPY and non-SPY runs. "
            "It reuses each component run's own deployment thresholds and post-count rules.",
        )
        if asset_model_runs.empty or asset_model_runs.loc[asset_model_runs["target_asset"] == "SPY"].empty:
            st.info("Save at least one SPY asset-model run before building a portfolio allocator.")
        else:
            portfolio_cols = st.columns(3)
            portfolio_run_name = portfolio_cols[0].text_input("Portfolio run name", value="portfolio-allocator-run")
            fallback_mode = portfolio_cols[1].selectbox("Fallback mode", options=["SPY", "FLAT"], index=0)
            allocator_transaction_cost = portfolio_cols[2].number_input(
                "Allocator round-trip cost (bps per side)",
                min_value=0.0,
                max_value=25.0,
                value=2.0,
                step=0.5,
            )
            spy_run_rows = asset_model_runs.loc[asset_model_runs["target_asset"] == "SPY"].copy()
            selected_spy_run_id = st.selectbox(
                "Pinned SPY run",
                options=spy_run_rows["run_id"].tolist(),
                format_func=lambda run_id: _live_run_option_label(
                    spy_run_rows.loc[spy_run_rows["run_id"] == run_id].iloc[0],
                ),
                key="portfolio-spy-run",
            )
            non_spy_assets = sorted(
                asset_model_runs.loc[asset_model_runs["target_asset"] != "SPY", "target_asset"].astype(str).unique().tolist(),
            )
            selected_assets = st.multiselect(
                "Additional assets",
                options=non_spy_assets,
                default=non_spy_assets[:2],
                format_func=lambda symbol: _target_asset_label(store, symbol),
                key="portfolio-assets",
            )
            selected_component_ids = [selected_spy_run_id]
            for asset_symbol in selected_assets:
                asset_rows = asset_model_runs.loc[asset_model_runs["target_asset"] == asset_symbol].copy()
                selected_run_id = st.selectbox(
                    f"{asset_symbol} component run",
                    options=asset_rows["run_id"].tolist(),
                    format_func=lambda run_id, asset_rows=asset_rows: _live_run_option_label(
                        asset_rows.loc[asset_rows["run_id"] == run_id].iloc[0],
                    ),
                    key=f"portfolio-component-{asset_symbol}",
                )
                selected_component_ids.append(selected_run_id)

            if st.button("Build saved-run portfolio allocator", use_container_width=True, disabled=not can_write):
                with st.spinner("Aligning component runs and backtesting the shared portfolio allocator..."):
                    component_bundles = {
                        run_id: bundle
                        for run_id in selected_component_ids
                        if (bundle := experiment_store.load_run(run_id)) is not None
                    }
                    portfolio_config = PortfolioRunConfig(
                        run_name=portfolio_run_name,
                        allocator_mode="saved_runs",
                        fallback_mode=str(fallback_mode).upper(),
                        transaction_cost_bps=float(allocator_transaction_cost),
                        component_run_ids=tuple(selected_component_ids),
                        universe_symbols=tuple(["SPY", *selected_assets]),
                    )
                    try:
                        portfolio_run, portfolio_artifacts = backtest_service.run_saved_run_allocator(
                            portfolio_config,
                            component_bundles,
                        )
                        experiment_store.save_portfolio_run(
                            run=portfolio_run,
                            config=portfolio_artifacts["config"],
                            trades=portfolio_artifacts["trades"],
                            decision_history=portfolio_artifacts["predictions"],
                            candidate_predictions=portfolio_artifacts["candidate_predictions"],
                            component_summary=portfolio_artifacts["windows"],
                            benchmarks=portfolio_artifacts["benchmarks"],
                            benchmark_curves=portfolio_artifacts["benchmark_curves"],
                            diagnostics=portfolio_artifacts["diagnostics"],
                            leakage_audit=portfolio_artifacts["leakage_audit"],
                        )
                    except RuntimeError as exc:
                        st.error(str(exc))
                        return
                st.success(f"Saved portfolio allocator run `{portfolio_run.run_id}`.")
    else:
        st.caption(
            "Train a portfolio-owned model suite across a selected asset subset, compare `per_asset` and `pooled` variants, "
            "pick one deployment winner, and save the portfolio run for live use.",
        )
        asset_session_features = store.read_frame("asset_session_features")
        if asset_session_features.empty:
            st.info(_refresh_required_message(settings, "Refresh datasets first so the asset-session feature dataset is available."))
        else:
            available_symbols = sorted(asset_session_features["asset_symbol"].dropna().astype(str).str.upper().unique().tolist())
            if "SPY" in available_symbols:
                available_symbols = ["SPY", *[symbol for symbol in available_symbols if symbol != "SPY"]]
            default_symbols = [symbol for symbol in ["SPY", "QQQ", "NVDA"] if symbol in available_symbols]
            if len(default_symbols) < 2:
                default_symbols = available_symbols[: min(len(available_symbols), 3)]
            joint_cols = st.columns(3)
            joint_run_name = joint_cols[0].text_input(
                "Joint portfolio run name",
                value="joint-portfolio-run",
                key="joint-portfolio-run-name",
            )
            joint_fallback_mode = joint_cols[1].selectbox("Fallback mode", options=["SPY", "FLAT"], index=0, key="joint-portfolio-fallback")
            joint_transaction_cost = joint_cols[2].number_input(
                "Round-trip cost (bps per side)",
                min_value=0.0,
                max_value=25.0,
                value=2.0,
                step=0.5,
                key="joint-portfolio-cost",
            )
            selected_symbols = st.multiselect(
                "Selected symbols",
                options=available_symbols,
                default=default_symbols,
                format_func=lambda symbol: _target_asset_label(store, symbol),
                key="joint-portfolio-symbols",
            )
            feature_versions = sorted(asset_session_features.get("feature_version", pd.Series(["asset-v1"])).dropna().astype(str).unique().tolist()) or ["asset-v1"]
            joint_feature_version = st.selectbox(
                "Feature version",
                options=feature_versions,
                index=0,
                key="joint-portfolio-feature-version",
            )
            joint_llm_enabled = st.checkbox(
                "Use semantic enrichment rows",
                value="llm" in joint_feature_version.lower(),
                key="joint-portfolio-llm-enabled",
            )
            narrative_mode_labels = {
                "baseline": "Baseline",
                "narrative_only": "Narrative only",
                "hybrid": "Hybrid",
            }
            default_narrative_modes = list(NARRATIVE_FEATURE_MODES) if joint_llm_enabled else ["baseline"]
            joint_narrative_feature_modes = st.multiselect(
                "Narrative feature modes",
                options=list(NARRATIVE_FEATURE_MODES),
                default=default_narrative_modes,
                format_func=lambda mode: narrative_mode_labels.get(str(mode), str(mode)),
                disabled=not joint_llm_enabled,
                key="joint-portfolio-narrative-feature-modes",
            )
            if not joint_llm_enabled:
                joint_narrative_feature_modes = ["baseline"]
            window_cols = st.columns(4)
            joint_train_window = int(
                window_cols[0].number_input(
                    "Train window",
                    min_value=20,
                    max_value=252,
                    value=90,
                    step=5,
                    key="joint-portfolio-train-window",
                ),
            )
            joint_validation_window = int(
                window_cols[1].number_input(
                    "Validation window",
                    min_value=10,
                    max_value=126,
                    value=30,
                    step=5,
                    key="joint-portfolio-validation-window",
                ),
            )
            joint_test_window = int(
                window_cols[2].number_input(
                    "Test window",
                    min_value=10,
                    max_value=126,
                    value=30,
                    step=5,
                    key="joint-portfolio-test-window",
                ),
            )
            joint_step_size = int(
                window_cols[3].number_input(
                    "Step size",
                    min_value=5,
                    max_value=126,
                    value=30,
                    step=5,
                    key="joint-portfolio-step-size",
                ),
            )
            grid_cols = st.columns(3)
            joint_threshold_grid = grid_cols[0].text_input(
                "Threshold grid",
                value="0.0, 0.001, 0.0025, 0.005",
                key="joint-portfolio-threshold-grid",
            )
            joint_minimum_grid = grid_cols[1].text_input(
                "Min post grid",
                value="1, 2, 3",
                key="joint-portfolio-minimum-grid",
            )
            joint_account_weight_grid = grid_cols[2].text_input(
                "Tracked-account weight grid",
                value="0.5, 1.0, 1.5",
                key="joint-portfolio-account-weight-grid",
            )
            topology_variants = st.multiselect(
                "Topology variants",
                options=["per_asset", "pooled"],
                default=["per_asset", "pooled"],
                key="joint-portfolio-topology-variants",
            )
            model_families = st.multiselect(
                "Model families",
                options=list(SUPPORTED_PORTFOLIO_MODEL_FAMILIES),
                default=["ridge", "elastic_net", "hist_gradient_boosting_regressor"],
                key="joint-portfolio-model-families",
            )

            if st.button("Build joint portfolio model", use_container_width=True, disabled=not can_write):
                if "SPY" not in selected_symbols and str(joint_fallback_mode).upper() == "SPY":
                    st.error("Include `SPY` in the selected symbols when fallback mode is `SPY`.")
                    return
                if len(selected_symbols) < 2:
                    st.error("Select at least two symbols for a joint portfolio run.")
                    return
                if not topology_variants:
                    st.error("Select at least one topology variant.")
                    return
                if not model_families:
                    st.error("Select at least one model family.")
                    return
                if not joint_narrative_feature_modes:
                    st.error("Select at least one narrative feature mode.")
                    return
                try:
                    portfolio_config = PortfolioRunConfig(
                        run_name=joint_run_name,
                        allocator_mode="joint_model",
                        fallback_mode=str(joint_fallback_mode).upper(),
                        transaction_cost_bps=float(joint_transaction_cost),
                        universe_symbols=tuple(selected_symbols),
                        selected_symbols=tuple(selected_symbols),
                        llm_enabled=bool(joint_llm_enabled),
                        feature_version=str(joint_feature_version),
                        train_window=joint_train_window,
                        validation_window=joint_validation_window,
                        test_window=joint_test_window,
                        step_size=joint_step_size,
                        threshold_grid=tuple(float(value) for value in _parse_grid(joint_threshold_grid, float)),
                        minimum_signal_grid=tuple(int(value) for value in _parse_grid(joint_minimum_grid, int)),
                        account_weight_grid=tuple(float(value) for value in _parse_grid(joint_account_weight_grid, float)),
                        model_families=tuple(str(value) for value in model_families),
                        topology_variants=tuple(str(value) for value in topology_variants),
                        narrative_feature_modes=tuple(str(value) for value in joint_narrative_feature_modes),
                    )
                except ValueError as exc:
                    st.error(f"Invalid grid input: {exc}")
                    return

                with st.spinner("Training joint portfolio model variants and selecting a deployment winner..."):
                    try:
                        portfolio_run, portfolio_artifacts = backtest_service.run_joint_model_allocator(
                            portfolio_config,
                            asset_session_features,
                        )
                        experiment_store.save_portfolio_run(
                            run=portfolio_run,
                            config=portfolio_artifacts["config"],
                            trades=portfolio_artifacts["trades"],
                            decision_history=portfolio_artifacts["predictions"],
                            candidate_predictions=portfolio_artifacts["candidate_predictions"],
                            component_summary=portfolio_artifacts["windows"],
                            benchmarks=portfolio_artifacts["benchmarks"],
                            benchmark_curves=portfolio_artifacts["benchmark_curves"],
                            diagnostics=portfolio_artifacts["diagnostics"],
                            leakage_audit=portfolio_artifacts["leakage_audit"],
                            variant_summary=portfolio_artifacts["variant_summary"],
                            portfolio_model_bundle=portfolio_artifacts["portfolio_model_bundle"],
                            importance=portfolio_artifacts["importance"],
                        )
                    except RuntimeError as exc:
                        st.error(str(exc))
                        return
                st.success(
                    f"Saved joint portfolio run `{portfolio_run.run_id}` with deployment variant "
                    f"`{portfolio_run.deployment_variant}`.",
                )

    if runs.empty:
        st.info("No experiment runs have been saved yet.")
        return

    leaderboard = runs.copy()
    leaderboard["total_return"] = leaderboard["metrics_json"].map(lambda metrics: metrics.get("total_return", 0.0))
    leaderboard["sharpe"] = leaderboard["metrics_json"].map(lambda metrics: metrics.get("sharpe", 0.0))
    leaderboard["sortino"] = leaderboard["metrics_json"].map(lambda metrics: metrics.get("sortino", 0.0))
    leaderboard["max_drawdown"] = leaderboard["metrics_json"].map(lambda metrics: metrics.get("max_drawdown", 0.0))
    leaderboard["robust_score"] = leaderboard["metrics_json"].map(lambda metrics: metrics.get("robust_score", 0.0))
    keep = [
        "run_id",
        "run_name",
        "run_type",
        "allocator_mode",
        "target_asset",
        "created_at",
        "total_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "robust_score",
    ]
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
    selected_settings = _comparison_settings(loaded)
    selected_run_type = selected_settings.get("run_type", "asset_model")
    st.caption(
        f"Selected run type: `{selected_run_type}` | "
        f"target asset: `{selected_settings.get('target_asset', 'SPY')}`",
    )

    feature_contributions = pd.DataFrame()
    post_attribution = pd.DataFrame()
    account_attribution = pd.DataFrame()
    if selected_run_type == "asset_model":
        feature_contributions = loaded["feature_contributions"]
        if feature_contributions.empty:
            feature_contributions = model_service.explain_predictions(loaded["model_artifact"], loaded["predictions"])
        post_attribution = loaded["post_attribution"]
        account_attribution = loaded["account_attribution"]
        if post_attribution.empty or account_attribution.empty:
            try:
                _, fallback_posts = _build_model_target_bundle(
                    store=store,
                    feature_service=feature_service,
                    posts=posts,
                    spy_market=spy,
                    tracked_accounts=tracked_accounts,
                    llm_enabled=bool(loaded["model_artifact"].metadata.get("llm_enabled", False)),
                    target_asset=selected_settings.get("target_asset", "SPY"),
                    feature_version=str(loaded.get("config", {}).get("feature_version", "v1")),
                )
                post_attribution = build_post_attribution(fallback_posts)
                account_attribution = build_account_attribution(post_attribution)
            except RuntimeError:
                post_attribution = pd.DataFrame()
                account_attribution = pd.DataFrame()

    _metric_row(loaded["metrics"])
    trade_view = loaded["trades"].copy()
    active_variant = str(selected_settings.get("deployment_variant", "") or "")
    if "variant_name" in trade_view.columns and active_variant:
        filtered = trade_view.loc[trade_view["variant_name"].astype(str) == active_variant].copy()
        if not filtered.empty:
            trade_view = filtered
    _render_equity_curve(trade_view, title="Walk-forward out-of-sample equity curve")

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
            curve_frame = run_bundle["trades"].copy()
            compare_settings = _comparison_settings(run_bundle)
            compare_variant = str(compare_settings.get("deployment_variant", "") or "")
            if "variant_name" in curve_frame.columns and compare_variant:
                filtered = curve_frame.loc[curve_frame["variant_name"].astype(str) == compare_variant].copy()
                if not filtered.empty:
                    curve_frame = filtered
            curves[run_id] = curve_frame[["next_session_date", "equity_curve"]].copy()
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
        benchmark_view = loaded["benchmarks"].copy()
        if "variant_name" in benchmark_view.columns:
            active_variant = str(selected_settings.get("deployment_variant", "") or "")
            if active_variant:
                filtered = benchmark_view.loc[benchmark_view["variant_name"].astype(str) == active_variant].copy()
                if not filtered.empty:
                    benchmark_view = filtered
        st.dataframe(benchmark_view, use_container_width=True, hide_index=True)
    if not loaded["benchmark_curves"].empty:
        curves = loaded["benchmark_curves"].copy()
        if "variant_name" in curves.columns:
            active_variant = str(selected_settings.get("deployment_variant", "") or "")
            if active_variant:
                filtered = curves.loc[curves["variant_name"].astype(str) == active_variant].copy()
                if not filtered.empty:
                    curves = filtered
            curves = curves.drop(columns=["variant_name"], errors="ignore")
        curve_fig = go.Figure()
        for column in curves.columns:
            if column == "next_session_date":
                continue
            curve_fig.add_trace(go.Scatter(x=curves["next_session_date"], y=curves[column], mode="lines", name=column))
        curve_fig.update_layout(title="Strategy vs. benchmark equity curves", xaxis_title="Trade date", yaxis_title="Equity")
        st.plotly_chart(curve_fig, use_container_width=True)

    if selected_run_type == "portfolio_allocator":
        selected_variant = str(selected_settings.get("deployment_variant", "") or "")
        variant_summary = _variant_summary_with_narrative_defaults(loaded.get("variant_summary", pd.DataFrame()).copy())
        if not variant_summary.empty:
            st.markdown("**Variant comparison**")
            st.dataframe(variant_summary, use_container_width=True, hide_index=True)
            variant_options = variant_summary["variant_name"].astype(str).tolist()
            if variant_options:
                selected_variant = st.selectbox(
                    "Inspect topology variant",
                    options=variant_options,
                    index=variant_options.index(selected_variant) if selected_variant in variant_options else 0,
                    key=f"portfolio-variant-{selected_run_id}",
                )
            narrative_lift = _build_narrative_lift_table(variant_summary)
            if not narrative_lift.empty:
                st.markdown("**Narrative lift vs. matching baseline**")
                st.dataframe(narrative_lift, use_container_width=True, hide_index=True)
        family_summary = _build_feature_family_summary(
            loaded,
            variant_name=selected_variant,
            importance=loaded.get("importance", pd.DataFrame()),
        )
        if not family_summary.empty:
            st.markdown("**Feature-family impact for selected variant**")
            st.dataframe(family_summary, use_container_width=True, hide_index=True)
        if not loaded["windows"].empty:
            allocator_windows = loaded["windows"].copy()
            if "variant_name" in allocator_windows.columns and selected_variant:
                filtered = allocator_windows.loc[allocator_windows["variant_name"].astype(str) == selected_variant].copy()
                if not filtered.empty:
                    allocator_windows = filtered
            title = "Allocator components" if selected_settings.get("allocator_mode") == "saved_runs" else "Joint portfolio training windows"
            st.markdown(f"**{title}**")
            st.dataframe(allocator_windows, use_container_width=True, hide_index=True)
        candidate_predictions = loaded.get("candidate_predictions", pd.DataFrame()).copy()
        if "variant_name" in candidate_predictions.columns and selected_variant:
            filtered = candidate_predictions.loc[candidate_predictions["variant_name"].astype(str) == selected_variant].copy()
            if not filtered.empty:
                candidate_predictions = filtered
        decision_rows = loaded["predictions"].copy()
        if "variant_name" in decision_rows.columns and selected_variant:
            filtered = decision_rows.loc[decision_rows["variant_name"].astype(str) == selected_variant].copy()
            if not filtered.empty:
                decision_rows = filtered
        decision_rows = decision_rows.sort_values("signal_session_date", ascending=False).reset_index(drop=True)
        if not decision_rows.empty:
            labels = decision_rows.apply(_portfolio_decision_option_label, axis=1).tolist()
            selected_label = st.selectbox(
                "Inspect allocator session",
                options=labels,
                index=0,
                key=f"portfolio-session-{selected_run_id}",
            )
            selected_decision = decision_rows.iloc[labels.index(selected_label)]
            session_candidates = _filter_for_session(
                candidate_predictions,
                _normalize_session_date(selected_decision.get("signal_session_date")),
            )
            _render_portfolio_session_panel(selected_decision, session_candidates)
        if not loaded["diagnostics"].empty:
            allocator_diag = loaded["diagnostics"].copy()
            if "variant_name" in allocator_diag.columns and selected_variant:
                filtered = allocator_diag.loc[allocator_diag["variant_name"].astype(str) == selected_variant].copy()
                if not filtered.empty:
                    allocator_diag = filtered
            diag_fig = go.Figure()
            diag_fig.add_trace(
                go.Scatter(
                    x=allocator_diag["signal_session_date"],
                    y=allocator_diag["winner_score"],
                    mode="lines+markers",
                    name="Winner score",
                ),
            )
            if "winner_gap_vs_runner_up" in allocator_diag.columns:
                diag_fig.add_trace(
                    go.Scatter(
                        x=allocator_diag["signal_session_date"],
                        y=allocator_diag["winner_gap_vs_runner_up"],
                        mode="lines",
                        name="Gap vs runner-up",
                    ),
                )
            diag_fig.update_layout(
                title="Portfolio allocator diagnostics",
                xaxis_title="Signal session",
                yaxis_title="Score",
            )
            st.plotly_chart(diag_fig, use_container_width=True)
            st.markdown("**Allocator decision diagnostics**")
            st.dataframe(allocator_diag.tail(30), use_container_width=True, hide_index=True)
    else:
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

    if selected_run_type == "portfolio_allocator" and loaded["leakage_audit"]:
        st.markdown("**Leakage audit**")
        st.json(loaded["leakage_audit"])

    with st.expander("Trades", expanded=False):
        st.dataframe(trade_view, use_container_width=True, hide_index=True)


def render_live_monitor(
    settings: AppSettings,
    store: DuckDBStore,
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
    feature_service: FeatureService,
    health_service: DataHealthService,
    model_service: ModelService,
    experiment_store: ExperimentStore,
) -> None:
    st.subheader("Live Monitor")
    st.caption(
        "Multi-asset decision console for pinned saved runs. It ranks current live candidates, "
        "applies the configured SPY/flat fallback, and keeps board history for debugging.",
    )
    can_write = _writes_enabled(settings)
    if is_public_mode(settings) and not can_write:
        st.info("Public visitors can inspect the live board and history, but polling refreshes and config changes require admin access.")
    remote_url = st.text_input(
        "Remote X / mentions CSV URL for polling",
        key="live_remote_x_url",
        value=st.session_state.get("remote_x_url", settings.remote_x_csv_url),
        disabled=not can_write,
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

    refresh_requested = st.button("Poll sources now", use_container_width=True, disabled=not can_write)
    if refresh_requested:
        with st.spinner("Refreshing sources and market data..."):
            _refresh_datasets(
                settings=settings,
                store=store,
                ingestion_service=ingestion_service,
                market_service=market_service,
                discovery_service=discovery_service,
                feature_service=feature_service,
                health_service=health_service,
                remote_url=remote_url,
                uploaded_files=[],
                incremental=True,
                refresh_mode="incremental",
            )
        st.success("Polling refresh complete.")

    runs = _normalize_runs_frame(experiment_store.list_runs())
    joint_portfolio_runs = runs.loc[
        (runs["run_type"].astype(str) == "portfolio_allocator")
        & (runs["allocator_mode"].astype(str) == "joint_model")
    ].copy() if not runs.empty else pd.DataFrame()
    if joint_portfolio_runs.empty:
        st.info("Save at least one joint portfolio run in Models & Backtests before using the live decision console.")
        return

    saved_config = experiment_store.load_live_monitor_config()
    if saved_config is not None and str(saved_config.mode or "portfolio_run") == "asset_model_set":
        st.warning("A legacy asset-pin live config was found. Save a portfolio-run config below to migrate the live console.")
    seeded_config = seed_live_monitor_config(runs)
    editor_config = saved_config if saved_config is not None and str(saved_config.mode or "portfolio_run") == "portfolio_run" else seeded_config
    if editor_config is None:
        st.info("Save at least one joint portfolio run before configuring the live decision console.")
        return

    st.markdown("**Pinned Portfolio Run**")
    if saved_config is None or str(saved_config.mode or "portfolio_run") != "portfolio_run":
        st.info("The selector below is seeded from the newest saved joint portfolio run. Save it to enable monitoring.")

    saved_config_errors = validate_live_monitor_config(saved_config, runs) if saved_config is not None else []
    if saved_config_errors:
        for message in saved_config_errors:
            st.warning(message)
        st.caption("Update and save the pinned portfolio run below to restore live monitoring.")

    portfolio_run_options = joint_portfolio_runs["run_id"].astype(str).tolist()
    configured_portfolio_run_id = str(editor_config.portfolio_run_id or "")
    if configured_portfolio_run_id not in portfolio_run_options:
        configured_portfolio_run_id = portfolio_run_options[0]
    fallback_default = str(editor_config.fallback_mode or "SPY").upper()
    fallback_index = 0 if fallback_default == "SPY" else 1
    active_config = saved_config if not saved_config_errors and saved_config is not None and str(saved_config.mode or "portfolio_run") == "portfolio_run" else None

    with st.form("live-monitor-config"):
        fallback_mode = st.radio(
            "Fallback mode",
            options=["SPY", "FLAT"],
            index=fallback_index,
            horizontal=True,
        )
        selected_portfolio_run_id = st.selectbox(
            "Pinned joint portfolio run",
            options=portfolio_run_options,
            index=portfolio_run_options.index(configured_portfolio_run_id),
            format_func=lambda run_id: _live_run_option_label(
                joint_portfolio_runs.loc[joint_portfolio_runs["run_id"].astype(str) == str(run_id)].iloc[0],
            ),
        )
        selected_row = joint_portfolio_runs.loc[
            joint_portfolio_runs["run_id"].astype(str) == str(selected_portfolio_run_id)
        ].iloc[0]
        selected_params = selected_row.get("selected_params_json", {}) or {}
        selected_symbols = selected_params.get("selected_symbols", [])
        deployment_variant = str(selected_params.get("deployment_variant", "") or "")
        deployment_narrative_mode = str(selected_params.get("deployment_narrative_feature_mode", "") or "")
        st.caption(
            f"Deployment variant: `{deployment_variant or 'n/a'}` | "
            f"narrative mode: `{deployment_narrative_mode or 'n/a'}` | "
            f"selected symbols: `{', '.join(selected_symbols) if selected_symbols else 'n/a'}`",
        )
        save_config = st.form_submit_button("Save pinned portfolio run", disabled=not can_write)

    if save_config:
        candidate_config = LiveMonitorConfig(
            mode="portfolio_run",
            fallback_mode=str(fallback_mode).upper(),
            portfolio_run_id=str(selected_portfolio_run_id),
            portfolio_run_name=str(selected_row.get("run_name", selected_portfolio_run_id) or selected_portfolio_run_id),
            deployment_variant=str(deployment_variant),
        )
        validation_errors = validate_live_monitor_config(candidate_config, runs)
        if validation_errors:
            for message in validation_errors:
                st.error(message)
        else:
            active_config = candidate_config
            experiment_store.save_live_monitor_config(active_config)
            st.success("Saved the pinned portfolio run.")

    if active_config is None:
        st.info("Save a valid pinned portfolio run above to enable live monitoring.")
        return

    config_errors = validate_live_monitor_config(active_config, runs)
    if config_errors:
        for message in config_errors:
            st.error(message)
        return

    paper_service = PaperTradingService(store)
    performance_service = PerformanceObservatoryService(store)
    active_run_rows = joint_portfolio_runs.loc[
        joint_portfolio_runs["run_id"].astype(str) == str(active_config.portfolio_run_id)
    ].copy()
    active_run_row = active_run_rows.iloc[0] if not active_run_rows.empty else pd.Series(dtype=object)
    active_run_name = str(
        active_run_row.get("run_name", active_config.portfolio_run_name or active_config.portfolio_run_id)
        or active_config.portfolio_run_name
        or active_config.portfolio_run_id
    )
    active_run_params = active_run_row.get("selected_params_json", {}) or {}
    active_transaction_cost_bps = float(active_run_params.get("transaction_cost_bps", 0.0) or 0.0)
    current_paper_config = paper_service.load_current_config()
    active_paper_config = current_paper_config if paper_config_matches_live(current_paper_config, active_config) else None

    posts = store.read_frame("normalized_posts")
    spy = store.read_frame("spy_daily")
    tracked_accounts = store.read_frame("tracked_accounts")
    if str(active_config.mode or "portfolio_run") == "asset_model_set" and (posts.empty or spy.empty):
        st.info(_refresh_required_message(settings, "Refresh datasets first so the live decision console has normalized posts and SPY market data."))
        return
    board_generated_at = pd.Timestamp.utcnow().floor("s")
    board, decision_history_row, explanation_lookup, live_warnings = _build_live_monitor_state(
        store=store,
        feature_service=feature_service,
        model_service=model_service,
        experiment_store=experiment_store,
        posts=posts,
        spy_market=spy,
        tracked_accounts=tracked_accounts,
        config=active_config,
        generated_at=board_generated_at,
    )
    for message in live_warnings:
        st.warning(message)
    if board.empty or decision_history_row.empty:
        st.info("No live candidate rows could be built from the pinned portfolio run.")
        return

    should_persist_snapshots = bool(refresh_requested and can_write)
    if poll_enabled and can_write:
        last_persisted = pd.to_datetime(st.session_state.get("live_monitor_last_persisted_at"), errors="coerce")
        if pd.isna(last_persisted) or (board_generated_at - last_persisted).total_seconds() >= max(int(poll_seconds * 0.8), 5):
            should_persist_snapshots = True
    if should_persist_snapshots:
        experiment_store.save_live_asset_snapshots(board)
        experiment_store.save_live_decision_snapshots(decision_history_row)
        st.session_state["live_monitor_last_persisted_at"] = board_generated_at.isoformat()

    paper_update = {"captured": 0, "settled": 0}
    if active_paper_config is not None and active_paper_config.enabled:
        paper_update = paper_service.process_live_history(active_paper_config, as_of=board_generated_at)

    decision_row = decision_history_row.iloc[0]
    winner_asset = str(decision_row.get("winning_asset", "") or "")
    winner_rows = board.loc[board["is_winner"]].copy()
    winner_row = winner_rows.iloc[0] if not winner_rows.empty else None
    winner_confidence = float(winner_row["confidence"]) if winner_row is not None else 0.0
    display_asset = winner_asset or "FLAT"
    tabs = st.tabs(["Current Decision", "Why This Asset Won", "History", "Paper Portfolio", "Performance Observatory"])

    with tabs[0]:
        metric_cols = st.columns(8)
        metric_cols[0].metric("Winning asset", display_asset)
        metric_cols[1].metric(
            "Signal session",
            f"{pd.Timestamp(decision_row['signal_session_date']):%Y-%m-%d}" if pd.notna(decision_row["signal_session_date"]) else "n/a",
        )
        metric_cols[2].metric("Decision source", str(decision_row["decision_source"]).upper())
        metric_cols[3].metric("Stance", str(decision_row["stance"]))
        metric_cols[4].metric("Winner score", f"{float(decision_row['winner_score']):+.3%}")
        metric_cols[5].metric("Winner confidence", f"{winner_confidence:.2f}")
        metric_cols[6].metric("Deployment variant", str(decision_row.get("deployment_variant", active_config.deployment_variant) or "n/a"))
        metric_cols[7].metric("Narrative mode", str(decision_row.get("narrative_feature_mode", "") or "n/a"))

        st.json(
            {
                "fallback_mode": str(decision_row["fallback_mode"]),
                "eligible_asset_count": int(decision_row["eligible_asset_count"]),
                "runner_up_asset": str(decision_row.get("runner_up_asset", "") or ""),
                "generated_at": str(pd.Timestamp(decision_row["generated_at"])),
            },
        )
        board_view = board.rename(
            columns={
                "variant_name": "Variant",
                "topology": "Topology",
                "narrative_feature_mode": "Narrative mode",
                "asset_symbol": "Asset",
                "run_name": "Run",
                "expected_return_score": "Score",
                "confidence": "Confidence",
                "threshold": "Threshold",
                "min_post_count": "Min posts",
                "post_count": "Posts",
                "qualifies": "Qualifies",
                "eligible_rank": "Eligible rank",
                "is_winner": "Winner",
                "decision_source": "Decision source",
                "stance": "Stance",
            },
        )
        st.markdown("**Live ranked board**")
        st.dataframe(
            board_view[
                [
                    "Variant",
                    "Topology",
                    "Narrative mode",
                    "Asset",
                    "Run",
                    "Score",
                    "Confidence",
                    "Threshold",
                    "Min posts",
                    "Posts",
                    "Qualifies",
                    "Eligible rank",
                    "Winner",
                    "Decision source",
                    "Stance",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tabs[1]:
        explain_options = board["asset_symbol"].astype(str).tolist()
        default_explain_asset = winner_asset or (explain_options[0] if explain_options else "")
        selected_asset = st.selectbox(
            "Explain asset",
            options=explain_options,
            index=explain_options.index(default_explain_asset) if default_explain_asset in explain_options else 0,
            key="live-monitor-explain-asset",
        )
        runner_frame = _build_live_runner_up_frame(board, decision_row)
        if not runner_frame.empty:
            st.markdown("**Winner vs runner-up**")
            st.dataframe(
                runner_frame.rename(
                    columns={
                        "asset_symbol": "Asset",
                        "run_name": "Run",
                        "score": "Score",
                        "confidence": "Confidence",
                        "threshold_gap": "Threshold gap",
                        "post_count": "Posts",
                        "qualifies": "Qualifies",
                        "winner": "Winner",
                    },
                ),
                use_container_width=True,
                hide_index=True,
            )
        selected_payload = explanation_lookup.get(selected_asset, {})
        prediction_row = selected_payload.get("prediction_row", pd.Series(dtype=object)).copy()
        if not prediction_row.empty:
            board_row = selected_payload.get("board_row")
            if board_row is not None:
                prediction_row["expected_return_score"] = board_row["expected_return_score"]
                prediction_row["prediction_confidence"] = board_row["confidence"]
            heading = "Why This Asset Won?" if selected_asset == winner_asset else f"Why {selected_asset}?"
            _render_signal_explanation_panel(
                prediction_row=prediction_row,
                feature_contributions=selected_payload.get("feature_contributions", pd.DataFrame()),
                post_attribution=selected_payload.get("post_attribution", pd.DataFrame()),
                account_attribution=selected_payload.get("account_attribution", pd.DataFrame()),
                heading=heading,
            )

    with tabs[2]:
        asset_history = store.read_frame("live_asset_snapshots")
        decision_history = store.read_frame("live_decision_snapshots")
        monitored_assets = set(board["asset_symbol"].astype(str).str.upper().tolist())
        if not asset_history.empty and "asset_symbol" in asset_history.columns:
            asset_history["asset_symbol"] = asset_history["asset_symbol"].astype(str).str.upper()
            asset_history = asset_history.loc[asset_history["asset_symbol"].isin(monitored_assets)].copy()
        if not asset_history.empty and "run_id" in asset_history.columns:
            asset_history = asset_history.loc[asset_history["run_id"].astype(str) == str(active_config.portfolio_run_id)].copy()
        if not decision_history.empty:
            decision_history = decision_history.sort_values("generated_at").reset_index(drop=True)

        if asset_history.empty:
            st.info("No persisted live board history yet. Use polling or the manual refresh button to build it up.")
        else:
            asset_history = asset_history.sort_values("generated_at").reset_index(drop=True)
            history_fig = go.Figure()
            for asset_symbol, group in asset_history.groupby("asset_symbol", sort=False):
                history_fig.add_trace(
                    go.Scatter(
                        x=group["generated_at"],
                        y=group["expected_return_score"],
                        mode="lines+markers",
                        name=str(asset_symbol),
                    ),
                )
            history_fig.update_layout(
                title="Live asset score history",
                xaxis_title="Generated at",
                yaxis_title="Expected next-session return",
            )
            st.plotly_chart(history_fig, use_container_width=True)

        st.markdown("**Recent live decisions**")
        if decision_history.empty:
            st.info("No persisted live decision history yet.")
        else:
            st.dataframe(decision_history.tail(25), use_container_width=True, hide_index=True)

        mapped_post_asset = st.selectbox(
            "Mapped-post view asset",
            options=board["asset_symbol"].astype(str).tolist(),
            index=board["asset_symbol"].astype(str).tolist().index(default_explain_asset) if default_explain_asset in board["asset_symbol"].astype(str).tolist() else 0,
            key="live-monitor-post-asset",
        )
        post_payload = explanation_lookup.get(mapped_post_asset, {})
        post_prediction = post_payload.get("prediction_row", pd.Series(dtype=object))
        mapped_posts = _filter_for_session(
            post_payload.get("post_attribution", pd.DataFrame()),
            _normalize_session_date(post_prediction.get("signal_session_date")),
        )
        st.markdown(f"**Latest mapped posts for {mapped_post_asset}**")
        if mapped_posts.empty:
            st.info(f"No mapped live posts are available for `{mapped_post_asset}`.")
        else:
            st.dataframe(
                mapped_posts[
                    [
                        "post_timestamp",
                        "author_handle",
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

    with tabs[3]:
        registry = paper_service.list_portfolios()
        active_paper_config = paper_service.load_current_config()
        active_matching_paper = active_paper_config if paper_config_matches_live(active_paper_config, active_config) else None
        current_journal = ensure_paper_decision_journal_frame(store.read_frame("paper_decision_journal"))
        current_trades = ensure_paper_trade_ledger_frame(store.read_frame("paper_trade_ledger"))
        current_equity = ensure_paper_equity_curve_frame(store.read_frame("paper_equity_curve"))
        current_benchmark = ensure_paper_benchmark_curve_frame(store.read_frame("paper_benchmark_curve"))

        st.markdown("**Paper trading controls**")
        if not can_write:
            st.info("Public visitors can inspect the paper portfolio history, but enabling, resetting, and archiving paper trading requires admin access.")

        if active_matching_paper is None:
            st.info("Paper trading is not enabled for the currently pinned portfolio run.")
            if can_write:
                with st.form("paper-portfolio-enable-form"):
                    starting_cash = st.number_input(
                        "Starting cash",
                        min_value=1000.0,
                        value=100000.0,
                        step=1000.0,
                    )
                    enable_paper = st.form_submit_button("Enable paper trading for pinned run")
                if enable_paper:
                    paper_service.upsert_current_for_live_config(
                        live_config=active_config,
                        portfolio_run_name=active_run_name,
                        transaction_cost_bps=active_transaction_cost_bps,
                        starting_cash=float(starting_cash),
                        enabled=True,
                        reset=False,
                        now=board_generated_at,
                    )
                    st.rerun()
        else:
            control_cols = st.columns(3)
            control_cols[0].metric("Paper portfolio", active_matching_paper.paper_portfolio_id)
            control_cols[1].metric("Status", "Enabled" if active_matching_paper.enabled else "Disabled")
            control_cols[2].metric("Starting cash", f"${active_matching_paper.starting_cash:,.0f}")
            toggle_label = "Disable paper trading" if active_matching_paper.enabled else "Enable paper trading"
            toggle_disabled = not can_write
            if control_cols[0].button(toggle_label, use_container_width=True, disabled=toggle_disabled):
                paper_service.upsert_current_for_live_config(
                    live_config=active_config,
                    portfolio_run_name=active_run_name,
                    transaction_cost_bps=active_matching_paper.transaction_cost_bps,
                    starting_cash=active_matching_paper.starting_cash,
                    enabled=not active_matching_paper.enabled,
                    reset=False,
                    now=board_generated_at,
                )
                st.rerun()
            if control_cols[1].button("Archive current portfolio", use_container_width=True, disabled=not can_write):
                paper_service.archive_current_config(now=board_generated_at)
                st.rerun()
            with control_cols[2].form("paper-portfolio-reset-form"):
                reset_cash = st.number_input(
                    "Reset starting cash",
                    min_value=1000.0,
                    value=float(active_matching_paper.starting_cash),
                    step=1000.0,
                )
                reset_paper = st.form_submit_button("Reset portfolio", disabled=not can_write)
            if reset_paper:
                paper_service.upsert_current_for_live_config(
                    live_config=active_config,
                    portfolio_run_name=active_run_name,
                    transaction_cost_bps=active_transaction_cost_bps or active_matching_paper.transaction_cost_bps,
                    starting_cash=float(reset_cash),
                    enabled=True,
                    reset=True,
                    now=board_generated_at,
                )
                st.rerun()

        if paper_update["captured"] or paper_update["settled"]:
            st.caption(
                f"Latest capture cycle: {paper_update['captured']} decision(s) captured, "
                f"{paper_update['settled']} decision(s) settled."
            )

        registry = paper_service.list_portfolios()
        current_journal = ensure_paper_decision_journal_frame(store.read_frame("paper_decision_journal"))
        current_trades = ensure_paper_trade_ledger_frame(store.read_frame("paper_trade_ledger"))
        current_equity = ensure_paper_equity_curve_frame(store.read_frame("paper_equity_curve"))
        current_benchmark = ensure_paper_benchmark_curve_frame(store.read_frame("paper_benchmark_curve"))

        if registry.empty:
            st.info("No paper portfolios have been created yet.")
        else:
            registry_view = registry.copy()
            registry_view["status"] = np.where(registry_view["archived_at"].notna(), "archived", np.where(registry_view["enabled"], "active", "disabled"))
            option_ids = registry_view["paper_portfolio_id"].astype(str).tolist()
            default_portfolio_id = (
                active_matching_paper.paper_portfolio_id
                if active_matching_paper is not None
                else option_ids[0]
            )
            selected_paper_id = st.selectbox(
                "View paper portfolio",
                options=option_ids,
                index=option_ids.index(default_portfolio_id) if default_portfolio_id in option_ids else 0,
                format_func=lambda paper_id: (
                    lambda row: f"{paper_id} | {row['status']} | {row['portfolio_run_name']} | {row['deployment_variant'] or 'n/a'}"
                )(registry_view.loc[registry_view["paper_portfolio_id"].astype(str) == str(paper_id)].iloc[0]),
                key="paper-portfolio-select",
            )
            selected_registry_row = registry_view.loc[
                registry_view["paper_portfolio_id"].astype(str) == str(selected_paper_id)
            ].iloc[0]
            selected_journal = current_journal.loc[
                current_journal["paper_portfolio_id"].astype(str) == str(selected_paper_id)
            ].copy()
            selected_trades = current_trades.loc[
                current_trades["paper_portfolio_id"].astype(str) == str(selected_paper_id)
            ].copy()
            selected_equity = current_equity.loc[
                current_equity["paper_portfolio_id"].astype(str) == str(selected_paper_id)
            ].copy()
            selected_benchmark = current_benchmark.loc[
                current_benchmark["paper_portfolio_id"].astype(str) == str(selected_paper_id)
            ].copy()

            latest_equity = float(selected_equity["equity"].iloc[-1]) if not selected_equity.empty else float(selected_registry_row["starting_cash"])
            cumulative_return = latest_equity / float(selected_registry_row["starting_cash"]) - 1.0 if float(selected_registry_row["starting_cash"]) else 0.0
            unsettled_count = int((selected_journal["settlement_status"].astype(str) == "pending").sum()) if not selected_journal.empty else 0
            settled_rows = selected_journal.loc[
                selected_journal["settlement_status"].astype(str).isin(["settled", "flat"])
            ].copy()
            last_settled = settled_rows["next_session_date"].max() if not settled_rows.empty else pd.NaT
            status_cols = st.columns(6)
            status_cols[0].metric("Portfolio status", str(selected_registry_row["status"]).title())
            status_cols[1].metric("Current equity", f"${latest_equity:,.2f}")
            status_cols[2].metric("Cumulative return", f"{cumulative_return:+.2%}")
            status_cols[3].metric("Realized trades", f"{len(selected_trades):,}")
            status_cols[4].metric("Open or unsettled", f"{unsettled_count:,}")
            status_cols[5].metric(
                "Last settled session",
                f"{pd.Timestamp(last_settled):%Y-%m-%d}" if pd.notna(last_settled) else "n/a",
            )

            capture_cols = st.columns(3)
            last_captured = selected_journal["generated_at"].max() if not selected_journal.empty else pd.NaT
            last_trade_settled = selected_trades["settled_at"].max() if not selected_trades.empty else pd.NaT
            capture_cols[0].metric("Scheduler capture", "Enabled" if settings.scheduler_enabled else "Disabled")
            capture_cols[1].metric(
                "Last paper decision",
                f"{pd.Timestamp(last_captured):%Y-%m-%d %H:%M UTC}" if pd.notna(last_captured) else "n/a",
            )
            capture_cols[2].metric(
                "Last trade settled",
                f"{pd.Timestamp(last_trade_settled):%Y-%m-%d %H:%M UTC}" if pd.notna(last_trade_settled) else "n/a",
            )

            if selected_equity.empty:
                st.info("No settled paper portfolio history yet.")
            else:
                paper_fig = go.Figure()
                paper_fig.add_trace(
                    go.Scatter(
                        x=selected_equity["next_session_date"],
                        y=selected_equity["equity"],
                        mode="lines+markers",
                        name="Paper portfolio",
                    ),
                )
                if not selected_benchmark.empty:
                    paper_fig.add_trace(
                        go.Scatter(
                            x=selected_benchmark["next_session_date"],
                            y=selected_benchmark["equity"],
                            mode="lines+markers",
                            name="SPY benchmark",
                        ),
                    )
                paper_fig.update_layout(
                    title="Paper portfolio equity vs SPY",
                    xaxis_title="Next session date",
                    yaxis_title="Equity",
                )
                st.plotly_chart(paper_fig, use_container_width=True)

            st.markdown("**Recent decision journal**")
            if selected_journal.empty:
                st.info("No paper decisions have been recorded for this portfolio yet.")
            else:
                st.dataframe(
                    selected_journal.tail(25),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("**Recent trade ledger**")
            if selected_trades.empty:
                st.info("No paper trades have settled for this portfolio yet.")
            else:
                st.dataframe(
                    selected_trades.tail(25),
                    use_container_width=True,
                    hide_index=True,
                )

    with tabs[4]:
        st.caption("Warn-only observability for the selected paper portfolio. These diagnostics do not gate live decisions, retrain models, or change trading behavior.")
        registry = paper_service.list_portfolios()
        if registry.empty:
            st.info("Create a paper portfolio before using the Performance Observatory.")
        else:
            registry_view = registry.copy()
            registry_view["status"] = np.where(
                registry_view["archived_at"].notna(),
                "archived",
                np.where(registry_view["enabled"], "active", "disabled"),
            )
            option_ids = registry_view["paper_portfolio_id"].astype(str).tolist()
            current_config = paper_service.load_current_config()
            default_performance_id = (
                current_config.paper_portfolio_id
                if current_config is not None and str(current_config.paper_portfolio_id) in option_ids
                else option_ids[0]
            )
            selected_performance_id = st.selectbox(
                "Performance portfolio",
                options=option_ids,
                index=option_ids.index(default_performance_id) if default_performance_id in option_ids else 0,
                format_func=lambda paper_id: (
                    lambda row: f"{paper_id} | {row['status']} | {row['portfolio_run_name']} | {row['deployment_variant'] or 'n/a'}"
                )(registry_view.loc[registry_view["paper_portfolio_id"].astype(str) == str(paper_id)].iloc[0]),
                key="performance-observatory-portfolio-select",
            )
            selected_performance_row = registry_view.loc[
                registry_view["paper_portfolio_id"].astype(str) == str(selected_performance_id)
            ].iloc[0]

            diagnostics = performance_service.load_latest_for_portfolio(selected_performance_id)
            if diagnostics.empty:
                diagnostics = performance_service.evaluate_paper_portfolio(
                    selected_performance_id,
                    generated_at=board_generated_at,
                )
                st.caption("No persisted observability snapshot exists yet, so the dashboard is showing a current in-memory evaluation.")
            else:
                latest_generated_at = diagnostics["generated_at"].max()
                st.caption(
                    "Showing the latest persisted observability snapshot"
                    + (f" from {pd.Timestamp(latest_generated_at):%Y-%m-%d %H:%M UTC}." if pd.notna(latest_generated_at) else "."),
                )

            observatory_registry = ensure_paper_portfolio_registry_frame(store.read_frame("paper_portfolio_registry"))
            observatory_journal = ensure_paper_decision_journal_frame(store.read_frame("paper_decision_journal"))
            observatory_trades = ensure_paper_trade_ledger_frame(store.read_frame("paper_trade_ledger"))
            observatory_equity = ensure_paper_equity_curve_frame(store.read_frame("paper_equity_curve"))
            observatory_benchmark = ensure_paper_benchmark_curve_frame(store.read_frame("paper_benchmark_curve"))
            summary = build_performance_summary(
                diagnostics=diagnostics,
                registry=observatory_registry,
                journal=observatory_journal,
                trades=observatory_trades,
                equity=observatory_equity,
                benchmark=observatory_benchmark,
                paper_portfolio_id=selected_performance_id,
            )

            def _pct_or_na(value: object) -> str:
                numeric = pd.to_numeric(value, errors="coerce")
                return f"{float(numeric):+.2%}" if pd.notna(numeric) else "n/a"

            def _num_or_na(value: object) -> str:
                numeric = pd.to_numeric(value, errors="coerce")
                return f"{float(numeric):.2f}" if pd.notna(numeric) else "n/a"

            perf_cols = st.columns(5)
            perf_cols[0].metric("Overall", str(summary.get("overall_severity", "ok")).upper())
            perf_cols[1].metric("Total return", _pct_or_na(summary.get("total_return")))
            perf_cols[2].metric("SPY return", _pct_or_na(summary.get("benchmark_return")))
            perf_cols[3].metric("Alpha", _pct_or_na(summary.get("alpha")))
            perf_cols[4].metric("Max drawdown", _pct_or_na(summary.get("max_drawdown")))
            quality_cols = st.columns(5)
            quality_cols[0].metric("Win rate", _pct_or_na(summary.get("win_rate")))
            quality_cols[1].metric("Trade count", f"{int(summary.get('trade_count', 0)):,}")
            quality_cols[2].metric("Pending decisions", f"{int(summary.get('pending_decisions', 0)):,}")
            quality_cols[3].metric("Fallback rate", _pct_or_na(summary.get("fallback_rate")))
            quality_cols[4].metric("Score/outcome corr", _num_or_na(summary.get("score_outcome_correlation")))

            equity_compare = build_equity_comparison_frame(
                observatory_equity,
                observatory_benchmark,
                selected_performance_id,
            )
            rolling_returns = build_rolling_return_frame(observatory_trades, selected_performance_id)
            score_outcomes = build_score_outcome_frame(
                observatory_journal,
                observatory_trades,
                selected_performance_id,
            )
            winner_distribution = build_winner_distribution_frame(observatory_journal, selected_performance_id)
            drift_frame = build_live_score_drift_frame(
                live_asset_snapshots=store.read_frame("live_asset_snapshots"),
                portfolio_run_id=str(selected_performance_row.get("portfolio_run_id", "") or ""),
                deployment_variant=str(selected_performance_row.get("deployment_variant", "") or ""),
            )

            chart_cols = st.columns(2)
            with chart_cols[0]:
                if equity_compare.empty:
                    st.info("No equity history is available yet.")
                else:
                    equity_fig = go.Figure()
                    equity_fig.add_trace(
                        go.Scatter(
                            x=equity_compare["next_session_date"],
                            y=equity_compare["paper_portfolio_equity"],
                            mode="lines+markers",
                            name="Paper portfolio",
                        ),
                    )
                    if "spy_benchmark_equity" in equity_compare.columns:
                        equity_fig.add_trace(
                            go.Scatter(
                                x=equity_compare["next_session_date"],
                                y=equity_compare["spy_benchmark_equity"],
                                mode="lines+markers",
                                name="SPY benchmark",
                            ),
                        )
                    equity_fig.update_layout(title="Equity vs SPY", xaxis_title="Session", yaxis_title="Equity")
                    st.plotly_chart(equity_fig, use_container_width=True)

            with chart_cols[1]:
                if rolling_returns.empty:
                    st.info("No settled trade returns are available yet.")
                else:
                    rolling_fig = go.Figure()
                    rolling_fig.add_trace(
                        go.Scatter(
                            x=rolling_returns["next_session_date"],
                            y=rolling_returns["rolling_net_return"],
                            mode="lines+markers",
                            name="Rolling net return",
                        ),
                    )
                    rolling_fig.update_layout(title="Rolling net return", xaxis_title="Session", yaxis_title="Return")
                    st.plotly_chart(rolling_fig, use_container_width=True)

            calibration_cols = st.columns(2)
            with calibration_cols[0]:
                if score_outcomes.empty:
                    st.info("No score/outcome pairs are available yet.")
                else:
                    score_fig = go.Figure()
                    for asset_symbol, group in score_outcomes.groupby(score_outcomes["winning_asset"].astype(str)):
                        score_fig.add_trace(
                            go.Scatter(
                                x=group["winner_score"],
                                y=group["net_return"],
                                mode="markers",
                                name=str(asset_symbol or "unknown"),
                            ),
                        )
                    score_fig.update_layout(title="Winner score vs realized return", xaxis_title="Winner score", yaxis_title="Net return")
                    st.plotly_chart(score_fig, use_container_width=True)

            with calibration_cols[1]:
                if winner_distribution.empty:
                    st.info("No paper decisions are available yet.")
                else:
                    winner_fig = go.Figure(
                        data=[
                            go.Bar(
                                x=winner_distribution["winning_asset"],
                                y=winner_distribution["decision_count"],
                                name="Decisions",
                            ),
                        ],
                    )
                    winner_fig.update_layout(title="Winner asset distribution", xaxis_title="Asset", yaxis_title="Decisions")
                    st.plotly_chart(winner_fig, use_container_width=True)

            if drift_frame.empty:
                st.info("No live score drift history is available yet.")
            else:
                drift_fig = go.Figure()
                for metric_name, group in drift_frame.groupby("metric_name", sort=False):
                    drift_fig.add_trace(
                        go.Bar(
                            x=group["asset_symbol"],
                            y=group["z_score"],
                            name=str(metric_name),
                        ),
                    )
                drift_fig.update_layout(
                    title="Live score drift by asset",
                    xaxis_title="Asset",
                    yaxis_title="Recent-vs-baseline z score",
                    barmode="group",
                )
                st.plotly_chart(drift_fig, use_container_width=True)
                st.dataframe(drift_frame, use_container_width=True, hide_index=True)

            score_buckets = build_score_bucket_outcome_frame(
                observatory_journal,
                observatory_trades,
                selected_performance_id,
            )
            if not score_buckets.empty:
                st.markdown("**Score bucket outcomes**")
                st.dataframe(score_buckets, use_container_width=True, hide_index=True)

            st.markdown("**Latest diagnostics**")
            normalized_diagnostics = ensure_performance_diagnostic_frame(diagnostics)
            if normalized_diagnostics.empty:
                st.info("No diagnostics are available for this portfolio yet.")
            else:
                severity_options = ["ok", "warn", "severe"]
                selected_severities = st.multiselect(
                    "Severity filter",
                    options=severity_options,
                    default=["warn", "severe"],
                    key="performance-observatory-severity-filter",
                )
                scope_options = sorted(normalized_diagnostics["scope_kind"].dropna().astype(str).unique().tolist())
                selected_scopes = st.multiselect(
                    "Scope filter",
                    options=scope_options,
                    default=scope_options,
                    key="performance-observatory-scope-filter",
                )
                diagnostics_view = normalized_diagnostics.copy()
                if selected_severities:
                    diagnostics_view = diagnostics_view.loc[diagnostics_view["severity"].isin(selected_severities)].copy()
                if selected_scopes:
                    diagnostics_view = diagnostics_view.loc[diagnostics_view["scope_kind"].astype(str).isin(selected_scopes)].copy()
                st.dataframe(diagnostics_view, use_container_width=True, hide_index=True)


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
        st.info(_refresh_required_message(settings, "Refresh datasets first so replay can rebuild historical features."))
        return

    selected_run_id = st.selectbox("Replay template run", options=runs["run_id"].tolist(), key="replay-run-id")
    loaded = experiment_store.load_run(selected_run_id)
    if loaded is None:
        st.warning("The selected replay template could not be loaded.")
        return

    run_config = _bundle_to_run_config(loaded)
    try:
        feature_rows, attribution_posts = _build_model_target_bundle(
            store=store,
            feature_service=feature_service,
            posts=posts,
            spy_market=spy,
            tracked_accounts=tracked_accounts,
            llm_enabled=run_config.llm_enabled,
            target_asset=run_config.target_asset,
            feature_version=run_config.feature_version,
        )
    except RuntimeError as exc:
        st.warning(str(exc))
        return
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
        attribution_posts.loc[
            pd.to_datetime(attribution_posts["session_date"], errors="coerce").dt.normalize()
            == pd.Timestamp(replay_prediction["signal_session_date"]).normalize()
        ].copy(),
    )
    session_account_attribution = build_account_attribution(session_post_attribution)

    full_history_match = _filter_for_session(loaded["predictions"], _normalize_session_date(replay_prediction["signal_session_date"]))
    full_history_row = full_history_match.iloc[0] if not full_history_match.empty else None

    metric_cols = st.columns(5)
    metric_cols[0].metric("Target asset", str(run_config.target_asset).upper())
    metric_cols[1].metric("Replay session", f"{pd.Timestamp(replay_prediction['signal_session_date']):%Y-%m-%d}")
    metric_cols[2].metric("Replay score", f"{float(replay_prediction['expected_return_score']):+.3%}")
    metric_cols[3].metric("Replay confidence", f"{float(replay_prediction['prediction_confidence']):.2f}")
    metric_cols[4].metric("Suggested stance", str(replay_prediction["suggested_stance"]))

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
            "target_asset": str(run_config.target_asset).upper(),
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
        "building next-session features for SPY and watchlist assets, and evaluating long/flat strategies with walk-forward testing.",
    )

    store = DuckDBStore(settings)
    ingestion_service = IngestionService()
    market_service = MarketDataService()
    discovery_service = DiscoveryService()
    health_service = DataHealthService()
    enrichment_service = LLMEnrichmentService(store)
    feature_service = FeatureService(enrichment_service)
    model_service = ModelService()
    backtest_service = BacktestService(model_service)
    experiment_store = ExperimentStore(store)

    _ensure_bootstrap(settings, store, ingestion_service, market_service, discovery_service, feature_service, health_service)

    _render_sidebar_access_panel(settings)
    pages = ["Research View", "Datasets", "Discovery", "Models & Backtests", "Historical Replay", "Live Monitor"]
    page = st.segmented_control(
        "Workbench",
        options=pages,
        default=pages[0],
        key="workbench_page",
        label_visibility="collapsed",
        width="stretch",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Storage: DuckDB + Parquet")
    st.sidebar.code(str(settings.state_root))
    st.sidebar.code(str(settings.db_path))
    if page is None:
        page = pages[0]

    if page == "Research View":
        render_research_view(settings, store, market_service, feature_service)
    elif page == "Datasets":
        render_datasets_view(settings, store, ingestion_service, market_service, discovery_service, feature_service, health_service)
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
            health_service,
            model_service,
            experiment_store,
        )
