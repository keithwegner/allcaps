from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from .config import AppSettings
from .enrichment import LLMEnrichmentService
from .experiments import ExperimentStore
from .health import DataHealthService, build_health_summary, build_health_trend_frame, ensure_refresh_history_frame
from .live_monitor import build_live_portfolio_run_state, validate_live_monitor_config
from .modeling import ModelService
from .paper_trading import (
    PaperTradingService,
    ensure_paper_benchmark_curve_frame,
    ensure_paper_decision_journal_frame,
    ensure_paper_equity_curve_frame,
    ensure_paper_portfolio_registry_frame,
    ensure_paper_trade_ledger_frame,
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
)
from .runtime import missing_core_datasets
from .research_asset_lab import build_research_asset_lab
from .research_workspace import build_research_workspace, detect_source_mode
from .run_explorer import build_run_comparison_payload, build_run_detail_payload
from .storage import DuckDBStore


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


def _frame_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    if limit is not None:
        out = out.head(max(0, int(limit)))
    return [_json_safe(record) for record in out.to_dict(orient="records")]


def create_app(settings: AppSettings | None = None, store: DuckDBStore | None = None) -> FastAPI:
    settings = settings or AppSettings()
    store = store or DuckDBStore(settings)
    health_service = DataHealthService()
    experiment_store = ExperimentStore(store)
    model_service = ModelService()
    enrichment_service = LLMEnrichmentService(store)
    paper_service = PaperTradingService(store)
    performance_service = PerformanceObservatoryService(store)

    app = FastAPI(
        title=f"{settings.title} API",
        version="0.1.0",
        description="Read-only API foundation for the web-first AllCaps frontend migration.",
    )
    app.state.settings = settings
    app.state.store = store

    if settings.api_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.api_cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "OPTIONS"],
            allow_headers=["*"],
        )

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        posts = store.read_frame("normalized_posts")
        registry = store.dataset_registry()
        return _json_safe(
            {
                "title": settings.title,
                "mode": "public" if settings.public_mode else "private",
                "state_root": str(settings.state_root),
                "db_path": str(settings.db_path),
                "source_mode": detect_source_mode(posts),
                "missing_core_datasets": missing_core_datasets(store),
                "dataset_count": int(len(registry)),
            },
        )

    @app.get("/api/research")
    def research(
        date_start: str | None = None,
        date_end: str | None = None,
        platforms: list[str] | None = Query(default=None),
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
        narrative_platforms: list[str] | None = Query(default=None),
        narrative_tracked_scope: str | None = None,
        narrative_bucket_field: str | None = None,
    ) -> dict[str, Any]:
        result = build_research_workspace(
            settings=settings,
            store=store,
            enrichment_service=enrichment_service,
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
        return _json_safe(result.payload)

    @app.get("/api/research/export")
    def research_export(
        date_start: str | None = None,
        date_end: str | None = None,
        platforms: list[str] | None = Query(default=None),
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
        narrative_platforms: list[str] | None = Query(default=None),
        narrative_tracked_scope: str | None = None,
        narrative_bucket_field: str | None = None,
    ) -> Response:
        result = build_research_workspace(
            settings=settings,
            store=store,
            enrichment_service=enrichment_service,
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
        return Response(
            content=result.export_bundle,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{result.export_filename}"'},
        )

    @app.get("/api/research/assets")
    def research_assets(
        date_start: str | None = None,
        date_end: str | None = None,
        platforms: list[str] | None = Query(default=None),
        include_reshares: bool | None = None,
        tracked_only: bool | None = None,
        trump_authored_only: bool | None = None,
        keyword: str | None = None,
        selected_asset: str | None = None,
        comparison_mode: str | None = None,
        benchmark_symbol: str | None = None,
        pre_sessions: int | None = None,
        post_sessions: int | None = None,
        intraday_session_date: str | None = None,
        intraday_anchor_post_id: str | None = None,
        before_minutes: int | None = None,
        after_minutes: int | None = None,
    ) -> dict[str, Any]:
        return _json_safe(
            build_research_asset_lab(
                settings=settings,
                store=store,
                date_start=date_start,
                date_end=date_end,
                platforms=platforms,
                include_reshares=include_reshares,
                tracked_only=tracked_only,
                trump_authored_only=trump_authored_only,
                keyword=keyword,
                selected_asset=selected_asset,
                comparison_mode=comparison_mode,
                benchmark_symbol=benchmark_symbol,
                pre_sessions=pre_sessions,
                post_sessions=post_sessions,
                intraday_session_date=intraday_session_date,
                intraday_anchor_post_id=intraday_anchor_post_id,
                before_minutes=before_minutes,
                after_minutes=after_minutes,
            ),
        )

    @app.get("/api/datasets/health")
    def dataset_health() -> dict[str, Any]:
        latest = store.read_frame("data_health_latest")
        if latest.empty:
            latest = health_service.evaluate_store(store)
        history = store.read_frame("data_health_history")
        refresh_history = ensure_refresh_history_frame(store.read_frame("refresh_history"))
        trend_source = history if not history.empty else latest
        summary = build_health_summary(latest, refresh_history)
        return {
            "summary": _json_safe(summary),
            "latest": _frame_records(latest),
            "history": _frame_records(history, limit=500),
            "trend": _frame_records(build_health_trend_frame(trend_source)),
            "refresh_history": _frame_records(refresh_history.tail(25)),
            "registry": _frame_records(store.dataset_registry()),
        }

    @app.get("/api/runs")
    def runs() -> dict[str, Any]:
        saved_runs = experiment_store.list_runs()
        return {
            "count": int(len(saved_runs)),
            "runs": _frame_records(saved_runs),
        }

    @app.get("/api/runs/compare")
    def compare_runs(
        run_ids: list[str] | None = Query(default=None),
        base_run_id: str | None = None,
    ) -> dict[str, Any]:
        return build_run_comparison_payload(
            experiment_store,
            run_ids or [],
            base_run_id=base_run_id,
        )

    @app.get("/api/runs/{run_id}")
    def run_detail(
        run_id: str,
        variant_name: str | None = None,
        session_date: str | None = None,
    ) -> dict[str, Any]:
        return build_run_detail_payload(
            experiment_store,
            run_id,
            variant_name=variant_name,
            session_date=session_date,
        )

    @app.get("/api/live/current")
    def live_current() -> dict[str, Any]:
        live_config = experiment_store.load_live_monitor_config()
        saved_runs = experiment_store.list_runs()
        if live_config is None:
            return {
                "configured": False,
                "errors": ["No live monitor config has been saved yet."],
                "warnings": [],
                "decision": None,
                "board": [],
            }

        errors = validate_live_monitor_config(live_config, saved_runs)
        if errors or str(live_config.mode or "portfolio_run") != "portfolio_run":
            return {
                "configured": False,
                "errors": errors or ["Only portfolio-run live configs are supported by the web API."],
                "warnings": [],
                "decision": None,
                "board": [],
            }

        board, decision, _explanations, warnings = build_live_portfolio_run_state(
            store=store,
            model_service=model_service,
            experiment_store=experiment_store,
            config=live_config,
            generated_at=pd.Timestamp.utcnow().floor("s"),
        )
        return {
            "configured": True,
            "errors": [],
            "warnings": warnings,
            "decision": _frame_records(decision)[0] if not decision.empty else None,
            "board": _frame_records(board),
        }

    @app.get("/api/paper/portfolios")
    def paper_portfolios() -> dict[str, Any]:
        registry = paper_service.list_portfolios()
        current_config = paper_service.load_current_config()
        return {
            "current_config": _json_safe(current_config.to_dict()) if current_config is not None else None,
            "portfolios": _frame_records(registry),
        }

    @app.get("/api/paper/{paper_portfolio_id}")
    def paper_portfolio(paper_portfolio_id: str) -> dict[str, Any]:
        registry = ensure_paper_portfolio_registry_frame(store.read_frame("paper_portfolio_registry"))
        journal = ensure_paper_decision_journal_frame(store.read_frame("paper_decision_journal"))
        trades = ensure_paper_trade_ledger_frame(store.read_frame("paper_trade_ledger"))
        equity = ensure_paper_equity_curve_frame(store.read_frame("paper_equity_curve"))
        benchmark = ensure_paper_benchmark_curve_frame(store.read_frame("paper_benchmark_curve"))
        return {
            "registry": _frame_records(registry.loc[registry["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)]),
            "decision_journal": _frame_records(journal.loc[journal["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].tail(100)),
            "trade_ledger": _frame_records(trades.loc[trades["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].tail(100)),
            "equity_curve": _frame_records(equity.loc[equity["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)]),
            "benchmark_curve": _frame_records(benchmark.loc[benchmark["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)]),
        }

    @app.get("/api/performance/{paper_portfolio_id}")
    def performance(paper_portfolio_id: str) -> dict[str, Any]:
        diagnostics = performance_service.load_latest_for_portfolio(paper_portfolio_id)
        persisted = not diagnostics.empty
        if diagnostics.empty:
            diagnostics = performance_service.evaluate_paper_portfolio(paper_portfolio_id)

        registry = ensure_paper_portfolio_registry_frame(store.read_frame("paper_portfolio_registry"))
        journal = ensure_paper_decision_journal_frame(store.read_frame("paper_decision_journal"))
        trades = ensure_paper_trade_ledger_frame(store.read_frame("paper_trade_ledger"))
        equity = ensure_paper_equity_curve_frame(store.read_frame("paper_equity_curve"))
        benchmark = ensure_paper_benchmark_curve_frame(store.read_frame("paper_benchmark_curve"))
        registry_rows = registry.loc[registry["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
        portfolio_run_id = str(registry_rows.iloc[0]["portfolio_run_id"]) if not registry_rows.empty else ""
        deployment_variant = str(registry_rows.iloc[0]["deployment_variant"]) if not registry_rows.empty else ""
        return {
            "persisted": persisted,
            "summary": _json_safe(
                build_performance_summary(
                    diagnostics=diagnostics,
                    registry=registry,
                    journal=journal,
                    trades=trades,
                    equity=equity,
                    benchmark=benchmark,
                    paper_portfolio_id=paper_portfolio_id,
                ),
            ),
            "diagnostics": _frame_records(diagnostics),
            "equity_comparison": _frame_records(build_equity_comparison_frame(equity, benchmark, paper_portfolio_id)),
            "rolling_returns": _frame_records(build_rolling_return_frame(trades, paper_portfolio_id)),
            "score_outcomes": _frame_records(build_score_outcome_frame(journal, trades, paper_portfolio_id)),
            "score_buckets": _frame_records(build_score_bucket_outcome_frame(journal, trades, paper_portfolio_id)),
            "winner_distribution": _frame_records(build_winner_distribution_frame(journal, paper_portfolio_id)),
            "drift": _frame_records(
                build_live_score_drift_frame(
                    live_asset_snapshots=store.read_frame("live_asset_snapshots"),
                    portfolio_run_id=portfolio_run_id,
                    deployment_variant=deployment_variant,
                ),
            ),
        }

    return app


app = create_app()
