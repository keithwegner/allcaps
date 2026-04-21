from __future__ import annotations

import secrets
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .access import verify_admin_password
from .backtesting import BacktestService
from .config import AppSettings
from .dataset_admin import (
    UploadedCsv,
    build_dataset_admin_payload,
    ensure_refresh_jobs_frame,
    save_dataset_watchlist,
    submit_refresh_job,
)
from .discovery import DiscoveryService
from .discovery_admin import (
    DiscoveryAdminError,
    DiscoveryOverrideMutation,
    create_discovery_override,
    delete_discovery_override,
)
from .discovery_workspace import build_discovery_workspace
from .enrichment import LLMEnrichmentService
from .experiments import ExperimentStore
from .features import FeatureService
from .health import DataHealthService, build_health_summary, build_health_trend_frame, ensure_refresh_history_frame
from .historical_replay import build_historical_replay_payload, build_historical_replay_session_payload
from .ingestion import IngestionService
from .live_monitor import build_live_portfolio_run_state, validate_live_monitor_config
from .live_ops import (
    apply_paper_action,
    build_live_config_from_run,
    build_live_ops_payload,
    run_live_capture,
)
from .market import MarketDataService
from .modeling import ModelService
from .model_training import (
    ModelTrainingRequest,
    build_model_training_payload,
    ensure_model_training_jobs_frame,
    submit_model_training_job,
)
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

ADMIN_TOKEN_TTL_SECONDS = 12 * 60 * 60


class AdminSessionRequest(BaseModel):
    password: str = ""


class DiscoveryOverrideRequest(BaseModel):
    account_id: str
    handle: str = ""
    display_name: str = ""
    source_platform: str = "X"
    action: str
    effective_from: str
    effective_to: str | None = None
    note: str = ""

    def to_mutation(self) -> DiscoveryOverrideMutation:
        return DiscoveryOverrideMutation(
            account_id=self.account_id,
            handle=self.handle,
            display_name=self.display_name,
            source_platform=self.source_platform,
            action=self.action,
            effective_from=self.effective_from,
            effective_to=self.effective_to,
            note=self.note,
        )


class LiveConfigSaveRequest(BaseModel):
    portfolio_run_id: str
    fallback_mode: str = "SPY"


class PaperCurrentActionRequest(BaseModel):
    action: str
    starting_cash: float | None = None


class WatchlistSaveRequest(BaseModel):
    symbols: list[str] = []
    reset: bool = False


class ModelTrainingJobRequest(BaseModel):
    workflow_mode: str
    run_name: str = ""
    target_asset: str = "SPY"
    feature_version: str = ""
    llm_enabled: bool = False
    train_window: int = 90
    validation_window: int = 30
    test_window: int = 30
    step_size: int = 30
    transaction_cost_bps: float = 2.0
    ridge_alpha: float = 1.0
    threshold_grid: str | list[float] = "0,0.001,0.0025,0.005"
    minimum_signal_grid: str | list[int] = "1,2,3"
    account_weight_grid: str | list[float] = "0.5,1.0,1.5"
    fallback_mode: str = "SPY"
    component_run_ids: list[str] = []
    selected_symbols: list[str] = []
    topology_variants: list[str] = []
    model_families: list[str] = []
    narrative_feature_modes: list[str] = []

    def to_training_request(self) -> ModelTrainingRequest:
        return ModelTrainingRequest(
            workflow_mode=self.workflow_mode,
            run_name=self.run_name,
            target_asset=self.target_asset,
            feature_version=self.feature_version,
            llm_enabled=self.llm_enabled,
            train_window=self.train_window,
            validation_window=self.validation_window,
            test_window=self.test_window,
            step_size=self.step_size,
            transaction_cost_bps=self.transaction_cost_bps,
            ridge_alpha=self.ridge_alpha,
            threshold_grid=self.threshold_grid,
            minimum_signal_grid=self.minimum_signal_grid,
            account_weight_grid=self.account_weight_grid,
            fallback_mode=self.fallback_mode,
            component_run_ids=tuple(self.component_run_ids),
            selected_symbols=tuple(self.selected_symbols),
            topology_variants=tuple(self.topology_variants),
            model_families=tuple(self.model_families),
            narrative_feature_modes=tuple(self.narrative_feature_modes),
        )


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


def create_app(
    settings: AppSettings | None = None,
    store: DuckDBStore | None = None,
    ingestion_service: IngestionService | None = None,
    market_service: MarketDataService | None = None,
    discovery_service: DiscoveryService | None = None,
    feature_service: FeatureService | None = None,
    health_service: DataHealthService | None = None,
    backtest_service: BacktestService | None = None,
    run_refresh_jobs_inline: bool = False,
    run_model_training_jobs_inline: bool = False,
) -> FastAPI:
    settings = settings or AppSettings()
    store = store or DuckDBStore(settings)
    health_service = health_service or DataHealthService()
    discovery_service = discovery_service or DiscoveryService()
    experiment_store = ExperimentStore(store)
    model_service = ModelService()
    enrichment_service = LLMEnrichmentService(store)
    ingestion_service = ingestion_service or IngestionService()
    market_service = market_service or MarketDataService()
    feature_service = feature_service or FeatureService(enrichment_service)
    backtest_service = backtest_service or BacktestService(model_service)
    paper_service = PaperTradingService(store)
    performance_service = PerformanceObservatoryService(store)

    app = FastAPI(
        title=f"{settings.title} API",
        version="0.1.0",
        description="Read-only API foundation for the web-first AllCaps frontend migration.",
    )
    app.state.settings = settings
    app.state.store = store
    app.state.admin_tokens = {}
    app.state.run_refresh_jobs_inline = bool(run_refresh_jobs_inline)
    app.state.run_model_training_jobs_inline = bool(run_model_training_jobs_inline)

    if settings.api_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.api_cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    def issue_admin_token() -> dict[str, Any]:
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + ADMIN_TOKEN_TTL_SECONDS
        app.state.admin_tokens[token] = expires_at
        return {
            "token": token,
            "token_type": "bearer",
            "expires_at": pd.Timestamp(expires_at, unit="s", tz="UTC").isoformat(),
            "expires_in_seconds": ADMIN_TOKEN_TTL_SECONDS,
            "mode": "public" if settings.public_mode else "private",
        }

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Admin token is required.")
        token = authorization.split(" ", 1)[1].strip()
        now = time.time()
        tokens = {
            key: expiry
            for key, expiry in dict(app.state.admin_tokens).items()
            if float(expiry) > now
        }
        app.state.admin_tokens = tokens
        expiry = tokens.get(token)
        if expiry is None or float(expiry) <= now:
            raise HTTPException(status_code=401, detail="Admin token is invalid or expired.")

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

    @app.post("/api/admin/session")
    def admin_session(request: AdminSessionRequest | None = None) -> dict[str, Any]:
        password = request.password if request is not None else ""
        if settings.public_mode and not verify_admin_password(settings, password):
            raise HTTPException(status_code=401, detail="Invalid admin password.")
        return _json_safe(issue_admin_token())

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

    @app.get("/api/discovery")
    def discovery() -> dict[str, Any]:
        return _json_safe(build_discovery_workspace(store, discovery_service=discovery_service, settings=settings))

    @app.post("/api/discovery/overrides")
    def create_discovery_override_endpoint(
        request: DiscoveryOverrideRequest,
        _: None = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            create_discovery_override(
                settings=settings,
                store=store,
                discovery_service=discovery_service,
                request=request.to_mutation(),
            )
        except DiscoveryAdminError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _json_safe(build_discovery_workspace(store, discovery_service=discovery_service, settings=settings))

    @app.delete("/api/discovery/overrides/{override_id}")
    def delete_discovery_override_endpoint(
        override_id: str,
        _: None = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            delete_discovery_override(
                settings=settings,
                store=store,
                discovery_service=discovery_service,
                override_id=override_id,
            )
        except DiscoveryAdminError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _json_safe(build_discovery_workspace(store, discovery_service=discovery_service, settings=settings))

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

    @app.get("/api/datasets/admin")
    def dataset_admin() -> dict[str, Any]:
        return _json_safe(
            build_dataset_admin_payload(
                settings=settings,
                store=store,
                health_service=health_service,
                public_mode=settings.public_mode,
            ),
        )

    @app.post("/api/datasets/watchlist")
    def save_watchlist_endpoint(
        request: WatchlistSaveRequest,
        _admin: None = Depends(require_admin),
    ) -> dict[str, Any]:
        save_dataset_watchlist(store, request.symbols, reset=request.reset)
        return _json_safe(
            build_dataset_admin_payload(
                settings=settings,
                store=store,
                health_service=health_service,
                public_mode=settings.public_mode,
            ),
        )

    @app.post("/api/datasets/refresh")
    async def start_dataset_refresh(
        refresh_mode: str = Form("incremental"),
        remote_url: str = Form(""),
        files: list[UploadFile] | None = File(default=None),
        _admin: None = Depends(require_admin),
    ) -> dict[str, Any]:
        uploaded_files: list[UploadedCsv] = []
        for upload in files or []:
            uploaded_files.append(UploadedCsv(name=upload.filename or "uploaded.csv", raw_bytes=await upload.read()))
        job_id, errors = submit_refresh_job(
            settings=settings,
            store=store,
            refresh_mode=refresh_mode,
            remote_url=remote_url,
            uploaded_files=uploaded_files,
            ingestion_service=ingestion_service,
            market_service=market_service,
            discovery_service=discovery_service,
            feature_service=feature_service,
            health_service=health_service,
            run_inline=bool(app.state.run_refresh_jobs_inline),
        )
        if errors:
            status_code = 409 if any("already running" in error for error in errors) else 400
            raise HTTPException(status_code=status_code, detail=errors)
        payload = build_dataset_admin_payload(
            settings=settings,
            store=store,
            health_service=health_service,
            public_mode=settings.public_mode,
            active_job_id=job_id,
        )
        payload["job_id"] = job_id
        return _json_safe(payload)

    @app.get("/api/datasets/jobs/{job_id}")
    def dataset_refresh_job(job_id: str) -> dict[str, Any]:
        jobs = ensure_refresh_jobs_frame(store.read_frame("dataset_refresh_jobs"))
        current = jobs.loc[jobs["job_id"].astype(str) == str(job_id)].copy()
        return _json_safe(
            {
                "job_id": job_id,
                "found": not current.empty,
                "job": _frame_records(current.tail(1))[0] if not current.empty else None,
                "recent_jobs": _frame_records(jobs.tail(10)),
            },
        )

    @app.get("/api/models/training")
    def model_training() -> dict[str, Any]:
        return _json_safe(
            build_model_training_payload(
                store=store,
                experiment_store=experiment_store,
            ),
        )

    @app.post("/api/models/jobs")
    def start_model_training_job(
        request: ModelTrainingJobRequest,
        _admin: None = Depends(require_admin),
    ) -> dict[str, Any]:
        job_id, errors = submit_model_training_job(
            store=store,
            experiment_store=experiment_store,
            feature_service=feature_service,
            backtest_service=backtest_service,
            request=request.to_training_request(),
            run_inline=bool(app.state.run_model_training_jobs_inline),
        )
        if errors:
            status_code = 409 if any("already running" in error for error in errors) else 400
            raise HTTPException(status_code=status_code, detail=errors)
        payload = build_model_training_payload(
            store=store,
            experiment_store=experiment_store,
            active_job_id=job_id,
        )
        payload["job_id"] = job_id
        return _json_safe(payload)

    @app.get("/api/models/jobs/{job_id}")
    def model_training_job(job_id: str) -> dict[str, Any]:
        jobs = ensure_model_training_jobs_frame(store.read_frame("model_training_jobs"))
        current = jobs.loc[jobs["job_id"].astype(str) == str(job_id)].copy()
        return _json_safe(
            {
                "job_id": job_id,
                "found": not current.empty,
                "job": _frame_records(current.tail(1))[0] if not current.empty else None,
                "recent_jobs": _frame_records(jobs.tail(10)),
            },
        )

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

    @app.get("/api/replay")
    def replay(run_id: str | None = None) -> dict[str, Any]:
        return _json_safe(
            build_historical_replay_payload(
                store=store,
                experiment_store=experiment_store,
                feature_service=feature_service,
                run_id=run_id,
            ),
        )

    @app.get("/api/replay/session")
    def replay_session(run_id: str, signal_session_date: str) -> dict[str, Any]:
        return _json_safe(
            build_historical_replay_session_payload(
                store=store,
                experiment_store=experiment_store,
                feature_service=feature_service,
                backtest_service=backtest_service,
                run_id=run_id,
                signal_session_date=signal_session_date,
            ),
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

    @app.get("/api/live/ops")
    def live_ops() -> dict[str, Any]:
        return _json_safe(
            build_live_ops_payload(
                store=store,
                experiment_store=experiment_store,
                model_service=model_service,
                paper_service=paper_service,
                performance_service=performance_service,
                public_mode=settings.public_mode,
            ),
        )

    @app.post("/api/live/config")
    def save_live_config(
        request: LiveConfigSaveRequest,
        _admin: None = Depends(require_admin),
    ) -> dict[str, Any]:
        config, errors = build_live_config_from_run(
            experiment_store.list_runs(),
            portfolio_run_id=request.portfolio_run_id,
            fallback_mode=request.fallback_mode,
        )
        if errors or config is None:
            raise HTTPException(status_code=400, detail=errors or ["Invalid live config."])
        experiment_store.save_live_monitor_config(config)
        return _json_safe(
            build_live_ops_payload(
                store=store,
                experiment_store=experiment_store,
                model_service=model_service,
                paper_service=paper_service,
                performance_service=performance_service,
                public_mode=settings.public_mode,
            ),
        )

    @app.post("/api/live/capture")
    def capture_live(
        _admin: None = Depends(require_admin),
    ) -> dict[str, Any]:
        capture_result = run_live_capture(
            store=store,
            experiment_store=experiment_store,
            model_service=model_service,
            paper_service=paper_service,
            performance_service=performance_service,
        )
        if capture_result.get("errors"):
            raise HTTPException(status_code=400, detail=capture_result["errors"])
        return _json_safe(
            build_live_ops_payload(
                store=store,
                experiment_store=experiment_store,
                model_service=model_service,
                paper_service=paper_service,
                performance_service=performance_service,
                capture_result=capture_result,
                public_mode=settings.public_mode,
            ),
        )

    @app.get("/api/paper/portfolios")
    def paper_portfolios() -> dict[str, Any]:
        registry = paper_service.list_portfolios()
        current_config = paper_service.load_current_config()
        return {
            "current_config": _json_safe(current_config.to_dict()) if current_config is not None else None,
            "portfolios": _frame_records(registry),
        }

    @app.post("/api/paper/current")
    def paper_current_action(
        request: PaperCurrentActionRequest,
        _admin: None = Depends(require_admin),
    ) -> dict[str, Any]:
        _config, errors = apply_paper_action(
            store=store,
            experiment_store=experiment_store,
            paper_service=paper_service,
            action=request.action,
            starting_cash=request.starting_cash,
        )
        if errors:
            raise HTTPException(status_code=400, detail=errors)
        return _json_safe(
            build_live_ops_payload(
                store=store,
                experiment_store=experiment_store,
                model_service=model_service,
                paper_service=paper_service,
                performance_service=performance_service,
                public_mode=settings.public_mode,
            ),
        )

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
