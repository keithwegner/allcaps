from __future__ import annotations

import os
import time
from contextlib import suppress
from dataclasses import dataclass

import pandas as pd

from .config import AppSettings
from .discovery import DiscoveryService
from .enrichment import LLMEnrichmentService
from .experiments import ExperimentStore
from .features import FeatureService
from .health import DataHealthService, ensure_refresh_history_frame
from .ingestion import IngestionService
from .live_monitor import build_live_portfolio_run_state, validate_live_monitor_config
from .market import MarketDataService
from .modeling import ModelService
from .paper_trading import PaperTradingService, paper_config_matches_live
from .runtime import missing_core_datasets, refresh_datasets
from .storage import DuckDBStore


@dataclass(frozen=True)
class SchedulerDecision:
    refresh_mode: str = ""
    incremental: bool = False

    @property
    def should_run(self) -> bool:
        return bool(self.refresh_mode)


def acquire_refresh_lock(settings: AppSettings) -> int | None:
    try:
        return os.open(str(settings.scheduler_lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None


def release_refresh_lock(settings: AppSettings, lock_fd: int | None) -> None:
    if lock_fd is None:
        return
    with suppress(OSError):
        os.close(lock_fd)
    with suppress(FileNotFoundError):
        settings.scheduler_lock_path.unlink()


def choose_scheduler_refresh(
    store: DuckDBStore,
    settings: AppSettings,
    now: pd.Timestamp | None = None,
) -> SchedulerDecision:
    if missing_core_datasets(store):
        return SchedulerDecision(refresh_mode="bootstrap", incremental=False)

    current_ts = pd.Timestamp.now(tz=settings.timezone) if now is None else pd.Timestamp(now)
    if current_ts.tzinfo is None:
        current_ts = current_ts.tz_localize(settings.timezone)
    else:
        current_ts = current_ts.tz_convert(settings.timezone)

    history = ensure_refresh_history_frame(store.read_frame("refresh_history"))
    if history.empty:
        return SchedulerDecision(refresh_mode="full", incremental=False)

    success_history = history.loc[history["status"].astype(str) == "success"].copy()
    if success_history.empty:
        return SchedulerDecision(refresh_mode="full", incremental=False)

    success_history["completed_at"] = pd.to_datetime(success_history["completed_at"], errors="coerce", utc=True)
    success_history = success_history.dropna(subset=["completed_at"]).copy()
    if success_history.empty:
        return SchedulerDecision(refresh_mode="full", incremental=False)
    success_history["completed_at_local"] = success_history["completed_at"].dt.tz_convert(settings.timezone)

    scheduled_full_at = current_ts.normalize() + pd.Timedelta(hours=settings.scheduler_full_hour, minutes=settings.scheduler_full_minute)
    full_history = success_history.loc[
        success_history["refresh_mode"].astype(str).isin(["bootstrap", "full"])
    ].copy()
    last_full = full_history["completed_at_local"].max() if not full_history.empty else pd.NaT
    if current_ts >= scheduled_full_at and (pd.isna(last_full) or pd.Timestamp(last_full) < scheduled_full_at):
        return SchedulerDecision(refresh_mode="full", incremental=False)

    last_success = success_history["completed_at_local"].max()
    if pd.isna(last_success):
        return SchedulerDecision(refresh_mode="incremental", incremental=True)
    elapsed_minutes = (current_ts - pd.Timestamp(last_success)).total_seconds() / 60.0
    if elapsed_minutes >= float(settings.scheduler_incremental_minutes):
        return SchedulerDecision(refresh_mode="incremental", incremental=True)
    return SchedulerDecision()


def run_scheduler_cycle(
    *,
    settings: AppSettings,
    store: DuckDBStore,
    decision: SchedulerDecision,
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
    feature_service: FeatureService,
    health_service: DataHealthService,
    experiment_store: ExperimentStore,
    model_service: ModelService,
    paper_service: PaperTradingService,
    generated_at: pd.Timestamp | None = None,
) -> SchedulerDecision:
    if not decision.should_run:
        return decision

    refresh_datasets(
        settings=settings,
        store=store,
        ingestion_service=ingestion_service,
        market_service=market_service,
        discovery_service=discovery_service,
        feature_service=feature_service,
        health_service=health_service,
        remote_url=settings.remote_x_csv_url,
        uploaded_files=[],
        incremental=decision.incremental,
        refresh_mode=decision.refresh_mode,
    )
    live_config = experiment_store.load_live_monitor_config()
    if live_config is None:
        return decision

    runs = experiment_store.list_runs()
    live_errors = validate_live_monitor_config(live_config, runs)
    if live_errors or str(live_config.mode or "portfolio_run") != "portfolio_run":
        return decision

    snapshot_time = pd.Timestamp.utcnow().floor("s") if generated_at is None else pd.Timestamp(generated_at).floor("s")
    board, decision_row, _, _warnings = build_live_portfolio_run_state(
        store=store,
        model_service=model_service,
        experiment_store=experiment_store,
        config=live_config,
        generated_at=snapshot_time,
    )
    if board.empty or decision_row.empty:
        return decision

    experiment_store.save_live_asset_snapshots(board)
    experiment_store.save_live_decision_snapshots(decision_row)
    paper_config = paper_service.load_current_config()
    if paper_config_matches_live(paper_config, live_config):
        paper_service.process_live_history(paper_config, as_of=snapshot_time)
    return decision


def run_scheduler_once(settings: AppSettings) -> SchedulerDecision:
    store = DuckDBStore(settings)
    decision = choose_scheduler_refresh(store, settings)
    if not decision.should_run:
        return decision

    lock_fd = acquire_refresh_lock(settings)
    if lock_fd is None:
        return SchedulerDecision()

    try:
        ingestion_service = IngestionService()
        market_service = MarketDataService()
        discovery_service = DiscoveryService()
        health_service = DataHealthService()
        enrichment_service = LLMEnrichmentService(store)
        feature_service = FeatureService(enrichment_service)
        experiment_store = ExperimentStore(store)
        model_service = ModelService()
        paper_service = PaperTradingService(store)
        return run_scheduler_cycle(
            settings=settings,
            store=store,
            decision=decision,
            ingestion_service=ingestion_service,
            market_service=market_service,
            discovery_service=discovery_service,
            feature_service=feature_service,
            health_service=health_service,
            experiment_store=experiment_store,
            model_service=model_service,
            paper_service=paper_service,
        )
    finally:
        release_refresh_lock(settings, lock_fd)


def main() -> None:
    settings = AppSettings()
    if not settings.scheduler_enabled:
        return
    while True:
        with suppress(Exception):
            run_scheduler_once(settings)
        time.sleep(settings.scheduler_loop_seconds)


if __name__ == "__main__":
    main()
