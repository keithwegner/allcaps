from __future__ import annotations

from dataclasses import dataclass
import threading
import uuid
from typing import Any

import pandas as pd

from .config import AppSettings
from .discovery import DiscoveryService
from .features import FeatureService
from .health import DataHealthService, build_health_summary, build_health_trend_frame, ensure_refresh_history_frame
from .ingestion import IngestionService
from .market import MarketDataService, normalize_symbols
from .research_workspace import detect_source_mode, source_mode_label
from .runtime import missing_core_datasets, refresh_datasets, save_watchlist, watchlist_symbols
from .scheduler import acquire_refresh_lock, release_refresh_lock
from .storage import DuckDBStore

DATASET_REFRESH_JOB_COLUMNS = [
    "job_id",
    "refresh_id",
    "refresh_mode",
    "status",
    "started_at",
    "completed_at",
    "error_message",
    "remote_url",
    "uploaded_file_count",
    "normalized_post_count",
    "asset_daily_row_count",
    "asset_intraday_row_count",
    "tracked_account_count",
]

VALID_REFRESH_MODES = {"bootstrap", "full", "incremental"}


@dataclass(frozen=True)
class UploadedCsv:
    name: str
    raw_bytes: bytes

    def getvalue(self) -> bytes:
        return self.raw_bytes


def ensure_refresh_jobs_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=DATASET_REFRESH_JOB_COLUMNS)
    out = frame.copy()
    for column in DATASET_REFRESH_JOB_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out["started_at"] = pd.to_datetime(out["started_at"], errors="coerce", utc=True)
    out["completed_at"] = pd.to_datetime(out["completed_at"], errors="coerce", utc=True)
    for column in ["uploaded_file_count", "normalized_post_count", "asset_daily_row_count", "asset_intraday_row_count", "tracked_account_count"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["job_id", "refresh_id", "refresh_mode", "status", "error_message", "remote_url"]:
        out[column] = out[column].fillna("").astype(str)
    return out[DATASET_REFRESH_JOB_COLUMNS].copy()


def _frame_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    if limit is not None:
        out = out.tail(max(0, int(limit)))
    return out.to_dict(orient="records")


def _count_rows(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key)
    if isinstance(value, pd.DataFrame):
        return int(len(value))
    return 0


def _job_row(
    *,
    job_id: str,
    refresh_mode: str,
    status: str,
    started_at: pd.Timestamp,
    completed_at: pd.Timestamp | pd.NaT = pd.NaT,
    error_message: str = "",
    remote_url: str = "",
    uploaded_file_count: int = 0,
    refresh_id: str = "",
    normalized_post_count: int | float | None = None,
    asset_daily_row_count: int | float | None = None,
    asset_intraday_row_count: int | float | None = None,
    tracked_account_count: int | float | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "job_id": str(job_id),
                "refresh_id": str(refresh_id or ""),
                "refresh_mode": str(refresh_mode),
                "status": str(status),
                "started_at": pd.Timestamp(started_at),
                "completed_at": completed_at,
                "error_message": str(error_message or ""),
                "remote_url": str(remote_url or ""),
                "uploaded_file_count": int(uploaded_file_count),
                "normalized_post_count": normalized_post_count,
                "asset_daily_row_count": asset_daily_row_count,
                "asset_intraday_row_count": asset_intraday_row_count,
                "tracked_account_count": tracked_account_count,
            },
        ],
        columns=DATASET_REFRESH_JOB_COLUMNS,
    )


def save_refresh_job(store: DuckDBStore, row: pd.DataFrame) -> pd.DataFrame:
    current = ensure_refresh_jobs_frame(store.read_frame("dataset_refresh_jobs"))
    combined = pd.concat([current, ensure_refresh_jobs_frame(row)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["job_id"], keep="last").reset_index(drop=True)
    store.save_frame(
        "dataset_refresh_jobs",
        combined,
        metadata={
            "row_count": int(len(combined)),
            "latest_status": str(combined.iloc[-1]["status"]) if not combined.empty else "",
        },
    )
    return combined


def _health_latest(store: DuckDBStore, health_service: DataHealthService) -> pd.DataFrame:
    latest = store.read_frame("data_health_latest")
    if latest.empty:
        latest = health_service.evaluate_store(store)
    return latest


def build_dataset_admin_payload(
    *,
    settings: AppSettings,
    store: DuckDBStore,
    health_service: DataHealthService,
    public_mode: bool = False,
    active_job_id: str = "",
) -> dict[str, Any]:
    posts = store.read_frame("normalized_posts")
    source_mode = detect_source_mode(posts)
    latest = _health_latest(store, health_service)
    history = store.read_frame("data_health_history")
    refresh_history = ensure_refresh_history_frame(store.read_frame("refresh_history"))
    trend_source = history if not history.empty else latest
    refresh_jobs = ensure_refresh_jobs_frame(store.read_frame("dataset_refresh_jobs"))
    watchlist = watchlist_symbols(store)
    asset_universe = store.read_frame("asset_universe")
    registry = store.dataset_registry()

    last_refresh = {}
    if not refresh_history.empty:
        order_col = "completed_at" if refresh_history["completed_at"].notna().any() else "started_at"
        last_refresh = refresh_history.sort_values(order_col).tail(1).to_dict(orient="records")[0]

    missing = missing_core_datasets(store)
    return {
        "admin": {
            "mode": "public" if public_mode else "private",
            "write_requires_unlock": bool(public_mode),
        },
        "status": {
            "operating_mode": source_mode_label(source_mode),
            "source_mode": source_mode,
            "state_root": str(settings.state_root),
            "db_path": str(settings.db_path),
            "scheduler_enabled": bool(settings.scheduler_enabled),
            "missing_core_datasets": missing,
            "missing_core_dataset_count": int(len(missing)),
            "last_refresh": last_refresh,
            "active_job_id": str(active_job_id or ""),
        },
        "watchlist_symbols": watchlist,
        "asset_universe": _frame_records(asset_universe, limit=500),
        "summary": build_health_summary(latest, refresh_history),
        "latest": _frame_records(latest, limit=500),
        "trend": _frame_records(build_health_trend_frame(trend_source), limit=100),
        "refresh_history": _frame_records(refresh_history, limit=50),
        "registry": _frame_records(registry, limit=500),
        "source_manifests": _frame_records(store.read_frame("source_manifests"), limit=200),
        "asset_market_manifest": _frame_records(store.read_frame("asset_market_manifest"), limit=300),
        "refresh_jobs": _frame_records(refresh_jobs, limit=50),
    }


def save_dataset_watchlist(store: DuckDBStore, symbols: list[str] | tuple[str, ...], reset: bool = False) -> dict[str, Any]:
    resolved_symbols = [] if reset else normalize_symbols(list(symbols))
    watchlist, asset_universe = save_watchlist(store, resolved_symbols)
    return {
        "watchlist_symbols": watchlist["symbol"].astype(str).tolist() if not watchlist.empty else [],
        "asset_universe": asset_universe.to_dict(orient="records") if not asset_universe.empty else [],
    }


def run_refresh_job(
    *,
    settings: AppSettings,
    store: DuckDBStore,
    job_id: str,
    refresh_mode: str,
    remote_url: str,
    uploaded_files: list[UploadedCsv],
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
    feature_service: FeatureService,
    health_service: DataHealthService,
    lock_fd: int | None,
) -> None:
    started_at = pd.Timestamp.now(tz="UTC")
    try:
        save_refresh_job(
            store,
            _job_row(
                job_id=job_id,
                refresh_mode=refresh_mode,
                status="running",
                started_at=started_at,
                remote_url=remote_url,
                uploaded_file_count=len(uploaded_files),
            ),
        )
        summary = refresh_datasets(
            settings=settings,
            store=store,
            ingestion_service=ingestion_service,
            market_service=market_service,
            discovery_service=discovery_service,
            feature_service=feature_service,
            health_service=health_service,
            remote_url=remote_url,
            uploaded_files=uploaded_files,
            incremental=refresh_mode == "incremental",
            refresh_mode=refresh_mode,
        )
        save_refresh_job(
            store,
            _job_row(
                job_id=job_id,
                refresh_id=str(summary.get("refresh_id", "")),
                refresh_mode=refresh_mode,
                status="success",
                started_at=started_at,
                completed_at=pd.Timestamp.now(tz="UTC"),
                remote_url=remote_url,
                uploaded_file_count=len(uploaded_files),
                normalized_post_count=_count_rows(summary, "posts"),
                asset_daily_row_count=_count_rows(summary, "asset_daily"),
                asset_intraday_row_count=_count_rows(summary, "asset_intraday"),
                tracked_account_count=_count_rows(summary, "tracked_accounts"),
            ),
        )
    except Exception as exc:
        save_refresh_job(
            store,
            _job_row(
                job_id=job_id,
                refresh_mode=refresh_mode,
                status="error",
                started_at=started_at,
                completed_at=pd.Timestamp.now(tz="UTC"),
                error_message=str(exc),
                remote_url=remote_url,
                uploaded_file_count=len(uploaded_files),
            ),
        )
    finally:
        release_refresh_lock(settings, lock_fd)


def submit_refresh_job(
    *,
    settings: AppSettings,
    store: DuckDBStore,
    refresh_mode: str,
    remote_url: str,
    uploaded_files: list[UploadedCsv],
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
    feature_service: FeatureService,
    health_service: DataHealthService,
    run_inline: bool = False,
) -> tuple[str, list[str]]:
    resolved_mode = str(refresh_mode or "").strip().lower()
    if resolved_mode not in VALID_REFRESH_MODES:
        return "", [f"Unsupported refresh mode: {refresh_mode}."]

    lock_fd = acquire_refresh_lock(settings)
    if lock_fd is None:
        return "", ["A dataset refresh is already running."]

    job_id = f"dataset-refresh-{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    save_refresh_job(
        store,
        _job_row(
            job_id=job_id,
            refresh_mode=resolved_mode,
            status="queued",
            started_at=pd.Timestamp.now(tz="UTC"),
            remote_url=remote_url,
            uploaded_file_count=len(uploaded_files),
        ),
    )
    if run_inline:
        run_refresh_job(
            settings=settings,
            store=store,
            job_id=job_id,
            refresh_mode=resolved_mode,
            remote_url=remote_url,
            uploaded_files=uploaded_files,
            ingestion_service=ingestion_service,
            market_service=market_service,
            discovery_service=discovery_service,
            feature_service=feature_service,
            health_service=health_service,
            lock_fd=lock_fd,
        )
    else:
        thread = threading.Thread(
            target=run_refresh_job,
            kwargs={
                "settings": settings,
                "store": store,
                "job_id": job_id,
                "refresh_mode": resolved_mode,
                "remote_url": remote_url,
                "uploaded_files": uploaded_files,
                "ingestion_service": ingestion_service,
                "market_service": market_service,
                "discovery_service": discovery_service,
                "feature_service": feature_service,
                "health_service": health_service,
                "lock_fd": lock_fd,
            },
            daemon=True,
        )
        thread.start()
    return job_id, []
