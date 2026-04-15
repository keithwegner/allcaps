from __future__ import annotations

from typing import Any

import pandas as pd

from .config import AppSettings, DEFAULT_ETF_SYMBOLS
from .contracts import MANUAL_OVERRIDE_COLUMNS
from .discovery import DiscoveryService
from .features import FeatureService
from .health import DataHealthService, create_refresh_id, ensure_refresh_history_frame, make_refresh_history_frame
from .ingestion import IngestionService, TruthSocialArchiveAdapter, XCsvAdapter
from .market import MarketDataService, build_asset_universe, build_watchlist_frame, normalize_symbols
from .storage import DuckDBStore

CORE_DATASET_NAMES = (
    "normalized_posts",
    "sp500_daily",
    "spy_daily",
    "asset_daily",
    "asset_intraday",
    "asset_post_mappings",
    "asset_session_features",
)


def watchlist_symbols(store: DuckDBStore) -> list[str]:
    watchlist = store.read_frame("asset_watchlist")
    if watchlist.empty or "symbol" not in watchlist.columns:
        return []
    return normalize_symbols(watchlist["symbol"].astype(str).tolist())


def save_watchlist(store: DuckDBStore, symbols: list[str] | tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def missing_core_datasets(store: DuckDBStore) -> list[str]:
    missing: list[str] = []
    for dataset_name in CORE_DATASET_NAMES:
        if store.read_frame(dataset_name).empty:
            missing.append(dataset_name)
    return missing


def build_source_adapters(
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


def rebuild_discovery_state(
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


def append_refresh_history(
    store: DuckDBStore,
    refresh_id: str,
    refresh_mode: str,
    status: str,
    started_at: pd.Timestamp,
    completed_at: pd.Timestamp,
    error_message: str = "",
) -> pd.DataFrame:
    refresh_row = make_refresh_history_frame(
        refresh_id=refresh_id,
        refresh_mode=refresh_mode,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        error_message=error_message,
    )
    store.append_frame(
        "refresh_history",
        refresh_row,
        dedupe_on=["refresh_id"],
        metadata={"latest_status": status, "latest_refresh_mode": refresh_mode},
    )
    return ensure_refresh_history_frame(store.read_frame("refresh_history"))


def refresh_datasets(
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
    started_at = pd.Timestamp.now(tz="UTC")
    resolved_mode = str(refresh_mode or ("incremental" if incremental else "full"))
    refresh_id = create_refresh_id(resolved_mode, started_at)

    try:
        adapters = build_source_adapters(settings, remote_url, uploaded_files)
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

        current_watchlist_symbols = watchlist_symbols(store)
        watchlist, asset_universe = save_watchlist(store, current_watchlist_symbols)
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
        tracked_accounts, ranking_history = rebuild_discovery_state(store, discovery_service, posts, as_of)
        prepared_posts = feature_service.prepare_session_posts(
            posts=posts,
            market_calendar=spy,
            tracked_accounts=tracked_accounts,
            llm_enabled=True,
        )
        asset_post_mappings = feature_service.build_asset_post_mappings(
            prepared_posts=prepared_posts,
            asset_universe=asset_universe,
            llm_enabled=True,
        )
        asset_session_features = feature_service.build_asset_session_dataset(
            asset_post_mappings=asset_post_mappings,
            asset_market=asset_daily,
            feature_version="asset-v1",
            llm_enabled=True,
            asset_universe=asset_universe,
        )
        store.save_frame(
            "asset_post_mappings",
            asset_post_mappings,
            metadata={"row_count": int(len(asset_post_mappings)), "match_mode": "rules_plus_semantic"},
        )
        store.save_frame(
            "asset_session_features",
            asset_session_features,
            metadata={"row_count": int(len(asset_session_features)), "feature_version": "asset-v1"},
        )
        completed_at = pd.Timestamp.now(tz="UTC")
        health_latest = health_service.persist_snapshot(store, refresh_id=refresh_id, generated_at=completed_at)
        refresh_history = append_refresh_history(
            store=store,
            refresh_id=refresh_id,
            refresh_mode=resolved_mode,
            status="success",
            started_at=started_at,
            completed_at=completed_at,
        )
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
            "asset_post_mappings": asset_post_mappings,
            "asset_session_features": asset_session_features,
            "tracked_accounts": tracked_accounts,
            "account_rankings": ranking_history,
            "data_health_latest": health_latest,
            "refresh_history": refresh_history,
            "refresh_id": refresh_id,
        }
    except Exception as exc:
        append_refresh_history(
            store=store,
            refresh_id=refresh_id,
            refresh_mode=resolved_mode,
            status="error",
            started_at=started_at,
            completed_at=pd.Timestamp.now(tz="UTC"),
            error_message=str(exc),
        )
        raise


def ensure_bootstrap(
    settings: AppSettings,
    store: DuckDBStore,
    ingestion_service: IngestionService,
    market_service: MarketDataService,
    discovery_service: DiscoveryService,
    feature_service: FeatureService,
    health_service: DataHealthService,
) -> dict[str, Any] | None:
    if store.read_frame("asset_watchlist").empty and store.read_frame("asset_universe").empty:
        save_watchlist(store, [])
    if not settings.auto_bootstrap_on_start:
        return None
    if not missing_core_datasets(store):
        return None
    return refresh_datasets(
        settings=settings,
        store=store,
        ingestion_service=ingestion_service,
        market_service=market_service,
        discovery_service=discovery_service,
        feature_service=feature_service,
        health_service=health_service,
        remote_url=settings.remote_x_csv_url,
        uploaded_files=[],
        incremental=False,
        refresh_mode="bootstrap",
    )
