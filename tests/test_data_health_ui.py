from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.config import AppSettings, EASTERN
from trump_workbench.contracts import MANUAL_OVERRIDE_COLUMNS, RANKING_HISTORY_COLUMNS, TRACKED_ACCOUNT_COLUMNS
from trump_workbench.health import DataHealthService, HEALTH_CHECK_COLUMNS, REFRESH_HISTORY_COLUMNS
from trump_workbench.storage import DuckDBStore
from trump_workbench.ui import (
    _append_refresh_history,
    _filter_health_rows,
    _load_data_health_view_state,
    _refresh_datasets,
)


class _FakeIngestionService:
    def __init__(self, full_posts: pd.DataFrame, incremental_posts: pd.DataFrame, source_manifest: pd.DataFrame) -> None:
        self.full_posts = full_posts
        self.incremental_posts = incremental_posts
        self.source_manifest = source_manifest

    def run_refresh(self, adapters):  # noqa: ANN001
        return self.full_posts.copy(), self.source_manifest.copy()

    def run_incremental_refresh(self, adapters, last_cursor):  # noqa: ANN001, ARG002
        return self.incremental_posts.copy(), self.source_manifest.copy()


class _FakeMarketService:
    def __init__(self, sp500: pd.DataFrame, spy: pd.DataFrame, asset_daily: pd.DataFrame, asset_intraday: pd.DataFrame, asset_market_manifest: pd.DataFrame) -> None:
        self.sp500 = sp500
        self.spy = spy
        self.asset_daily = asset_daily
        self.asset_intraday = asset_intraday
        self.asset_market_manifest = asset_market_manifest

    def load_sp500_daily(self, start: str, end: str) -> pd.DataFrame:  # noqa: ARG002
        return self.sp500.copy()

    def load_spy_daily(self, start: str, end: str) -> pd.DataFrame:  # noqa: ARG002
        return self.spy.copy()

    def load_assets_daily(self, symbols, start: str, end: str):  # noqa: ANN001, ARG002
        manifest = self.asset_market_manifest.loc[self.asset_market_manifest["dataset_kind"] == "daily"].reset_index(drop=True)
        return self.asset_daily.copy(), manifest

    def load_assets_intraday(self, symbols, interval: str = "5m", lookback_days: int = 30):  # noqa: ANN001, ARG002
        manifest = self.asset_market_manifest.loc[self.asset_market_manifest["dataset_kind"] == f"intraday_{interval}"].reset_index(drop=True)
        return self.asset_intraday.copy(), manifest


class _FakeDiscoveryService:
    def __init__(self, tracked_accounts: pd.DataFrame, ranking_history: pd.DataFrame) -> None:
        self.tracked_accounts = tracked_accounts
        self.ranking_history = ranking_history

    def normalize_manual_overrides(self, overrides: pd.DataFrame | None) -> pd.DataFrame:
        if overrides is None or overrides.empty:
            return pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)
        return overrides.copy()

    def refresh_accounts(self, posts: pd.DataFrame, existing_accounts: pd.DataFrame, as_of: pd.Timestamp, manual_overrides: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # noqa: ARG002
        return self.tracked_accounts.copy(), self.ranking_history.copy()


class _FakeFeatureService:
    def __init__(self, prepared_posts: pd.DataFrame, asset_post_mappings: pd.DataFrame, asset_session_features: pd.DataFrame) -> None:
        self.prepared_posts = prepared_posts
        self.asset_post_mappings = asset_post_mappings
        self.asset_session_features = asset_session_features

    def prepare_session_posts(self, posts: pd.DataFrame, market_calendar: pd.DataFrame, tracked_accounts: pd.DataFrame, llm_enabled: bool) -> pd.DataFrame:  # noqa: ARG002
        return self.prepared_posts.copy()

    def build_asset_post_mappings(self, prepared_posts: pd.DataFrame, asset_universe: pd.DataFrame, llm_enabled: bool) -> pd.DataFrame:  # noqa: ARG002
        return self.asset_post_mappings.copy()

    def build_asset_session_dataset(self, asset_post_mappings: pd.DataFrame, asset_market: pd.DataFrame, feature_version: str, llm_enabled: bool, asset_universe: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG002
        return self.asset_session_features.copy()


class DataHealthUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(self.settings)
        self.health_service = DataHealthService()
        post_ts = pd.Timestamp("2026-04-15 10:00:00", tz=EASTERN)
        self.posts = pd.DataFrame(
            [
                {
                    "source_platform": "X",
                    "source_type": "x_csv",
                    "author_account_id": "acct-a",
                    "author_handle": "macroa",
                    "author_display_name": "Macro A",
                    "author_is_trump": False,
                    "post_id": "post-1",
                    "post_url": "https://x.com/macroa/status/1",
                    "post_timestamp": post_ts,
                    "raw_text": "Trump growth",
                    "cleaned_text": "Trump growth",
                    "is_reshare": False,
                    "has_media": False,
                    "replies_count": 1,
                    "reblogs_count": 2,
                    "favourites_count": 3,
                    "mentions_trump": True,
                    "source_provenance": "unit-test",
                    "engagement_score": 6.0,
                    "sentiment_score": 0.3,
                    "sentiment_label": "positive",
                },
            ],
        )
        self.source_manifest = pd.DataFrame(
            [
                {
                    "source": "Truth Social archive",
                    "provenance": "unit-test",
                    "post_count": 1,
                    "coverage_start": post_ts,
                    "coverage_end": post_ts,
                    "status": "ok",
                    "detail": "",
                },
            ],
        )
        self.sp500 = pd.DataFrame([{"trade_date": pd.Timestamp("2026-04-15"), "close": 6100.0}])
        self.spy = pd.DataFrame(
            [{"trade_date": pd.Timestamp("2026-04-15"), "open": 600.0, "high": 605.0, "low": 598.0, "close": 603.0, "volume": 1_000_000}],
        )
        self.asset_daily = pd.DataFrame(
            [
                {"symbol": "SPY", "trade_date": pd.Timestamp("2026-04-15"), "open": 600.0, "high": 605.0, "low": 598.0, "close": 603.0, "volume": 1_000_000},
                {"symbol": "QQQ", "trade_date": pd.Timestamp("2026-04-15"), "open": 500.0, "high": 505.0, "low": 498.0, "close": 504.0, "volume": 900_000},
                {"symbol": "XLK", "trade_date": pd.Timestamp("2026-04-15"), "open": 200.0, "high": 201.0, "low": 198.0, "close": 200.5, "volume": 700_000},
                {"symbol": "XLF", "trade_date": pd.Timestamp("2026-04-15"), "open": 40.0, "high": 41.0, "low": 39.5, "close": 40.5, "volume": 500_000},
                {"symbol": "XLE", "trade_date": pd.Timestamp("2026-04-15"), "open": 80.0, "high": 81.0, "low": 79.0, "close": 80.5, "volume": 400_000},
                {"symbol": "SMH", "trade_date": pd.Timestamp("2026-04-15"), "open": 250.0, "high": 252.0, "low": 248.0, "close": 251.0, "volume": 300_000},
            ],
        )
        self.asset_intraday = pd.DataFrame(
            [
                {"symbol": "SPY", "timestamp": post_ts, "open": 600.0, "high": 601.0, "low": 599.0, "close": 600.5, "volume": 1000, "interval": "5m"},
                {"symbol": "QQQ", "timestamp": post_ts, "open": 500.0, "high": 501.0, "low": 499.0, "close": 500.5, "volume": 900, "interval": "5m"},
                {"symbol": "XLK", "timestamp": post_ts, "open": 200.0, "high": 201.0, "low": 199.0, "close": 200.5, "volume": 800, "interval": "5m"},
                {"symbol": "XLF", "timestamp": post_ts, "open": 40.0, "high": 40.5, "low": 39.5, "close": 40.2, "volume": 700, "interval": "5m"},
                {"symbol": "XLE", "timestamp": post_ts, "open": 80.0, "high": 81.0, "low": 79.0, "close": 80.4, "volume": 600, "interval": "5m"},
                {"symbol": "SMH", "timestamp": post_ts, "open": 250.0, "high": 251.0, "low": 249.0, "close": 250.5, "volume": 500, "interval": "5m"},
            ],
        )
        self.asset_market_manifest = pd.DataFrame(
            [
                {"symbol": "SPY", "dataset_kind": "daily", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15"), "end_at": pd.Timestamp("2026-04-15"), "detail": ""},
                {"symbol": "SPY", "dataset_kind": "intraday_5m", "row_count": 1, "status": "ok", "start_at": post_ts, "end_at": post_ts, "detail": ""},
                {"symbol": "QQQ", "dataset_kind": "daily", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15"), "end_at": pd.Timestamp("2026-04-15"), "detail": ""},
                {"symbol": "QQQ", "dataset_kind": "intraday_5m", "row_count": 1, "status": "ok", "start_at": post_ts, "end_at": post_ts, "detail": ""},
                {"symbol": "XLK", "dataset_kind": "daily", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15"), "end_at": pd.Timestamp("2026-04-15"), "detail": ""},
                {"symbol": "XLK", "dataset_kind": "intraday_5m", "row_count": 1, "status": "ok", "start_at": post_ts, "end_at": post_ts, "detail": ""},
                {"symbol": "XLF", "dataset_kind": "daily", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15"), "end_at": pd.Timestamp("2026-04-15"), "detail": ""},
                {"symbol": "XLF", "dataset_kind": "intraday_5m", "row_count": 1, "status": "ok", "start_at": post_ts, "end_at": post_ts, "detail": ""},
                {"symbol": "XLE", "dataset_kind": "daily", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15"), "end_at": pd.Timestamp("2026-04-15"), "detail": ""},
                {"symbol": "XLE", "dataset_kind": "intraday_5m", "row_count": 1, "status": "ok", "start_at": post_ts, "end_at": post_ts, "detail": ""},
                {"symbol": "SMH", "dataset_kind": "daily", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15"), "end_at": pd.Timestamp("2026-04-15"), "detail": ""},
                {"symbol": "SMH", "dataset_kind": "intraday_5m", "row_count": 1, "status": "ok", "start_at": post_ts, "end_at": post_ts, "detail": ""},
            ],
        )
        self.tracked_accounts = pd.DataFrame(
            [
                {
                    "version_id": "v1",
                    "account_id": "acct-a",
                    "handle": "macroa",
                    "display_name": "Macro A",
                    "source_platform": "X",
                    "discovery_score": 1.0,
                    "status": "active",
                    "first_seen_at": pd.Timestamp("2026-04-10"),
                    "last_seen_at": pd.Timestamp("2026-04-15"),
                    "effective_from": pd.Timestamp("2026-04-15"),
                    "effective_to": pd.NaT,
                    "auto_included": True,
                    "provenance": "unit-test",
                    "mention_count": 1,
                    "engagement_mean": 10.0,
                    "active_days": 1,
                },
            ],
            columns=TRACKED_ACCOUNT_COLUMNS,
        )
        self.ranking_history = pd.DataFrame(columns=RANKING_HISTORY_COLUMNS)
        self.asset_post_mappings = pd.DataFrame(
            [
                {
                    "asset_symbol": "QQQ",
                    "session_date": pd.Timestamp("2026-04-15"),
                    "post_id": "post-1",
                    "reaction_anchor_ts": post_ts,
                    "mapping_reason": "same-session",
                    "author_handle": "macroa",
                    "author_display_name": "Macro A",
                    "author_account_id": "acct-a",
                    "author_is_trump": False,
                    "source_platform": "X",
                    "cleaned_text": "Trump growth",
                    "mentions_trump": True,
                    "engagement_score": 6.0,
                    "sentiment_score": 0.3,
                    "sentiment_label": "positive",
                    "semantic_topic": "markets",
                    "semantic_policy_bucket": "economy",
                    "semantic_stance": "supportive",
                    "semantic_market_relevance": 0.8,
                    "semantic_urgency": 0.2,
                    "semantic_primary_asset": "QQQ",
                    "semantic_asset_targets": "QQQ",
                    "semantic_confidence": 0.7,
                    "semantic_summary": "Market-related post",
                    "semantic_schema_version": "v1",
                    "semantic_provider": "heuristic",
                    "is_active_tracked_account": True,
                    "tracked_discovery_score": 1.0,
                    "tracked_account_status": "active",
                    "rule_match_score": 0.4,
                    "semantic_match_score": 0.3,
                    "asset_relevance_score": 0.7,
                    "match_reasons": "ticker",
                    "match_rank": 1,
                    "is_primary_asset": True,
                    "asset_display_name": "QQQ",
                    "asset_type": "etf",
                    "asset_source": "core_etf",
                },
            ],
        )
        self.asset_session_features = pd.DataFrame(
            [
                {
                    "asset_symbol": "QQQ",
                    "signal_session_date": pd.Timestamp("2026-04-15"),
                    "next_session_date": pd.Timestamp("2026-04-16"),
                    "post_count": 1,
                    "target_next_session_return": 0.01,
                    "target_available": True,
                },
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _save_required_frames(self) -> None:
        self.store.save_frame("normalized_posts", self.posts, metadata={"row_count": len(self.posts)})
        self.store.save_frame("source_manifests", self.source_manifest, metadata={"row_count": len(self.source_manifest)})
        self.store.save_frame("sp500_daily", self.sp500, metadata={"row_count": len(self.sp500)})
        self.store.save_frame("spy_daily", self.spy, metadata={"row_count": len(self.spy)})
        self.store.save_frame(
            "asset_universe",
            pd.DataFrame(
                [
                    {"symbol": "SPY", "display_name": "SPY", "asset_type": "etf", "source": "core_etf", "is_default": True, "is_watchlist": False},
                    {"symbol": "QQQ", "display_name": "QQQ", "asset_type": "etf", "source": "core_etf", "is_default": True, "is_watchlist": False},
                    {"symbol": "XLK", "display_name": "XLK", "asset_type": "etf", "source": "core_etf", "is_default": True, "is_watchlist": False},
                    {"symbol": "XLF", "display_name": "XLF", "asset_type": "etf", "source": "core_etf", "is_default": True, "is_watchlist": False},
                    {"symbol": "XLE", "display_name": "XLE", "asset_type": "etf", "source": "core_etf", "is_default": True, "is_watchlist": False},
                    {"symbol": "SMH", "display_name": "SMH", "asset_type": "etf", "source": "core_etf", "is_default": True, "is_watchlist": False},
                ],
            ),
            metadata={"row_count": 6},
        )
        self.store.save_frame("asset_daily", self.asset_daily, metadata={"row_count": len(self.asset_daily)})
        self.store.save_frame("asset_intraday", self.asset_intraday, metadata={"row_count": len(self.asset_intraday)})
        self.store.save_frame("asset_market_manifest", self.asset_market_manifest, metadata={"row_count": len(self.asset_market_manifest)})
        self.store.save_frame("tracked_accounts", self.tracked_accounts, metadata={"row_count": len(self.tracked_accounts)})
        self.store.save_frame("asset_post_mappings", self.asset_post_mappings, metadata={"row_count": len(self.asset_post_mappings)})
        self.store.save_frame("asset_session_features", self.asset_session_features, metadata={"row_count": len(self.asset_session_features)})

    def test_append_refresh_history_records_success_and_failure_rows(self) -> None:
        first = _append_refresh_history(
            self.store,
            refresh_id="refresh-1",
            refresh_mode="full",
            status="success",
            started_at=pd.Timestamp("2026-04-15 12:00:00", tz="UTC"),
            completed_at=pd.Timestamp("2026-04-15 12:01:00", tz="UTC"),
        )
        second = _append_refresh_history(
            self.store,
            refresh_id="refresh-2",
            refresh_mode="incremental",
            status="error",
            started_at=pd.Timestamp("2026-04-15 12:05:00", tz="UTC"),
            completed_at=pd.Timestamp("2026-04-15 12:06:00", tz="UTC"),
            error_message="network timeout",
        )

        history = self.store.read_frame("refresh_history")
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 2)
        self.assertEqual(list(history["status"]), ["success", "error"])
        self.assertEqual(list(history["refresh_mode"]), ["full", "incremental"])

    def test_load_data_health_view_state_computes_in_memory_snapshot_without_history(self) -> None:
        self._save_required_frames()

        state = _load_data_health_view_state(self.store, self.health_service)

        self.assertFalse(state["latest"].empty)
        self.assertTrue(state["history"].empty)
        self.assertEqual(state["refresh_history"].columns.tolist(), REFRESH_HISTORY_COLUMNS)
        self.assertIn("overall_severity", state["summary"])
        self.assertFalse(state["trend"].empty)

    def test_filter_health_rows_filters_severity_and_scope_kind(self) -> None:
        rows = pd.DataFrame(
            [
                {"snapshot_id": "a", "generated_at": pd.Timestamp("2026-04-15", tz="UTC"), "refresh_id": "r1", "scope_kind": "dataset", "scope_key": "normalized_posts", "check_name": "row_count_anomaly", "severity": "warn", "observed_value": 120.0, "baseline_value": 100.0, "detail": ""},
                {"snapshot_id": "a", "generated_at": pd.Timestamp("2026-04-15", tz="UTC"), "refresh_id": "r1", "scope_kind": "asset_intraday", "scope_key": "QQQ", "check_name": "intraday_lag_minutes", "severity": "severe", "observed_value": 130.0, "baseline_value": 30.0, "detail": ""},
                {"snapshot_id": "a", "generated_at": pd.Timestamp("2026-04-15", tz="UTC"), "refresh_id": "r1", "scope_kind": "dataset", "scope_key": "spy_daily", "check_name": "duplicate_rate", "severity": "ok", "observed_value": 0.0, "baseline_value": 0.005, "detail": ""},
            ],
            columns=HEALTH_CHECK_COLUMNS,
        )

        filtered = _filter_health_rows(rows, ["warn", "severe"], "dataset")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["scope_key"], "normalized_posts")

    def test_refresh_datasets_persists_health_snapshots_for_full_and_incremental_runs(self) -> None:
        ingestion_service = _FakeIngestionService(self.posts, self.posts, self.source_manifest)
        market_service = _FakeMarketService(self.sp500, self.spy, self.asset_daily, self.asset_intraday, self.asset_market_manifest)
        discovery_service = _FakeDiscoveryService(self.tracked_accounts, self.ranking_history)
        feature_service = _FakeFeatureService(self.posts, self.asset_post_mappings, self.asset_session_features)

        first = _refresh_datasets(
            settings=self.settings,
            store=self.store,
            ingestion_service=ingestion_service,
            market_service=market_service,
            discovery_service=discovery_service,
            feature_service=feature_service,
            health_service=self.health_service,
            remote_url="",
            uploaded_files=[],
            incremental=False,
            refresh_mode="full",
        )
        second = _refresh_datasets(
            settings=self.settings,
            store=self.store,
            ingestion_service=ingestion_service,
            market_service=market_service,
            discovery_service=discovery_service,
            feature_service=feature_service,
            health_service=self.health_service,
            remote_url="",
            uploaded_files=[],
            incremental=True,
            refresh_mode="incremental",
        )

        latest = self.store.read_frame("data_health_latest")
        history = self.store.read_frame("data_health_history")
        refresh_history = self.store.read_frame("refresh_history")

        self.assertFalse(first["data_health_latest"].empty)
        self.assertFalse(second["data_health_latest"].empty)
        self.assertFalse(latest.empty)
        self.assertGreaterEqual(history["snapshot_id"].nunique(), 2)
        self.assertEqual(len(refresh_history), 2)
        self.assertEqual(list(refresh_history["status"]), ["success", "success"])


if __name__ == "__main__":
    unittest.main()
