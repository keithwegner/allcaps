from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from trump_workbench.api import create_app
from trump_workbench.config import AppSettings
from trump_workbench.contracts import BacktestRun, MANUAL_OVERRIDE_COLUMNS, RANKING_HISTORY_COLUMNS, TRACKED_ACCOUNT_COLUMNS
from trump_workbench.experiments import ExperimentStore
from trump_workbench.health import HEALTH_CHECK_COLUMNS
from trump_workbench.scheduler import acquire_refresh_lock, release_refresh_lock
from trump_workbench.paper_trading import (
    PAPER_BENCHMARK_CURVE_COLUMNS,
    PAPER_DECISION_JOURNAL_COLUMNS,
    PAPER_EQUITY_CURVE_COLUMNS,
    PAPER_PORTFOLIO_REGISTRY_COLUMNS,
    PAPER_TRADE_LEDGER_COLUMNS,
)
from trump_workbench.storage import DuckDBStore


class FakeIngestionService:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def _posts(self, post_id: str = "truth-refresh-1") -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source_platform": "Truth Social",
                    "source_type": "truth_archive",
                    "author_account_id": "truth-account",
                    "author_handle": "realDonaldTrump",
                    "author_display_name": "Donald Trump",
                    "author_is_trump": True,
                    "post_id": post_id,
                    "post_url": "https://truthsocial.com/@realDonaldTrump/posts/refresh",
                    "post_timestamp": pd.Timestamp("2025-02-03 08:00:00", tz="America/New_York"),
                    "raw_text": "Synthetic refresh post",
                    "cleaned_text": "Synthetic refresh post",
                    "is_reshare": False,
                    "has_media": False,
                    "replies_count": 0,
                    "reblogs_count": 0,
                    "favourites_count": 0,
                    "mentions_trump": False,
                    "source_provenance": "unit-test",
                    "engagement_score": 0.0,
                    "sentiment_score": 0.2,
                    "sentiment_label": "positive",
                },
            ],
        )

    def _manifest(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source": "fake-truth",
                    "status": "ok",
                    "post_count": 1,
                    "error_message": "",
                },
            ],
        )

    def run_refresh(self, adapters) -> tuple[pd.DataFrame, pd.DataFrame]:
        del adapters
        if self.fail:
            raise RuntimeError("fake ingestion failed")
        return self._posts(), self._manifest()

    def run_incremental_refresh(self, adapters, last_cursor) -> tuple[pd.DataFrame, pd.DataFrame]:
        del adapters, last_cursor
        if self.fail:
            raise RuntimeError("fake ingestion failed")
        return self._posts("truth-refresh-2"), self._manifest()


class FakeMarketDataService:
    def load_sp500_daily(self, start: str, end: str) -> pd.DataFrame:
        del start, end
        return pd.DataFrame(
            [
                {"trade_date": pd.Timestamp("2025-02-03"), "close": 100.0},
                {"trade_date": pd.Timestamp("2025-02-04"), "close": 101.0},
            ],
        )

    def load_spy_daily(self, start: str, end: str) -> pd.DataFrame:
        del start, end
        return pd.DataFrame(
            [
                {"trade_date": pd.Timestamp("2025-02-03"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
                {"trade_date": pd.Timestamp("2025-02-04"), "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 1100},
            ],
        )

    def load_assets_daily(self, symbols: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        del start, end
        rows = []
        manifest_rows = []
        for symbol in symbols:
            rows.extend(
                [
                    {"symbol": symbol, "trade_date": pd.Timestamp("2025-02-03"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
                    {"symbol": symbol, "trade_date": pd.Timestamp("2025-02-04"), "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 1100},
                ],
            )
            manifest_rows.append(
                {
                    "symbol": symbol,
                    "dataset_kind": "daily",
                    "row_count": 2,
                    "status": "ok",
                    "start_at": pd.Timestamp("2025-02-03"),
                    "end_at": pd.Timestamp("2025-02-04"),
                    "detail": "",
                },
            )
        return pd.DataFrame(rows), pd.DataFrame(manifest_rows)

    def load_assets_intraday(self, symbols: list[str], interval: str = "5m", lookback_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
        del lookback_days
        rows = []
        manifest_rows = []
        for symbol in symbols:
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": pd.Timestamp("2025-02-04 14:30:00", tz="UTC"),
                    "open": 100.0,
                    "high": 100.5,
                    "low": 99.8,
                    "close": 100.2,
                    "volume": 100,
                    "interval": interval,
                },
            )
            manifest_rows.append(
                {
                    "symbol": symbol,
                    "dataset_kind": "intraday",
                    "row_count": 1,
                    "status": "ok",
                    "start_at": pd.Timestamp("2025-02-04 14:30:00", tz="UTC"),
                    "end_at": pd.Timestamp("2025-02-04 14:30:00", tz="UTC"),
                    "detail": "",
                },
            )
        return pd.DataFrame(rows), pd.DataFrame(manifest_rows)


class FakeFeatureService:
    def prepare_session_posts(self, posts: pd.DataFrame, market_calendar: pd.DataFrame, tracked_accounts: pd.DataFrame, llm_enabled: bool) -> pd.DataFrame:
        del market_calendar, tracked_accounts, llm_enabled
        return posts.assign(session_date=pd.Timestamp("2025-02-03"))

    def build_asset_post_mappings(self, prepared_posts: pd.DataFrame, asset_universe: pd.DataFrame, llm_enabled: bool) -> pd.DataFrame:
        del llm_enabled
        symbols = asset_universe["symbol"].astype(str).tolist() if not asset_universe.empty else ["SPY"]
        return pd.DataFrame(
            [
                {
                    "asset_symbol": symbols[0],
                    "post_id": str(prepared_posts.iloc[0]["post_id"]) if not prepared_posts.empty else "post-1",
                    "session_date": pd.Timestamp("2025-02-03"),
                    "asset_relevance_score": 1.0,
                    "mapping_reason": "unit-test",
                },
            ],
        )

    def build_asset_session_dataset(
        self,
        asset_post_mappings: pd.DataFrame,
        asset_market: pd.DataFrame,
        feature_version: str,
        llm_enabled: bool,
        asset_universe: pd.DataFrame,
    ) -> pd.DataFrame:
        del asset_post_mappings, asset_market, feature_version, llm_enabled
        symbols = asset_universe["symbol"].astype(str).tolist() if not asset_universe.empty else ["SPY"]
        return pd.DataFrame(
            [
                {
                    "asset_symbol": symbol,
                    "signal_session_date": pd.Timestamp("2025-02-03"),
                    "next_session_date": pd.Timestamp("2025-02-04"),
                    "post_count": 1,
                    "target_next_session_return": 0.01,
                    "target_available": True,
                }
                for symbol in symbols
            ],
        )


class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(self.settings)
        self.experiments = ExperimentStore(self.store)
        self.client = TestClient(create_app(settings=self.settings, store=self.store))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _save_truth_posts(self) -> None:
        self.store.save_frame(
            "normalized_posts",
            pd.DataFrame(
                [
                    {
                        "source_platform": "Truth Social",
                        "post_id": "truth-1",
                        "post_timestamp": pd.Timestamp("2026-04-20 12:00:00", tz="UTC"),
                        "author_is_trump": True,
                        "cleaned_text": "Market update",
                    },
                ],
            ),
        )

    def _save_research_frames(self, include_x: bool = True) -> None:
        rows = [
            {
                "source_platform": "Truth Social",
                "source_type": "truth_archive",
                "author_account_id": "truth-account",
                "author_handle": "realDonaldTrump",
                "author_display_name": "Donald Trump",
                "author_is_trump": True,
                "post_id": "truth-1",
                "post_url": "https://truthsocial.com/@realDonaldTrump/posts/1",
                "post_timestamp": pd.Timestamp("2025-02-03 08:00:00", tz="America/New_York"),
                "raw_text": "The stock market and economy are looking strong.",
                "cleaned_text": "The stock market and economy are looking strong.",
                "is_reshare": False,
                "has_media": False,
                "replies_count": 0,
                "reblogs_count": 0,
                "favourites_count": 0,
                "mentions_trump": False,
                "source_provenance": "unit-test",
                "engagement_score": 0.0,
                "sentiment_score": 0.7,
                "sentiment_label": "positive",
            },
        ]
        if include_x:
            rows.append(
                {
                    "source_platform": "X",
                    "source_type": "x_csv",
                    "author_account_id": "acct-macro",
                    "author_handle": "macroalpha",
                    "author_display_name": "Macro Alpha",
                    "author_is_trump": False,
                    "post_id": "x-1",
                    "post_url": "https://x.com/macroalpha/status/1",
                    "post_timestamp": pd.Timestamp("2025-02-03 10:00:00", tz="America/New_York"),
                    "raw_text": "Trump tariff headlines could pressure semiconductors.",
                    "cleaned_text": "Trump tariff headlines could pressure semiconductors.",
                    "is_reshare": False,
                    "has_media": False,
                    "replies_count": 0,
                    "reblogs_count": 0,
                    "favourites_count": 0,
                    "mentions_trump": True,
                    "source_provenance": "unit-test",
                    "engagement_score": 0.0,
                    "sentiment_score": -0.4,
                    "sentiment_label": "negative",
                },
            )
        self.store.save_frame("normalized_posts", pd.DataFrame(rows))
        self.store.save_frame(
            "sp500_daily",
            pd.DataFrame(
                [
                    {"trade_date": pd.Timestamp("2025-02-03"), "close": 100.0},
                    {"trade_date": pd.Timestamp("2025-02-04"), "close": 101.0},
                    {"trade_date": pd.Timestamp("2025-02-05"), "close": 99.0},
                ],
            ),
        )
        self.store.save_frame(
            "asset_universe",
            pd.DataFrame(
                [
                    {"symbol": "SPY", "display_name": "SPDR S&P 500 ETF", "asset_type": "etf", "source": "core_etf"},
                    {"symbol": "QQQ", "display_name": "Invesco QQQ Trust", "asset_type": "etf", "source": "core_etf"},
                    {"symbol": "NVDA", "display_name": "NVIDIA", "asset_type": "equity", "source": "watchlist"},
                ],
            ),
        )

    def _save_discovery_frames(self) -> None:
        self._save_research_frames(include_x=True)
        self.store.save_frame(
            "tracked_accounts",
            pd.DataFrame(
                [
                    {
                        "version_id": "acct-macro-v1",
                        "account_id": "acct-macro",
                        "handle": "macroalpha",
                        "display_name": "Macro Alpha",
                        "source_platform": "X",
                        "discovery_score": 12.0,
                        "status": "active",
                        "first_seen_at": pd.Timestamp("2025-02-03 10:00:00", tz="America/New_York"),
                        "last_seen_at": pd.Timestamp("2025-02-03 10:00:00", tz="America/New_York"),
                        "effective_from": pd.Timestamp("2025-02-03"),
                        "effective_to": pd.NaT,
                        "auto_included": True,
                        "provenance": "discovery_auto_include",
                        "mention_count": 3,
                        "engagement_mean": 45.0,
                        "active_days": 1,
                    },
                    {
                        "version_id": "acct-policy-v1",
                        "account_id": "acct-policy",
                        "handle": "policywatch",
                        "display_name": "Policy Watch",
                        "source_platform": "X",
                        "discovery_score": 9.0,
                        "status": "pinned",
                        "first_seen_at": pd.Timestamp("2025-02-01 10:00:00", tz="America/New_York"),
                        "last_seen_at": pd.Timestamp("2025-02-03 10:00:00", tz="America/New_York"),
                        "effective_from": pd.Timestamp("2025-02-01"),
                        "effective_to": pd.NaT,
                        "auto_included": False,
                        "provenance": "manual_override:pin",
                        "mention_count": 1,
                        "engagement_mean": 10.0,
                        "active_days": 1,
                    },
                ],
                columns=TRACKED_ACCOUNT_COLUMNS,
            ),
        )
        self.store.save_frame(
            "account_rankings",
            pd.DataFrame(
                [
                    {
                        "author_account_id": "acct-macro",
                        "author_handle": "macroalpha",
                        "author_display_name": "Macro Alpha",
                        "source_platform": "X",
                        "discovery_score": 10.0,
                        "mention_count": 2,
                        "engagement_mean": 30.0,
                        "active_days": 1,
                        "ranked_at": pd.Timestamp("2025-02-02"),
                        "discovery_rank": 2,
                        "final_selected": True,
                        "selected_status": "active",
                        "suppressed_by_override": False,
                        "pinned_by_override": False,
                    },
                    {
                        "author_account_id": "acct-macro",
                        "author_handle": "macroalpha",
                        "author_display_name": "Macro Alpha",
                        "source_platform": "X",
                        "discovery_score": 12.0,
                        "mention_count": 3,
                        "engagement_mean": 45.0,
                        "active_days": 1,
                        "ranked_at": pd.Timestamp("2025-02-03"),
                        "discovery_rank": 1,
                        "final_selected": True,
                        "selected_status": "active",
                        "suppressed_by_override": False,
                        "pinned_by_override": False,
                    },
                    {
                        "author_account_id": "acct-policy",
                        "author_handle": "policywatch",
                        "author_display_name": "Policy Watch",
                        "source_platform": "X",
                        "discovery_score": 9.0,
                        "mention_count": 1,
                        "engagement_mean": 10.0,
                        "active_days": 1,
                        "ranked_at": pd.Timestamp("2025-02-03"),
                        "discovery_rank": 2,
                        "final_selected": True,
                        "selected_status": "pinned",
                        "suppressed_by_override": False,
                        "pinned_by_override": True,
                    },
                    {
                        "author_account_id": "acct-muted",
                        "author_handle": "muted",
                        "author_display_name": "Muted Account",
                        "source_platform": "X",
                        "discovery_score": 3.0,
                        "mention_count": 1,
                        "engagement_mean": 1.0,
                        "active_days": 1,
                        "ranked_at": pd.Timestamp("2025-02-03"),
                        "discovery_rank": 3,
                        "final_selected": False,
                        "selected_status": "excluded",
                        "suppressed_by_override": True,
                        "pinned_by_override": False,
                    },
                ],
                columns=RANKING_HISTORY_COLUMNS,
            ),
        )
        self.store.save_frame(
            "manual_account_overrides",
            pd.DataFrame(
                [
                    {
                        "override_id": "",
                        "account_id": "acct-policy",
                        "handle": "policywatch",
                        "display_name": "Policy Watch",
                        "source_platform": "X",
                        "action": "PIN",
                        "effective_from": pd.Timestamp("2025-02-01"),
                        "effective_to": pd.NaT,
                        "note": "Always inspect",
                        "created_at": pd.Timestamp("2025-02-01 12:00:00", tz="UTC"),
                    },
                    {
                        "override_id": "",
                        "account_id": "acct-muted",
                        "handle": "muted",
                        "display_name": "Muted Account",
                        "source_platform": "X",
                        "action": "suppress",
                        "effective_from": pd.Timestamp("2025-02-03"),
                        "effective_to": pd.NaT,
                        "note": "Low quality",
                        "created_at": pd.Timestamp("2025-02-03 12:00:00", tz="UTC"),
                    },
                ],
                columns=MANUAL_OVERRIDE_COLUMNS,
            ),
        )

    def _save_run_record(self) -> None:
        self.store.save_run_record(
            run_id="run-1",
            run_name="Portfolio run",
            target_asset="PORTFOLIO",
            run_type="portfolio_allocator",
            allocator_mode="joint_model",
            config_hash="hash-1",
            metrics={"total_return": 0.1},
            selected_params={"deployment_variant": "per_asset", "selected_symbols": ["SPY"]},
            artifact_paths={
                "summary_path": str(self.settings.artifact_dir / "runs" / "run-1" / "summary.json"),
                "trades_path": "",
                "predictions_path": "",
                "windows_path": "",
                "importance_path": "",
                "model_path": "",
            },
        )

    def _save_asset_run_artifacts(self, run_id: str = "asset-run-1") -> None:
        self.experiments.save_run(
            run=BacktestRun(
                run_id=run_id,
                run_name="SPY asset model",
                target_asset="SPY",
                config_hash="asset-hash",
                train_window=60,
                validation_window=20,
                test_window=20,
                metrics={
                    "total_return": 0.08,
                    "sharpe": 1.2,
                    "sortino": 1.4,
                    "max_drawdown": -0.03,
                    "robust_score": 1.6,
                    "trade_count": 2,
                },
                selected_params={"threshold": 0.001, "min_post_count": 1, "account_weight": 1.0},
            ),
            config={
                "run_name": "SPY asset model",
                "target_asset": "SPY",
                "feature_version": "v1",
                "llm_enabled": True,
                "train_window": 60,
                "validation_window": 20,
                "test_window": 20,
                "transaction_cost_bps": 2.0,
            },
            trades=pd.DataFrame(
                {
                    "signal_session_date": [pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-04")],
                    "next_session_date": [pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-05")],
                    "trade_taken": [True, True],
                    "net_return": [0.01, -0.002],
                    "equity_curve": [1.01, 1.00798],
                },
            ),
            predictions=pd.DataFrame(
                {
                    "signal_session_date": [pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-04")],
                    "next_session_date": [pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-05")],
                    "expected_return_score": [0.012, 0.004],
                    "prediction_confidence": [0.7, 0.55],
                    "post_count": [4, 2],
                    "target_next_session_return": [0.01, -0.002],
                    "suggested_stance": ["LONG", "LONG"],
                },
            ),
            windows=pd.DataFrame({"window_id": [1], "train_start": [pd.Timestamp("2025-01-01")]}),
            importance=pd.DataFrame(
                {
                    "feature_name": ["post_count", "semantic_relevance_avg"],
                    "coefficient": [0.2, 0.4],
                    "abs_coefficient": [0.2, 0.4],
                },
            ),
            model_artifact={
                "model_version": "asset-linear-v1",
                "feature_names": ["post_count", "semantic_relevance_avg"],
                "intercept": 0.0,
                "coefficients": [0.2, 0.4],
                "means": [2.0, 0.5],
                "stds": [1.0, 0.2],
                "residual_std": 0.01,
                "train_rows": 30,
                "metadata": {"run_type": "asset_model", "target_asset": "SPY", "llm_enabled": True},
            },
            feature_contributions=pd.DataFrame(
                {
                    "signal_session_date": [pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-04")],
                    "feature_name": ["semantic_relevance_avg", "post_count"],
                    "feature_family": ["semantic", "activity"],
                    "raw_value": [0.9, 2.0],
                    "coefficient": [0.4, 0.2],
                    "contribution": [0.01, -0.002],
                    "contribution_share": [0.83, 0.17],
                },
            ),
            post_attribution=pd.DataFrame(
                {
                    "signal_session_date": [pd.Timestamp("2025-02-04")],
                    "post_timestamp": [pd.Timestamp("2025-02-04 12:00:00")],
                    "author_handle": ["realDonaldTrump"],
                    "author_is_trump": [True],
                    "is_active_tracked_account": [False],
                    "sentiment_score": [0.5],
                    "engagement_score": [10.0],
                    "post_signal_score": [0.4],
                    "post_preview": ["Market update"],
                },
            ),
            account_attribution=pd.DataFrame(
                {
                    "signal_session_date": [pd.Timestamp("2025-02-04")],
                    "author_handle": ["realDonaldTrump"],
                    "author_is_trump": [True],
                    "is_active_tracked_account": [False],
                    "post_count": [1],
                    "avg_sentiment": [0.5],
                    "net_post_signal": [0.4],
                    "total_engagement": [10.0],
                },
            ),
            benchmarks=pd.DataFrame(
                {
                    "benchmark_name": ["strategy", "always_long_spy"],
                    "total_return": [0.08, 0.05],
                    "sharpe": [1.2, 0.8],
                },
            ),
            diagnostics=pd.DataFrame(
                {
                    "signal_session_date": [pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-04")],
                    "expected_return_score": [0.012, 0.004],
                    "actual_next_session_return": [0.01, -0.002],
                    "absolute_error": [0.002, 0.006],
                },
            ),
            benchmark_curves=pd.DataFrame(
                {
                    "next_session_date": [pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-05")],
                    "strategy": [1.01, 1.00798],
                    "always_long_spy": [1.005, 1.003],
                },
            ),
            leakage_audit={"overall_pass": True},
        )

    def _save_portfolio_run_artifacts(self, run_id: str = "portfolio-run-1") -> None:
        self.experiments.save_portfolio_run(
            run=BacktestRun(
                run_id=run_id,
                run_name="Portfolio Alpha",
                target_asset="PORTFOLIO",
                config_hash="portfolio-hash",
                train_window=60,
                validation_window=20,
                test_window=20,
                metrics={
                    "total_return": 0.12,
                    "sharpe": 1.5,
                    "sortino": 1.7,
                    "max_drawdown": -0.025,
                    "robust_score": 2.1,
                    "trade_count": 2,
                },
                selected_params={
                    "fallback_mode": "SPY",
                    "selected_symbols": ["SPY", "QQQ"],
                    "deployment_variant": "per_asset_hybrid",
                    "deployment_narrative_feature_mode": "hybrid",
                },
                run_type="portfolio_allocator",
                allocator_mode="joint_model",
                fallback_mode="SPY",
                deployment_variant="per_asset_hybrid",
                selected_symbols=["SPY", "QQQ"],
                topology_variants=["per_asset"],
                narrative_feature_modes=["baseline", "hybrid"],
                model_families=["ridge"],
            ),
            config={
                "allocator_mode": "joint_model",
                "fallback_mode": "SPY",
                "selected_symbols": ["SPY", "QQQ"],
                "topology_variants": ["per_asset"],
                "narrative_feature_modes": ["baseline", "hybrid"],
                "model_families": ["ridge"],
                "transaction_cost_bps": 2.0,
            },
            trades=pd.DataFrame(
                {
                    "variant_name": ["per_asset_hybrid", "per_asset_hybrid"],
                    "signal_session_date": [pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-04")],
                    "next_session_date": [pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-05")],
                    "selected_asset": ["QQQ", "SPY"],
                    "trade_taken": [True, True],
                    "net_return": [0.015, 0.004],
                    "equity_curve": [1.015, 1.01906],
                },
            ),
            decision_history=pd.DataFrame(
                {
                    "variant_name": ["per_asset_hybrid", "per_asset_hybrid"],
                    "signal_session_date": [pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-04")],
                    "next_session_date": [pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-05")],
                    "winning_asset": ["QQQ", "SPY"],
                    "winning_run_id": ["qqq-model", "spy-model"],
                    "decision_source": ["eligible", "eligible"],
                    "fallback_mode": ["SPY", "SPY"],
                    "stance": ["LONG QQQ NEXT SESSION", "LONG SPY NEXT SESSION"],
                    "eligible_asset_count": [2, 1],
                    "runner_up_asset": ["SPY", "QQQ"],
                    "winner_score": [0.025, 0.012],
                    "runner_up_score": [0.01, 0.002],
                },
            ),
            candidate_predictions=pd.DataFrame(
                {
                    "variant_name": ["per_asset_hybrid", "per_asset_hybrid"],
                    "signal_session_date": [pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-04")],
                    "next_session_date": [pd.Timestamp("2025-02-05"), pd.Timestamp("2025-02-05")],
                    "asset_symbol": ["SPY", "QQQ"],
                    "run_id": ["spy-model", "qqq-model"],
                    "run_name": ["SPY model", "QQQ model"],
                    "expected_return_score": [0.012, 0.002],
                    "confidence": [0.7, 0.55],
                    "threshold": [0.001, 0.001],
                    "min_post_count": [1, 1],
                    "post_count": [3, 1],
                    "tradeable": [True, True],
                    "signal_qualifies": [True, True],
                    "qualifies": [True, True],
                    "eligible_rank": [1, 2],
                    "is_winner": [True, False],
                    "decision_source": ["eligible", "eligible"],
                    "stance": ["LONG SPY NEXT SESSION", "FLAT"],
                },
            ),
            component_summary=pd.DataFrame(
                {
                    "variant_name": ["per_asset_hybrid"],
                    "window_id": [1],
                    "model_family": ["ridge"],
                    "account_weight": [1.0],
                },
            ),
            benchmarks=pd.DataFrame(
                {
                    "variant_name": ["per_asset_hybrid", "per_asset_hybrid"],
                    "benchmark_name": ["strategy", "always_long_spy"],
                    "total_return": [0.12, 0.06],
                },
            ),
            benchmark_curves=pd.DataFrame(
                {
                    "variant_name": ["per_asset_hybrid", "per_asset_hybrid"],
                    "next_session_date": [pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-05")],
                    "strategy": [1.015, 1.01906],
                    "always_long_spy": [1.006, 1.01],
                },
            ),
            diagnostics=pd.DataFrame(
                {
                    "variant_name": ["per_asset_hybrid", "per_asset_hybrid"],
                    "signal_session_date": [pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-04")],
                    "winner_score": [0.025, 0.012],
                    "winner_gap_vs_runner_up": [0.015, 0.01],
                },
            ),
            leakage_audit={"overall_pass": True},
            variant_summary=pd.DataFrame(
                {
                    "variant_name": ["per_asset_baseline", "per_asset_hybrid"],
                    "topology": ["per_asset", "per_asset"],
                    "narrative_feature_mode": ["baseline", "hybrid"],
                    "model_family": ["ridge", "ridge"],
                    "validation_robust_score": [1.0, 1.4],
                    "validation_total_return": [0.04, 0.06],
                    "test_robust_score": [0.9, 1.2],
                    "test_total_return": [0.03, 0.05],
                    "deployment_winner": [False, True],
                },
            ),
            portfolio_model_bundle={
                "deployment_variant": "per_asset_hybrid",
                "selected_symbols": ["SPY", "QQQ"],
                "narrative_feature_modes": ["baseline", "hybrid"],
                "variants": {
                    "per_asset_hybrid": {
                        "variant_name": "per_asset_hybrid",
                        "topology": "per_asset",
                        "narrative_feature_mode": "hybrid",
                        "selected_symbols": ["SPY", "QQQ"],
                        "threshold": 0.001,
                        "min_post_count": 1,
                        "model_family": "ridge",
                        "models": {
                            "SPY": {
                                "model_version": "spy-live-test",
                                "feature_names": ["post_count", "semantic_relevance_avg"],
                                "intercept": 0.0,
                                "coefficients": [0.004, 0.0],
                                "means": [0.0, 0.0],
                                "stds": [1.0, 1.0],
                                "residual_std": 0.05,
                            },
                            "QQQ": {
                                "model_version": "qqq-live-test",
                                "feature_names": ["post_count", "policy_trade_count"],
                                "intercept": 0.0,
                                "coefficients": [0.006, 0.0],
                                "means": [0.0, 0.0],
                                "stds": [1.0, 1.0],
                                "residual_std": 0.05,
                            },
                        },
                    },
                },
            },
            importance=pd.DataFrame(
                {
                    "variant_name": ["per_asset_hybrid", "per_asset_hybrid"],
                    "feature_name": ["semantic_relevance_avg", "policy_trade_count"],
                    "importance": [0.7, 0.3],
                },
            ),
        )

    def _save_paper_frames(self) -> None:
        signal_date = pd.Timestamp("2026-04-17", tz="UTC")
        self.store.save_frame(
            "paper_portfolio_registry",
            pd.DataFrame(
                [
                    {
                        "paper_portfolio_id": "paper-1",
                        "portfolio_run_id": "run-1",
                        "portfolio_run_name": "Portfolio run",
                        "deployment_variant": "per_asset",
                        "fallback_mode": "SPY",
                        "transaction_cost_bps": 2.0,
                        "starting_cash": 100000.0,
                        "enabled": True,
                        "created_at": signal_date,
                        "archived_at": pd.NaT,
                    },
                ],
                columns=PAPER_PORTFOLIO_REGISTRY_COLUMNS,
            ),
        )
        self.store.save_frame(
            "paper_decision_journal",
            pd.DataFrame(
                [
                    {
                        "paper_portfolio_id": "paper-1",
                        "generated_at": signal_date,
                        "signal_session_date": signal_date,
                        "next_session_date": signal_date + pd.Timedelta(days=1),
                        "decision_cutoff_ts": signal_date + pd.Timedelta(days=1, hours=13, minutes=30),
                        "portfolio_run_id": "run-1",
                        "portfolio_run_name": "Portfolio run",
                        "deployment_variant": "per_asset",
                        "winning_asset": "SPY",
                        "winning_run_id": "run-1",
                        "decision_source": "eligible",
                        "fallback_mode": "SPY",
                        "stance": "LONG",
                        "winner_score": 0.02,
                        "runner_up_asset": "",
                        "runner_up_score": 0.0,
                        "eligible_asset_count": 1,
                        "settlement_status": "settled",
                        "settled_at": signal_date + pd.Timedelta(days=1, hours=21),
                    },
                ],
                columns=PAPER_DECISION_JOURNAL_COLUMNS,
            ),
        )
        self.store.save_frame(
            "paper_trade_ledger",
            pd.DataFrame(
                [
                    {
                        "paper_portfolio_id": "paper-1",
                        "signal_session_date": signal_date,
                        "next_session_date": signal_date + pd.Timedelta(days=1),
                        "asset_symbol": "SPY",
                        "run_id": "run-1",
                        "decision_source": "eligible",
                        "stance": "LONG",
                        "next_session_open": 100.0,
                        "next_session_close": 101.0,
                        "gross_return": 0.01,
                        "net_return": 0.0096,
                        "benchmark_return": 0.01,
                        "transaction_cost_bps": 2.0,
                        "starting_equity": 100000.0,
                        "ending_equity": 100960.0,
                        "settled_at": signal_date + pd.Timedelta(days=1, hours=21),
                    },
                ],
                columns=PAPER_TRADE_LEDGER_COLUMNS,
            ),
        )
        self.store.save_frame(
            "paper_equity_curve",
            pd.DataFrame(
                [
                    {
                        "paper_portfolio_id": "paper-1",
                        "signal_session_date": signal_date,
                        "next_session_date": signal_date + pd.Timedelta(days=1),
                        "equity": 100960.0,
                        "return_pct": 0.0096,
                        "settled_at": signal_date + pd.Timedelta(days=1, hours=21),
                    },
                ],
                columns=PAPER_EQUITY_CURVE_COLUMNS,
            ),
        )
        self.store.save_frame(
            "paper_benchmark_curve",
            pd.DataFrame(
                [
                    {
                        "paper_portfolio_id": "paper-1",
                        "benchmark_name": "always_long_spy",
                        "signal_session_date": signal_date,
                        "next_session_date": signal_date + pd.Timedelta(days=1),
                        "equity": 101000.0,
                        "return_pct": 0.01,
                        "settled_at": signal_date + pd.Timedelta(days=1, hours=21),
                    },
                ],
                columns=PAPER_BENCHMARK_CURVE_COLUMNS,
            ),
        )

    def _admin_token(self, client: TestClient | None = None) -> str:
        response = (client or self.client).post("/api/admin/session", json={})
        self.assertEqual(response.status_code, 200)
        return str(response.json()["token"])

    def _save_live_ops_fixture(self) -> None:
        self._save_portfolio_run_artifacts()
        signal_date = pd.Timestamp("2025-02-04")
        next_date = pd.Timestamp("2025-02-05")
        self.store.save_frame(
            "asset_session_features",
            pd.DataFrame(
                [
                    {
                        "asset_symbol": "SPY",
                        "signal_session_date": signal_date,
                        "next_session_date": next_date,
                        "post_count": 3,
                        "semantic_relevance_avg": 0.5,
                        "policy_trade_count": 0.0,
                        "target_available": True,
                        "tradeable": True,
                        "next_session_open": 100.0,
                        "next_session_close": 101.0,
                        "next_session_open_ts": pd.Timestamp("2025-02-05 14:30:00", tz="UTC"),
                    },
                    {
                        "asset_symbol": "QQQ",
                        "signal_session_date": signal_date,
                        "next_session_date": next_date,
                        "post_count": 2,
                        "semantic_relevance_avg": 0.1,
                        "policy_trade_count": 1.0,
                        "target_available": True,
                        "tradeable": True,
                        "next_session_open": 200.0,
                        "next_session_close": 204.0,
                        "next_session_open_ts": pd.Timestamp("2025-02-05 14:30:00", tz="UTC"),
                    },
                ],
            ),
        )
        self.store.save_frame(
            "asset_daily",
            pd.DataFrame(
                [
                    {"symbol": "SPY", "trade_date": next_date, "open": 100.0, "close": 101.0},
                    {"symbol": "QQQ", "trade_date": next_date, "open": 200.0, "close": 204.0},
                ],
            ),
        )
        self.store.save_frame("asset_post_mappings", pd.DataFrame())

    def test_status_reports_source_mode_and_state_paths(self) -> None:
        self._save_truth_posts()

        response = self.client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source_mode"]["mode"], "truth_only")
        self.assertIn(".workbench", payload["db_path"])

    def test_research_endpoint_returns_empty_state_without_core_data(self) -> None:
        response = self.client.get("/api/research")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ready"])
        self.assertIn("Refresh datasets first", payload["message"])
        self.assertEqual(payload["source_mode"]["mode"], "unknown")
        self.assertEqual(payload["session_rows"], [])

    def test_research_endpoint_seeds_truth_only_defaults_without_cache_writes(self) -> None:
        self._save_research_frames(include_x=False)

        response = self.client.get("/api/research?date_start=2025-02-03&date_end=2025-02-05")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["source_mode"]["mode"], "truth_only")
        self.assertEqual(payload["filters"]["platforms"], ["Truth Social"])
        self.assertTrue(payload["filters"]["trump_authored_only"])
        self.assertEqual(payload["headline_metrics"]["posts_in_view"], 1)
        self.assertEqual(len(payload["post_rows"]), 1)
        self.assertIn("social_activity", payload["charts"])
        self.assertTrue(self.store.read_frame("semantic_cache").empty)

    def test_research_endpoint_applies_explicit_filters_and_narrative_filters(self) -> None:
        self._save_research_frames(include_x=True)

        response = self.client.get(
            "/api/research",
            params=[
                ("date_start", "2025-02-03"),
                ("date_end", "2025-02-05"),
                ("platforms", "X"),
                ("trump_authored_only", "false"),
                ("keyword", "tariff"),
                ("narrative_topic", "trade"),
            ],
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source_mode"]["mode"], "truth_plus_x")
        self.assertEqual(payload["filters"]["platforms"], ["X"])
        self.assertFalse(payload["filters"]["trump_authored_only"])
        self.assertEqual(len(payload["post_rows"]), 1)
        self.assertEqual(payload["post_rows"][0]["author_handle"], "macroalpha")
        self.assertEqual(payload["narrative_frequency"][0]["semantic_topic"], "trade")
        self.assertEqual(payload["narrative_metrics"]["narrative_tagged_posts"], 1)

    def test_research_export_endpoint_returns_filtered_zip_bundle(self) -> None:
        self._save_research_frames(include_x=False)

        response = self.client.get("/api/research/export?date_start=2025-02-03&date_end=2025-02-05")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")
        self.assertIn("research-pack-20250203-20250205.zip", response.headers["content-disposition"])
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertIn("manifest.json", archive.namelist())
            self.assertIn("sessions.csv", archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["source_mode"]["mode"], "truth_only")
            self.assertTrue(manifest["filters"]["trump_authored_only"])
            self.assertEqual(manifest["headline_metrics"]["posts_in_view"], 1)

    def test_discovery_endpoint_returns_empty_state_without_posts(self) -> None:
        response = self.client.get("/api/discovery")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ready"])
        self.assertIn("Refresh datasets", payload["message"])
        self.assertEqual(payload["source_mode"]["mode"], "unknown")
        self.assertEqual(payload["active_accounts"], [])
        self.assertEqual(payload["latest_rankings"], [])
        self.assertIn("top_discovered_accounts", payload["charts"])

    def test_discovery_endpoint_explains_truth_only_mode(self) -> None:
        self._save_truth_posts()

        response = self.client.get("/api/discovery")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["source_mode"]["mode"], "truth_only")
        self.assertIn("Discovery ranks non-Trump X accounts", payload["message"])
        self.assertEqual(payload["summary"]["x_candidate_post_count"], 0)

    def test_discovery_endpoint_returns_rankings_active_accounts_and_overrides(self) -> None:
        self._save_discovery_frames()

        response = self.client.get("/api/discovery")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["message"], "")
        self.assertEqual(payload["summary"]["x_candidate_post_count"], 1)
        self.assertEqual(payload["summary"]["active_account_count"], 2)
        self.assertEqual(payload["summary"]["latest_ranking_count"], 3)
        self.assertEqual(payload["summary"]["pin_override_count"], 1)
        self.assertEqual(payload["summary"]["suppress_override_count"], 1)
        self.assertEqual(payload["latest_rankings"][0]["author_account_id"], "acct-macro")
        self.assertEqual(payload["latest_rankings"][0]["discovery_score"], 12.0)
        self.assertEqual({row["handle"] for row in payload["active_accounts"]}, {"macroalpha", "policywatch"})
        self.assertEqual({row["action"] for row in payload["override_history"]}, {"pin", "suppress"})
        self.assertGreaterEqual(len(payload["recent_ranking_history"]), 1)
        self.assertIsNotNone(payload["latest_ranked_at"])

    def test_discovery_endpoint_handles_schema_less_ranking_history(self) -> None:
        self._save_research_frames(include_x=True)
        self.store.save_frame("account_rankings", pd.DataFrame({"other": [1]}))

        response = self.client.get("/api/discovery")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ready"])
        self.assertIn("rankings have not been built", payload["message"])
        self.assertEqual(payload["latest_rankings"], [])

    def test_dataset_health_returns_summary_rows_and_registry(self) -> None:
        self._save_truth_posts()

        response = self.client.get("/api/datasets/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("overall_severity", payload["summary"])
        self.assertIsInstance(payload["latest"], list)
        self.assertGreaterEqual(len(payload["registry"]), 1)

    def test_dataset_admin_returns_safe_empty_state_payload(self) -> None:
        response = self.client.get("/api/datasets/admin")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["admin"]["mode"], "private")
        self.assertIn("status", payload)
        self.assertGreaterEqual(payload["status"]["missing_core_dataset_count"], 1)
        self.assertIn("normalized_posts", payload["status"]["missing_core_datasets"])
        self.assertIn("overall_severity", payload["summary"])
        self.assertEqual(payload["watchlist_symbols"], [])
        self.assertEqual(payload["refresh_jobs"], [])

    def test_dataset_watchlist_endpoint_requires_admin_and_persists_symbols(self) -> None:
        rejected = self.client.post("/api/datasets/watchlist", json={"symbols": ["NVDA"], "reset": False})
        token = self._admin_token()

        saved = self.client.post(
            "/api/datasets/watchlist",
            json={"symbols": ["nvda", "TSLA"], "reset": False},
            headers={"Authorization": f"Bearer {token}"},
        )
        reset = self.client.post(
            "/api/datasets/watchlist",
            json={"symbols": [], "reset": True},
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(rejected.status_code, 401)
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["watchlist_symbols"], ["NVDA", "TSLA"])
        self.assertIn("SPY", {row["symbol"] for row in saved.json()["asset_universe"]})
        self.assertIn("NVDA", {row["symbol"] for row in saved.json()["asset_universe"]})
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(reset.json()["watchlist_symbols"], [])

    def test_dataset_refresh_rejects_overlapping_lock(self) -> None:
        token = self._admin_token()
        lock_fd = acquire_refresh_lock(self.settings)
        self.assertIsNotNone(lock_fd)
        try:
            response = self.client.post(
                "/api/datasets/refresh",
                data={"refresh_mode": "full", "remote_url": ""},
                headers={"Authorization": f"Bearer {token}"},
            )
        finally:
            release_refresh_lock(self.settings, lock_fd)

        self.assertEqual(response.status_code, 409)
        self.assertIn("already running", str(response.json()["detail"]))

    def test_dataset_refresh_job_persists_success_health_and_history(self) -> None:
        client = TestClient(
            create_app(
                settings=self.settings,
                store=self.store,
                ingestion_service=FakeIngestionService(),
                market_service=FakeMarketDataService(),
                feature_service=FakeFeatureService(),
                run_refresh_jobs_inline=True,
            ),
        )
        token = self._admin_token(client)

        response = client.post(
            "/api/datasets/refresh",
            data={"refresh_mode": "full", "remote_url": ""},
            files=[("files", ("mentions.csv", b"author,text\nmacro,hello\n", "text/csv"))],
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["job_id"].startswith("dataset-refresh-"))
        jobs = self.store.read_frame("dataset_refresh_jobs")
        self.assertFalse(jobs.empty)
        self.assertEqual(str(jobs.iloc[-1]["status"]), "success")
        self.assertEqual(int(jobs.iloc[-1]["uploaded_file_count"]), 1)
        self.assertFalse(self.store.read_frame("normalized_posts").empty)
        self.assertFalse(self.store.read_frame("refresh_history").empty)
        self.assertFalse(self.store.read_frame("data_health_latest").empty)
        self.assertFalse(self.store.read_frame("data_health_history").empty)

    def test_dataset_refresh_job_persists_error_without_overwriting_prior_health(self) -> None:
        prior_health = pd.DataFrame(
            [
                {
                    "snapshot_id": "prior",
                    "generated_at": pd.Timestamp("2025-02-01", tz="UTC"),
                    "refresh_id": "prior-refresh",
                    "scope_kind": "dataset",
                    "scope_key": "normalized_posts",
                    "check_name": "prior_check",
                    "severity": "ok",
                    "observed_value": 1.0,
                    "baseline_value": 1.0,
                    "detail": "prior snapshot",
                },
            ],
            columns=HEALTH_CHECK_COLUMNS,
        )
        self.store.save_frame("data_health_latest", prior_health)
        client = TestClient(
            create_app(
                settings=self.settings,
                store=self.store,
                ingestion_service=FakeIngestionService(fail=True),
                market_service=FakeMarketDataService(),
                feature_service=FakeFeatureService(),
                run_refresh_jobs_inline=True,
            ),
        )
        token = self._admin_token(client)

        response = client.post(
            "/api/datasets/refresh",
            data={"refresh_mode": "bootstrap", "remote_url": ""},
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 200)
        jobs = self.store.read_frame("dataset_refresh_jobs")
        self.assertEqual(str(jobs.iloc[-1]["status"]), "error")
        self.assertIn("fake ingestion failed", str(jobs.iloc[-1]["error_message"]))
        refresh_history = self.store.read_frame("refresh_history")
        self.assertEqual(str(refresh_history.iloc[-1]["status"]), "error")
        latest = self.store.read_frame("data_health_latest")
        self.assertEqual(str(latest.iloc[0]["check_name"]), "prior_check")

    def test_runs_and_live_current_handle_empty_live_config(self) -> None:
        self._save_run_record()

        runs_response = self.client.get("/api/runs")
        live_response = self.client.get("/api/live/current")

        self.assertEqual(runs_response.status_code, 200)
        self.assertEqual(runs_response.json()["count"], 1)
        self.assertEqual(live_response.status_code, 200)
        self.assertFalse(live_response.json()["configured"])
        self.assertIn("No live monitor config", live_response.json()["errors"][0])

    def test_admin_session_auth_modes_and_protected_writes(self) -> None:
        private_response = self.client.post("/api/admin/session", json={})
        self.assertEqual(private_response.status_code, 200)
        self.assertIn("token", private_response.json())
        protected_response = self.client.post("/api/live/config", json={"portfolio_run_id": "missing", "fallback_mode": "SPY"})
        self.assertEqual(protected_response.status_code, 401)

        with tempfile.TemporaryDirectory() as temp_dir:
            public_settings = AppSettings(base_dir=Path(temp_dir), public_mode=True, admin_password="secret")
            public_client = TestClient(create_app(settings=public_settings, store=DuckDBStore(public_settings)))
            bad = public_client.post("/api/admin/session", json={"password": "wrong"})
            good = public_client.post("/api/admin/session", json={"password": "secret"})
            self.assertEqual(bad.status_code, 401)
            self.assertEqual(good.status_code, 200)

    def test_live_ops_config_save_validates_and_returns_payload(self) -> None:
        self._save_portfolio_run_artifacts()
        token = self._admin_token()

        bad = self.client.post(
            "/api/live/config",
            json={"portfolio_run_id": "missing", "fallback_mode": "SPY"},
            headers={"Authorization": f"Bearer {token}"},
        )
        good = self.client.post(
            "/api/live/config",
            json={"portfolio_run_id": "portfolio-run-1", "fallback_mode": "FLAT"},
            headers={"Authorization": f"Bearer {token}"},
        )
        ops = self.client.get("/api/live/ops")

        self.assertEqual(bad.status_code, 400)
        self.assertEqual(good.status_code, 200)
        self.assertEqual(good.json()["current_config"]["portfolio_run_id"], "portfolio-run-1")
        self.assertEqual(good.json()["current_config"]["fallback_mode"], "FLAT")
        self.assertEqual(ops.status_code, 200)
        self.assertEqual(ops.json()["run_options"][0]["run_id"], "portfolio-run-1")

    def test_live_ops_capture_persists_snapshots_without_refresh(self) -> None:
        self._save_live_ops_fixture()
        token = self._admin_token()
        self.client.post(
            "/api/live/config",
            json={"portfolio_run_id": "portfolio-run-1", "fallback_mode": "SPY"},
            headers={"Authorization": f"Bearer {token}"},
        )

        response = self.client.post("/api/live/capture", json={}, headers={"Authorization": f"Bearer {token}"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["capture_result"]["persisted_assets"], 2)
        self.assertEqual(payload["capture_result"]["persisted_decisions"], 1)
        self.assertFalse(self.store.read_frame("live_asset_snapshots").empty)
        self.assertFalse(self.store.read_frame("live_decision_snapshots").empty)
        self.assertTrue(self.store.read_frame("refresh_history").empty)

    def test_paper_current_actions_manage_active_portfolio(self) -> None:
        self._save_live_ops_fixture()
        token = self._admin_token()
        self.client.post(
            "/api/live/config",
            json={"portfolio_run_id": "portfolio-run-1", "fallback_mode": "SPY"},
            headers={"Authorization": f"Bearer {token}"},
        )

        enabled = self.client.post(
            "/api/paper/current",
            json={"action": "enable", "starting_cash": 123000},
            headers={"Authorization": f"Bearer {token}"},
        )
        disabled = self.client.post(
            "/api/paper/current",
            json={"action": "disable"},
            headers={"Authorization": f"Bearer {token}"},
        )
        reset = self.client.post(
            "/api/paper/current",
            json={"action": "reset", "starting_cash": 125000},
            headers={"Authorization": f"Bearer {token}"},
        )
        archived = self.client.post(
            "/api/paper/current",
            json={"action": "archive"},
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(enabled.status_code, 200)
        self.assertEqual(enabled.json()["paper"]["active_config"]["starting_cash"], 123000.0)
        self.assertEqual(disabled.status_code, 200)
        self.assertFalse(disabled.json()["paper"]["active_config"]["enabled"])
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(reset.json()["paper"]["active_config"]["starting_cash"], 125000.0)
        self.assertEqual(archived.status_code, 200)
        self.assertIsNone(archived.json()["paper"]["active_config"])
        registry = self.store.read_frame("paper_portfolio_registry")
        self.assertGreaterEqual(len(registry), 2)
        self.assertTrue(registry["archived_at"].notna().any())

    def test_run_compare_endpoint_handles_empty_store(self) -> None:
        response = self.client.get("/api/runs/compare")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["scorecard"], [])
        self.assertEqual(payload["change_notes"], [])

    def test_asset_run_detail_returns_model_inspection_payload(self) -> None:
        self._save_asset_run_artifacts()

        response = self.client.get("/api/runs/asset-run-1?session_date=2025-02-04")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["found"])
        self.assertEqual(payload["settings"]["run_type"], "asset_model")
        self.assertEqual(payload["settings"]["target_asset"], "SPY")
        self.assertEqual(payload["model_artifact"]["feature_count"], 2)
        self.assertIn("equity", payload["charts"])
        self.assertEqual(len(payload["tables"]["feature_importance"]), 2)
        self.assertEqual(payload["selected_session"]["session_date"], "2025-02-04")
        self.assertEqual(payload["selected_session"]["prediction"][0]["expected_return_score"], 0.004)
        self.assertGreaterEqual(len(payload["selected_session"]["feature_contributions"]), 1)
        self.assertEqual(payload["leakage_audit"]["overall_pass"], True)

    def test_portfolio_run_detail_returns_variant_and_candidate_payload(self) -> None:
        self._save_portfolio_run_artifacts()

        response = self.client.get("/api/runs/portfolio-run-1?variant_name=per_asset_hybrid&session_date=2025-02-04")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["found"])
        self.assertEqual(payload["settings"]["run_type"], "portfolio_allocator")
        self.assertEqual(payload["settings"]["deployment_variant"], "per_asset_hybrid")
        self.assertEqual(payload["settings"]["deployment_narrative_feature_mode"], "hybrid")
        self.assertGreaterEqual(len(payload["tables"]["variant_summary"]), 2)
        self.assertEqual(payload["tables"]["narrative_lift"][0]["narrative_feature_mode"], "hybrid")
        self.assertGreaterEqual(len(payload["tables"]["feature_family_summary"]), 1)
        self.assertEqual(payload["selected_session"]["decision"][0]["winning_asset"], "SPY")
        self.assertEqual(len(payload["selected_session"]["candidates"]), 2)
        self.assertIn("diagnostics", payload["charts"])

    def test_run_comparison_endpoint_returns_mixed_run_diffs(self) -> None:
        self._save_asset_run_artifacts()
        self._save_portfolio_run_artifacts()

        response = self.client.get(
            "/api/runs/compare",
            params=[
                ("run_ids", "asset-run-1"),
                ("run_ids", "portfolio-run-1"),
                ("base_run_id", "asset-run-1"),
            ],
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["base_run_id"], "asset-run-1")
        self.assertEqual(len(payload["scorecard"]), 2)
        self.assertGreaterEqual(len(payload["setting_diffs"]), 1)
        self.assertGreaterEqual(len(payload["feature_diffs"]), 2)
        self.assertGreaterEqual(len(payload["benchmark_deltas"]), 1)
        self.assertTrue(any("portfolio-run-1" in note for note in payload["change_notes"]))

    def test_paper_and_performance_endpoints_return_portfolio_slices(self) -> None:
        self._save_paper_frames()

        portfolios_response = self.client.get("/api/paper/portfolios")
        paper_response = self.client.get("/api/paper/paper-1")
        performance_response = self.client.get("/api/performance/paper-1")

        self.assertEqual(portfolios_response.status_code, 200)
        self.assertEqual(len(portfolios_response.json()["portfolios"]), 1)
        self.assertEqual(paper_response.status_code, 200)
        self.assertEqual(len(paper_response.json()["trade_ledger"]), 1)
        self.assertEqual(performance_response.status_code, 200)
        self.assertIn("total_return", performance_response.json()["summary"])
        self.assertGreaterEqual(len(performance_response.json()["diagnostics"]), 1)


if __name__ == "__main__":
    unittest.main()
