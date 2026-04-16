from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from trump_workbench.backtesting import BacktestService
from trump_workbench.config import DEFAULT_ETF_SYMBOLS, EASTERN, AppSettings
from trump_workbench.contracts import BacktestRun, LiveMonitorConfig, MANUAL_OVERRIDE_COLUMNS, PortfolioRunConfig
from trump_workbench.experiments import ExperimentStore
from trump_workbench.features import ASSET_POST_MAPPING_COLUMNS
from trump_workbench.health import DataHealthService, ensure_refresh_history_frame
from trump_workbench.live_monitor import build_live_portfolio_run_state
from trump_workbench.modeling import ModelService
from trump_workbench.paper_trading import (
    PaperTradingService,
    ensure_paper_benchmark_curve_frame,
    ensure_paper_decision_journal_frame,
    ensure_paper_equity_curve_frame,
    ensure_paper_portfolio_registry_frame,
    ensure_paper_trade_ledger_frame,
)
from trump_workbench.portfolio import rank_portfolio_candidates
from trump_workbench.runtime import refresh_datasets
from trump_workbench.scheduler import SchedulerDecision, run_scheduler_cycle
from trump_workbench.storage import DuckDBStore


class _FakeIngestionService:
    def __init__(self, posts: pd.DataFrame, source_manifest: pd.DataFrame) -> None:
        self.posts = posts
        self.source_manifest = source_manifest

    def run_refresh(self, adapters):  # noqa: ANN001, ARG002
        return self.posts.copy(), self.source_manifest.copy()

    def run_incremental_refresh(self, adapters, last_cursor):  # noqa: ANN001, ARG002
        return self.posts.copy(), self.source_manifest.copy()


class _FakeMarketService:
    def __init__(
        self,
        sp500: pd.DataFrame,
        spy: pd.DataFrame,
        asset_daily: pd.DataFrame,
        asset_intraday: pd.DataFrame,
        asset_market_manifest: pd.DataFrame,
    ) -> None:
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

    def refresh_accounts(
        self,
        posts: pd.DataFrame,
        existing_accounts: pd.DataFrame,
        as_of: pd.Timestamp,
        manual_overrides: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:  # noqa: ARG002
        return self.tracked_accounts.copy(), self.ranking_history.copy()


class _FakeFeatureService:
    def __init__(
        self,
        prepared_posts: pd.DataFrame,
        asset_post_mappings: pd.DataFrame,
        asset_session_features: pd.DataFrame,
    ) -> None:
        self.prepared_posts = prepared_posts
        self.asset_post_mappings = asset_post_mappings
        self.asset_session_features = asset_session_features

    def prepare_session_posts(
        self,
        posts: pd.DataFrame,
        market_calendar: pd.DataFrame,
        tracked_accounts: pd.DataFrame,
        llm_enabled: bool,
    ) -> pd.DataFrame:  # noqa: ARG002
        return self.prepared_posts.copy()

    def build_asset_post_mappings(
        self,
        prepared_posts: pd.DataFrame,
        asset_universe: pd.DataFrame,
        llm_enabled: bool,
    ) -> pd.DataFrame:  # noqa: ARG002
        return self.asset_post_mappings.copy()

    def build_asset_session_dataset(
        self,
        asset_post_mappings: pd.DataFrame,
        asset_market: pd.DataFrame,
        feature_version: str,
        llm_enabled: bool,
        asset_universe: pd.DataFrame,
    ) -> pd.DataFrame:  # noqa: ARG002
        return self.asset_session_features.copy()


class IntegrationWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(self.settings)
        self.model_service = ModelService()
        self.backtests = BacktestService(self.model_service)
        self.experiments = ExperimentStore(self.store)
        self.paper_service = PaperTradingService(self.store)
        self.health_service = DataHealthService()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _joint_feature_rows(symbols: list[str], periods: int = 48) -> pd.DataFrame:
        signal_dates = pd.bdate_range("2026-01-02", periods=periods)
        rows: list[dict[str, object]] = []
        for idx, signal_date in enumerate(signal_dates):
            next_session_date = signal_dates[idx + 1] if idx + 1 < len(signal_dates) else signal_date + pd.offsets.BDay(1)
            strong_asset = symbols[idx % len(symbols)]
            for asset_offset, asset_symbol in enumerate(symbols):
                score = 0.8 if asset_symbol == strong_asset else -0.2
                target_return = 0.010 if asset_symbol == strong_asset else -0.002
                open_price = 100.0 + asset_offset * 50.0 + idx
                close_price = open_price * (1.0 + target_return)
                next_open_ts = pd.Timestamp(f"{pd.Timestamp(next_session_date).date()} 09:30", tz=EASTERN).tz_convert("UTC")
                rows.append(
                    {
                        "signal_session_date": signal_date,
                        "next_session_date": next_session_date,
                        "asset_symbol": asset_symbol,
                        "feature_version": "asset-v1",
                        "llm_enabled": False,
                        "target_next_session_return": target_return,
                        "target_available": True,
                        "tradeable": True,
                        "next_session_open": open_price,
                        "next_session_close": close_price,
                        "next_session_open_ts": next_open_ts,
                        "post_count": 4 + asset_offset,
                        "trump_post_count": 1,
                        "tracked_account_post_count": 2,
                        "tracked_weighted_mentions": 1.0 + score,
                        "tracked_weighted_engagement": 5.0 + score,
                        "sentiment_avg": score,
                        "sentiment_close": score * 0.9,
                        "sentiment_range": 0.2 + asset_offset * 0.05,
                        "mention_concentration": 0.3 + asset_offset * 0.02,
                        "author_diversity": 2.0 + asset_offset,
                        "session_return": 0.001 * (idx % 5),
                        "rolling_vol_5d": 0.01 + asset_offset * 0.001,
                        "close_vs_ma_5": score * 0.1,
                        "volume_z_5": 0.2 + asset_offset * 0.03,
                    },
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _posts() -> pd.DataFrame:
        post_ts = pd.Timestamp("2026-01-02 10:00:00", tz=EASTERN)
        return pd.DataFrame(
            [
                {
                    "source_platform": "X",
                    "source_type": "x_csv",
                    "author_account_id": "acct-macro",
                    "author_handle": "macroalpha",
                    "author_display_name": "Macro Alpha",
                    "author_is_trump": False,
                    "post_id": "post-1",
                    "post_url": "https://x.com/macroalpha/status/1",
                    "post_timestamp": post_ts,
                    "raw_text": "Trump says tech stocks and the broad market are moving.",
                    "cleaned_text": "Trump says tech stocks and the broad market are moving.",
                    "is_reshare": False,
                    "has_media": False,
                    "replies_count": 1,
                    "reblogs_count": 2,
                    "favourites_count": 3,
                    "mentions_trump": True,
                    "source_provenance": "integration-test",
                    "engagement_score": 6.0,
                    "sentiment_score": 0.4,
                    "sentiment_label": "positive",
                },
            ],
        )

    @staticmethod
    def _source_manifest(posts: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source": "Synthetic X",
                    "provenance": "integration-test",
                    "post_count": int(len(posts)),
                    "coverage_start": posts["post_timestamp"].min(),
                    "coverage_end": posts["post_timestamp"].max(),
                    "status": "ok",
                    "detail": "",
                },
            ],
        )

    @staticmethod
    def _tracked_accounts() -> tuple[pd.DataFrame, pd.DataFrame]:
        ranked_at = pd.Timestamp("2026-01-02 12:00:00", tz=EASTERN)
        tracked = pd.DataFrame(
            [
                {
                    "version_id": "v1",
                    "account_id": "acct-macro",
                    "handle": "macroalpha",
                    "display_name": "Macro Alpha",
                    "source_platform": "X",
                    "discovery_score": 1.0,
                    "status": "active",
                    "first_seen_at": ranked_at,
                    "last_seen_at": ranked_at,
                    "effective_from": ranked_at,
                    "effective_to": pd.NaT,
                    "auto_included": True,
                    "provenance": "integration-test",
                    "mention_count": 1,
                    "engagement_mean": 6.0,
                    "active_days": 1,
                },
            ],
        )
        rankings = pd.DataFrame(
            [
                {
                    "author_account_id": "acct-macro",
                    "author_handle": "macroalpha",
                    "author_display_name": "Macro Alpha",
                    "source_platform": "X",
                    "discovery_score": 1.0,
                    "mention_count": 1,
                    "engagement_mean": 6.0,
                    "active_days": 1,
                    "ranked_at": ranked_at,
                    "discovery_rank": 1,
                    "final_selected": True,
                    "selected_status": "active",
                    "suppressed_by_override": False,
                    "pinned_by_override": False,
                },
            ],
        )
        return tracked, rankings

    @staticmethod
    def _asset_post_mappings(feature_rows: pd.DataFrame) -> pd.DataFrame:
        latest_session = pd.to_datetime(feature_rows["signal_session_date"]).max()
        rows: list[dict[str, object]] = []
        for asset_symbol in sorted(feature_rows["asset_symbol"].astype(str).str.upper().unique().tolist()):
            row = {
                "asset_symbol": asset_symbol,
                "asset_display_name": asset_symbol,
                "asset_type": "etf",
                "asset_source": "core_etf",
                "session_date": latest_session,
                "post_id": f"post-{asset_symbol.lower()}",
                "post_timestamp": pd.Timestamp(f"{latest_session.date()} 10:00", tz=EASTERN),
                "reaction_anchor_ts": pd.Timestamp(f"{latest_session.date()} 16:00", tz=EASTERN),
                "mapping_reason": "integration fixture",
                "author_account_id": "acct-macro",
                "author_handle": "macroalpha",
                "author_display_name": "Macro Alpha",
                "author_is_trump": False,
                "source_platform": "X",
                "cleaned_text": f"Trump market narrative for {asset_symbol}",
                "mentions_trump": True,
                "engagement_score": 10.0,
                "sentiment_score": 0.6,
                "sentiment_label": "positive",
                "semantic_topic": "markets",
                "semantic_policy_bucket": "economy",
                "semantic_stance": "supportive",
                "semantic_market_relevance": "high",
                "semantic_urgency": "medium",
                "semantic_primary_asset": asset_symbol,
                "semantic_asset_targets": asset_symbol,
                "semantic_confidence": 0.8,
                "semantic_summary": f"{asset_symbol} narrative",
                "semantic_schema_version": "narrative-v1",
                "semantic_provider": "heuristic",
                "is_active_tracked_account": True,
                "tracked_discovery_score": 1.0,
                "tracked_account_status": "active",
                "rule_match_score": 1.0,
                "semantic_match_score": 0.5,
                "asset_relevance_score": 1.0,
                "match_reasons": "integration",
                "match_rank": 1,
                "is_primary_asset": True,
            }
            rows.append(row)
        return pd.DataFrame(rows, columns=ASSET_POST_MAPPING_COLUMNS)

    @staticmethod
    def _market_frames(feature_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        selected_daily = feature_rows[
            ["asset_symbol", "next_session_date", "next_session_open", "next_session_close"]
        ].rename(
            columns={
                "asset_symbol": "symbol",
                "next_session_date": "trade_date",
                "next_session_open": "open",
                "next_session_close": "close",
            },
        )
        selected_daily = selected_daily.dropna(subset=["trade_date"]).drop_duplicates(["symbol", "trade_date"], keep="last")
        selected_daily["high"] = selected_daily[["open", "close"]].max(axis=1) * 1.01
        selected_daily["low"] = selected_daily[["open", "close"]].min(axis=1) * 0.99
        selected_daily["volume"] = 1_000_000
        selected_daily = selected_daily[["symbol", "trade_date", "open", "high", "low", "close", "volume"]]

        trade_dates = pd.to_datetime(selected_daily["trade_date"]).drop_duplicates().sort_values().tolist()
        filler_rows: list[dict[str, object]] = []
        for symbol in DEFAULT_ETF_SYMBOLS:
            if symbol in set(selected_daily["symbol"].astype(str)):
                continue
            for idx, trade_date in enumerate(trade_dates):
                open_price = 50.0 + idx
                close_price = open_price * 1.001
                filler_rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "open": open_price,
                        "high": close_price * 1.01,
                        "low": open_price * 0.99,
                        "close": close_price,
                        "volume": 500_000,
                    },
                )
        asset_daily = pd.concat([selected_daily, pd.DataFrame(filler_rows)], ignore_index=True)
        asset_daily = asset_daily.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        spy = asset_daily.loc[asset_daily["symbol"] == "SPY", ["trade_date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        sp500 = spy[["trade_date", "close"]].copy()

        latest_intraday_ts = pd.Timestamp("2026-04-15 15:55:00", tz="UTC")
        asset_intraday = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "timestamp": latest_intraday_ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 100_000,
                    "interval": "5m",
                }
                for symbol in DEFAULT_ETF_SYMBOLS
            ],
        )
        manifest_rows: list[dict[str, object]] = []
        for symbol in DEFAULT_ETF_SYMBOLS:
            daily_count = int((asset_daily["symbol"].astype(str) == symbol).sum())
            manifest_rows.append(
                {
                    "symbol": symbol,
                    "dataset_kind": "daily",
                    "row_count": daily_count,
                    "status": "ok",
                    "start_at": asset_daily.loc[asset_daily["symbol"] == symbol, "trade_date"].min(),
                    "end_at": asset_daily.loc[asset_daily["symbol"] == symbol, "trade_date"].max(),
                    "detail": "",
                },
            )
            manifest_rows.append(
                {
                    "symbol": symbol,
                    "dataset_kind": "intraday_5m",
                    "row_count": 1,
                    "status": "ok",
                    "start_at": latest_intraday_ts,
                    "end_at": latest_intraday_ts,
                    "detail": "",
                },
            )
        asset_market_manifest = pd.DataFrame(manifest_rows)
        return sp500, spy, asset_daily, asset_intraday, asset_market_manifest

    def _fake_services(self, feature_rows: pd.DataFrame):
        posts = self._posts()
        source_manifest = self._source_manifest(posts)
        tracked_accounts, ranking_history = self._tracked_accounts()
        asset_post_mappings = self._asset_post_mappings(feature_rows)
        sp500, spy, asset_daily, asset_intraday, asset_market_manifest = self._market_frames(feature_rows)
        return {
            "ingestion_service": _FakeIngestionService(posts, source_manifest),
            "market_service": _FakeMarketService(sp500, spy, asset_daily, asset_intraday, asset_market_manifest),
            "discovery_service": _FakeDiscoveryService(tracked_accounts, ranking_history),
            "feature_service": _FakeFeatureService(posts, asset_post_mappings, feature_rows),
        }

    def _refresh_with_fakes(self, feature_rows: pd.DataFrame, refresh_mode: str = "full") -> None:
        services = self._fake_services(feature_rows)
        refresh_datasets(
            settings=self.settings,
            store=self.store,
            ingestion_service=services["ingestion_service"],
            market_service=services["market_service"],
            discovery_service=services["discovery_service"],
            feature_service=services["feature_service"],
            health_service=self.health_service,
            remote_url="",
            uploaded_files=[],
            incremental=refresh_mode == "incremental",
            refresh_mode=refresh_mode,
        )

    def _save_joint_portfolio_run(self, feature_rows: pd.DataFrame) -> BacktestRun:
        config = PortfolioRunConfig(
            run_name="integration-joint",
            allocator_mode="joint_model",
            fallback_mode="SPY",
            transaction_cost_bps=2.0,
            universe_symbols=("SPY", "QQQ"),
            selected_symbols=("SPY", "QQQ"),
            llm_enabled=False,
            feature_version="asset-v1",
            train_window=16,
            validation_window=8,
            test_window=8,
            step_size=8,
            threshold_grid=(0.0,),
            minimum_signal_grid=(1,),
            account_weight_grid=(1.0,),
            model_families=("ridge",),
            topology_variants=("per_asset", "pooled"),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
            run, artifacts = self.backtests.run_joint_model_allocator(config, feature_rows)
        self.experiments.save_portfolio_run(
            run=run,
            config=artifacts["config"],
            trades=artifacts["trades"],
            decision_history=artifacts["predictions"],
            candidate_predictions=artifacts["candidate_predictions"],
            component_summary=artifacts["windows"],
            benchmarks=artifacts["benchmarks"],
            benchmark_curves=artifacts["benchmark_curves"],
            diagnostics=artifacts["diagnostics"],
            leakage_audit=artifacts["leakage_audit"],
            variant_summary=artifacts["variant_summary"],
            portfolio_model_bundle=artifacts["portfolio_model_bundle"],
        )
        return run

    def _save_simple_portfolio_run(self) -> BacktestRun:
        run = BacktestRun(
            run_id="scheduler-portfolio",
            run_name="Scheduler portfolio",
            target_asset="PORTFOLIO",
            config_hash="scheduler-portfolio",
            train_window=16,
            validation_window=8,
            test_window=8,
            metrics={"robust_score": 1.0, "total_return": 0.1, "max_drawdown": -0.01},
            selected_params={
                "fallback_mode": "SPY",
                "transaction_cost_bps": 2.0,
                "selected_symbols": ["SPY", "QQQ"],
                "deployment_variant": "per_asset",
            },
            run_type="portfolio_allocator",
            allocator_mode="joint_model",
            fallback_mode="SPY",
            deployment_variant="per_asset",
            universe_symbols=["SPY", "QQQ"],
            selected_symbols=["SPY", "QQQ"],
            topology_variants=["per_asset"],
            model_families=["ridge"],
        )
        model_payload = {
            "model_family": "ridge",
            "feature_names": ["sentiment_avg"],
            "intercept": 0.0,
            "coefficients": [0.01],
            "means": [0.0],
            "stds": [1.0],
            "residual_std": 0.1,
            "train_rows": 10,
            "explanation_kind": "linear_exact",
            "feature_importances": [0.01],
        }
        self.experiments.save_portfolio_run(
            run=run,
            config={
                "allocator_mode": "joint_model",
                "feature_version": "asset-v1",
                "llm_enabled": False,
                "selected_symbols": ["SPY", "QQQ"],
            },
            trades=pd.DataFrame(),
            decision_history=pd.DataFrame(),
            candidate_predictions=pd.DataFrame(),
            component_summary=pd.DataFrame(),
            benchmarks=pd.DataFrame(),
            benchmark_curves=pd.DataFrame(),
            diagnostics=pd.DataFrame(),
            leakage_audit={"overall_pass": True},
            variant_summary=pd.DataFrame(
                [{"variant_name": "per_asset", "deployment_winner": True, "validation_robust_score": 1.0}],
            ),
            portfolio_model_bundle={
                "deployment_variant": "per_asset",
                "selected_symbols": ["SPY", "QQQ"],
                "fallback_mode": "SPY",
                "feature_version": "asset-v1",
                "llm_enabled": False,
                "variants": {
                    "per_asset": {
                        "variant_name": "per_asset",
                        "topology": "per_asset",
                        "model_family": "ridge",
                        "threshold": 0.0,
                        "min_post_count": 1,
                        "account_weight": 1.0,
                        "selected_symbols": ["SPY", "QQQ"],
                        "feature_version": "asset-v1",
                        "llm_enabled": False,
                        "models": {
                            "SPY": {**model_payload, "model_version": "scheduler-spy", "metadata": {"target_asset": "SPY"}},
                            "QQQ": {**model_payload, "model_version": "scheduler-qqq", "metadata": {"target_asset": "QQQ"}},
                        },
                    },
                },
            },
        )
        return run

    def _live_config_for_run(self, run: BacktestRun) -> LiveMonitorConfig:
        config = LiveMonitorConfig(
            mode="portfolio_run",
            fallback_mode=run.fallback_mode or "SPY",
            portfolio_run_id=run.run_id,
            portfolio_run_name=run.run_name,
            deployment_variant=run.deployment_variant,
        )
        self.experiments.save_live_monitor_config(config)
        return config

    def test_refresh_joint_live_and_paper_workflow_persists_cross_feature_state(self) -> None:
        feature_rows = self._joint_feature_rows(["SPY", "QQQ"])
        self._refresh_with_fakes(feature_rows)

        for dataset_name in [
            "normalized_posts",
            "source_manifests",
            "sp500_daily",
            "spy_daily",
            "asset_universe",
            "asset_daily",
            "asset_intraday",
            "asset_post_mappings",
            "asset_session_features",
            "refresh_history",
            "data_health_latest",
            "data_health_history",
        ]:
            self.assertFalse(self.store.read_frame(dataset_name).empty, dataset_name)
        refresh_history = ensure_refresh_history_frame(self.store.read_frame("refresh_history"))
        self.assertEqual(refresh_history.iloc[-1]["status"], "success")

        run = self._save_joint_portfolio_run(feature_rows)
        live_config = self._live_config_for_run(run)
        latest_cutoff = pd.to_datetime(feature_rows["next_session_open_ts"], errors="coerce", utc=True).max()
        generated_at = latest_cutoff - pd.Timedelta(minutes=30)
        board, decision, explanation_lookup, warnings_list = build_live_portfolio_run_state(
            store=self.store,
            model_service=self.model_service,
            experiment_store=self.experiments,
            config=live_config,
            generated_at=generated_at,
        )

        self.assertFalse(warnings_list)
        self.assertFalse(board.empty)
        self.assertFalse(decision.empty)
        self.assertEqual(decision.iloc[0]["deployment_variant"], run.deployment_variant)
        self.assertEqual(set(board["asset_symbol"].astype(str)), {"SPY", "QQQ"})
        self.assertFalse(explanation_lookup[str(decision.iloc[0]["winning_asset"])].get("feature_contributions").empty)

        _, expected_decision = rank_portfolio_candidates(board, fallback_mode=live_config.fallback_mode, require_tradeable=False)
        self.assertEqual(str(decision.iloc[0]["winning_asset"]), str(expected_decision.iloc[0]["winning_asset"]))
        self.assertAlmostEqual(float(decision.iloc[0]["winner_score"]), float(expected_decision.iloc[0]["winner_score"]))

        self.experiments.save_live_asset_snapshots(board)
        self.experiments.save_live_decision_snapshots(decision)
        paper_config = self.paper_service.upsert_current_for_live_config(
            live_config=live_config,
            portfolio_run_name=run.run_name,
            transaction_cost_bps=2.0,
            starting_cash=100000.0,
            enabled=True,
            now=generated_at,
        )
        self.paper_service.process_live_history(paper_config, as_of=latest_cutoff + pd.Timedelta(hours=8))

        journal = ensure_paper_decision_journal_frame(self.store.read_frame("paper_decision_journal"))
        trades = ensure_paper_trade_ledger_frame(self.store.read_frame("paper_trade_ledger"))
        equity = ensure_paper_equity_curve_frame(self.store.read_frame("paper_equity_curve"))
        benchmark = ensure_paper_benchmark_curve_frame(self.store.read_frame("paper_benchmark_curve"))

        portfolio_journal = journal.loc[journal["paper_portfolio_id"].astype(str) == paper_config.paper_portfolio_id]
        portfolio_trades = trades.loc[trades["paper_portfolio_id"].astype(str) == paper_config.paper_portfolio_id]
        self.assertEqual(len(portfolio_journal), 1)
        self.assertEqual(len(portfolio_trades), 1)
        self.assertEqual(len(equity.loc[equity["paper_portfolio_id"].astype(str) == paper_config.paper_portfolio_id]), 1)
        self.assertEqual(len(benchmark.loc[benchmark["paper_portfolio_id"].astype(str) == paper_config.paper_portfolio_id]), 1)

        trade = portfolio_trades.iloc[0]
        asset_daily = self.store.read_frame("asset_daily")
        expected_price = asset_daily.loc[
            (asset_daily["symbol"].astype(str) == str(trade["asset_symbol"]))
            & (pd.to_datetime(asset_daily["trade_date"]) == pd.Timestamp(trade["next_session_date"]).tz_localize(None)),
        ].iloc[0]
        self.assertAlmostEqual(float(trade["next_session_open"]), float(expected_price["open"]))
        self.assertAlmostEqual(float(trade["next_session_close"]), float(expected_price["close"]))

        archived_id = paper_config.paper_portfolio_id
        self.paper_service.upsert_current_for_live_config(
            live_config=LiveMonitorConfig(
                mode="portfolio_run",
                fallback_mode="SPY",
                portfolio_run_id="replacement-portfolio",
                portfolio_run_name="Replacement",
                deployment_variant="per_asset",
            ),
            portfolio_run_name="Replacement",
            transaction_cost_bps=2.0,
            starting_cash=125000.0,
            enabled=True,
            now=latest_cutoff + pd.Timedelta(hours=9),
        )
        registry = ensure_paper_portfolio_registry_frame(self.store.read_frame("paper_portfolio_registry"))
        archived_row = registry.loc[registry["paper_portfolio_id"].astype(str) == archived_id].iloc[0]
        self.assertTrue(pd.notna(archived_row["archived_at"]))
        updated_trades = ensure_paper_trade_ledger_frame(self.store.read_frame("paper_trade_ledger"))
        self.assertFalse(updated_trades.loc[updated_trades["paper_portfolio_id"].astype(str) == archived_id].empty)

    def test_scheduler_cycle_uses_injected_services_and_processes_paper_history(self) -> None:
        feature_rows = self._joint_feature_rows(["SPY", "QQQ"])
        self._refresh_with_fakes(feature_rows)
        run = self._save_simple_portfolio_run()
        live_config = self._live_config_for_run(run)

        cutoff = pd.to_datetime(feature_rows["next_session_open_ts"], errors="coerce", utc=True).max()
        pre_open_generated_at = cutoff - pd.Timedelta(minutes=30)
        board, _, _, warnings_list = build_live_portfolio_run_state(
            store=self.store,
            model_service=self.model_service,
            experiment_store=self.experiments,
            config=live_config,
            generated_at=pre_open_generated_at,
        )
        self.assertFalse(warnings_list)
        self.experiments.save_live_asset_snapshots(board)
        paper_config = self.paper_service.upsert_current_for_live_config(
            live_config=live_config,
            portfolio_run_name=run.run_name,
            transaction_cost_bps=2.0,
            starting_cash=100000.0,
            enabled=True,
            now=pre_open_generated_at,
        )

        services = self._fake_services(feature_rows)
        decision = run_scheduler_cycle(
            settings=self.settings,
            store=self.store,
            decision=SchedulerDecision(refresh_mode="full", incremental=False),
            ingestion_service=services["ingestion_service"],
            market_service=services["market_service"],
            discovery_service=services["discovery_service"],
            feature_service=services["feature_service"],
            health_service=self.health_service,
            experiment_store=self.experiments,
            model_service=self.model_service,
            paper_service=self.paper_service,
            generated_at=cutoff + pd.Timedelta(hours=8),
        )

        self.assertTrue(decision.should_run)
        self.assertGreaterEqual(len(self.store.read_frame("live_asset_snapshots")), len(board) * 2)
        self.assertFalse(self.store.read_frame("live_decision_snapshots").empty)
        journal = ensure_paper_decision_journal_frame(self.store.read_frame("paper_decision_journal"))
        trades = ensure_paper_trade_ledger_frame(self.store.read_frame("paper_trade_ledger"))
        portfolio_journal = journal.loc[journal["paper_portfolio_id"].astype(str) == paper_config.paper_portfolio_id]
        portfolio_trades = trades.loc[trades["paper_portfolio_id"].astype(str) == paper_config.paper_portfolio_id]
        self.assertEqual(len(portfolio_journal), 1)
        self.assertEqual(str(portfolio_journal.iloc[0]["settlement_status"]), "settled")
        self.assertEqual(len(portfolio_trades), 1)
        self.assertEqual(str(portfolio_trades.iloc[0]["asset_symbol"]), "QQQ")


if __name__ == "__main__":
    unittest.main()
