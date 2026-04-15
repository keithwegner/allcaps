from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.backtesting import BacktestService
from trump_workbench.config import AppSettings
from trump_workbench.contracts import BacktestRun, LiveMonitorConfig, LiveMonitorPinnedRun, ModelRunConfig
from trump_workbench.discovery import DiscoveryService
from trump_workbench.enrichment import LLMEnrichmentService
from trump_workbench.experiments import ExperimentStore
from trump_workbench.explanations import build_account_attribution, build_post_attribution
from trump_workbench.features import FeatureService
from trump_workbench.ingestion import XCsvAdapter
from trump_workbench.live_monitor import rank_live_asset_snapshots, seed_live_monitor_config, validate_live_monitor_config
from trump_workbench.modeling import ModelService
from trump_workbench.storage import DuckDBStore
from trump_workbench.ui import _build_live_monitor_state, _build_live_runner_up_frame


class LiveMonitorLogicTests(unittest.TestCase):
    def test_seed_live_monitor_config_uses_latest_run_per_asset(self) -> None:
        runs = pd.DataFrame(
            [
                {
                    "run_id": "qqq-new",
                    "run_name": "QQQ newer",
                    "target_asset": "QQQ",
                    "created_at": pd.Timestamp("2026-04-14 10:00:00"),
                },
                {
                    "run_id": "spy-run",
                    "run_name": "SPY baseline",
                    "target_asset": "SPY",
                    "created_at": pd.Timestamp("2026-04-14 09:00:00"),
                },
                {
                    "run_id": "qqq-old",
                    "run_name": "QQQ older",
                    "target_asset": "QQQ",
                    "created_at": pd.Timestamp("2026-04-13 09:00:00"),
                },
            ],
        )

        config = seed_live_monitor_config(runs)

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.mode, "asset_model_set")
        self.assertEqual(config.pinned_runs[0].asset_symbol, "SPY")
        self.assertEqual(config.pinned_runs[1].run_id, "qqq-new")

    def test_validate_live_monitor_config_rejects_missing_spy_duplicate_assets_and_mismatch(self) -> None:
        runs = pd.DataFrame(
            [
                {"run_id": "spy-run", "run_name": "SPY", "target_asset": "SPY"},
                {"run_id": "qqq-run", "run_name": "QQQ", "target_asset": "QQQ"},
            ],
        )
        bad_config = LiveMonitorConfig(
            mode="asset_model_set",
            fallback_mode="INVALID",
            pinned_runs=[
                LiveMonitorPinnedRun(asset_symbol="QQQ", run_id="qqq-run"),
                LiveMonitorPinnedRun(asset_symbol="QQQ", run_id="spy-run"),
            ],
        )

        errors = validate_live_monitor_config(bad_config, runs)

        self.assertTrue(any("Fallback mode" in error for error in errors))
        self.assertTrue(any("Only one pinned run" in error for error in errors))
        self.assertTrue(any("Exactly one pinned `SPY`" in error for error in errors))

    def test_validate_live_monitor_config_requires_joint_portfolio_run(self) -> None:
        runs = pd.DataFrame(
            [
                {
                    "run_id": "portfolio-1",
                    "run_name": "Joint portfolio",
                    "run_type": "portfolio_allocator",
                    "allocator_mode": "joint_model",
                    "target_asset": "PORTFOLIO",
                },
            ],
        )
        config = LiveMonitorConfig(
            mode="portfolio_run",
            fallback_mode="SPY",
            portfolio_run_id="missing",
            deployment_variant="pooled",
        )

        errors = validate_live_monitor_config(config, runs)

        self.assertTrue(any("not available anymore" in error for error in errors))

    def test_rank_live_asset_snapshots_multiple_assets_choose_highest_eligible_score(self) -> None:
        board, decision = rank_live_asset_snapshots(
            pd.DataFrame(
                [
                    {
                        "generated_at": pd.Timestamp("2026-04-14 10:00:00"),
                        "signal_session_date": pd.Timestamp("2026-04-14"),
                        "next_session_date": pd.Timestamp("2026-04-15"),
                        "asset_symbol": "SPY",
                        "run_id": "spy-run",
                        "run_name": "SPY",
                        "feature_version": "v1",
                        "model_version": "spy-model",
                        "expected_return_score": 0.010,
                        "confidence": 0.60,
                        "threshold": 0.001,
                        "min_post_count": 1,
                        "post_count": 5,
                    },
                    {
                        "generated_at": pd.Timestamp("2026-04-14 10:00:00"),
                        "signal_session_date": pd.Timestamp("2026-04-14"),
                        "next_session_date": pd.Timestamp("2026-04-15"),
                        "asset_symbol": "QQQ",
                        "run_id": "qqq-run",
                        "run_name": "QQQ",
                        "feature_version": "asset-v1",
                        "model_version": "qqq-model",
                        "expected_return_score": 0.015,
                        "confidence": 0.55,
                        "threshold": 0.001,
                        "min_post_count": 1,
                        "post_count": 5,
                    },
                ],
            ),
            fallback_mode="SPY",
        )

        self.assertEqual(decision.iloc[0]["winning_asset"], "QQQ")
        self.assertEqual(decision.iloc[0]["decision_source"], "eligible")
        self.assertTrue(bool(board.loc[board["asset_symbol"] == "QQQ", "is_winner"].iloc[0]))

    def test_rank_live_asset_snapshots_only_spy_qualifies(self) -> None:
        board, decision = rank_live_asset_snapshots(
            pd.DataFrame(
                [
                    {
                        "generated_at": pd.Timestamp("2026-04-14 10:00:00"),
                        "signal_session_date": pd.Timestamp("2026-04-14"),
                        "next_session_date": pd.Timestamp("2026-04-15"),
                        "asset_symbol": "SPY",
                        "run_id": "spy-run",
                        "run_name": "SPY",
                        "feature_version": "v1",
                        "model_version": "spy-model",
                        "expected_return_score": 0.004,
                        "confidence": 0.60,
                        "threshold": 0.001,
                        "min_post_count": 1,
                        "post_count": 5,
                    },
                    {
                        "generated_at": pd.Timestamp("2026-04-14 10:00:00"),
                        "signal_session_date": pd.Timestamp("2026-04-14"),
                        "next_session_date": pd.Timestamp("2026-04-15"),
                        "asset_symbol": "QQQ",
                        "run_id": "qqq-run",
                        "run_name": "QQQ",
                        "feature_version": "asset-v1",
                        "model_version": "qqq-model",
                        "expected_return_score": 0.0005,
                        "confidence": 0.55,
                        "threshold": 0.001,
                        "min_post_count": 1,
                        "post_count": 5,
                    },
                ],
            ),
            fallback_mode="SPY",
        )

        self.assertEqual(decision.iloc[0]["winning_asset"], "SPY")
        self.assertEqual(decision.iloc[0]["decision_source"], "eligible")
        self.assertTrue(bool(board.loc[board["asset_symbol"] == "SPY", "qualifies"].iloc[0]))
        self.assertFalse(bool(board.loc[board["asset_symbol"] == "QQQ", "qualifies"].iloc[0]))

    def test_rank_live_asset_snapshots_fallback_spy_when_nothing_qualifies(self) -> None:
        board, decision = rank_live_asset_snapshots(
            pd.DataFrame(
                [
                    {
                        "generated_at": pd.Timestamp("2026-04-14 10:00:00"),
                        "signal_session_date": pd.Timestamp("2026-04-14"),
                        "next_session_date": pd.Timestamp("2026-04-15"),
                        "asset_symbol": "SPY",
                        "run_id": "spy-run",
                        "run_name": "SPY",
                        "feature_version": "v1",
                        "model_version": "spy-model",
                        "expected_return_score": -0.001,
                        "confidence": 0.60,
                        "threshold": 0.005,
                        "min_post_count": 10,
                        "post_count": 1,
                    },
                    {
                        "generated_at": pd.Timestamp("2026-04-14 10:00:00"),
                        "signal_session_date": pd.Timestamp("2026-04-14"),
                        "next_session_date": pd.Timestamp("2026-04-15"),
                        "asset_symbol": "QQQ",
                        "run_id": "qqq-run",
                        "run_name": "QQQ",
                        "feature_version": "asset-v1",
                        "model_version": "qqq-model",
                        "expected_return_score": 0.001,
                        "confidence": 0.55,
                        "threshold": 0.005,
                        "min_post_count": 10,
                        "post_count": 1,
                    },
                ],
            ),
            fallback_mode="SPY",
        )

        self.assertEqual(decision.iloc[0]["winning_asset"], "SPY")
        self.assertEqual(decision.iloc[0]["decision_source"], "fallback")
        self.assertEqual(decision.iloc[0]["stance"], "LONG SPY NEXT SESSION")
        self.assertTrue(bool(board.loc[board["asset_symbol"] == "SPY", "is_winner"].iloc[0]))

    def test_rank_live_asset_snapshots_fallback_flat_when_nothing_qualifies(self) -> None:
        board, decision = rank_live_asset_snapshots(
            pd.DataFrame(
                [
                    {
                        "generated_at": pd.Timestamp("2026-04-14 10:00:00"),
                        "signal_session_date": pd.Timestamp("2026-04-14"),
                        "next_session_date": pd.Timestamp("2026-04-15"),
                        "asset_symbol": "SPY",
                        "run_id": "spy-run",
                        "run_name": "SPY",
                        "feature_version": "v1",
                        "model_version": "spy-model",
                        "expected_return_score": 0.0005,
                        "confidence": 0.60,
                        "threshold": 0.005,
                        "min_post_count": 10,
                        "post_count": 1,
                    },
                ],
            ),
            fallback_mode="FLAT",
        )

        self.assertEqual(decision.iloc[0]["winning_asset"], "")
        self.assertEqual(decision.iloc[0]["decision_source"], "fallback")
        self.assertEqual(decision.iloc[0]["stance"], "FLAT")
        self.assertFalse(bool(board["is_winner"].any()))


class LiveMonitorStateIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(self.settings)
        self.discovery = DiscoveryService()
        self.enrichment = LLMEnrichmentService(self.store)
        self.features = FeatureService(self.enrichment)
        self.model_service = ModelService()
        self.backtests = BacktestService(self.model_service)
        self.experiments = ExperimentStore(self.store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _build_market(self) -> pd.DataFrame:
        dates = pd.bdate_range("2025-02-03", periods=140)
        opens = []
        closes = []
        volumes = []
        price = 100.0
        for idx, _ in enumerate(dates):
            daily_signal = 0.004 if idx % 4 in (0, 1) else -0.002
            open_price = price
            close_price = open_price * (1.0 + daily_signal)
            opens.append(open_price)
            closes.append(close_price)
            volumes.append(1_000_000 + idx * 1_000)
            price = close_price
        return pd.DataFrame(
            {
                "trade_date": dates,
                "open": opens,
                "high": [value * 1.01 for value in closes],
                "low": [value * 0.99 for value in opens],
                "close": closes,
                "volume": volumes,
            },
        )

    def _build_asset_market(self, symbols: list[str]) -> pd.DataFrame:
        base = self._build_market()
        frames: list[pd.DataFrame] = []
        for offset, symbol in enumerate(symbols, start=1):
            frame = base.copy()
            scale = 1.0 + offset * 0.08
            frame["symbol"] = symbol
            frame["open"] = frame["open"] * scale
            frame["high"] = frame["high"] * scale
            frame["low"] = frame["low"] * scale
            frame["close"] = frame["close"] * scale
            frame["volume"] = frame["volume"] + offset * 10_000
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _build_x_csv(self) -> bytes:
        rows = ["timestamp,text,url,is_retweet,author_handle,author_name,author_id,mentions_trump,replies_count,reblogs_count,favourites_count"]
        dates = pd.bdate_range("2025-02-03", periods=140)
        for idx, date in enumerate(dates[:-1]):
            positive = (idx + 1) % 4 in (0, 1)
            text = (
                "Trump says Nasdaq and big tech are leading the market higher."
                if positive
                else "Trump trade pressure is weighing on energy and financials."
            )
            handle = "macroalpha" if positive else "policybeta"
            row = ",".join(
                [
                    f"{date:%Y-%m-%d}T15:30:00-05:00",
                    text,
                    f"https://x.com/{handle}/status/{1000 + idx}",
                    "false",
                    handle,
                    handle.title(),
                    handle,
                    "true",
                    str(10 + idx % 5),
                    str(20 + idx % 7),
                    str(50 + idx % 11),
                ],
            )
            rows.append(row)
        return "\n".join(rows).encode("utf-8")

    def test_build_live_monitor_state_generates_ranked_board_and_asset_specific_explanations(self) -> None:
        adapter = XCsvAdapter(
            settings=self.settings,
            name="synthetic-x",
            provenance="unit-test",
            raw_bytes=self._build_x_csv(),
        )
        posts, _ = adapter.fetch_history()
        tracked, _ = self.discovery.refresh_accounts(posts, pd.DataFrame(), as_of=posts["post_timestamp"].max(), auto_include_top_n=2)
        spy_market = self._build_market()
        asset_market = self._build_asset_market(["QQQ"])
        asset_universe = pd.DataFrame(
            [
                {
                    "symbol": "QQQ",
                    "display_name": "Invesco QQQ Trust",
                    "asset_type": "etf",
                    "source": "core_etf",
                },
            ],
        )
        self.store.save_frame("asset_universe", asset_universe, metadata={"row_count": 1})
        self.store.save_frame("asset_daily", asset_market, metadata={"row_count": int(len(asset_market))})

        prepared_posts = self.features.prepare_session_posts(posts, spy_market, tracked, llm_enabled=True)
        spy_dataset = self.features.build_session_dataset(
            posts=posts,
            spy_market=spy_market,
            tracked_accounts=tracked,
            feature_version="v1",
            llm_enabled=True,
            prepared_posts=prepared_posts,
        )
        spy_dataset["target_asset"] = "SPY"
        asset_post_mappings = self.features.build_asset_post_mappings(prepared_posts, asset_universe, llm_enabled=True)
        qqq_dataset = self.features.build_asset_session_dataset(
            asset_post_mappings=asset_post_mappings,
            asset_market=asset_market,
            feature_version="asset-v1",
            llm_enabled=True,
            asset_universe=asset_universe,
        )
        qqq_dataset = qqq_dataset.loc[qqq_dataset["asset_symbol"] == "QQQ"].copy()
        qqq_dataset["target_asset"] = "QQQ"

        spy_config = ModelRunConfig(
            run_name="spy-live-run",
            target_asset="SPY",
            llm_enabled=True,
            train_window=60,
            validation_window=20,
            test_window=20,
            step_size=20,
            threshold_grid=(0.0, 0.001),
            minimum_signal_grid=(1, 2),
            account_weight_grid=(0.5, 1.0),
        )
        spy_run, spy_artifacts = self.backtests.run_walk_forward(spy_config, spy_dataset)
        spy_saved_run = BacktestRun(
            run_id=spy_run.run_id,
            run_name=spy_run.run_name,
            target_asset="SPY",
            config_hash=spy_run.config_hash,
            train_window=spy_run.train_window,
            validation_window=spy_run.validation_window,
            test_window=spy_run.test_window,
            metrics=spy_run.metrics,
            selected_params={"threshold": 0.50, "min_post_count": 999, "account_weight": 1.0},
        )
        self.experiments.save_run(
            run=spy_saved_run,
            config=spy_artifacts["config"],
            trades=spy_artifacts["trades"],
            predictions=spy_artifacts["predictions"],
            windows=spy_artifacts["windows"],
            importance=spy_artifacts["importance"],
            model_artifact=spy_artifacts["model_artifact"],
            feature_contributions=spy_artifacts["feature_contributions"],
            post_attribution=build_post_attribution(prepared_posts),
            account_attribution=build_account_attribution(build_post_attribution(prepared_posts)),
            benchmarks=spy_artifacts["benchmarks"],
            diagnostics=spy_artifacts["diagnostics"],
            benchmark_curves=spy_artifacts["benchmark_curves"],
            leakage_audit=spy_artifacts["leakage_audit"],
        )

        qqq_config = ModelRunConfig(
            run_name="qqq-live-run",
            target_asset="QQQ",
            feature_version="asset-v1",
            llm_enabled=True,
            train_window=60,
            validation_window=20,
            test_window=20,
            step_size=20,
            threshold_grid=(0.0, 0.001),
            minimum_signal_grid=(1, 2),
            account_weight_grid=(0.5, 1.0),
        )
        qqq_run, qqq_artifacts = self.backtests.run_walk_forward(qqq_config, qqq_dataset)
        qqq_saved_run = BacktestRun(
            run_id=qqq_run.run_id,
            run_name=qqq_run.run_name,
            target_asset="QQQ",
            config_hash=qqq_run.config_hash,
            train_window=qqq_run.train_window,
            validation_window=qqq_run.validation_window,
            test_window=qqq_run.test_window,
            metrics=qqq_run.metrics,
            selected_params={"threshold": -1.0, "min_post_count": 0, "account_weight": 1.0},
        )
        qqq_post_attribution = build_post_attribution(asset_post_mappings.loc[asset_post_mappings["asset_symbol"] == "QQQ"].copy())
        self.experiments.save_run(
            run=qqq_saved_run,
            config=qqq_artifacts["config"],
            trades=qqq_artifacts["trades"],
            predictions=qqq_artifacts["predictions"],
            windows=qqq_artifacts["windows"],
            importance=qqq_artifacts["importance"],
            model_artifact=qqq_artifacts["model_artifact"],
            feature_contributions=qqq_artifacts["feature_contributions"],
            post_attribution=qqq_post_attribution,
            account_attribution=build_account_attribution(qqq_post_attribution),
            benchmarks=qqq_artifacts["benchmarks"],
            diagnostics=qqq_artifacts["diagnostics"],
            benchmark_curves=qqq_artifacts["benchmark_curves"],
            leakage_audit=qqq_artifacts["leakage_audit"],
        )

        live_config = LiveMonitorConfig(
            mode="asset_model_set",
            fallback_mode="SPY",
            pinned_runs=[
                LiveMonitorPinnedRun(asset_symbol="SPY", run_id=spy_saved_run.run_id, run_name=spy_saved_run.run_name),
                LiveMonitorPinnedRun(asset_symbol="QQQ", run_id=qqq_saved_run.run_id, run_name=qqq_saved_run.run_name),
            ],
        )
        board, decision, explanation_lookup, warnings = _build_live_monitor_state(
            store=self.store,
            feature_service=self.features,
            model_service=self.model_service,
            experiment_store=self.experiments,
            posts=posts,
            spy_market=spy_market,
            tracked_accounts=tracked,
            config=live_config,
            generated_at=pd.Timestamp("2026-04-14 10:00:00"),
        )

        self.assertFalse(warnings)
        self.assertSetEqual(set(board["asset_symbol"].astype(str)), {"SPY", "QQQ"})
        self.assertEqual(decision.iloc[0]["winning_asset"], "QQQ")
        self.assertEqual(decision.iloc[0]["decision_source"], "eligible")
        self.assertFalse(bool(board.loc[board["asset_symbol"] == "SPY", "qualifies"].iloc[0]))
        self.assertTrue(bool(board.loc[board["asset_symbol"] == "QQQ", "qualifies"].iloc[0]))
        self.assertIn("QQQ", explanation_lookup)
        self.assertEqual(str(explanation_lookup["QQQ"]["prediction_row"]["target_asset"]), "QQQ")
        self.assertFalse(explanation_lookup["QQQ"]["post_attribution"].empty)

        runner_frame = _build_live_runner_up_frame(board, decision.iloc[0])
        self.assertEqual(len(runner_frame), 2)
        self.assertIn("threshold_gap", runner_frame.columns.tolist())

    def test_build_live_monitor_state_from_portfolio_run_uses_saved_models(self) -> None:
        asset_session_features = pd.DataFrame(
            [
                {
                    "signal_session_date": pd.Timestamp("2025-05-01"),
                    "next_session_date": pd.Timestamp("2025-05-02"),
                    "asset_symbol": "SPY",
                    "feature_version": "asset-v1",
                    "llm_enabled": False,
                    "target_available": False,
                    "tradeable": True,
                    "next_session_open": 100.0,
                    "next_session_close": 101.0,
                    "post_count": 3,
                    "tracked_weighted_mentions": 1.0,
                    "tracked_weighted_engagement": 2.0,
                    "tracked_account_post_count": 1,
                    "sentiment_avg": 0.1,
                },
                {
                    "signal_session_date": pd.Timestamp("2025-05-01"),
                    "next_session_date": pd.Timestamp("2025-05-02"),
                    "asset_symbol": "QQQ",
                    "feature_version": "asset-v1",
                    "llm_enabled": False,
                    "target_available": False,
                    "tradeable": True,
                    "next_session_open": 200.0,
                    "next_session_close": 203.0,
                    "post_count": 6,
                    "tracked_weighted_mentions": 2.0,
                    "tracked_weighted_engagement": 4.0,
                    "tracked_account_post_count": 2,
                    "sentiment_avg": 0.6,
                },
            ],
        )
        asset_post_mappings = pd.DataFrame(
            [
                {
                    "asset_symbol": "SPY",
                    "session_date": pd.Timestamp("2025-05-01"),
                    "post_timestamp": pd.Timestamp("2025-05-01 10:00:00"),
                    "author_account_id": "acct-spy",
                    "author_handle": "macrospy",
                    "author_display_name": "Macro Spy",
                    "source_platform": "X",
                    "author_is_trump": False,
                    "is_active_tracked_account": True,
                    "tracked_account_status": "active",
                    "mentions_trump": True,
                    "sentiment_score": 0.1,
                    "engagement_score": 10.0,
                    "tracked_discovery_score": 1.0,
                    "cleaned_text": "Trump broad market chatter",
                    "post_url": "https://example.com/spy",
                },
                {
                    "asset_symbol": "QQQ",
                    "session_date": pd.Timestamp("2025-05-01"),
                    "post_timestamp": pd.Timestamp("2025-05-01 10:05:00"),
                    "author_account_id": "acct-qqq",
                    "author_handle": "macroqqq",
                    "author_display_name": "Macro QQQ",
                    "source_platform": "X",
                    "author_is_trump": False,
                    "is_active_tracked_account": True,
                    "tracked_account_status": "active",
                    "mentions_trump": True,
                    "sentiment_score": 0.6,
                    "engagement_score": 25.0,
                    "tracked_discovery_score": 1.5,
                    "cleaned_text": "Trump tech momentum chatter",
                    "post_url": "https://example.com/qqq",
                },
            ],
        )
        self.store.save_frame("asset_session_features", asset_session_features, metadata={"row_count": len(asset_session_features)})
        self.store.save_frame("asset_post_mappings", asset_post_mappings, metadata={"row_count": len(asset_post_mappings)})

        portfolio_run = BacktestRun(
            run_id="portfolio-live",
            run_name="Portfolio live",
            target_asset="PORTFOLIO",
            config_hash="portfolio-live",
            train_window=20,
            validation_window=10,
            test_window=10,
            metrics={"robust_score": 1.0},
            selected_params={
                "fallback_mode": "SPY",
                "deployment_variant": "per_asset",
                "selected_symbols": ["SPY", "QQQ"],
            },
            run_type="portfolio_allocator",
            allocator_mode="joint_model",
            fallback_mode="SPY",
            deployment_variant="per_asset",
            selected_symbols=["SPY", "QQQ"],
            universe_symbols=["SPY", "QQQ"],
            topology_variants=["per_asset", "pooled"],
            model_families=["ridge"],
        )
        self.experiments.save_portfolio_run(
            run=portfolio_run,
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
                {
                    "variant_name": ["per_asset", "pooled"],
                    "validation_robust_score": [1.2, 1.1],
                    "deployment_winner": [True, False],
                },
            ),
            portfolio_model_bundle={
                "deployment_variant": "per_asset",
                "selected_symbols": ["SPY", "QQQ"],
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
                        "models": {
                            "SPY": {
                                "model_version": "spy-live-model",
                                "feature_names": ["post_count", "sentiment_avg"],
                                "intercept": -0.01,
                                "coefficients": [0.001, 0.002],
                                "means": [0.0, 0.0],
                                "stds": [1.0, 1.0],
                                "residual_std": 0.1,
                                "train_rows": 10,
                                "metadata": {"target_asset": "SPY"},
                            },
                            "QQQ": {
                                "model_version": "qqq-live-model",
                                "feature_names": ["post_count", "sentiment_avg"],
                                "intercept": -0.01,
                                "coefficients": [0.002, 0.01],
                                "means": [0.0, 0.0],
                                "stds": [1.0, 1.0],
                                "residual_std": 0.1,
                                "train_rows": 10,
                                "metadata": {"target_asset": "QQQ"},
                            },
                        },
                    },
                },
            },
        )

        live_config = LiveMonitorConfig(
            mode="portfolio_run",
            fallback_mode="SPY",
            portfolio_run_id="portfolio-live",
            portfolio_run_name="Portfolio live",
            deployment_variant="per_asset",
        )
        board, decision, explanation_lookup, warnings = _build_live_monitor_state(
            store=self.store,
            feature_service=self.features,
            model_service=self.model_service,
            experiment_store=self.experiments,
            posts=pd.DataFrame(),
            spy_market=pd.DataFrame(),
            tracked_accounts=pd.DataFrame(),
            config=live_config,
            generated_at=pd.Timestamp("2026-04-14 10:00:00"),
        )

        self.assertFalse(warnings)
        self.assertEqual(decision.iloc[0]["winning_asset"], "QQQ")
        self.assertEqual(decision.iloc[0]["deployment_variant"], "per_asset")
        self.assertSetEqual(set(board["asset_symbol"].astype(str)), {"SPY", "QQQ"})
        self.assertIn("QQQ", explanation_lookup)
        self.assertFalse(explanation_lookup["QQQ"]["feature_contributions"].empty)
        self.assertFalse(explanation_lookup["QQQ"]["post_attribution"].empty)


if __name__ == "__main__":
    unittest.main()
