from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from trump_workbench.config import APP_TITLE, AppSettings, CURRENT_TERM_START, DEFAULT_ETF_SYMBOLS, EASTERN
from trump_workbench.contracts import (
    BacktestRun,
    LinearModelArtifact,
    LiveMonitorConfig,
    LiveMonitorPinnedRun,
    ModelRunConfig,
    NormalizedPost,
    PortfolioRunConfig,
    PredictionSnapshot,
    SessionFeatureRow,
    TrackedAccount,
)


class ConfigAndContractsTests(unittest.TestCase):
    def test_app_settings_creates_expected_directories_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = AppSettings(base_dir=Path(temp_dir))

            self.assertEqual(settings.title, APP_TITLE)
            self.assertEqual(settings.timezone, EASTERN)
            self.assertEqual(settings.term_start, CURRENT_TERM_START)
            self.assertEqual(settings.code_root, settings.base_dir)
            self.assertEqual(settings.state_root, settings.base_dir)
            self.assertTrue(settings.cache_dir.exists())
            self.assertTrue(settings.workbench_dir.exists())
            self.assertTrue(settings.lake_dir.exists())
            self.assertTrue(settings.artifact_dir.exists())
            self.assertEqual(settings.db_path, settings.workbench_dir / "workbench.duckdb")
            self.assertEqual(settings.truth_cache_file, settings.cache_dir / "truth_archive.csv")
            self.assertEqual(settings.local_x_path, settings.base_dir / "data" / "realDonaldTrump_x_current_term.csv")
            self.assertEqual(settings.local_mentions_path, settings.base_dir / "data" / "influential_x_mentions.csv")
            self.assertEqual(settings.x_template_path, settings.base_dir / "templates" / "x_posts_template.csv")
            self.assertEqual(settings.mention_template_path, settings.base_dir / "templates" / "x_mentions_template.csv")
            self.assertEqual(DEFAULT_ETF_SYMBOLS, ("SPY", "QQQ", "XLK", "XLF", "XLE", "SMH"))
            self.assertFalse(settings.public_mode)
            self.assertTrue(settings.auto_bootstrap_on_start)

    def test_app_settings_supports_state_root_and_hosting_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as code_dir, tempfile.TemporaryDirectory() as state_dir:
            env = {
                "ALLCAPS_STATE_DIR": state_dir,
                "ALLCAPS_PUBLIC_MODE": "true",
                "ALLCAPS_ADMIN_PASSWORD": "secret",
                "ALLCAPS_AUTO_BOOTSTRAP_ON_START": "false",
                "ALLCAPS_SCHEDULER_ENABLED": "true",
                "ALLCAPS_SCHEDULER_INCREMENTAL_MINUTES": "45",
                "ALLCAPS_SCHEDULER_FULL_HOUR": "4",
                "ALLCAPS_SCHEDULER_FULL_MINUTE": "15",
                "ALLCAPS_SCHEDULER_LOOP_SECONDS": "90",
                "ALLCAPS_REMOTE_X_CSV_URL": "https://example.com/mentions.csv",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                settings = AppSettings(base_dir=Path(code_dir))

            self.assertEqual(settings.code_root, Path(code_dir))
            self.assertEqual(settings.state_root, Path(state_dir).resolve())
            self.assertEqual(settings.cache_dir, Path(state_dir).resolve() / ".cache")
            self.assertEqual(settings.db_path, Path(state_dir).resolve() / ".workbench" / "workbench.duckdb")
            self.assertEqual(settings.local_x_path, Path(code_dir) / "data" / "realDonaldTrump_x_current_term.csv")
            self.assertTrue(settings.public_mode)
            self.assertEqual(settings.admin_password, "secret")
            self.assertFalse(settings.auto_bootstrap_on_start)
            self.assertTrue(settings.scheduler_enabled)
            self.assertEqual(settings.scheduler_incremental_minutes, 45)
            self.assertEqual(settings.scheduler_full_hour, 4)
            self.assertEqual(settings.scheduler_full_minute, 15)
            self.assertEqual(settings.scheduler_loop_seconds, 90)
            self.assertEqual(settings.remote_x_csv_url, "https://example.com/mentions.csv")

    def test_contracts_round_trip_to_dict_helpers(self) -> None:
        post = NormalizedPost(
            source_platform="X",
            source_type="x_csv",
            author_account_id="acct-a",
            author_handle="macroa",
            author_display_name="Macro A",
            author_is_trump=False,
            post_id="1",
            post_url="https://x.com/macroa/status/1",
            post_timestamp=pd.Timestamp("2025-02-03 10:00:00", tz="America/New_York"),
            raw_text="Trump growth",
            cleaned_text="Trump growth",
            is_reshare=False,
            has_media=False,
            replies_count=1,
            reblogs_count=2,
            favourites_count=3,
            mentions_trump=True,
            source_provenance="unit-test",
            engagement_score=6.0,
        )
        tracked = TrackedAccount(
            version_id="v1",
            account_id="acct-a",
            handle="macroa",
            display_name="Macro A",
            source_platform="X",
            discovery_score=1.5,
            status="active",
            first_seen_at=pd.Timestamp("2025-02-01"),
            last_seen_at=pd.Timestamp("2025-02-03"),
            effective_from=pd.Timestamp("2025-02-03"),
            effective_to=None,
            auto_included=True,
            provenance="discovery_auto_include",
            mention_count=3,
            engagement_mean=20.0,
            active_days=2,
        )
        feature_row = SessionFeatureRow(
            trade_date=pd.Timestamp("2025-02-03"),
            feature_version="v1",
            has_posts=True,
            post_count=2,
            trump_post_count=1,
            x_post_count=1,
            tracked_account_post_count=1,
            mention_post_count=1,
            target_next_session_return=0.01,
        )
        config = ModelRunConfig(run_name="test")
        portfolio_config = PortfolioRunConfig(
            run_name="portfolio",
            allocator_mode="joint_model",
            component_run_ids=("spy-run", "qqq-run"),
            universe_symbols=("SPY", "QQQ"),
            selected_symbols=("SPY", "QQQ"),
            model_families=("ridge", "hist_gradient_boosting_regressor"),
            topology_variants=("per_asset", "pooled"),
            deployment_variant="pooled",
        )
        snapshot = PredictionSnapshot(
            signal_session_date=pd.Timestamp("2025-02-03"),
            next_session_date=pd.Timestamp("2025-02-04"),
            target_asset="SPY",
            expected_return_score=0.01,
            feature_version="v1",
            model_version="linear-v1",
            confidence=0.6,
            generated_at=pd.Timestamp("2025-02-03 16:00:00"),
            stance="long",
        )
        backtest_run = BacktestRun(
            run_id="run-1",
            run_name="test",
            target_asset="SPY",
            config_hash="hash",
            train_window=60,
            validation_window=20,
            test_window=20,
            metrics={"robust_score": 1.0},
            selected_params={"threshold": 0.001},
        )
        artifact = LinearModelArtifact(
            model_version="linear-v1",
            feature_names=["x1"],
            intercept=0.1,
            coefficients=[0.2],
            means=[0.0],
            stds=[1.0],
            residual_std=0.3,
            train_rows=10,
            metadata={"source": "unit-test"},
        )
        live_config = LiveMonitorConfig(
            mode="portfolio_run",
            fallback_mode="FLAT",
            portfolio_run_id="portfolio-1",
            portfolio_run_name="Portfolio baseline",
            deployment_variant="pooled",
            pinned_runs=[
                LiveMonitorPinnedRun(
                    asset_symbol="SPY",
                    run_id="run-1",
                    run_name="SPY baseline",
                    model_version="linear-v1",
                    pinned_at="2026-04-14T01:02:03",
                ),
            ],
        )

        self.assertEqual(post.to_dict()["post_id"], "1")
        self.assertEqual(tracked.to_dict()["status"], "active")
        self.assertEqual(feature_row.to_dict()["post_count"], 2)
        self.assertEqual(config.to_dict()["target_asset"], "SPY")
        self.assertEqual(config.to_dict()["threshold_grid"], list(config.threshold_grid))
        self.assertEqual(portfolio_config.to_dict()["component_run_ids"], ["spy-run", "qqq-run"])
        self.assertEqual(portfolio_config.to_dict()["selected_symbols"], ["SPY", "QQQ"])
        self.assertEqual(snapshot.to_dict()["target_asset"], "SPY")
        self.assertEqual(snapshot.to_dict()["stance"], "long")
        self.assertEqual(backtest_run.to_dict()["run_id"], "run-1")
        self.assertEqual(backtest_run.to_dict()["run_type"], "asset_model")
        round_trip_live = LiveMonitorConfig.from_dict(live_config.to_dict())
        self.assertEqual(round_trip_live.mode, "portfolio_run")
        self.assertEqual(round_trip_live.portfolio_run_id, "portfolio-1")
        self.assertEqual(round_trip_live.deployment_variant, "pooled")
        self.assertEqual(round_trip_live.pinned_runs[0].asset_symbol, "SPY")
        round_trip_artifact = LinearModelArtifact.from_dict(artifact.to_dict())
        self.assertEqual(round_trip_artifact.feature_names, ["x1"])
        self.assertEqual(round_trip_artifact.model_family, "custom_linear")
        self.assertEqual(
            LiveMonitorConfig.from_dict({"fallback_mode": "SPY", "pinned_runs": [live_config.pinned_runs[0].to_dict()]}).mode,
            "asset_model_set",
        )


if __name__ == "__main__":
    unittest.main()
