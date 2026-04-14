from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.config import APP_TITLE, AppSettings, CURRENT_TERM_START, DEFAULT_ETF_SYMBOLS, EASTERN
from trump_workbench.contracts import (
    BacktestRun,
    LinearModelArtifact,
    ModelRunConfig,
    NormalizedPost,
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

        self.assertEqual(post.to_dict()["post_id"], "1")
        self.assertEqual(tracked.to_dict()["status"], "active")
        self.assertEqual(feature_row.to_dict()["post_count"], 2)
        self.assertEqual(config.to_dict()["target_asset"], "SPY")
        self.assertEqual(config.to_dict()["threshold_grid"], list(config.threshold_grid))
        self.assertEqual(snapshot.to_dict()["target_asset"], "SPY")
        self.assertEqual(snapshot.to_dict()["stance"], "long")
        self.assertEqual(backtest_run.to_dict()["run_id"], "run-1")
        self.assertEqual(LinearModelArtifact.from_dict(artifact.to_dict()).feature_names, ["x1"])


if __name__ == "__main__":
    unittest.main()
