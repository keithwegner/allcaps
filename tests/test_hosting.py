from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from trump_workbench.access import ADMIN_SESSION_KEY, app_mode_label, verify_admin_password, writes_enabled
from trump_workbench.config import AppSettings
from trump_workbench.health import make_refresh_history_frame
from trump_workbench.runtime import CORE_DATASET_NAMES, ensure_bootstrap
from trump_workbench.scheduler import (
    SchedulerDecision,
    acquire_refresh_lock,
    choose_scheduler_refresh,
    release_refresh_lock,
    run_scheduler_cycle,
    run_scheduler_once,
)
from trump_workbench.storage import DuckDBStore


class HostingBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(self.settings)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _seed_core_datasets(self) -> None:
        for dataset_name in CORE_DATASET_NAMES:
            self.store.save_frame(dataset_name, pd.DataFrame([{"value": 1}]), metadata={"row_count": 1})

    def test_public_mode_requires_admin_session_for_writes(self) -> None:
        settings = AppSettings(
            base_dir=Path(self.temp_dir.name),
            public_mode=True,
            admin_password="secret",
        )

        self.assertFalse(writes_enabled(settings, {}))
        self.assertEqual(app_mode_label(settings, {}), "public_read_only")
        self.assertTrue(verify_admin_password(settings, "secret"))
        self.assertFalse(verify_admin_password(settings, "wrong"))
        self.assertTrue(writes_enabled(settings, {ADMIN_SESSION_KEY: True}))
        self.assertEqual(app_mode_label(settings, {ADMIN_SESSION_KEY: True}), "admin")

    def test_ensure_bootstrap_skips_when_auto_bootstrap_is_disabled(self) -> None:
        settings = AppSettings(
            base_dir=Path(self.temp_dir.name),
            auto_bootstrap_on_start=False,
        )

        result = ensure_bootstrap(
            settings=settings,
            store=self.store,
            ingestion_service=None,
            market_service=None,
            discovery_service=None,
            feature_service=None,
            health_service=None,
        )

        self.assertIsNone(result)
        self.assertFalse(self.store.read_frame("asset_universe").empty)
        self.assertTrue(self.store.read_frame("normalized_posts").empty)

    def test_scheduler_prefers_bootstrap_when_core_datasets_are_missing(self) -> None:
        decision = choose_scheduler_refresh(self.store, self.settings, now=pd.Timestamp("2026-04-15 10:00:00", tz="America/New_York"))

        self.assertEqual(decision.refresh_mode, "bootstrap")
        self.assertFalse(decision.incremental)

    def test_scheduler_chooses_full_refresh_at_nightly_cutoff(self) -> None:
        self._seed_core_datasets()
        history = make_refresh_history_frame(
            refresh_id="incremental-1",
            refresh_mode="incremental",
            status="success",
            started_at=pd.Timestamp("2026-04-14 13:00:00", tz="UTC"),
            completed_at=pd.Timestamp("2026-04-14 13:05:00", tz="UTC"),
        )
        self.store.save_frame("refresh_history", history, metadata={"latest_status": "success"})

        decision = choose_scheduler_refresh(
            self.store,
            AppSettings(base_dir=Path(self.temp_dir.name), scheduler_full_hour=3, scheduler_full_minute=0),
            now=pd.Timestamp("2026-04-15 03:30:00", tz="America/New_York"),
        )

        self.assertEqual(decision.refresh_mode, "full")
        self.assertFalse(decision.incremental)

    def test_scheduler_chooses_incremental_refresh_when_interval_elapsed(self) -> None:
        self._seed_core_datasets()
        history = make_refresh_history_frame(
            refresh_id="full-1",
            refresh_mode="full",
            status="success",
            started_at=pd.Timestamp("2026-04-15 05:00:00", tz="UTC"),
            completed_at=pd.Timestamp("2026-04-15 05:05:00", tz="UTC"),
        )
        self.store.save_frame("refresh_history", history, metadata={"latest_status": "success"})

        decision = choose_scheduler_refresh(
            self.store,
            AppSettings(base_dir=Path(self.temp_dir.name), scheduler_incremental_minutes=30, scheduler_full_hour=23),
            now=pd.Timestamp("2026-04-15 01:45:00", tz="America/New_York"),
        )

        self.assertEqual(decision.refresh_mode, "incremental")
        self.assertTrue(decision.incremental)

    def test_refresh_lock_prevents_overlap(self) -> None:
        lock_fd = acquire_refresh_lock(self.settings)
        self.assertIsNotNone(lock_fd)
        second_lock_fd = acquire_refresh_lock(self.settings)
        self.assertIsNone(second_lock_fd)
        release_refresh_lock(self.settings, lock_fd)
        third_lock_fd = acquire_refresh_lock(self.settings)
        self.assertIsNotNone(third_lock_fd)
        release_refresh_lock(self.settings, third_lock_fd)

    def test_release_refresh_lock_ignores_missing_lock(self) -> None:
        release_refresh_lock(self.settings, None)

        self.assertFalse(self.settings.scheduler_lock_path.exists())

    def test_scheduler_chooses_full_when_history_is_missing_or_invalid(self) -> None:
        self._seed_core_datasets()

        empty_history_decision = choose_scheduler_refresh(
            self.store,
            self.settings,
            now=pd.Timestamp("2026-04-15 01:00:00"),
        )

        error_history = make_refresh_history_frame(
            refresh_id="failed-1",
            refresh_mode="incremental",
            status="error",
            started_at=pd.Timestamp("2026-04-15 01:00:00", tz="UTC"),
            completed_at=pd.Timestamp("2026-04-15 01:01:00", tz="UTC"),
            error_message="boom",
        )
        self.store.save_frame("refresh_history", error_history, metadata={"latest_status": "error"})
        error_history_decision = choose_scheduler_refresh(
            self.store,
            self.settings,
            now=pd.Timestamp("2026-04-15 01:05:00", tz="America/New_York"),
        )

        invalid_history = make_refresh_history_frame(
            refresh_id="bad-1",
            refresh_mode="incremental",
            status="success",
            started_at=pd.Timestamp("2026-04-15 01:00:00", tz="UTC"),
            completed_at=pd.NaT,
        )
        self.store.save_frame("refresh_history", invalid_history, metadata={"latest_status": "success"})
        invalid_history_decision = choose_scheduler_refresh(
            self.store,
            self.settings,
            now=pd.Timestamp("2026-04-15 01:10:00", tz="America/New_York"),
        )

        self.assertEqual(empty_history_decision.refresh_mode, "full")
        self.assertEqual(error_history_decision.refresh_mode, "full")
        self.assertEqual(invalid_history_decision.refresh_mode, "full")

    def test_scheduler_skips_when_recent_success_is_fresh(self) -> None:
        self._seed_core_datasets()
        history = make_refresh_history_frame(
            refresh_id="full-1",
            refresh_mode="full",
            status="success",
            started_at=pd.Timestamp("2026-04-15 06:00:00", tz="UTC"),
            completed_at=pd.Timestamp("2026-04-15 06:05:00", tz="UTC"),
        )
        self.store.save_frame("refresh_history", history, metadata={"latest_status": "success"})

        decision = choose_scheduler_refresh(
            self.store,
            AppSettings(
                base_dir=Path(self.temp_dir.name),
                scheduler_incremental_minutes=30,
                scheduler_full_hour=1,
                scheduler_full_minute=0,
            ),
            now=pd.Timestamp("2026-04-15 02:15:00", tz="America/New_York"),
        )

        self.assertFalse(decision.should_run)

    def test_run_scheduler_cycle_skips_refresh_when_no_decision(self) -> None:
        with patch("trump_workbench.scheduler.refresh_datasets") as refresh_mock:
            result = run_scheduler_cycle(
                settings=self.settings,
                store=self.store,
                decision=SchedulerDecision(),
                ingestion_service=Mock(),
                market_service=Mock(),
                discovery_service=Mock(),
                feature_service=Mock(),
                health_service=Mock(),
                experiment_store=Mock(),
                model_service=Mock(),
                paper_service=Mock(),
            )

        self.assertFalse(result.should_run)
        refresh_mock.assert_not_called()

    def test_run_scheduler_cycle_returns_after_refresh_without_live_config(self) -> None:
        experiment_store = Mock()
        experiment_store.load_live_monitor_config.return_value = None

        with patch("trump_workbench.scheduler.refresh_datasets") as refresh_mock:
            result = run_scheduler_cycle(
                settings=self.settings,
                store=self.store,
                decision=SchedulerDecision(refresh_mode="incremental", incremental=True),
                ingestion_service=Mock(),
                market_service=Mock(),
                discovery_service=Mock(),
                feature_service=Mock(),
                health_service=Mock(),
                experiment_store=experiment_store,
                model_service=Mock(),
                paper_service=Mock(),
            )

        self.assertEqual(result.refresh_mode, "incremental")
        refresh_mock.assert_called_once()
        experiment_store.list_runs.assert_not_called()

    def test_run_scheduler_cycle_persists_live_paper_and_performance_outputs(self) -> None:
        board = pd.DataFrame(
            [
                {
                    "generated_at": pd.Timestamp("2026-04-15 12:00:00", tz="UTC"),
                    "asset_symbol": "SPY",
                    "expected_return_score": 0.01,
                },
            ],
        )
        decision_row = pd.DataFrame(
            [
                {
                    "generated_at": pd.Timestamp("2026-04-15 12:00:00", tz="UTC"),
                    "signal_session_date": pd.Timestamp("2026-04-14", tz="UTC"),
                    "winning_asset": "SPY",
                    "stance": "LONG",
                },
            ],
        )
        live_config = SimpleNamespace(
            mode="portfolio_run",
            portfolio_run_id="run-1",
            deployment_variant="per_asset__baseline",
            fallback_mode="SPY",
        )
        paper_config = SimpleNamespace(paper_portfolio_id="paper-1")
        experiment_store = Mock()
        experiment_store.load_live_monitor_config.return_value = live_config
        experiment_store.list_runs.return_value = pd.DataFrame([{"run_id": "run-1"}])
        paper_service = Mock()
        paper_service.load_current_config.return_value = paper_config
        performance_service = Mock()

        with (
            patch("trump_workbench.scheduler.refresh_datasets") as refresh_mock,
            patch("trump_workbench.scheduler.validate_live_monitor_config", return_value=[]),
            patch("trump_workbench.scheduler.build_live_portfolio_run_state", return_value=(board, decision_row, None, [])),
            patch("trump_workbench.scheduler.paper_config_matches_live", return_value=True),
        ):
            result = run_scheduler_cycle(
                settings=self.settings,
                store=self.store,
                decision=SchedulerDecision(refresh_mode="full", incremental=False),
                ingestion_service=Mock(),
                market_service=Mock(),
                discovery_service=Mock(),
                feature_service=Mock(),
                health_service=Mock(),
                experiment_store=experiment_store,
                model_service=Mock(),
                paper_service=paper_service,
                performance_service=performance_service,
                generated_at=pd.Timestamp("2026-04-15 12:00:00", tz="UTC"),
            )

        self.assertEqual(result.refresh_mode, "full")
        refresh_mock.assert_called_once()
        experiment_store.save_live_asset_snapshots.assert_called_once()
        experiment_store.save_live_decision_snapshots.assert_called_once()
        paper_service.process_live_history.assert_called_once_with(
            paper_config,
            as_of=pd.Timestamp("2026-04-15 12:00:00", tz="UTC"),
        )
        performance_service.persist_snapshot.assert_called_once_with(
            "paper-1",
            generated_at=pd.Timestamp("2026-04-15 12:00:00", tz="UTC"),
        )

    def test_run_scheduler_once_returns_empty_decision_when_lock_is_held(self) -> None:
        lock_fd = acquire_refresh_lock(self.settings)
        self.assertIsNotNone(lock_fd)
        try:
            decision = run_scheduler_once(self.settings)
        finally:
            release_refresh_lock(self.settings, lock_fd)

        self.assertFalse(decision.should_run)

    def test_run_scheduler_once_releases_lock_after_cycle_failure(self) -> None:
        with (
            patch(
                "trump_workbench.scheduler.choose_scheduler_refresh",
                return_value=SchedulerDecision(refresh_mode="full", incremental=False),
            ),
            patch("trump_workbench.scheduler.run_scheduler_cycle", side_effect=RuntimeError("cycle failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "cycle failed"):
                run_scheduler_once(self.settings)

        lock_fd = acquire_refresh_lock(self.settings)
        self.assertIsNotNone(lock_fd)
        release_refresh_lock(self.settings, lock_fd)


if __name__ == "__main__":
    unittest.main()
