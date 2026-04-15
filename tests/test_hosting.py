from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.access import ADMIN_SESSION_KEY, app_mode_label, verify_admin_password, writes_enabled
from trump_workbench.config import AppSettings
from trump_workbench.health import make_refresh_history_frame
from trump_workbench.runtime import CORE_DATASET_NAMES, ensure_bootstrap
from trump_workbench.scheduler import acquire_refresh_lock, choose_scheduler_refresh, release_refresh_lock
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


if __name__ == "__main__":
    unittest.main()
