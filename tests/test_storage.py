from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.config import AppSettings
from trump_workbench.storage import DuckDBStore


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(self.settings)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_save_read_append_and_registry_round_trip(self) -> None:
        first = pd.DataFrame({"id": [1], "value": ["a"]})
        second = pd.DataFrame({"id": [1, 2], "value": ["updated", "b"]})

        saved_path = self.store.save_frame("sample", first, metadata={"source": "test"})
        appended_path = self.store.append_frame("sample", second, dedupe_on=["id"], metadata={"source": "append"})
        loaded = self.store.read_frame("sample").sort_values("id").reset_index(drop=True)
        registry = self.store.dataset_registry()

        self.assertEqual(saved_path, appended_path)
        self.assertEqual(loaded["value"].tolist(), ["updated", "b"])
        self.assertEqual(registry.iloc[0]["dataset_name"], "sample")
        self.assertEqual(registry.iloc[0]["row_count"], 2)
        self.assertIn("append", registry.iloc[0]["metadata_json"])

    def test_missing_frames_and_json_artifacts_are_handled(self) -> None:
        self.assertTrue(self.store.read_frame("missing").empty)
        self.assertIsNone(self.store.read_json_artifact("missing.json"))

        path = self.store.save_json_artifact("reports/status.json", {"ok": True})
        payload = self.store.read_json_artifact("reports/status.json")

        self.assertTrue(path.exists())
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(self.store.dataset_path("dataset"), self.settings.lake_dir / "dataset.parquet")
        self.assertEqual(self.store.artifact_path("reports", "status.json"), self.settings.artifact_dir / "reports" / "status.json")

    def test_save_run_record_and_list_records(self) -> None:
        self.assertTrue(self.store.list_run_records().empty)

        self.store.save_run_record(
            run_id="run-1",
            run_name="coverage test",
            config_hash="hash-1",
            metrics={"sharpe": 1.2},
            selected_params={"threshold": 0.001},
            artifact_paths={
                "summary_path": "/tmp/summary.json",
                "trades_path": "/tmp/trades.parquet",
                "predictions_path": "/tmp/predictions.parquet",
                "windows_path": "/tmp/windows.parquet",
                "importance_path": "/tmp/importance.parquet",
                "model_path": "/tmp/model.json",
            },
        )

        records = self.store.list_run_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records.iloc[0]["metrics_json"]["sharpe"], 1.2)
        self.assertEqual(records.iloc[0]["selected_params_json"]["threshold"], 0.001)


if __name__ == "__main__":
    unittest.main()
