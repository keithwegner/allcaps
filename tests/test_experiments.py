from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.config import AppSettings
from trump_workbench.contracts import BacktestRun
from trump_workbench.experiments import ExperimentStore
from trump_workbench.storage import DuckDBStore


class ExperimentStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(settings)
        self.experiments = ExperimentStore(self.store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _sample_run(run_id: str = "run-123", target_asset: str = "SPY") -> BacktestRun:
        return BacktestRun(
            run_id=run_id,
            run_name="unit-test-run",
            target_asset=target_asset,
            config_hash="config-hash",
            train_window=60,
            validation_window=20,
            test_window=20,
            metrics={"robust_score": 1.23},
            selected_params={"threshold": 0.001},
        )

    @staticmethod
    def _sample_frame(column: str = "value") -> pd.DataFrame:
        return pd.DataFrame({column: [1, 2]})

    def test_load_methods_return_none_when_no_runs_exist(self) -> None:
        self.assertIsNone(self.experiments.load_run("missing"))
        self.assertIsNone(self.experiments.load_latest_model_artifact())
        self.assertIsNone(self.experiments.load_latest_model_artifact(target_asset="QQQ"))

    def test_save_load_run_and_prediction_snapshots(self) -> None:
        run = self._sample_run()
        saved = self.experiments.save_run(
            run=run,
            config={"feature_version": "v1", "target_asset": "SPY"},
            trades=self._sample_frame("trade_id"),
            predictions=self._sample_frame("prediction"),
            windows=self._sample_frame("window"),
            importance=self._sample_frame("feature"),
            model_artifact={
                "model_version": "linear-v1",
                "feature_names": ["x1"],
                "intercept": 0.1,
                "coefficients": [0.2],
                "means": [0.0],
                "stds": [1.0],
                "residual_std": 0.3,
                "train_rows": 10,
                "metadata": {"source": "unit-test", "target_asset": "SPY"},
            },
            feature_contributions=self._sample_frame("contribution"),
            post_attribution=self._sample_frame("post_signal_score"),
            account_attribution=self._sample_frame("net_post_signal"),
            benchmarks=self._sample_frame("benchmark_name"),
            diagnostics=self._sample_frame("error"),
            benchmark_curves=self._sample_frame("equity"),
            leakage_audit={"overall_pass": True},
        )

        loaded = self.experiments.load_run(run.run_id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(saved.summary_path.exists())
        self.assertEqual(loaded["metrics"]["robust_score"], 1.23)
        self.assertEqual(loaded["config"]["feature_version"], "v1")
        self.assertEqual(loaded["config"]["target_asset"], "SPY")
        self.assertEqual(loaded["run"]["target_asset"], "SPY")
        self.assertEqual(loaded["run"]["run_name"], "unit-test-run")
        self.assertEqual(loaded["model_artifact"].feature_names, ["x1"])
        self.assertFalse(loaded["feature_contributions"].empty)
        self.assertFalse(loaded["post_attribution"].empty)
        self.assertFalse(loaded["account_attribution"].empty)
        self.assertFalse(loaded["benchmarks"].empty)
        self.assertTrue(bool(loaded["leakage_audit"]["overall_pass"]))

        latest = self.experiments.load_latest_model_artifact()
        self.assertIsNotNone(latest)
        assert latest is not None
        artifact, params = latest
        self.assertEqual(artifact.model_version, "linear-v1")
        self.assertEqual(params["threshold"], 0.001)

        qqq_run = self._sample_run(run_id="run-qqq", target_asset="QQQ")
        self.experiments.save_run(
            run=qqq_run,
            config={"feature_version": "asset-v1", "target_asset": "QQQ"},
            trades=self._sample_frame("trade_id"),
            predictions=self._sample_frame("prediction"),
            windows=self._sample_frame("window"),
            importance=self._sample_frame("feature"),
            model_artifact={
                "model_version": "linear-qqq-v1",
                "feature_names": ["x1"],
                "intercept": 0.1,
                "coefficients": [0.2],
                "means": [0.0],
                "stds": [1.0],
                "residual_std": 0.3,
                "train_rows": 10,
                "metadata": {"source": "unit-test", "target_asset": "QQQ"},
            },
            feature_contributions=self._sample_frame("contribution"),
            post_attribution=self._sample_frame("post_signal_score"),
            account_attribution=self._sample_frame("net_post_signal"),
            benchmarks=self._sample_frame("benchmark_name"),
            diagnostics=self._sample_frame("error"),
            benchmark_curves=self._sample_frame("equity"),
            leakage_audit={"overall_pass": True},
        )
        latest_spy = self.experiments.load_latest_model_artifact(target_asset="SPY")
        latest_qqq = self.experiments.load_latest_model_artifact(target_asset="QQQ")
        self.assertIsNotNone(latest_spy)
        self.assertIsNotNone(latest_qqq)
        assert latest_spy is not None
        assert latest_qqq is not None
        self.assertEqual(latest_spy[0].metadata["target_asset"], "SPY")
        self.assertEqual(latest_qqq[0].metadata["target_asset"], "QQQ")
        self.assertIsNone(self.experiments.load_latest_model_artifact(target_asset="XLE"))

        self.experiments.save_prediction_snapshots(pd.DataFrame())
        snapshots = pd.DataFrame(
            {
                "signal_session_date": [pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-03")],
                "generated_at": [
                    pd.Timestamp("2025-02-03 16:00:00"),
                    pd.Timestamp("2025-02-03 16:00:00"),
                    pd.Timestamp("2025-02-03 16:00:00"),
                ],
                "target_asset": ["SPY", "SPY", "QQQ"],
                "expected_return_score": [0.01, 0.02, 0.03],
            },
        )
        self.experiments.save_prediction_snapshots(snapshots)
        stored_snapshots = self.store.read_frame("prediction_snapshots")
        self.assertEqual(len(stored_snapshots), 2)
        self.assertSetEqual(set(stored_snapshots["target_asset"].astype(str)), {"SPY", "QQQ"})
        self.assertAlmostEqual(
            float(
                stored_snapshots.loc[
                    stored_snapshots["target_asset"].astype(str) == "SPY",
                    "expected_return_score",
                ].iloc[0],
            ),
            0.02,
        )

        saved.benchmarks_path.unlink()
        saved.diagnostics_path.unlink()
        saved.benchmark_curves_path.unlink()
        saved.leakage_audit_path.unlink()
        saved.feature_contributions_path.unlink()
        saved.post_attribution_path.unlink()
        saved.account_attribution_path.unlink()

        loaded_without_optional = self.experiments.load_run(run.run_id)
        assert loaded_without_optional is not None
        self.assertTrue(loaded_without_optional["feature_contributions"].empty)
        self.assertTrue(loaded_without_optional["post_attribution"].empty)
        self.assertTrue(loaded_without_optional["account_attribution"].empty)
        self.assertTrue(loaded_without_optional["benchmarks"].empty)
        self.assertTrue(loaded_without_optional["diagnostics"].empty)
        self.assertTrue(loaded_without_optional["benchmark_curves"].empty)
        self.assertEqual(loaded_without_optional["leakage_audit"], {})


if __name__ == "__main__":
    unittest.main()
