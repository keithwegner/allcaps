from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from trump_workbench.api import create_app
from trump_workbench.config import AppSettings
from trump_workbench.paper_trading import (
    PAPER_BENCHMARK_CURVE_COLUMNS,
    PAPER_DECISION_JOURNAL_COLUMNS,
    PAPER_EQUITY_CURVE_COLUMNS,
    PAPER_PORTFOLIO_REGISTRY_COLUMNS,
    PAPER_TRADE_LEDGER_COLUMNS,
)
from trump_workbench.storage import DuckDBStore


class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(self.settings)
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

    def test_status_reports_source_mode_and_state_paths(self) -> None:
        self._save_truth_posts()

        response = self.client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source_mode"]["mode"], "truth_only")
        self.assertIn(".workbench", payload["db_path"])

    def test_dataset_health_returns_summary_rows_and_registry(self) -> None:
        self._save_truth_posts()

        response = self.client.get("/api/datasets/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("overall_severity", payload["summary"])
        self.assertIsInstance(payload["latest"], list)
        self.assertGreaterEqual(len(payload["registry"]), 1)

    def test_runs_and_live_current_handle_empty_live_config(self) -> None:
        self._save_run_record()

        runs_response = self.client.get("/api/runs")
        live_response = self.client.get("/api/live/current")

        self.assertEqual(runs_response.status_code, 200)
        self.assertEqual(runs_response.json()["count"], 1)
        self.assertEqual(live_response.status_code, 200)
        self.assertFalse(live_response.json()["configured"])
        self.assertIn("No live monitor config", live_response.json()["errors"][0])

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

