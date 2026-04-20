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
