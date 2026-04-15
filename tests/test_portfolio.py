from __future__ import annotations

import unittest

import pandas as pd

from trump_workbench.backtesting import BacktestService
from trump_workbench.contracts import PortfolioRunConfig
from trump_workbench.modeling import ModelService
from trump_workbench.portfolio import build_portfolio_decision_history, rank_portfolio_candidates


class PortfolioDecisionTests(unittest.TestCase):
    def test_rank_portfolio_candidates_breaks_ties_deterministically(self) -> None:
        board, decision = rank_portfolio_candidates(
            pd.DataFrame(
                [
                    {
                        "signal_session_date": pd.Timestamp("2025-02-03"),
                        "next_session_date": pd.Timestamp("2025-02-04"),
                        "asset_symbol": "SPY",
                        "run_id": "spy-run",
                        "run_name": "SPY run",
                        "expected_return_score": 0.01,
                        "confidence": 0.6,
                        "threshold": 0.001,
                        "min_post_count": 1,
                        "post_count": 2,
                    },
                    {
                        "signal_session_date": pd.Timestamp("2025-02-03"),
                        "next_session_date": pd.Timestamp("2025-02-04"),
                        "asset_symbol": "AAPL",
                        "run_id": "aapl-run",
                        "run_name": "AAPL run",
                        "expected_return_score": 0.01,
                        "confidence": 0.6,
                        "threshold": 0.001,
                        "min_post_count": 1,
                        "post_count": 2,
                    },
                ],
            ),
            fallback_mode="SPY",
        )

        self.assertEqual(decision.iloc[0]["winning_asset"], "AAPL")
        self.assertEqual(int(board.loc[board["asset_symbol"] == "AAPL", "eligible_rank"].iloc[0]), 1)

    def test_build_portfolio_decision_history_handles_missing_sessions_and_untradable_assets(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "signal_session_date": pd.Timestamp("2025-02-03"),
                    "next_session_date": pd.Timestamp("2025-02-04"),
                    "asset_symbol": "SPY",
                    "run_id": "spy-run",
                    "run_name": "SPY",
                    "expected_return_score": 0.01,
                    "confidence": 0.55,
                    "threshold": 0.001,
                    "min_post_count": 1,
                    "post_count": 4,
                    "tradeable": True,
                },
                {
                    "signal_session_date": pd.Timestamp("2025-02-03"),
                    "next_session_date": pd.Timestamp("2025-02-04"),
                    "asset_symbol": "QQQ",
                    "run_id": "qqq-run",
                    "run_name": "QQQ",
                    "expected_return_score": 0.02,
                    "confidence": 0.6,
                    "threshold": 0.001,
                    "min_post_count": 1,
                    "post_count": 4,
                    "tradeable": True,
                },
                {
                    "signal_session_date": pd.Timestamp("2025-02-04"),
                    "next_session_date": pd.Timestamp("2025-02-05"),
                    "asset_symbol": "SPY",
                    "run_id": "spy-run",
                    "run_name": "SPY",
                    "expected_return_score": 0.015,
                    "confidence": 0.55,
                    "threshold": 0.001,
                    "min_post_count": 1,
                    "post_count": 4,
                    "tradeable": True,
                },
                {
                    "signal_session_date": pd.Timestamp("2025-02-04"),
                    "next_session_date": pd.Timestamp("2025-02-05"),
                    "asset_symbol": "QQQ",
                    "run_id": "qqq-run",
                    "run_name": "QQQ",
                    "expected_return_score": 0.03,
                    "confidence": 0.6,
                    "threshold": 0.001,
                    "min_post_count": 1,
                    "post_count": 4,
                    "tradeable": False,
                },
            ],
        )

        board, decision_history = build_portfolio_decision_history(
            candidates,
            fallback_mode="SPY",
            require_tradeable=True,
        )

        self.assertEqual(len(decision_history), 2)
        self.assertEqual(decision_history.iloc[0]["winning_asset"], "QQQ")
        self.assertEqual(decision_history.iloc[1]["winning_asset"], "SPY")
        self.assertFalse(bool(board.loc[(board["signal_session_date"] == pd.Timestamp("2025-02-04")) & (board["asset_symbol"] == "QQQ"), "qualifies"].iloc[0]))
        self.assertTrue(bool(board.loc[(board["signal_session_date"] == pd.Timestamp("2025-02-04")) & (board["asset_symbol"] == "QQQ"), "signal_qualifies"].iloc[0]))


class PortfolioAllocatorIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backtests = BacktestService(ModelService())

    @staticmethod
    def _component_bundle(
        run_id: str,
        target_asset: str,
        predictions: pd.DataFrame,
        threshold: float = 0.001,
        min_post_count: int = 1,
    ) -> dict[str, object]:
        return {
            "run": {
                "run_id": run_id,
                "run_name": f"{target_asset} model",
                "target_asset": target_asset,
                "run_type": "asset_model",
            },
            "config": {
                "target_asset": target_asset,
                "feature_version": "v1" if target_asset == "SPY" else "asset-v1",
            },
            "selected_params": {
                "threshold": threshold,
                "min_post_count": min_post_count,
            },
            "predictions": predictions,
            "leakage_audit": {"overall_pass": True},
        }

    def test_saved_run_allocator_rotates_between_assets_and_persists_benchmarks(self) -> None:
        spy_predictions = pd.DataFrame(
            {
                "signal_session_date": [pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-05")],
                "next_session_date": [pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-05"), pd.Timestamp("2025-02-06")],
                "next_session_open": [100.0, 101.0, 102.0],
                "next_session_close": [101.0, 102.0, 103.0],
                "target_available": [True, True, True],
                "tradeable": [True, True, True],
                "expected_return_score": [0.01, 0.015, 0.0005],
                "prediction_confidence": [0.55, 0.56, 0.54],
                "post_count": [4, 4, 1],
                "model_version": ["spy-v1", "spy-v1", "spy-v1"],
            },
        )
        qqq_predictions = pd.DataFrame(
            {
                "signal_session_date": [pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-05")],
                "next_session_date": [pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-05"), pd.Timestamp("2025-02-06")],
                "next_session_open": [200.0, None, 202.0],
                "next_session_close": [204.0, None, 202.5],
                "target_available": [True, False, True],
                "tradeable": [True, False, True],
                "expected_return_score": [0.02, 0.03, 0.0002],
                "prediction_confidence": [0.60, 0.62, 0.58],
                "post_count": [4, 4, 1],
                "model_version": ["qqq-v1", "qqq-v1", "qqq-v1"],
            },
        )
        aapl_predictions = pd.DataFrame(
            {
                "signal_session_date": [pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-05")],
                "next_session_date": [pd.Timestamp("2025-02-05"), pd.Timestamp("2025-02-06")],
                "next_session_open": [50.0, 51.0],
                "next_session_close": [50.2, 51.2],
                "target_available": [True, True],
                "tradeable": [True, True],
                "expected_return_score": [0.014, 0.0001],
                "prediction_confidence": [0.53, 0.52],
                "post_count": [4, 1],
                "model_version": ["aapl-v1", "aapl-v1"],
            },
        )

        component_bundles = {
            "spy-run": self._component_bundle("spy-run", "SPY", spy_predictions),
            "qqq-run": self._component_bundle("qqq-run", "QQQ", qqq_predictions),
            "aapl-run": self._component_bundle("aapl-run", "AAPL", aapl_predictions),
        }
        config = PortfolioRunConfig(
            run_name="portfolio-test",
            allocator_mode="saved_runs",
            fallback_mode="SPY",
            transaction_cost_bps=2.0,
            component_run_ids=("spy-run", "qqq-run", "aapl-run"),
            universe_symbols=("SPY", "QQQ", "AAPL"),
        )

        run, artifacts = self.backtests.run_saved_run_allocator(config, component_bundles)

        self.assertEqual(run.run_type, "portfolio_allocator")
        self.assertEqual(run.allocator_mode, "saved_runs")
        self.assertEqual(run.target_asset, "PORTFOLIO")
        self.assertEqual(artifacts["predictions"]["winning_asset"].tolist(), ["QQQ", "SPY", "SPY"])
        self.assertEqual(artifacts["trades"]["selected_asset"].tolist(), ["QQQ", "SPY", "SPY"])
        self.assertTrue((artifacts["trades"]["trade_taken"].astype(bool)).all())
        self.assertIn("always_long_spy", artifacts["benchmarks"]["benchmark_name"].tolist())
        self.assertIn("always_long_qqq", artifacts["benchmarks"]["benchmark_name"].tolist())
        self.assertIn("always_long_aapl", artifacts["benchmarks"]["benchmark_name"].tolist())
        self.assertFalse(artifacts["candidate_predictions"].empty)
        self.assertTrue(artifacts["leakage_audit"]["overall_pass"])


if __name__ == "__main__":
    unittest.main()
