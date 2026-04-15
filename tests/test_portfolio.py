from __future__ import annotations

import unittest
import warnings

import numpy as np
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

    @staticmethod
    def _joint_feature_rows(symbols: list[str], periods: int = 72) -> pd.DataFrame:
        signal_dates = pd.bdate_range("2025-02-03", periods=periods)
        rows: list[dict[str, object]] = []
        for idx, signal_date in enumerate(signal_dates):
            next_session_date = signal_dates[idx + 1] if idx + 1 < len(signal_dates) else pd.NaT
            regime = idx % len(symbols)
            for asset_offset, asset_symbol in enumerate(symbols):
                strong_asset = symbols[regime]
                score = 0.8 if asset_symbol == strong_asset else -0.2
                target_return = 0.012 if asset_symbol == strong_asset else -0.002
                open_price = 100.0 + asset_offset * 25.0 + idx
                close_price = open_price * (1.0 + target_return) if pd.notna(next_session_date) else np.nan
                rows.append(
                    {
                        "signal_session_date": signal_date,
                        "next_session_date": next_session_date,
                        "asset_symbol": asset_symbol,
                        "feature_version": "asset-v1",
                        "llm_enabled": False,
                        "target_next_session_return": target_return if pd.notna(next_session_date) else np.nan,
                        "target_available": pd.notna(next_session_date),
                        "tradeable": pd.notna(next_session_date),
                        "next_session_open": open_price if pd.notna(next_session_date) else np.nan,
                        "next_session_close": close_price,
                        "post_count": 4 + asset_offset,
                        "trump_post_count": 1,
                        "tracked_account_post_count": 2,
                        "tracked_weighted_mentions": 1.0 + score,
                        "tracked_weighted_engagement": 5.0 + score,
                        "sentiment_avg": score,
                        "sentiment_close": score * 0.9,
                        "sentiment_range": 0.2 + asset_offset * 0.05,
                        "mention_concentration": 0.3 + asset_offset * 0.02,
                        "author_diversity": 2.0 + asset_offset,
                        "session_return": 0.001 * (idx % 5),
                        "rolling_vol_5d": 0.01 + asset_offset * 0.001,
                        "close_vs_ma_5": score * 0.1,
                        "volume_z_5": 0.2 + asset_offset * 0.03,
                    },
                )
        return pd.DataFrame(rows)

    def test_joint_model_allocator_trains_variants_and_selects_deployment(self) -> None:
        feature_rows = self._joint_feature_rows(["SPY", "QQQ", "NVDA"])
        config = PortfolioRunConfig(
            run_name="joint-portfolio",
            allocator_mode="joint_model",
            fallback_mode="SPY",
            transaction_cost_bps=2.0,
            universe_symbols=("SPY", "QQQ", "NVDA"),
            selected_symbols=("SPY", "QQQ", "NVDA"),
            llm_enabled=False,
            feature_version="asset-v1",
            train_window=24,
            validation_window=12,
            test_window=12,
            step_size=12,
            threshold_grid=(0.0, 0.001),
            minimum_signal_grid=(1, 2),
            account_weight_grid=(0.5, 1.0),
            model_families=("ridge", "hist_gradient_boosting_regressor"),
            topology_variants=("per_asset", "pooled"),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
            run, artifacts = self.backtests.run_joint_model_allocator(config, feature_rows)

        self.assertEqual(run.run_type, "portfolio_allocator")
        self.assertEqual(run.allocator_mode, "joint_model")
        self.assertIn(run.deployment_variant, {"per_asset", "pooled"})
        self.assertSetEqual(set(artifacts["variant_summary"]["variant_name"].astype(str)), {"per_asset", "pooled"})
        self.assertIn("deployment_winner", artifacts["variant_summary"].columns)
        self.assertSetEqual(set(artifacts["candidate_predictions"]["variant_name"].astype(str)), {"per_asset", "pooled"})
        self.assertSetEqual(set(artifacts["predictions"]["variant_name"].astype(str)), {"per_asset", "pooled"})
        self.assertSetEqual(set(artifacts["trades"]["variant_name"].astype(str)), {"per_asset", "pooled"})
        self.assertSetEqual(set(artifacts["benchmarks"]["variant_name"].astype(str)), {"per_asset", "pooled"})
        self.assertIn(run.deployment_variant, artifacts["portfolio_model_bundle"]["variants"])
        self.assertEqual(
            artifacts["portfolio_model_bundle"]["deployment_variant"],
            run.deployment_variant,
        )
        self.assertTrue(artifacts["leakage_audit"]["overall_pass"])


if __name__ == "__main__":
    unittest.main()
