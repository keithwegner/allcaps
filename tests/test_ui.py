from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.config import AppSettings
from trump_workbench.contracts import RANKING_HISTORY_COLUMNS
from trump_workbench.contracts import LinearModelArtifact
from trump_workbench.storage import DuckDBStore
from trump_workbench.ui import (
    _build_benchmark_delta_table,
    _build_replay_comparison_frame,
    _build_feature_diff_table,
    _build_metric_comparison_table,
    _build_setting_diff_table,
    _bundle_to_run_config,
    _eligible_replay_sessions,
    _latest_ranking_snapshot,
    _replay_option_label,
    _save_watchlist,
    _summarize_run_changes,
    _watchlist_symbols,
    _watchlist_text_value,
)


class UiHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = DuckDBStore(AppSettings(base_dir=Path(self.temp_dir.name)))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _run_bundle(
        run_id: str,
        *,
        llm_enabled: bool,
        threshold: float,
        min_post_count: int,
        total_return: float,
        robust_score: float,
        features: list[str],
    ) -> dict[str, object]:
        return {
            "run": {"run_id": run_id, "run_name": run_id},
            "config": {
                "feature_version": "v1",
                "llm_enabled": llm_enabled,
                "train_window": 90,
                "validation_window": 30,
                "test_window": 30,
                "step_size": 30,
                "ridge_alpha": 1.0,
                "transaction_cost_bps": 2.0,
                "threshold_grid": [0.0, 0.001, 0.0025],
                "minimum_signal_grid": [1, 2, 3],
                "account_weight_grid": [0.5, 1.0, 1.5],
            },
            "selected_params": {
                "threshold": threshold,
                "min_post_count": min_post_count,
                "account_weight": 1.0,
            },
            "metrics": {
                "total_return": total_return,
                "sharpe": 1.0 + total_return,
                "sortino": 1.2 + total_return,
                "max_drawdown": -0.1,
                "robust_score": robust_score,
                "trade_count": 20,
            },
            "model_artifact": LinearModelArtifact(
                model_version=f"{run_id}-model",
                feature_names=features,
                intercept=0.1,
                coefficients=[0.2 for _ in features],
                means=[0.0 for _ in features],
                stds=[1.0 for _ in features],
                residual_std=0.3,
                train_rows=100,
                metadata={"llm_enabled": llm_enabled},
            ),
            "benchmarks": pd.DataFrame(
                [
                    {"benchmark_name": "strategy", "total_return": total_return, "robust_score": robust_score},
                    {"benchmark_name": "always_long", "total_return": total_return - 0.02, "robust_score": robust_score - 0.1},
                    {"benchmark_name": "trump_only", "total_return": total_return - 0.01, "robust_score": robust_score - 0.05},
                ],
            ),
        }

    def test_latest_ranking_snapshot_handles_empty_or_schema_less_frames(self) -> None:
        empty_snapshot = _latest_ranking_snapshot(pd.DataFrame())
        schema_less_snapshot = _latest_ranking_snapshot(pd.DataFrame({"other": [1]}))

        self.assertTrue(empty_snapshot.empty)
        self.assertTrue(schema_less_snapshot.empty)
        self.assertListEqual(list(empty_snapshot.columns), RANKING_HISTORY_COLUMNS)
        self.assertListEqual(list(schema_less_snapshot.columns), RANKING_HISTORY_COLUMNS)

    def test_latest_ranking_snapshot_sorts_and_dedupes(self) -> None:
        ranking_history = pd.DataFrame(
            [
                {
                    "author_account_id": "acct-a",
                    "author_handle": "macroa",
                    "author_display_name": "Macro A",
                    "source_platform": "X",
                    "discovery_score": 10.0,
                    "mention_count": 3,
                    "engagement_mean": 50.0,
                    "active_days": 2,
                    "ranked_at": "2025-02-04",
                    "discovery_rank": 2,
                    "final_selected": True,
                    "selected_status": "active",
                    "suppressed_by_override": False,
                    "pinned_by_override": False,
                },
                {
                    "author_account_id": "acct-a",
                    "author_handle": "macroa",
                    "author_display_name": "Macro A",
                    "source_platform": "X",
                    "discovery_score": 11.0,
                    "mention_count": 4,
                    "engagement_mean": 60.0,
                    "active_days": 3,
                    "ranked_at": "2025-02-05",
                    "discovery_rank": 1,
                    "final_selected": True,
                    "selected_status": "active",
                    "suppressed_by_override": False,
                    "pinned_by_override": False,
                },
                {
                    "author_account_id": "acct-b",
                    "author_handle": "policyb",
                    "author_display_name": "Policy B",
                    "source_platform": "X",
                    "discovery_score": 9.0,
                    "mention_count": 2,
                    "engagement_mean": 30.0,
                    "active_days": 1,
                    "ranked_at": "2025-02-03",
                    "discovery_rank": 3,
                    "final_selected": False,
                    "selected_status": "excluded",
                    "suppressed_by_override": False,
                    "pinned_by_override": False,
                },
            ],
        )

        snapshot = _latest_ranking_snapshot(ranking_history)

        self.assertEqual(len(snapshot), 2)
        self.assertEqual(snapshot.iloc[0]["author_account_id"], "acct-a")
        self.assertEqual(snapshot.iloc[0]["discovery_score"], 11.0)

    def test_watchlist_helpers_save_symbols_and_build_text(self) -> None:
        watchlist, asset_universe = _save_watchlist(self.store, [" msft ", "spy", "nvda", "NVDA"])

        self.assertEqual(watchlist["symbol"].tolist(), ["MSFT", "NVDA"])
        self.assertIn("SPY", asset_universe["symbol"].tolist())
        self.assertIn("QQQ", asset_universe["symbol"].tolist())
        self.assertEqual(_watchlist_symbols(self.store), ["MSFT", "NVDA"])
        self.assertEqual(_watchlist_text_value(self.store), "MSFT, NVDA")

    def test_run_comparison_helpers_build_diffs_and_summary(self) -> None:
        bundles = {
            "base-run": self._run_bundle(
                "base-run",
                llm_enabled=False,
                threshold=0.001,
                min_post_count=2,
                total_return=0.10,
                robust_score=1.5,
                features=["sentiment_avg", "post_count", "prev_return_1d"],
            ),
            "alt-run": self._run_bundle(
                "alt-run",
                llm_enabled=True,
                threshold=0.0025,
                min_post_count=1,
                total_return=0.14,
                robust_score=1.9,
                features=["sentiment_avg", "post_count", "prev_return_1d", "semantic_market_relevance_avg"],
            ),
        }

        metric_table = _build_metric_comparison_table("base-run", bundles)
        setting_diff = _build_setting_diff_table("base-run", bundles)
        feature_diff = _build_feature_diff_table("base-run", bundles)
        benchmark_diff = _build_benchmark_delta_table("base-run", bundles)
        notes = _summarize_run_changes("base-run", bundles)

        self.assertEqual(metric_table.iloc[0]["run_id"], "alt-run")
        self.assertAlmostEqual(float(metric_table.iloc[0]["delta_robust_score_vs_base"]), 0.4)
        self.assertIn("deploy_threshold", setting_diff["setting"].tolist())
        self.assertIn("llm_enabled", setting_diff["setting"].tolist())
        alt_feature_row = feature_diff.loc[feature_diff["run_id"] == "alt-run"].iloc[0]
        self.assertEqual(int(alt_feature_row["unique_vs_base_count"]), 1)
        self.assertIn("semantic_market_relevance_avg", str(alt_feature_row["unique_vs_base"]))
        strategy_row = benchmark_diff.loc[
            (benchmark_diff["run_id"] == "alt-run") & (benchmark_diff["benchmark_name"] == "strategy")
        ].iloc[0]
        self.assertAlmostEqual(float(strategy_row["delta_total_return_vs_base"]), 0.04)
        self.assertEqual(len(notes), 1)
        self.assertIn("LLM on vs off", notes[0])
        self.assertIn("threshold 0.001 -> 0.0025", notes[0])

    def test_replay_helpers_build_session_list_config_and_summary(self) -> None:
        feature_rows = pd.DataFrame(
            {
                "signal_session_date": pd.bdate_range("2025-02-03", periods=30),
                "target_available": [True] * 29 + [False],
                "post_count": [idx % 5 for idx in range(30)],
            },
        )
        eligible = _eligible_replay_sessions(feature_rows, min_history_rows=20)
        self.assertEqual(pd.Timestamp(eligible.iloc[0]["signal_session_date"]), pd.Timestamp("2025-03-03"))
        label = _replay_option_label(eligible.iloc[0])
        self.assertIn("prior train rows 20", label)

        bundle = self._run_bundle(
            "replay-run",
            llm_enabled=True,
            threshold=0.0025,
            min_post_count=1,
            total_return=0.12,
            robust_score=1.8,
            features=["sentiment_avg", "semantic_market_relevance_avg"],
        )
        config = _bundle_to_run_config(bundle)
        self.assertTrue(config.llm_enabled)
        self.assertEqual(config.run_name, "replay-run")
        self.assertEqual(config.threshold_grid, (0.0, 0.001, 0.0025))

        replay_row = pd.Series(
            {
                "expected_return_score": 0.012,
                "prediction_confidence": 0.61,
                "deployment_threshold": 0.0025,
                "deployment_min_post_count": 1,
                "training_rows_used": 55,
                "target_next_session_return": 0.009,
            },
        )
        full_history_row = pd.Series(
            {
                "expected_return_score": 0.02,
                "prediction_confidence": 0.7,
            },
        )
        summary = _build_replay_comparison_frame(replay_row, full_history_row)
        self.assertIn("Replay vs full-history drift", summary["metric"].tolist())
        drift_value = float(summary.loc[summary["metric"] == "Replay vs full-history drift", "value"].iloc[0])
        self.assertAlmostEqual(drift_value, -0.008)


if __name__ == "__main__":
    unittest.main()
