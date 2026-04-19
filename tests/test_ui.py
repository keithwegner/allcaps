from __future__ import annotations

import inspect
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
    _bundle_feature_names,
    _build_feature_diff_table,
    _build_feature_family_summary,
    _build_metric_comparison_table,
    _build_narrative_lift_table,
    _build_setting_diff_table,
    _bundle_to_run_config,
    _eligible_replay_sessions,
    _latest_ranking_snapshot,
    _replay_option_label,
    _save_watchlist,
    _source_mode,
    _source_mode_label,
    _summarize_run_changes,
    _variant_summary_with_narrative_defaults,
    _watchlist_symbols,
    _watchlist_text_value,
    render_datasets_view,
    render_discovery_view,
    render_models_view,
    render_research_view,
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
        target_asset: str = "SPY",
        llm_enabled: bool,
        threshold: float,
        min_post_count: int,
        total_return: float,
        robust_score: float,
        features: list[str],
    ) -> dict[str, object]:
        return {
            "run": {"run_id": run_id, "run_name": run_id, "target_asset": target_asset},
            "config": {
                "target_asset": target_asset,
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
                metadata={"llm_enabled": llm_enabled, "target_asset": target_asset},
            ),
            "benchmarks": pd.DataFrame(
                [
                    {"benchmark_name": "strategy", "target_asset": target_asset, "total_return": total_return, "robust_score": robust_score},
                    {
                        "benchmark_name": f"always_long_{target_asset.lower()}",
                        "target_asset": target_asset,
                        "total_return": total_return - 0.02,
                        "robust_score": robust_score - 0.1,
                    },
                    {"benchmark_name": "trump_only", "target_asset": target_asset, "total_return": total_return - 0.01, "robust_score": robust_score - 0.05},
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

    def test_source_mode_detects_empty_truth_only_and_mixed_sources(self) -> None:
        empty_mode = _source_mode(pd.DataFrame())
        self.assertEqual(empty_mode["mode"], "unknown")
        self.assertEqual(_source_mode_label(empty_mode), "No source data")

        truth_only = _source_mode(pd.DataFrame({"source_platform": ["Truth Social", "Truth Social"]}))
        self.assertEqual(truth_only["mode"], "truth_only")
        self.assertTrue(truth_only["has_truth_posts"])
        self.assertFalse(truth_only["has_x_posts"])
        self.assertEqual(truth_only["truth_post_count"], 2)
        self.assertEqual(truth_only["x_post_count"], 0)
        self.assertEqual(_source_mode_label(truth_only), "Truth Social-only")

        mixed = _source_mode(pd.DataFrame({"source_platform": ["Truth Social", "X", "X"]}))
        self.assertEqual(mixed["mode"], "truth_plus_x")
        self.assertTrue(mixed["has_truth_posts"])
        self.assertTrue(mixed["has_x_posts"])
        self.assertEqual(mixed["truth_post_count"], 1)
        self.assertEqual(mixed["x_post_count"], 2)
        self.assertEqual(_source_mode_label(mixed), "Truth Social + X mentions")

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
                target_asset="SPY",
                llm_enabled=False,
                threshold=0.001,
                min_post_count=2,
                total_return=0.10,
                robust_score=1.5,
                features=["sentiment_avg", "post_count", "prev_return_1d"],
            ),
            "alt-run": self._run_bundle(
                "alt-run",
                target_asset="QQQ",
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
        self.assertEqual(metric_table.iloc[0]["target_asset"], "QQQ")
        self.assertAlmostEqual(float(metric_table.iloc[0]["delta_robust_score_vs_base"]), 0.4)
        self.assertIn("deploy_threshold", setting_diff["setting"].tolist())
        self.assertIn("llm_enabled", setting_diff["setting"].tolist())
        self.assertIn("target_asset", setting_diff["setting"].tolist())
        alt_feature_row = feature_diff.loc[feature_diff["run_id"] == "alt-run"].iloc[0]
        self.assertEqual(alt_feature_row["target_asset"], "QQQ")
        self.assertEqual(int(alt_feature_row["unique_vs_base_count"]), 1)
        self.assertIn("semantic_market_relevance_avg", str(alt_feature_row["unique_vs_base"]))
        strategy_row = benchmark_diff.loc[
            (benchmark_diff["run_id"] == "alt-run") & (benchmark_diff["benchmark_name"] == "strategy")
        ].iloc[0]
        self.assertEqual(strategy_row["target_asset"], "QQQ")
        self.assertAlmostEqual(float(strategy_row["delta_total_return_vs_base"]), 0.04)
        self.assertEqual(len(notes), 1)
        self.assertIn("target asset SPY -> QQQ", notes[0])
        self.assertIn("LLM on vs off", notes[0])
        self.assertIn("threshold 0.001 -> 0.0025", notes[0])

    def test_bundle_feature_names_handles_joint_portfolio_bundle(self) -> None:
        bundle = {
            "run": {
                "run_id": "portfolio-run",
                "run_name": "portfolio-run",
                "run_type": "portfolio_allocator",
                "allocator_mode": "joint_model",
                "target_asset": "PORTFOLIO",
            },
            "config": {
                "allocator_mode": "joint_model",
                "selected_symbols": ["SPY", "QQQ"],
                "topology_variants": ["per_asset", "pooled"],
                "model_families": ["ridge"],
            },
            "selected_params": {
                "deployment_variant": "pooled",
            },
            "metrics": {"total_return": 0.12, "sharpe": 1.1, "sortino": 1.2, "max_drawdown": -0.1, "robust_score": 1.5, "trade_count": 10},
            "model_artifact": LinearModelArtifact(
                model_version="portfolio-placeholder",
                feature_names=[],
            ),
            "portfolio_model_bundle": {
                "deployment_variant": "pooled",
                "variants": {
                    "pooled": {
                        "models": {
                            "pooled": {
                                "feature_names": ["post_count", "asset_indicator__spy", "sentiment_avg"],
                            },
                        },
                    },
                },
            },
            "benchmarks": pd.DataFrame(),
        }

        self.assertEqual(
            _bundle_feature_names(bundle),
            ["asset_indicator__spy", "post_count", "sentiment_avg"],
        )

    def test_narrative_variant_helpers_build_lift_and_family_summary(self) -> None:
        variant_summary = pd.DataFrame(
            {
                "variant_name": ["per_asset_baseline", "per_asset_hybrid", "pooled_baseline"],
                "topology": ["per_asset", "per_asset", "pooled"],
                "narrative_feature_mode": ["baseline", "hybrid", "baseline"],
                "validation_robust_score": [1.0, 1.25, 0.9],
                "validation_total_return": [0.08, 0.1, 0.07],
                "test_robust_score": [0.8, 0.95, 0.7],
                "test_total_return": [0.06, 0.09, 0.05],
            },
        )
        lift = _build_narrative_lift_table(variant_summary)

        self.assertEqual(lift.iloc[0]["variant_name"], "per_asset_hybrid")
        self.assertAlmostEqual(float(lift.iloc[0]["validation_robust_lift"]), 0.25)
        self.assertAlmostEqual(float(lift.iloc[0]["test_return_lift"]), 0.03)

        legacy_summary = _variant_summary_with_narrative_defaults(pd.DataFrame({"variant_name": ["per_asset"]}))
        self.assertEqual(legacy_summary.iloc[0]["narrative_feature_mode"], "unspecified")

        bundle = {
            "run": {
                "run_id": "portfolio-run",
                "run_name": "portfolio-run",
                "run_type": "portfolio_allocator",
                "allocator_mode": "joint_model",
                "target_asset": "PORTFOLIO",
            },
            "config": {},
            "selected_params": {"deployment_variant": "per_asset_hybrid"},
            "model_artifact": LinearModelArtifact(model_version="portfolio-placeholder", feature_names=[]),
            "portfolio_model_bundle": {
                "deployment_variant": "per_asset_hybrid",
                "variants": {
                    "per_asset_hybrid": {
                        "models": {
                            "SPY": {
                                "feature_names": [
                                    "post_count",
                                    "sentiment_avg",
                                    "semantic_market_relevance_avg",
                                    "policy_trade",
                                ],
                            },
                        },
                    },
                },
            },
        }
        importance = pd.DataFrame(
            {
                "variant_name": ["per_asset_hybrid", "per_asset_hybrid", "per_asset_hybrid"],
                "feature_name": ["semantic_market_relevance_avg", "policy_trade", "post_count"],
                "abs_coefficient": [0.5, 0.2, 0.1],
            },
        )
        family_summary = _build_feature_family_summary(bundle, variant_name="per_asset_hybrid", importance=importance)
        self.assertEqual(family_summary.iloc[0]["feature_family"], "semantic")
        self.assertIn("policy", family_summary["feature_family"].tolist())

    def test_joint_portfolio_controls_have_unique_widget_keys(self) -> None:
        source = inspect.getsource(render_models_view)
        required_keys = [
            "joint-portfolio-run-name",
            "joint-portfolio-feature-version",
            "joint-portfolio-llm-enabled",
            "joint-portfolio-narrative-feature-modes",
            "joint-portfolio-train-window",
            "joint-portfolio-validation-window",
            "joint-portfolio-test-window",
            "joint-portfolio-step-size",
            "joint-portfolio-threshold-grid",
            "joint-portfolio-minimum-grid",
            "joint-portfolio-account-weight-grid",
            "joint-portfolio-topology-variants",
            "joint-portfolio-model-families",
        ]

        for widget_key in required_keys:
            self.assertIn(widget_key, source)

    def test_research_view_exposes_trump_authored_filter(self) -> None:
        source = inspect.getsource(render_research_view)

        self.assertIn("Trump-authored only", source)
        self.assertIn("research_trump_authored_only", source)
        self.assertIn("trump_authored_only=trump_authored_only", source)

    def test_research_view_seeds_truth_only_defaults(self) -> None:
        source = inspect.getsource(render_research_view)

        self.assertIn("research_source_mode_seeded", source)
        self.assertIn("research_platforms", source)
        self.assertIn("Truth Social-only mode detected", source)
        self.assertIn('source_mode_name == "truth_only"', source)

    def test_research_view_exposes_export_download_for_filtered_slice(self) -> None:
        source = inspect.getsource(render_research_view)

        self.assertIn("Export current research pack", source)
        self.assertIn("st.download_button", source)
        self.assertIn("build_research_export_bundle", source)
        self.assertIn("sessions=session_table", source)
        self.assertIn("posts=post_table", source)

    def test_datasets_view_exposes_operating_mode(self) -> None:
        source = inspect.getsource(render_datasets_view)

        self.assertIn("Operating mode", source)
        self.assertIn("_source_mode_label(source_mode)", source)

    def test_discovery_view_explains_truth_only_empty_state(self) -> None:
        source = inspect.getsource(render_discovery_view)

        self.assertIn("This dataset is currently Truth Social-only", source)
        self.assertIn("Discovery ranks non-Trump X accounts that mention Trump", source)
        self.assertIn("Load X/mention CSVs", source)

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
            target_asset="QQQ",
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
        self.assertEqual(config.target_asset, "QQQ")
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
