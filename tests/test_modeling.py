from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trump_workbench.contracts import LinearModelArtifact, ModelRunConfig
from trump_workbench.modeling import (
    ASSET_INDICATOR_PREFIX,
    ModelService,
    add_asset_indicator_columns,
    classify_feature_family,
    normalize_narrative_feature_mode,
)


class ModelFeatureSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ModelService()
        self.frame = pd.DataFrame(
            {
                "signal_session_date": pd.date_range("2025-02-03", periods=3),
                "asset_symbol": ["SPY", "QQQ", "NVDA"],
                "target_next_session_return": [0.01, -0.002, 0.005],
                "post_count": [3, 4, 5],
                "sentiment_avg": [0.2, -0.1, 0.4],
                "session_return": [0.001, -0.002, 0.003],
                "semantic_market_relevance_avg": [0.7, 0.3, 0.9],
                "semantic_topic_trade": [1.0, 0.0, 1.0],
                "policy_trade": [1.0, 0.0, 0.0],
                "semantic_matched_post_count": [2, 0, 3],
                "primary_match_post_count": [1, 0, 2],
                "asset_relevance_score_avg": [0.8, 0.1, 0.6],
                "asset_semantic_match_score_avg": [0.7, 0.0, 0.5],
            },
        )

    def test_baseline_excludes_narrative_columns(self) -> None:
        features = self.service.select_feature_columns(self.frame, llm_enabled=True, feature_mode="baseline")

        self.assertIn("post_count", features)
        self.assertIn("sentiment_avg", features)
        self.assertNotIn("semantic_market_relevance_avg", features)
        self.assertNotIn("semantic_topic_trade", features)
        self.assertNotIn("policy_trade", features)
        self.assertNotIn("semantic_matched_post_count", features)
        self.assertNotIn("asset_semantic_match_score_avg", features)

    def test_narrative_only_uses_narrative_columns(self) -> None:
        features = self.service.select_feature_columns(self.frame, llm_enabled=True, feature_mode="narrative_only")

        self.assertNotIn("post_count", features)
        self.assertNotIn("sentiment_avg", features)
        self.assertIn("semantic_market_relevance_avg", features)
        self.assertIn("semantic_topic_trade", features)
        self.assertIn("policy_trade", features)
        self.assertIn("semantic_matched_post_count", features)
        self.assertIn("asset_semantic_match_score_avg", features)

    def test_hybrid_includes_baseline_and_narrative_columns(self) -> None:
        features = self.service.select_feature_columns(self.frame, llm_enabled=True, feature_mode="hybrid")

        self.assertIn("post_count", features)
        self.assertIn("sentiment_avg", features)
        self.assertIn("semantic_market_relevance_avg", features)
        self.assertIn("policy_trade", features)

    def test_missing_narrative_columns_degrade_safely(self) -> None:
        baseline_frame = self.frame[["signal_session_date", "target_next_session_return", "post_count", "sentiment_avg"]].copy()

        features = self.service.select_feature_columns(
            baseline_frame,
            llm_enabled=True,
            feature_mode="narrative_only",
        )

        self.assertEqual(features, [])


class ModelTrainingAndPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ModelService()
        dates = pd.date_range("2025-02-03", periods=24, freq="B")
        index = np.arange(len(dates), dtype=float)
        self.frame = pd.DataFrame(
            {
                "signal_session_date": dates,
                "next_session_date": dates + pd.offsets.BDay(1),
                "asset_symbol": ["SPY", "QQQ", "NVDA"] * 8,
                "target_next_session_return": (index - 11.5) / 2500.0,
                "post_count": (index % 5) + 1,
                "sentiment_avg": np.linspace(-0.45, 0.55, len(dates)),
                "session_return": np.linspace(-0.012, 0.014, len(dates)),
                "tracked_author_count": (index % 4) + 1,
                "engagement_score_avg": np.linspace(0.1, 1.2, len(dates)),
                "semantic_market_relevance_avg": np.linspace(0.2, 0.9, len(dates)),
                "policy_trade": [1.0, 0.0, 0.0] * 8,
            },
        )
        self.feature_columns = ["post_count", "sentiment_avg", "session_return", "tracked_author_count"]

    def test_feature_mode_and_family_helpers_cover_expected_branches(self) -> None:
        self.assertEqual(normalize_narrative_feature_mode("narrative-only"), "narrative_only")
        self.assertEqual(normalize_narrative_feature_mode("bad-mode"), "hybrid")
        self.assertEqual(classify_feature_family("policy_trade"), "policy")
        self.assertEqual(classify_feature_family("semantic_topic_trade"), "semantic")
        self.assertEqual(classify_feature_family(f"{ASSET_INDICATOR_PREFIX}spy"), "asset_identity")
        self.assertEqual(classify_feature_family("rolling_vol_5d"), "market_context")
        self.assertEqual(classify_feature_family("sentiment_avg"), "social_sentiment")
        self.assertEqual(classify_feature_family("engagement_score_avg"), "social_engagement")
        self.assertEqual(classify_feature_family("tracked_author_count"), "account_structure")
        self.assertEqual(classify_feature_family("mention_post_count"), "activity")
        self.assertEqual(classify_feature_family("unclassified_feature"), "other")

        with_indicators = add_asset_indicator_columns(self.frame[["asset_symbol"]].head(3), ["spy", "qqq"])

        self.assertEqual(with_indicators[f"{ASSET_INDICATOR_PREFIX}spy"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(with_indicators[f"{ASSET_INDICATOR_PREFIX}qqq"].tolist(), [0.0, 1.0, 0.0])

    def test_custom_train_predict_and_explain_round_trip(self) -> None:
        artifact, importance = self.service.train(
            ModelRunConfig(run_name="unit-model", llm_enabled=True, target_asset="SPY"),
            self.frame,
            "custom-v1",
        )

        predictions = self.service.predict(artifact, self.frame.head(3).drop(columns=["tracked_author_count"]))
        explanation = self.service.explain_predictions(artifact, predictions.head(1))

        self.assertEqual(artifact.model_family, "custom_linear")
        self.assertFalse(importance.empty)
        self.assertIn("expected_return_score", predictions.columns)
        self.assertIn("tracked_author_count", predictions.columns)
        self.assertFalse(explanation.empty)
        self.assertEqual(set(explanation["model_version"]), {"custom-v1"})

    def test_train_rejects_empty_targets_and_missing_features(self) -> None:
        no_targets = self.frame.copy()
        no_targets["target_next_session_return"] = pd.NA
        no_features = self.frame[["signal_session_date", "target_next_session_return"]].copy()

        with self.assertRaisesRegex(RuntimeError, "No trainable rows"):
            self.service.train(ModelRunConfig(run_name="empty"), no_targets, "empty-v1")
        with self.assertRaisesRegex(RuntimeError, "No numeric feature columns"):
            self.service.train(ModelRunConfig(run_name="no-features"), no_features, "no-features-v1")

    def test_sklearn_linear_families_fit_predict_and_explain(self) -> None:
        for family in ["ridge", "lasso", "elastic_net"]:
            with self.subTest(family=family):
                artifact, importance = self.service.train_with_family(
                    self.frame,
                    llm_enabled=True,
                    model_family=family,
                    model_version=f"{family}-v1",
                    metadata={"family": family},
                    feature_columns=self.feature_columns,
                    regularization=0.01,
                )
                predictions = self.service.predict(artifact, self.frame.head(4))
                explanation = self.service.explain_predictions(artifact, predictions.head(2))

                self.assertEqual(artifact.model_family, family)
                self.assertEqual(artifact.explanation_kind, "linear_exact")
                self.assertEqual(set(artifact.feature_names), set(self.feature_columns))
                self.assertFalse(importance.empty)
                self.assertTrue(np.isfinite(predictions["expected_return_score"]).all())
                self.assertFalse(explanation.empty)

    def test_tree_families_fit_with_serialized_estimators(self) -> None:
        for family, compute_importance in [
            ("random_forest_regressor", True),
            ("hist_gradient_boosting_regressor", False),
        ]:
            with self.subTest(family=family):
                artifact, importance = self.service.train_with_family(
                    self.frame,
                    llm_enabled=True,
                    model_family=family,
                    model_version=f"{family}-v1",
                    metadata={"family": family},
                    feature_columns=self.feature_columns,
                    compute_importance=compute_importance,
                )
                predictions = self.service.predict(artifact, self.frame.head(4))

                self.assertEqual(artifact.model_family, family)
                self.assertEqual(artifact.explanation_kind, "importance_proxy")
                self.assertTrue(artifact.serialized_estimator_b64)
                self.assertEqual(len(artifact.feature_importances), len(self.feature_columns))
                self.assertFalse(importance.empty)
                self.assertTrue(np.isfinite(predictions["expected_return_score"]).all())
                if family == "hist_gradient_boosting_regressor":
                    self.assertEqual(artifact.feature_importances, [0.0 for _ in self.feature_columns])

    def test_train_with_family_rejects_invalid_inputs(self) -> None:
        no_targets = self.frame.copy()
        no_targets["target_next_session_return"] = pd.NA
        no_features = self.frame[["signal_session_date", "target_next_session_return"]].copy()

        with self.assertRaisesRegex(RuntimeError, "No trainable rows"):
            self.service.train_with_family(
                no_targets,
                llm_enabled=True,
                model_family="ridge",
                model_version="empty-v1",
                metadata={},
            )
        with self.assertRaisesRegex(RuntimeError, "No numeric feature columns"):
            self.service.train_with_family(
                no_features,
                llm_enabled=True,
                model_family="ridge",
                model_version="no-features-v1",
                metadata={},
            )
        with self.assertRaisesRegex(RuntimeError, "Unsupported model family"):
            self.service.train_with_family(
                self.frame,
                llm_enabled=True,
                model_family="unsupported",
                model_version="bad-v1",
                metadata={},
                feature_columns=self.feature_columns,
            )

    def test_predict_empty_and_importance_proxy_explanations_are_safe(self) -> None:
        artifact = LinearModelArtifact(
            model_version="proxy-v1",
            feature_names=["post_count", "sentiment_avg"],
            means=[0.0],
            stds=[0.0],
            residual_std=0.2,
            model_family="random_forest_regressor",
            feature_importances=[0.75],
            explanation_kind="importance_proxy",
        )
        rows = pd.DataFrame(
            {
                "signal_session_date": pd.date_range("2025-03-03", periods=2),
                "next_session_date": pd.date_range("2025-03-04", periods=2),
                "model_version": ["proxy-v1", "proxy-v1"],
                "expected_return_score": [0.01, -0.002],
                "prediction_confidence": [0.8, 0.7],
                "post_count": [3.0, 0.0],
                "sentiment_avg": [0.2, -0.1],
            },
        )

        empty_predictions = self.service.predict(artifact, pd.DataFrame())
        explanation = self.service.explain_predictions(artifact, rows)

        self.assertEqual(list(empty_predictions.columns), ["expected_return_score", "prediction_confidence", "model_version"])
        self.assertEqual(len(explanation), 4)
        self.assertTrue(np.isfinite(explanation["contribution_share"]).all())
        self.assertEqual(set(explanation["feature_family"]), {"activity", "social_sentiment"})


if __name__ == "__main__":
    unittest.main()
