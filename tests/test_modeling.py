from __future__ import annotations

import unittest

import pandas as pd

from trump_workbench.modeling import ModelService


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


if __name__ == "__main__":
    unittest.main()
