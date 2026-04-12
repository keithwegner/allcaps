from __future__ import annotations

import unittest

import pandas as pd

from trump_workbench.contracts import LinearModelArtifact
from trump_workbench.explanations import build_account_attribution, build_post_attribution
from trump_workbench.modeling import ModelService


class ExplainabilityTests(unittest.TestCase):
    def test_model_service_builds_feature_contributions(self) -> None:
        artifact = LinearModelArtifact(
            model_version="linear-test",
            feature_names=["sentiment_avg", "prev_return_1d", "tracked_account_post_count"],
            intercept=0.05,
            coefficients=[0.4, -0.2, 0.1],
            means=[0.0, 0.0, 0.0],
            stds=[1.0, 1.0, 1.0],
            residual_std=0.2,
            train_rows=25,
            metadata={"llm_enabled": False},
        )
        rows = pd.DataFrame(
            {
                "signal_session_date": [pd.Timestamp("2025-02-03")],
                "next_session_date": [pd.Timestamp("2025-02-04")],
                "sentiment_avg": [0.5],
                "prev_return_1d": [-0.1],
                "tracked_account_post_count": [2.0],
            },
        )

        explanation = ModelService().explain_predictions(artifact, rows)

        self.assertEqual(len(explanation), 3)
        self.assertAlmostEqual(explanation["contribution"].sum(), 0.42, places=6)
        self.assertSetEqual(set(explanation["feature_family"]), {"social_sentiment", "market_context", "account_structure"})
        top_feature = explanation.sort_values("abs_contribution", ascending=False).iloc[0]
        self.assertEqual(top_feature["feature_name"], "sentiment_avg")
        self.assertAlmostEqual(float(top_feature["contribution"]), 0.2, places=6)

    def test_post_and_account_attribution_rank_session_inputs(self) -> None:
        mapped_posts = pd.DataFrame(
            {
                "session_date": [pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-04")],
                "post_timestamp": [
                    pd.Timestamp("2025-02-03 10:00:00", tz="America/New_York"),
                    pd.Timestamp("2025-02-03 11:00:00", tz="America/New_York"),
                    pd.Timestamp("2025-02-04 09:45:00", tz="America/New_York"),
                ],
                "author_account_id": ["acct-trump", "acct-a", "acct-a"],
                "author_handle": ["realdonaldtrump", "macroa", "macroa"],
                "author_display_name": ["Donald Trump", "Macro A", "Macro A"],
                "source_platform": ["Truth Social", "X", "X"],
                "author_is_trump": [True, False, False],
                "is_active_tracked_account": [False, True, True],
                "tracked_account_status": ["none", "active", "active"],
                "tracked_discovery_score": [0.0, 2.0, 2.0],
                "mentions_trump": [True, True, True],
                "sentiment_score": [0.8, 0.2, -0.6],
                "engagement_score": [150.0, 20.0, 10.0],
                "cleaned_text": [
                    "Markets love the announcement.",
                    "Trump mention from a tracked macro account.",
                    "Negative follow-up mention.",
                ],
                "post_url": [
                    "https://truthsocial.com/post/1",
                    "https://x.com/macroa/status/1",
                    "https://x.com/macroa/status/2",
                ],
            },
        )

        post_attribution = build_post_attribution(mapped_posts)
        account_attribution = build_account_attribution(post_attribution)

        self.assertEqual(len(post_attribution), 3)
        self.assertEqual(post_attribution.iloc[0]["author_handle"], "realdonaldtrump")
        self.assertGreater(post_attribution.iloc[0]["post_signal_score"], post_attribution.iloc[1]["post_signal_score"])

        session_accounts = account_attribution.loc[account_attribution["signal_session_date"] == pd.Timestamp("2025-02-03")]
        self.assertEqual(len(session_accounts), 2)
        macro_row = session_accounts.loc[session_accounts["author_handle"] == "macroa"].iloc[0]
        self.assertEqual(int(macro_row["post_count"]), 1)
        self.assertTrue(bool(macro_row["is_active_tracked_account"]))


if __name__ == "__main__":
    unittest.main()
