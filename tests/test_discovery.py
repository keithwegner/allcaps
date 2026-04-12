from __future__ import annotations

import unittest

import pandas as pd

from trump_workbench.discovery import DiscoveryService


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        timestamps = pd.to_datetime(
            [
                "2025-02-03 10:00-05:00",
                "2025-02-03 14:00-05:00",
                "2025-02-04 11:00-05:00",
                "2025-02-05 11:00-05:00",
            ],
            utc=True,
        ).tz_convert("America/New_York")
        self.posts = pd.DataFrame(
            {
                "source_platform": ["X"] * 4,
                "mentions_trump": [True] * 4,
                "author_is_trump": [False] * 4,
                "author_account_id": ["acct-a", "acct-a", "acct-b", "acct-c"],
                "author_handle": ["macroa", "macroa", "policyb", "newc"],
                "author_display_name": ["Macro A", "Macro A", "Policy B", "New C"],
                "post_id": ["1", "2", "3", "4"],
                "post_timestamp": timestamps,
                "engagement_score": [100.0, 80.0, 30.0, 10.0],
                "sentiment_score": [0.4, 0.2, -0.1, 0.1],
            },
        )
        self.service = DiscoveryService()

    def test_rank_candidates_orders_by_discovery_score(self) -> None:
        ranked = self.service.rank_candidates(self.posts)
        self.assertEqual(ranked.iloc[0]["author_account_id"], "acct-a")

    def test_refresh_accounts_creates_and_deactivates_versions(self) -> None:
        tracked, _ = self.service.refresh_accounts(self.posts, pd.DataFrame(), as_of=self.posts["post_timestamp"].max(), auto_include_top_n=2)
        self.assertEqual(len(tracked), 2)
        reduced_posts = self.posts.loc[self.posts["author_account_id"].isin(["acct-b"])].copy()
        tracked2, _ = self.service.refresh_accounts(reduced_posts, tracked, as_of=pd.Timestamp("2025-02-10", tz="America/New_York"), auto_include_top_n=1)
        inactive = tracked2.loc[tracked2["status"] == "inactive"]
        self.assertFalse(inactive.empty)


if __name__ == "__main__":
    unittest.main()
