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
                "2025-02-10 11:00-05:00",
            ],
            utc=True,
        ).tz_convert("America/New_York")
        self.posts = pd.DataFrame(
            {
                "source_platform": ["X"] * 5,
                "mentions_trump": [True] * 5,
                "author_is_trump": [False] * 5,
                "author_account_id": ["acct-a", "acct-a", "acct-b", "acct-c", "acct-b"],
                "author_handle": ["macroa", "macroa", "policyb", "newc", "policyb"],
                "author_display_name": ["Macro A", "Macro A", "Policy B", "New C", "Policy B"],
                "post_id": ["1", "2", "3", "4", "5"],
                "post_timestamp": timestamps,
                "engagement_score": [100.0, 80.0, 30.0, 10.0, 90.0],
                "sentiment_score": [0.4, 0.2, -0.1, 0.1, 0.5],
            },
        )
        self.service = DiscoveryService()

    def test_rank_candidates_respects_as_of_cutoff(self) -> None:
        ranked = self.service.rank_candidates(self.posts, as_of=pd.Timestamp("2025-02-05 23:59", tz="America/New_York"))
        self.assertNotIn("5", ranked["mention_count"].astype(str).tolist())
        self.assertEqual(ranked.iloc[0]["author_account_id"], "acct-a")

    def test_refresh_accounts_builds_historical_versions_with_overrides(self) -> None:
        overrides = self.service.add_manual_override(
            overrides=pd.DataFrame(),
            account_id="acct-c",
            handle="newc",
            display_name="New C",
            action="pin",
            effective_from=pd.Timestamp("2025-02-05"),
        )
        tracked, ranking_history = self.service.refresh_accounts(
            self.posts,
            pd.DataFrame(),
            as_of=self.posts["post_timestamp"].max(),
            auto_include_top_n=1,
            manual_overrides=overrides,
        )
        active = self.service.current_active_accounts(tracked, as_of=pd.Timestamp("2025-02-05"))
        self.assertIn("acct-c", active["account_id"].astype(str).tolist())
        self.assertTrue((ranking_history["selected_status"] == "pinned").any())

    def test_suppress_override_removes_account_from_current_universe(self) -> None:
        overrides = self.service.add_manual_override(
            overrides=pd.DataFrame(),
            account_id="acct-a",
            handle="macroa",
            display_name="Macro A",
            action="suppress",
            effective_from=pd.Timestamp("2025-02-04"),
        )
        tracked, _ = self.service.refresh_accounts(
            self.posts,
            pd.DataFrame(),
            as_of=self.posts["post_timestamp"].max(),
            auto_include_top_n=2,
            manual_overrides=overrides,
        )
        active = self.service.current_active_accounts(tracked, as_of=pd.Timestamp("2025-02-05"))
        self.assertNotIn("acct-a", active["account_id"].astype(str).tolist())


if __name__ == "__main__":
    unittest.main()
