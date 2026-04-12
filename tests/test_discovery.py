from __future__ import annotations

import unittest

import pandas as pd

from trump_workbench.contracts import RANKING_HISTORY_COLUMNS, TRACKED_ACCOUNT_COLUMNS
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

    def test_normalize_and_remove_manual_overrides(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "account_id": "acct-a",
                    "handle": "macroa",
                    "display_name": "Macro A",
                    "action": "PIN",
                    "effective_from": "2025-02-03",
                    "created_at": "2025-02-01T10:00:00Z",
                },
                {
                    "account_id": "acct-b",
                    "handle": "policyb",
                    "display_name": "Policy B",
                    "action": "ignore",
                    "effective_from": "2025-02-03",
                    "created_at": "2025-02-01T10:00:00Z",
                },
            ],
        )

        normalized = self.service.normalize_manual_overrides(raw)
        self.assertEqual(len(normalized), 1)
        self.assertTrue(bool(normalized.iloc[0]["override_id"]))

        remaining = self.service.remove_manual_override(normalized, normalized.iloc[0]["override_id"])
        self.assertTrue(remaining.empty)
        self.assertTrue(self.service.remove_manual_override(pd.DataFrame(), "missing").empty)
        self.assertTrue(self.service.overrides_in_effect(pd.DataFrame(), pd.Timestamp("2025-02-03")).empty)

    def test_refresh_accounts_handles_empty_posts_and_pin_without_history(self) -> None:
        overrides = self.service.add_manual_override(
            overrides=pd.DataFrame(),
            account_id="acct-z",
            handle="ghost",
            display_name="Ghost Account",
            action="pin",
            effective_from=pd.Timestamp("2025-02-07"),
        )

        tracked, ranking_history = self.service.refresh_accounts(
            posts=self.posts.iloc[0:0].copy(),
            existing_accounts=pd.DataFrame(),
            as_of=pd.Timestamp("2025-02-07", tz="America/New_York"),
            manual_overrides=overrides,
            auto_include_top_n=1,
        )

        self.assertEqual(tracked.iloc[0]["status"], "pinned")
        self.assertEqual(int(tracked.iloc[0]["mention_count"]), 0)
        self.assertEqual(int(ranking_history.iloc[0]["discovery_rank"]), -1)
        self.assertTrue(self.service.current_active_accounts(pd.DataFrame(), as_of=pd.Timestamp("2025-02-07")).empty)

    def test_rank_candidates_returns_empty_for_non_matching_inputs(self) -> None:
        self.assertTrue(self.service.rank_candidates(pd.DataFrame()).empty)

        non_candidates = self.posts.copy()
        non_candidates["mentions_trump"] = False
        self.assertTrue(self.service.rank_candidates(non_candidates).empty)

    def test_refresh_accounts_returns_stable_empty_schemas(self) -> None:
        tracked, ranking_history = self.service.refresh_accounts(
            posts=pd.DataFrame(),
            existing_accounts=pd.DataFrame(),
            as_of=pd.Timestamp("2025-02-07", tz="America/New_York"),
        )

        self.assertListEqual(list(tracked.columns), TRACKED_ACCOUNT_COLUMNS)
        self.assertListEqual(list(ranking_history.columns), RANKING_HISTORY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
