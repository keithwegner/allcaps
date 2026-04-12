from __future__ import annotations

import unittest

import pandas as pd

from trump_workbench.contracts import RANKING_HISTORY_COLUMNS
from trump_workbench.ui import _latest_ranking_snapshot


class UiHelperTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
