from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.config import AppSettings
from trump_workbench.enrichment import LLMEnrichmentService
from trump_workbench.features import FeatureService, map_posts_to_trade_sessions
from trump_workbench.storage import DuckDBStore


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(settings)
        self.feature_service = FeatureService(LLMEnrichmentService(self.store))
        self.market = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2025-02-03", "2025-02-04", "2025-02-05"]),
                "open": [100.0, 101.0, 103.0],
                "high": [101.0, 104.0, 105.0],
                "low": [99.0, 100.0, 102.0],
                "close": [101.0, 103.0, 104.0],
                "volume": [1000.0, 1200.0, 1100.0],
            },
        )
        self.posts = pd.DataFrame(
            {
                "source_platform": ["Truth Social", "X"],
                "source_type": ["truth_archive", "x_csv"],
                "author_account_id": ["trump", "acct-a"],
                "author_handle": ["realDonaldTrump", "macroa"],
                "author_display_name": ["Donald Trump", "Macro A"],
                "author_is_trump": [True, False],
                "post_id": ["1", "2"],
                "post_url": ["", ""],
                "post_timestamp": pd.to_datetime(
                    ["2025-02-03 08:00-05:00", "2025-02-03 16:30-05:00"],
                    utc=True,
                ).tz_convert("America/New_York"),
                "raw_text": ["great growth", "Trump tariff concern"],
                "cleaned_text": ["great growth", "Trump tariff concern"],
                "is_reshare": [False, False],
                "has_media": [False, False],
                "replies_count": [0, 0],
                "reblogs_count": [0, 0],
                "favourites_count": [0, 0],
                "mentions_trump": [False, True],
                "source_provenance": ["truth", "x"],
                "engagement_score": [0.0, 10.0],
                "sentiment_score": [0.8, -0.3],
                "sentiment_label": ["positive", "negative"],
            },
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_map_posts_to_trade_sessions_handles_after_close(self) -> None:
        mapped = map_posts_to_trade_sessions(self.posts, self.market[["trade_date"]])
        self.assertEqual(str(mapped.iloc[0]["session_date"].date()), "2025-02-03")
        self.assertEqual(str(mapped.iloc[1]["session_date"].date()), "2025-02-04")

    def test_build_session_dataset_aligns_next_session_target(self) -> None:
        feature_rows = self.feature_service.build_session_dataset(
            posts=self.posts,
            spy_market=self.market,
            tracked_accounts=pd.DataFrame(),
            feature_version="v1",
            llm_enabled=False,
        )
        first = feature_rows.iloc[0]
        expected = self.market.iloc[1]["close"] / self.market.iloc[1]["open"] - 1.0
        self.assertAlmostEqual(first["target_next_session_return"], expected)
        self.assertTrue(bool(feature_rows.loc[feature_rows["signal_session_date"] == pd.Timestamp("2025-02-04"), "feature_cutoff_before_next_open"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
