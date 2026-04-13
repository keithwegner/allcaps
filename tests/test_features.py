from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.config import AppSettings
from trump_workbench.enrichment import LLMEnrichmentService
from trump_workbench.features import (
    ASSET_POST_MAPPING_COLUMNS,
    FeatureService,
    latest_feature_preview,
    map_posts_to_trade_sessions,
    preview_post_texts,
)
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

        empty_session = feature_rows.loc[feature_rows["signal_session_date"] == pd.Timestamp("2025-02-05")].iloc[0]
        self.assertFalse(bool(empty_session["has_posts"]))
        self.assertEqual(empty_session["post_count"], 0)

    def test_map_posts_to_trade_sessions_handles_empty_weekend_and_end_of_range(self) -> None:
        empty_posts = self.posts.iloc[0:0].copy()
        empty_mapped = map_posts_to_trade_sessions(empty_posts, self.market[["trade_date"]])
        self.assertTrue(empty_mapped.empty)
        self.assertIn("mapping_reason", empty_mapped.columns)

        weekend_and_late = pd.DataFrame(
            {
                "post_timestamp": pd.to_datetime(
                    [
                        "2025-02-02 12:00-05:00",
                        "2025-02-05 18:00-05:00",
                    ],
                    utc=True,
                ).tz_convert("America/New_York"),
            },
        )
        mapped = map_posts_to_trade_sessions(weekend_and_late, self.market[["trade_date"]])
        self.assertEqual(mapped.iloc[0]["mapping_reason"], "weekend/holiday -> next session open")
        self.assertEqual(str(mapped.iloc[0]["session_date"].date()), "2025-02-03")
        self.assertEqual(len(mapped), 1)

    def test_build_session_dataset_returns_empty_for_empty_market(self) -> None:
        feature_rows = self.feature_service.build_session_dataset(
            posts=self.posts,
            spy_market=pd.DataFrame(),
            tracked_accounts=pd.DataFrame(),
            feature_version="v1",
            llm_enabled=False,
        )
        self.assertTrue(feature_rows.empty)

    def test_flag_tracked_posts_handles_suppressed_invalid_and_blank_accounts(self) -> None:
        mapped = map_posts_to_trade_sessions(self.posts, self.market[["trade_date"]])
        tracked_accounts = pd.DataFrame(
            [
                {
                    "account_id": "acct-a",
                    "effective_from": pd.Timestamp("2025-02-01"),
                    "effective_to": pd.Timestamp("2025-02-05"),
                    "discovery_score": 2.5,
                    "status": "suppressed",
                    "provenance": "manual_override:suppress",
                },
                {
                    "account_id": "",
                    "effective_from": pd.Timestamp("2025-02-01"),
                    "effective_to": pd.NaT,
                    "discovery_score": 5.0,
                    "status": "active",
                    "provenance": "bad-row",
                },
                {
                    "account_id": "acct-a",
                    "effective_from": pd.NaT,
                    "effective_to": pd.NaT,
                    "discovery_score": 4.0,
                    "status": "active",
                    "provenance": "bad-date",
                },
            ],
        )

        flagged = self.feature_service._flag_tracked_posts(mapped, tracked_accounts)
        x_row = flagged.loc[flagged["author_account_id"] == "acct-a"].iloc[0]

        self.assertFalse(bool(x_row["is_active_tracked_account"]))
        self.assertEqual(x_row["tracked_discovery_score"], 0.0)
        self.assertEqual(x_row["tracked_account_status"], "suppressed")

    def test_feature_preview_helpers(self) -> None:
        feature_rows = self.feature_service.build_session_dataset(
            posts=self.posts,
            spy_market=self.market,
            tracked_accounts=pd.DataFrame(),
            feature_version="v1",
            llm_enabled=False,
        )

        preview = latest_feature_preview(feature_rows)
        self.assertEqual(preview["post_count"], 0)
        self.assertIn("prev 1d", preview["market_context"])
        self.assertEqual(latest_feature_preview(pd.DataFrame()), {})

        text_preview = preview_post_texts(self.posts, max_items=2)
        self.assertIn("@realDonaldTrump", text_preview)
        self.assertEqual(preview_post_texts(pd.DataFrame()), "")

    def test_build_asset_post_mappings_supports_rule_and_semantic_matches(self) -> None:
        prepared_posts = pd.DataFrame(
            [
                {
                    "session_date": pd.Timestamp("2025-02-03"),
                    "post_id": "post-1",
                    "post_timestamp": pd.Timestamp("2025-02-03 10:00", tz="America/New_York"),
                    "author_account_id": "acct-1",
                    "author_handle": "macro1",
                    "author_display_name": "Macro One",
                    "author_is_trump": False,
                    "source_platform": "X",
                    "cleaned_text": "NVIDIA keeps winning the AI chip race.",
                    "mentions_trump": True,
                    "engagement_score": 15.0,
                    "sentiment_score": 0.7,
                    "sentiment_label": "positive",
                    "semantic_topic": "markets",
                    "semantic_policy_bucket": "economy",
                    "semantic_market_relevance": 0.9,
                    "semantic_urgency": 0.2,
                    "is_active_tracked_account": True,
                    "tracked_discovery_score": 4.2,
                    "tracked_account_status": "active",
                },
                {
                    "session_date": pd.Timestamp("2025-02-04"),
                    "post_id": "post-2",
                    "post_timestamp": pd.Timestamp("2025-02-04 11:00", tz="America/New_York"),
                    "author_account_id": "acct-2",
                    "author_handle": "macro2",
                    "author_display_name": "Macro Two",
                    "author_is_trump": False,
                    "source_platform": "X",
                    "cleaned_text": "Tariff pressure is rising after another Trump trade warning.",
                    "mentions_trump": True,
                    "engagement_score": 20.0,
                    "sentiment_score": -0.1,
                    "sentiment_label": "negative",
                    "semantic_topic": "trade",
                    "semantic_policy_bucket": "trade",
                    "semantic_market_relevance": 0.8,
                    "semantic_urgency": 0.4,
                    "is_active_tracked_account": False,
                    "tracked_discovery_score": 0.0,
                    "tracked_account_status": "none",
                },
            ],
        )
        asset_universe = pd.DataFrame(
            [
                {"symbol": "SPY", "display_name": "SPDR S&P 500 ETF Trust", "asset_type": "etf", "source": "core_etf"},
                {"symbol": "QQQ", "display_name": "Invesco QQQ Trust", "asset_type": "etf", "source": "core_etf"},
                {"symbol": "XLE", "display_name": "Energy Select Sector SPDR Fund", "asset_type": "etf", "source": "core_etf"},
                {"symbol": "SMH", "display_name": "VanEck Semiconductor ETF", "asset_type": "etf", "source": "core_etf"},
                {"symbol": "NVDA", "display_name": "NVIDIA Corporation", "asset_type": "equity", "source": "watchlist"},
            ],
        )

        mappings = self.feature_service.build_asset_post_mappings(prepared_posts, asset_universe, llm_enabled=True)

        nvda_post = mappings.loc[(mappings["post_id"] == "post-1") & (mappings["asset_symbol"] == "NVDA")].iloc[0]
        smh_post = mappings.loc[(mappings["post_id"] == "post-1") & (mappings["asset_symbol"] == "SMH")].iloc[0]
        xle_post = mappings.loc[(mappings["post_id"] == "post-2") & (mappings["asset_symbol"] == "XLE")].iloc[0]

        self.assertGreater(float(nvda_post["rule_match_score"]), 0.0)
        self.assertGreater(float(nvda_post["semantic_match_score"]), 0.0)
        self.assertIn("rule:nvidia", str(nvda_post["match_reasons"]).lower())
        self.assertEqual(int(nvda_post["match_rank"]), 1)
        self.assertTrue(bool(nvda_post["is_primary_asset"]))
        self.assertGreater(float(smh_post["rule_match_score"]), 0.0)
        self.assertEqual(float(xle_post["rule_match_score"]), 0.0)
        self.assertGreater(float(xle_post["semantic_match_score"]), 0.0)
        self.assertFalse(
            bool(
                (
                    (mappings["post_id"] == "post-2")
                    & (mappings["asset_symbol"] == "NVDA")
                ).any()
            ),
        )

    def test_build_asset_session_dataset_generates_rows_per_asset(self) -> None:
        asset_post_mappings = pd.DataFrame(
            [
                {
                    "asset_symbol": "NVDA",
                    "asset_display_name": "NVIDIA Corporation",
                    "asset_type": "equity",
                    "asset_source": "watchlist",
                    "session_date": pd.Timestamp("2025-02-03"),
                    "post_id": "post-1",
                    "post_timestamp": pd.Timestamp("2025-02-03 10:00", tz="America/New_York"),
                    "author_account_id": "acct-1",
                    "author_handle": "macro1",
                    "author_display_name": "Macro One",
                    "author_is_trump": False,
                    "source_platform": "X",
                    "cleaned_text": "NVIDIA keeps winning the AI chip race.",
                    "mentions_trump": True,
                    "engagement_score": 15.0,
                    "sentiment_score": 0.7,
                    "sentiment_label": "positive",
                    "semantic_topic": "markets",
                    "semantic_policy_bucket": "economy",
                    "semantic_market_relevance": 0.9,
                    "semantic_urgency": 0.2,
                    "is_active_tracked_account": True,
                    "tracked_discovery_score": 4.2,
                    "tracked_account_status": "active",
                    "rule_match_score": 0.6,
                    "semantic_match_score": 0.18,
                    "asset_relevance_score": 0.78,
                    "match_reasons": "rule:nvidia, topic:markets",
                    "match_rank": 1,
                    "is_primary_asset": True,
                },
                {
                    "asset_symbol": "SPY",
                    "asset_display_name": "SPDR S&P 500 ETF Trust",
                    "asset_type": "etf",
                    "asset_source": "core_etf",
                    "session_date": pd.Timestamp("2025-02-03"),
                    "post_id": "post-2",
                    "post_timestamp": pd.Timestamp("2025-02-03 11:00", tz="America/New_York"),
                    "author_account_id": "acct-2",
                    "author_handle": "macro2",
                    "author_display_name": "Macro Two",
                    "author_is_trump": False,
                    "source_platform": "X",
                    "cleaned_text": "Tariffs are starting to hit the broad market.",
                    "mentions_trump": True,
                    "engagement_score": 12.0,
                    "sentiment_score": -0.2,
                    "sentiment_label": "negative",
                    "semantic_topic": "trade",
                    "semantic_policy_bucket": "trade",
                    "semantic_market_relevance": 0.8,
                    "semantic_urgency": 0.3,
                    "is_active_tracked_account": False,
                    "tracked_discovery_score": 0.0,
                    "tracked_account_status": "none",
                    "rule_match_score": 0.6,
                    "semantic_match_score": 0.47,
                    "asset_relevance_score": 1.0,
                    "match_reasons": "rule:broad market, topic:trade, policy:trade",
                    "match_rank": 1,
                    "is_primary_asset": True,
                },
            ],
            columns=ASSET_POST_MAPPING_COLUMNS,
        )
        asset_market = pd.DataFrame(
            [
                {"symbol": "NVDA", "trade_date": pd.Timestamp("2025-02-03"), "open": 200.0, "high": 203.0, "low": 199.0, "close": 202.0, "volume": 1_000_000.0},
                {"symbol": "NVDA", "trade_date": pd.Timestamp("2025-02-04"), "open": 205.0, "high": 207.0, "low": 204.0, "close": 206.0, "volume": 1_100_000.0},
                {"symbol": "NVDA", "trade_date": pd.Timestamp("2025-02-05"), "open": 207.0, "high": 208.0, "low": 205.0, "close": 205.0, "volume": 1_050_000.0},
                {"symbol": "SPY", "trade_date": pd.Timestamp("2025-02-03"), "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 2_000_000.0},
                {"symbol": "SPY", "trade_date": pd.Timestamp("2025-02-04"), "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 2_100_000.0},
                {"symbol": "SPY", "trade_date": pd.Timestamp("2025-02-05"), "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 2_050_000.0},
            ],
        )
        asset_universe = pd.DataFrame(
            [
                {"symbol": "NVDA", "display_name": "NVIDIA Corporation", "asset_type": "equity", "source": "watchlist"},
                {"symbol": "SPY", "display_name": "SPDR S&P 500 ETF Trust", "asset_type": "etf", "source": "core_etf"},
            ],
        )

        dataset = self.feature_service.build_asset_session_dataset(
            asset_post_mappings=asset_post_mappings,
            asset_market=asset_market,
            feature_version="asset-v1",
            llm_enabled=True,
            asset_universe=asset_universe,
        )

        self.assertEqual(len(dataset), 6)
        nvda_row = dataset.loc[
            (dataset["asset_symbol"] == "NVDA") & (dataset["signal_session_date"] == pd.Timestamp("2025-02-03"))
        ].iloc[0]
        self.assertEqual(int(nvda_row["post_count"]), 1)
        self.assertEqual(int(nvda_row["rule_matched_post_count"]), 1)
        self.assertEqual(int(nvda_row["primary_match_post_count"]), 1)
        self.assertAlmostEqual(float(nvda_row["asset_relevance_score_avg"]), 0.78)
        self.assertAlmostEqual(float(nvda_row["target_next_session_return"]), 206.0 / 205.0 - 1.0)
        self.assertTrue(bool(nvda_row["tradeable"]))

        nvda_empty_row = dataset.loc[
            (dataset["asset_symbol"] == "NVDA") & (dataset["signal_session_date"] == pd.Timestamp("2025-02-05"))
        ].iloc[0]
        self.assertFalse(bool(nvda_empty_row["has_posts"]))
        self.assertEqual(int(nvda_empty_row["post_count"]), 0)
        self.assertFalse(bool(nvda_empty_row["tradeable"]))

    def test_build_asset_session_dataset_handles_empty_mappings_schema(self) -> None:
        asset_market = pd.DataFrame(
            [
                {"symbol": "SPY", "trade_date": pd.Timestamp("2025-02-03"), "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 2_000_000.0},
                {"symbol": "SPY", "trade_date": pd.Timestamp("2025-02-04"), "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 2_100_000.0},
            ],
        )

        dataset = self.feature_service.build_asset_session_dataset(
            asset_post_mappings=pd.DataFrame(),
            asset_market=asset_market,
            feature_version="asset-v1",
            llm_enabled=False,
        )

        self.assertEqual(len(dataset), 2)
        self.assertTrue((dataset["post_count"] == 0).all())
        self.assertFalse(bool(dataset.iloc[-1]["tradeable"]))


if __name__ == "__main__":
    unittest.main()
