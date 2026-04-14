from __future__ import annotations

import unittest

import pandas as pd

from trump_workbench.research import (
    _format_post_summary,
    aggregate_research_sessions,
    build_asset_comparison_chart,
    build_asset_comparison_frame,
    build_combined_chart,
    build_event_frame,
    build_intraday_chart,
    filter_posts,
    get_intraday_window,
    make_asset_mapping_table,
    make_asset_session_table,
    make_post_table,
    make_session_table,
)


class ResearchTests(unittest.TestCase):
    def setUp(self) -> None:
        timestamps = pd.to_datetime(
            [
                "2025-02-03 08:00-05:00",
                "2025-02-03 13:15-05:00",
                "2025-02-04 10:00-05:00",
            ],
            utc=True,
        ).tz_convert("America/New_York")
        self.posts = pd.DataFrame(
            {
                "source_platform": ["Truth Social", "X", "X"],
                "author_handle": ["realDonaldTrump", "macroalpha", "resharer"],
                "author_display_name": ["Donald Trump", "Macro Alpha", "Re Sharer"],
                "post_timestamp": timestamps,
                "cleaned_text": ["Great growth", "Trump tariff concern", "RT Trump update"],
                "is_reshare": [False, False, True],
                "author_is_trump": [True, False, False],
                "is_active_tracked_account": [False, True, False],
                "session_date": pd.to_datetime(["2025-02-03", "2025-02-03", "2025-02-04"]),
                "mapping_reason": ["pre-market", "regular", "regular"],
                "mentions_trump": [False, True, True],
                "sentiment_score": [0.8, -0.2, 0.1],
                "sentiment_label": ["positive", "negative", "positive"],
                "post_url": ["https://truth/1", "https://x.com/macro/status/1", "https://x.com/re/status/2"],
            },
        )
        self.market = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2025-02-03", "2025-02-04", "2025-02-05"]),
                "close": [100.0, 101.0, 102.0],
            },
        )
        self.asset_market = pd.DataFrame(
            {
                "symbol": ["SPY", "SPY", "SPY", "NVDA", "NVDA", "NVDA"],
                "trade_date": pd.to_datetime(
                    ["2025-02-03", "2025-02-04", "2025-02-05", "2025-02-03", "2025-02-04", "2025-02-05"],
                ),
                "close": [100.0, 101.0, 102.0, 200.0, 210.0, 205.0],
            },
        )
        self.asset_session_features = pd.DataFrame(
            {
                "asset_symbol": ["NVDA", "NVDA"],
                "signal_session_date": pd.to_datetime(["2025-02-03", "2025-02-04"]),
                "post_count": [2, 1],
                "rule_matched_post_count": [2, 1],
                "semantic_matched_post_count": [1, 1],
                "primary_match_post_count": [2, 1],
                "asset_relevance_score_avg": [0.72, 0.61],
                "sentiment_avg": [0.3, -0.1],
                "target_next_session_return": [0.05, -0.01],
                "target_available": [True, True],
            },
        )
        self.asset_post_mappings = pd.DataFrame(
            {
                "asset_symbol": ["NVDA", "NVDA"],
                "session_date": pd.to_datetime(["2025-02-03", "2025-02-04"]),
                "post_timestamp": pd.to_datetime(
                    ["2025-02-03 10:00-05:00", "2025-02-04 11:00-05:00"],
                    utc=True,
                ).tz_convert("America/New_York"),
                "author_handle": ["macroalpha", "macrobeta"],
                "author_display_name": ["Macro Alpha", "Macro Beta"],
                "asset_relevance_score": [0.8, 0.55],
                "rule_match_score": [0.6, 0.4],
                "semantic_match_score": [0.2, 0.15],
                "match_rank": [1, 1],
                "is_primary_asset": [True, True],
                "match_reasons": ["rule:nvidia, topic:markets", "rule:semiconductor, policy:economy"],
                "cleaned_text": ["NVIDIA keeps rallying after strong AI demand.", "Semiconductor policy talk is picking up."],
            },
        )

    def test_filter_posts_applies_platform_keyword_and_tracking_filters(self) -> None:
        self.assertTrue(
            filter_posts(
                pd.DataFrame(),
                pd.Timestamp("2025-02-03"),
                pd.Timestamp("2025-02-05"),
                include_reshares=False,
                platforms=["X"],
                keyword="Trump",
                tracked_only=True,
            ).empty,
        )

        filtered = filter_posts(
            self.posts,
            pd.Timestamp("2025-02-03"),
            pd.Timestamp("2025-02-04"),
            include_reshares=False,
            platforms=["X"],
            keyword="tariff",
            tracked_only=True,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["author_handle"], "macroalpha")

    def test_research_aggregation_tables_and_charts(self) -> None:
        summary = _format_post_summary(self.posts.iloc[0])
        self.assertIn("@realDonaldTrump", summary)

        sessions = aggregate_research_sessions(self.posts)
        self.assertEqual(sessions.iloc[0]["post_count"], 2)
        self.assertIn("@macroalpha", sessions.iloc[0]["sample_posts"])

        events = build_event_frame(self.market, sessions)
        self.assertTrue(bool(events.iloc[0]["has_posts"]))
        self.assertEqual(int(events.iloc[-1]["post_count"]), 0)

        chart = build_combined_chart(events, scale_markers=True)
        self.assertGreaterEqual(len(chart.data), 2)
        self.assertEqual(chart.layout.title.text, "Research View: social activity vs. market baseline")

        session_table = make_session_table(events)
        self.assertIn("sp500_close", session_table.columns)
        self.assertIn("session_return", session_table.columns)

        post_table = make_post_table(self.posts)
        self.assertIn("post_text", post_table.columns)
        self.assertEqual(post_table.iloc[0]["session_date"].isoformat(), "2025-02-03")

    def test_asset_comparison_helpers_and_tables(self) -> None:
        comparison = build_asset_comparison_frame(
            self.asset_market,
            "NVDA",
            date_start=pd.Timestamp("2025-02-03"),
            date_end=pd.Timestamp("2025-02-05"),
        )
        self.assertEqual(len(comparison), 3)
        self.assertAlmostEqual(float(comparison.iloc[-1]["spy_normalized_return"]), 0.02)
        self.assertAlmostEqual(float(comparison.iloc[-1]["asset_normalized_return"]), 0.025)

        price_chart = build_asset_comparison_chart(comparison, "NVDA", mode="price")
        normalized_chart = build_asset_comparison_chart(comparison, "NVDA", mode="normalized")
        self.assertEqual(price_chart.layout.title.text, "SPY vs. NVDA price overlay")
        self.assertEqual(normalized_chart.layout.title.text, "SPY vs. NVDA normalized returns")
        self.assertEqual(len(price_chart.data), 2)
        self.assertEqual(len(normalized_chart.data), 2)

        asset_session_table = make_asset_session_table(self.asset_session_features, "NVDA")
        self.assertIn("trade_date", asset_session_table.columns)
        self.assertIn("next_session_return", asset_session_table.columns)
        self.assertEqual(len(asset_session_table), 2)

        asset_mapping_table = make_asset_mapping_table(self.asset_post_mappings, "NVDA")
        self.assertIn("post_text", asset_mapping_table.columns)
        self.assertIn("match_reasons", asset_mapping_table.columns)
        self.assertEqual(asset_mapping_table.iloc[0]["asset_symbol"], "NVDA")

    def test_intraday_helpers(self) -> None:
        intraday = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2025-02-03 09:25-05:00",
                        "2025-02-03 09:30-05:00",
                        "2025-02-03 09:35-05:00",
                    ],
                    utc=True,
                ).tz_convert("America/New_York"),
                "close": [100.0, 100.5, 101.0],
            },
        )
        anchor = pd.Timestamp("2025-02-03 09:30:00", tz="America/New_York")

        window = get_intraday_window(intraday, anchor, before_minutes=5, after_minutes=5)
        chart = build_intraday_chart(window, anchor, "Intraday")

        self.assertEqual(len(window), 3)
        self.assertEqual(chart.layout.title.text, "Intraday")


if __name__ == "__main__":
    unittest.main()
