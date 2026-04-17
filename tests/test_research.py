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
    build_event_study_chart,
    build_event_study_frame,
    build_intraday_comparison_chart,
    build_intraday_comparison_frame,
    build_intraday_chart,
    build_narrative_asset_heatmap_chart,
    build_narrative_asset_heatmap_frame,
    build_narrative_frequency_chart,
    build_narrative_frequency_frame,
    build_narrative_return_chart,
    build_narrative_return_frame,
    filter_posts,
    filter_narrative_rows,
    get_intraday_window,
    make_asset_mapping_table,
    make_asset_session_table,
    make_narrative_event_table,
    make_narrative_post_table,
    make_post_table,
    make_session_table,
    narrative_urgency_band,
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
                "semantic_topic": ["markets", "trade", "campaign"],
                "semantic_policy_bucket": ["economy", "trade", "economy"],
                "semantic_stance": ["positive", "negative", "positive"],
                "semantic_market_relevance": [0.7, 0.9, 0.4],
                "semantic_urgency": [0.2, 0.8, 0.3],
                "semantic_primary_asset": ["SPY", "NVDA", "SPY"],
                "semantic_asset_targets": ["SPY,QQQ", "NVDA,XLE", "SPY"],
                "semantic_confidence": [0.71, 0.89, 0.5],
                "semantic_summary": [
                    "Markets narrative with broad-market focus.",
                    "Trade narrative with NVDA focus.",
                    "Campaign narrative with broad-market focus.",
                ],
                "semantic_provider": ["heuristic-cache", "heuristic-cache", "heuristic-cache"],
                "semantic_cache_hit": [True, False, True],
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
                "symbol": ["SPY", "SPY", "SPY", "NVDA", "NVDA", "NVDA", "QQQ", "QQQ", "QQQ"],
                "trade_date": pd.to_datetime(
                    [
                        "2025-02-03", "2025-02-04", "2025-02-05",
                        "2025-02-03", "2025-02-04", "2025-02-05",
                        "2025-02-03", "2025-02-04", "2025-02-05",
                    ],
                ),
                "close": [100.0, 101.0, 102.0, 200.0, 210.0, 205.0, 300.0, 306.0, 309.0],
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
                "post_id": ["nvda-1", "nvda-2"],
                "asset_symbol": ["NVDA", "NVDA"],
                "session_date": pd.to_datetime(["2025-02-03", "2025-02-04"]),
                "post_timestamp": pd.to_datetime(
                    ["2025-02-03 10:00-05:00", "2025-02-04 11:00-05:00"],
                    utc=True,
                ).tz_convert("America/New_York"),
                "author_handle": ["macroalpha", "macrobeta"],
                "author_display_name": ["Macro Alpha", "Macro Beta"],
                "reaction_anchor_ts": pd.to_datetime(
                    ["2025-02-03 10:00-05:00", "2025-02-04 11:00-05:00"],
                    utc=True,
                ).tz_convert("America/New_York"),
                "mapping_reason": ["during regular hours -> same session", "during regular hours -> same session"],
                "asset_relevance_score": [0.8, 0.55],
                "rule_match_score": [0.6, 0.4],
                "semantic_match_score": [0.2, 0.15],
                "semantic_topic": ["markets", "trade"],
                "semantic_policy_bucket": ["economy", "trade"],
                "semantic_stance": ["positive", "negative"],
                "semantic_market_relevance": [0.9, 0.8],
                "semantic_urgency": [0.3, 0.6],
                "semantic_primary_asset": ["NVDA", "NVDA"],
                "semantic_asset_targets": ["NVDA,SMH,QQQ", "NVDA,XLE"],
                "semantic_confidence": [0.84, 0.77],
                "semantic_summary": ["Markets narrative with NVDA focus.", "Trade narrative with NVDA focus."],
                "semantic_provider": ["heuristic-cache", "heuristic-cache"],
                "match_rank": [1, 1],
                "is_primary_asset": [True, True],
                "match_reasons": ["rule:nvidia, topic:markets", "rule:semiconductor, policy:economy"],
                "cleaned_text": ["NVIDIA keeps rallying after strong AI demand.", "Semiconductor policy talk is picking up."],
            },
        )
        self.asset_intraday = pd.DataFrame(
            {
                "symbol": ["SPY", "SPY", "SPY", "NVDA", "NVDA", "NVDA", "QQQ", "QQQ", "QQQ"],
                "timestamp": pd.to_datetime(
                    [
                        "2025-02-03 09:30-05:00", "2025-02-03 09:35-05:00", "2025-02-03 09:40-05:00",
                        "2025-02-03 09:30-05:00", "2025-02-03 09:35-05:00", "2025-02-03 09:40-05:00",
                        "2025-02-03 09:30-05:00", "2025-02-03 09:35-05:00", "2025-02-03 09:40-05:00",
                    ],
                    utc=True,
                ).tz_convert("America/New_York"),
                "close": [100.0, 100.4, 100.8, 200.0, 202.0, 204.0, 300.0, 301.5, 303.0],
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
                trump_authored_only=False,
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
            trump_authored_only=False,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["author_handle"], "macroalpha")

        trump_only = filter_posts(
            self.posts,
            pd.Timestamp("2025-02-03"),
            pd.Timestamp("2025-02-04"),
            include_reshares=True,
            platforms=["Truth Social", "X"],
            keyword="",
            tracked_only=False,
            trump_authored_only=True,
        )

        self.assertEqual(trump_only["author_handle"].tolist(), ["realDonaldTrump"])
        self.assertTrue(trump_only["author_is_trump"].astype(bool).all())

    def test_trump_authored_filter_composes_with_other_filters(self) -> None:
        x_only = filter_posts(
            self.posts,
            pd.Timestamp("2025-02-03"),
            pd.Timestamp("2025-02-04"),
            include_reshares=True,
            platforms=["X"],
            keyword="",
            tracked_only=False,
            trump_authored_only=True,
        )
        self.assertTrue(x_only.empty)

        tracked_and_trump_only = filter_posts(
            self.posts,
            pd.Timestamp("2025-02-03"),
            pd.Timestamp("2025-02-04"),
            include_reshares=True,
            platforms=["Truth Social", "X"],
            keyword="",
            tracked_only=True,
            trump_authored_only=True,
        )
        self.assertEqual(tracked_and_trump_only["author_handle"].tolist(), ["realDonaldTrump"])

        keyword_and_trump_only = filter_posts(
            self.posts,
            pd.Timestamp("2025-02-03"),
            pd.Timestamp("2025-02-04"),
            include_reshares=True,
            platforms=["Truth Social", "X"],
            keyword="growth",
            tracked_only=False,
            trump_authored_only=True,
        )
        self.assertEqual(keyword_and_trump_only["author_handle"].tolist(), ["realDonaldTrump"])

    def test_trump_authored_filter_handles_missing_author_flag(self) -> None:
        posts_without_author_flag = self.posts.drop(columns=["author_is_trump"])

        filtered = filter_posts(
            posts_without_author_flag,
            pd.Timestamp("2025-02-03"),
            pd.Timestamp("2025-02-04"),
            include_reshares=True,
            platforms=["Truth Social", "X"],
            keyword="",
            tracked_only=False,
            trump_authored_only=True,
        )

        self.assertTrue(filtered.empty)

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
            benchmark_symbol="QQQ",
            date_start=pd.Timestamp("2025-02-03"),
            date_end=pd.Timestamp("2025-02-05"),
        )
        self.assertEqual(len(comparison), 3)
        self.assertAlmostEqual(float(comparison.iloc[-1]["spy_normalized_return"]), 0.02)
        self.assertAlmostEqual(float(comparison.iloc[-1]["asset_normalized_return"]), 0.025)
        self.assertAlmostEqual(float(comparison.iloc[-1]["benchmark_normalized_return"]), 0.03)

        price_chart = build_asset_comparison_chart(comparison, "NVDA", mode="price")
        normalized_chart = build_asset_comparison_chart(comparison, "NVDA", mode="normalized")
        self.assertEqual(price_chart.layout.title.text, "SPY vs. NVDA price overlay")
        self.assertEqual(normalized_chart.layout.title.text, "SPY vs. NVDA normalized returns")
        self.assertEqual(len(price_chart.data), 3)
        self.assertEqual(len(normalized_chart.data), 3)

        asset_session_table = make_asset_session_table(self.asset_session_features, "NVDA")
        self.assertIn("trade_date", asset_session_table.columns)
        self.assertIn("next_session_return", asset_session_table.columns)
        self.assertEqual(len(asset_session_table), 2)

        asset_mapping_table = make_asset_mapping_table(self.asset_post_mappings, "NVDA")
        self.assertIn("post_text", asset_mapping_table.columns)
        self.assertIn("match_reasons", asset_mapping_table.columns)
        self.assertIn("reaction_anchor_et", asset_mapping_table.columns)
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

    def test_event_study_and_intraday_comparison_helpers(self) -> None:
        event_study = build_event_study_frame(
            asset_market=self.asset_market,
            asset_session_features=self.asset_session_features,
            selected_symbol="NVDA",
            benchmark_symbol="QQQ",
            pre_sessions=1,
            post_sessions=1,
        )
        self.assertEqual(sorted(event_study["symbol"].unique().tolist()), ["NVDA", "QQQ", "SPY"])
        self.assertEqual(sorted(event_study["relative_session"].unique().tolist()), [-1, 0, 1])
        event_chart = build_event_study_chart(event_study, "NVDA")
        self.assertEqual(event_chart.layout.title.text, "SPY vs. NVDA event study")

        anchor = pd.Timestamp("2025-02-03 09:35:00", tz="America/New_York")
        intraday_comparison = build_intraday_comparison_frame(
            intraday_frame=self.asset_intraday,
            selected_symbol="NVDA",
            anchor_ts=anchor,
            before_minutes=5,
            after_minutes=5,
            benchmark_symbol="QQQ",
        )
        self.assertEqual(sorted(intraday_comparison["symbol"].unique().tolist()), ["NVDA", "QQQ", "SPY"])
        spy_anchor = intraday_comparison.loc[
            (intraday_comparison["symbol"] == "SPY") & (intraday_comparison["timestamp"] == anchor),
            "normalized_return",
        ].iloc[0]
        self.assertAlmostEqual(float(spy_anchor), 0.0)
        intraday_chart = build_intraday_comparison_chart(intraday_comparison, "NVDA", anchor)
        self.assertEqual(intraday_chart.layout.title.text, "SPY vs. NVDA intraday reaction")

    def test_narrative_helpers_and_filters(self) -> None:
        self.assertEqual(narrative_urgency_band(0.1), "low")
        self.assertEqual(narrative_urgency_band(0.5), "medium")
        self.assertEqual(narrative_urgency_band(0.9), "high")

        filtered = filter_narrative_rows(
            self.posts,
            topic="trade",
            policy_bucket="trade",
            stance="negative",
            urgency_band="high",
            narrative_asset="NVDA",
            platforms=["X"],
            tracked_scope="Tracked accounts only",
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["author_handle"], "macroalpha")

        frequency = build_narrative_frequency_frame(self.posts)
        self.assertEqual(set(frequency["semantic_topic"].tolist()), {"campaign", "markets", "trade"})
        frequency_chart = build_narrative_frequency_chart(frequency)
        self.assertEqual(frequency_chart.layout.title.text, "Narrative frequency over time")

        returns = build_narrative_return_frame(self.posts, self.market, bucket_field="semantic_topic")
        self.assertIn("avg_next_session_return", returns.columns)
        return_chart = build_narrative_return_chart(returns, "semantic_topic")
        self.assertEqual(return_chart.layout.title.text, "Next-session return by narrative bucket")

        heatmap = build_narrative_asset_heatmap_frame(self.asset_post_mappings)
        self.assertEqual(sorted(heatmap["asset_symbol"].unique().tolist()), ["NVDA"])
        heatmap_chart = build_narrative_asset_heatmap_chart(heatmap)
        self.assertEqual(heatmap_chart.layout.title.text, "Asset-by-narrative heatmap")

        narrative_post_table = make_narrative_post_table(self.posts)
        self.assertIn("semantic_summary", narrative_post_table.columns)
        self.assertIn("semantic_cache_hit", narrative_post_table.columns)

        narrative_event_table = make_narrative_event_table(self.posts, self.market)
        self.assertIn("next_session_return", narrative_event_table.columns)
        self.assertIn("primary_topics", narrative_event_table.columns)


if __name__ == "__main__":
    unittest.main()
