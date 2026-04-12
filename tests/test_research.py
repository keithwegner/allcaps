from __future__ import annotations

import unittest

import pandas as pd

from trump_workbench.research import (
    _format_post_summary,
    aggregate_research_sessions,
    build_combined_chart,
    build_event_frame,
    build_intraday_chart,
    filter_posts,
    get_intraday_window,
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
