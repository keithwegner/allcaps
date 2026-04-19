from __future__ import annotations

import io
import json
import unittest
import zipfile

import pandas as pd
import plotly.graph_objects as go

from trump_workbench.research_exports import (
    EXPORT_TABLE_COLUMNS,
    build_research_export_bundle,
    build_research_export_manifest,
    research_export_filename,
)


class ResearchExportTests(unittest.TestCase):
    def test_research_export_bundle_contains_manifest_chart_summary_and_csvs(self) -> None:
        manifest = build_research_export_manifest(
            filters={
                "date_start": "2025-02-03",
                "date_end": "2025-02-05",
                "platforms": ["Truth Social"],
                "keyword": "",
                "include_reshares": False,
                "tracked_only": False,
                "trump_authored_only": True,
            },
            source_mode={
                "mode": "truth_only",
                "has_truth_posts": True,
                "has_x_posts": False,
                "truth_post_count": 2,
                "x_post_count": 0,
            },
            headline_metrics={
                "sessions_with_posts": 1,
                "posts_in_view": 1,
                "truth_posts": 1,
                "tracked_x_posts": 0,
                "mean_sentiment": 0.8,
                "sp500_change": 0.02,
            },
            generated_at=pd.Timestamp("2026-04-19 12:00:00", tz="UTC"),
        )
        chart = go.Figure()
        chart.update_layout(title="Research View: social activity vs. market baseline")
        sessions = pd.DataFrame(
            {
                "trade_date": [pd.Timestamp("2025-02-03").date()],
                "post_count": [1],
                "sp500_close": [100.0],
                "sentiment_avg": [0.8],
            },
        )
        posts = pd.DataFrame(
            {
                "source_platform": ["Truth Social"],
                "author_handle": ["realDonaldTrump"],
                "post_time_et": ["2025-02-03 08:00"],
                "session_date": [pd.Timestamp("2025-02-03").date()],
                "sentiment_score": [0.8],
                "post_text": ["Great growth"],
            },
        )

        bundle = build_research_export_bundle(
            manifest=manifest,
            chart=chart,
            sessions=sessions,
            posts=posts,
            narrative_frequency=pd.DataFrame(),
            narrative_returns=pd.DataFrame(),
            narrative_asset_heatmap=pd.DataFrame(),
            narrative_posts=pd.DataFrame(),
            narrative_events=pd.DataFrame(),
        )

        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {
                    "manifest.json",
                    "summary.md",
                    "social_activity_chart.html",
                    "sessions.csv",
                    "posts.csv",
                    "narrative_frequency.csv",
                    "narrative_returns.csv",
                    "narrative_asset_heatmap.csv",
                    "narrative_posts.csv",
                    "narrative_events.csv",
                },
            )
            manifest_payload = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest_payload["schema_version"], "research-export-v1")
            self.assertTrue(manifest_payload["filters"]["trump_authored_only"])
            self.assertEqual(manifest_payload["filters"]["platforms"], ["Truth Social"])
            self.assertEqual(manifest_payload["source_mode"]["mode"], "truth_only")
            self.assertEqual(manifest_payload["headline_metrics"]["posts_in_view"], 1)
            self.assertEqual(manifest_payload["files"]["sessions.csv"]["rows"], 1)
            self.assertEqual(
                manifest_payload["files"]["narrative_frequency.csv"]["columns"],
                EXPORT_TABLE_COLUMNS["narrative_frequency.csv"],
            )

            chart_html = archive.read("social_activity_chart.html").decode("utf-8")
            self.assertIn("Research View: social activity vs. market baseline", chart_html)
            summary = archive.read("summary.md").decode("utf-8")
            self.assertIn("Trump-authored only: True", summary)

            session_rows = pd.read_csv(io.BytesIO(archive.read("sessions.csv")))
            self.assertEqual(session_rows.iloc[0]["post_count"], 1)
            self.assertIn("sp500_close", session_rows.columns)
            narrative_frequency = pd.read_csv(io.BytesIO(archive.read("narrative_frequency.csv")))
            self.assertEqual(narrative_frequency.columns.tolist(), EXPORT_TABLE_COLUMNS["narrative_frequency.csv"])

    def test_research_export_filename_uses_date_range(self) -> None:
        self.assertEqual(
            research_export_filename(pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-05")),
            "research-pack-20250203-20250205.zip",
        )


if __name__ == "__main__":
    unittest.main()
