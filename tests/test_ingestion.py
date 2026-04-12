from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from trump_workbench.config import AppSettings
from trump_workbench.contracts import NORMALIZED_POST_COLUMNS
from trump_workbench.ingestion import (
    HTTP_HEADERS,
    IngestionService,
    TruthSocialArchiveAdapter,
    XCsvAdapter,
    _ensure_normalized_schema,
    _request_text,
)


class _FakeResponse:
    def __init__(self, text: str = "", content: bytes | None = None, headers: dict[str, str] | None = None) -> None:
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None


class _DummyAdapter:
    def __init__(self, name: str, posts: pd.DataFrame) -> None:
        self.name = name
        self._posts = posts

    def fetch_history(self) -> tuple[pd.DataFrame, dict[str, object]]:
        return self._posts.copy(), {"source": self.name, "post_count": int(len(self._posts))}

    def fetch_since(self, last_cursor: pd.Timestamp | None) -> tuple[pd.DataFrame, dict[str, object]]:
        posts = self._posts.copy()
        if last_cursor is not None:
            posts = posts.loc[posts["post_timestamp"] > last_cursor].copy()
        return posts.reset_index(drop=True), {"source": self.name, "incremental": True, "post_count": int(len(posts))}


class IngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = AppSettings(base_dir=Path(self.temp_dir.name))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _normalized_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "source_platform": ["X"],
                "source_type": ["x_csv"],
                "author_account_id": ["acct-a"],
                "author_handle": ["macroa"],
                "author_display_name": ["Macro A"],
                "author_is_trump": [False],
                "post_id": ["1"],
                "post_url": ["https://x.com/macroa/status/1"],
                "post_timestamp": [pd.Timestamp("2025-02-03 15:00:00", tz="America/New_York")],
                "raw_text": ["Trump growth"],
                "cleaned_text": ["Trump growth"],
                "is_reshare": [False],
                "has_media": [False],
                "replies_count": [1],
                "reblogs_count": [2],
                "favourites_count": [3],
                "mentions_trump": [True],
                "source_provenance": ["unit-test"],
                "engagement_score": [6.0],
                "sentiment_score": [0.2],
                "sentiment_label": ["positive"],
            },
        )[NORMALIZED_POST_COLUMNS]

    def test_request_text_uses_expected_headers(self) -> None:
        response = _FakeResponse("hello", headers={"Last-Modified": "today"})
        with mock.patch("trump_workbench.ingestion.requests.get", return_value=response) as mocked_get:
            text, headers = _request_text("https://example.com/data.csv", timeout=12)

        self.assertEqual(text, "hello")
        self.assertEqual(headers["Last-Modified"], "today")
        mocked_get.assert_called_once_with(
            "https://example.com/data.csv",
            headers=HTTP_HEADERS,
            timeout=12,
        )

    def test_ensure_normalized_schema_fills_defaults_dedupes_and_filters(self) -> None:
        raw = pd.DataFrame(
            {
                "source_platform": ["X", "X", "X", "X"],
                "post_timestamp": [
                    "2025-02-03T10:00:00-05:00",
                    "2025-02-03T10:00:00-05:00",
                    "bad timestamp",
                    "2025-01-01T10:00:00-05:00",
                ],
                "cleaned_text": ["Alpha", "Alpha", "Broken", "Old"],
                "raw_text": ["Alpha", "Alpha", "Broken", "Old"],
                "author_handle": ["macro", "macro", "bad", "old"],
                "author_display_name": ["Macro", "Macro", "Bad", "Old"],
                "replies_count": [1, 2, 0, 0],
                "reblogs_count": [0, 1, 0, 0],
                "favourites_count": [0, 3, 0, 0],
                "post_url": ["", "", "", ""],
            },
        )

        normalized = _ensure_normalized_schema(raw)

        self.assertEqual(len(normalized), 1)
        row = normalized.iloc[0]
        self.assertEqual(row["engagement_score"], 6.0)
        self.assertFalse(bool(row["mentions_trump"]))
        self.assertTrue(bool(row["post_id"]))
        self.assertTrue(bool(row["author_account_id"]))

    def test_truth_social_fetch_history_uses_network_and_fetch_since_filters(self) -> None:
        csv_text = (
            "created_at,content,url,media,id,replies_count,reblogs_count,favourites_count\n"
            "2025-02-03T14:00:00Z,,https://truth/1,image.jpg,1,1,2,3\n"
            "2025-02-04T15:00:00Z,Great growth,https://truth/2,,2,0,1,4\n"
        )
        adapter = TruthSocialArchiveAdapter(settings=self.settings)

        with mock.patch(
            "trump_workbench.ingestion._request_text",
            return_value=(csv_text, {"Last-Modified": "Mon"}),
        ):
            posts, meta = adapter.fetch_history()
            newer_posts, newer_meta = adapter.fetch_since(pd.Timestamp("2025-02-03 12:00:00", tz="America/New_York"))

        self.assertEqual(len(posts), 2)
        self.assertEqual(posts.iloc[0]["cleaned_text"], "[media-only post]")
        self.assertTrue(bool(posts.iloc[0]["has_media"]))
        self.assertEqual(meta["last_modified"], "Mon")
        self.assertTrue(self.settings.truth_cache_file.exists())
        self.assertEqual(len(newer_posts), 1)
        self.assertTrue(bool(newer_meta["incremental"]))

    def test_truth_social_fetch_history_falls_back_to_cache(self) -> None:
        cached_csv = (
            "created_at,content,url,media,id\n"
            "2025-02-03T14:00:00Z,Cache hit,https://truth/1,,1\n"
        )
        self.settings.truth_cache_file.write_text(cached_csv, encoding="utf-8")
        adapter = TruthSocialArchiveAdapter(settings=self.settings)

        with mock.patch("trump_workbench.ingestion._request_text", side_effect=RuntimeError("offline")):
            posts, meta = adapter.fetch_history()

        self.assertEqual(len(posts), 1)
        self.assertIn("local cache", str(meta["provenance"]))

    def test_truth_social_fetch_history_raises_for_missing_created_at(self) -> None:
        adapter = TruthSocialArchiveAdapter(settings=self.settings)
        bad_csv = "content,url\nmissing timestamp,https://truth/1\n"

        with mock.patch("trump_workbench.ingestion._request_text", return_value=(bad_csv, {})):
            with self.assertRaises(RuntimeError):
                adapter.fetch_history()

    def test_truth_social_fetch_history_raises_when_no_sources_are_available(self) -> None:
        adapter = TruthSocialArchiveAdapter(settings=self.settings)

        with mock.patch("trump_workbench.ingestion._request_text", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                adapter.fetch_history()

    def test_x_csv_adapter_parses_flexible_columns_and_filters_incremental(self) -> None:
        raw_bytes = (
            "Created At,Body,Screen Name,Name,Reply Count,Retweet Count,Like Count,Status URL\n"
            "2025-02-03T15:00:00-05:00,Trump trade deal,macroalpha,Macro Alpha,1,2,5,https://x.com/macroalpha/status/1\n"
            "2025-02-03T16:30:00-05:00,RT @other: Trump tariff risk,marketwatch,Market Watch,0,1,1,https://x.com/marketwatch/status/2\n"
            "2025-02-04T10:00:00-05:00,Donald Trump says jobs,realDonaldTrump,Donald J. Trump,2,3,10,https://x.com/realdonaldtrump/status/3\n"
        ).encode("utf-8")
        adapter = XCsvAdapter(
            settings=self.settings,
            name="X test",
            provenance="unit-test",
            raw_bytes=raw_bytes,
        )

        posts, _ = adapter.fetch_history()
        newer_posts, newer_meta = adapter.fetch_since(pd.Timestamp("2025-02-03 23:59:00", tz="America/New_York"))

        self.assertEqual(len(posts), 3)
        self.assertTrue(bool(posts.iloc[0]["mentions_trump"]))
        self.assertTrue(bool(posts.iloc[1]["is_reshare"]))
        self.assertTrue(bool(posts.iloc[2]["author_is_trump"]))
        self.assertFalse(bool(posts.iloc[2]["mentions_trump"]))
        self.assertTrue(bool(posts.iloc[0]["author_account_id"]))
        self.assertEqual(len(newer_posts), 1)
        self.assertTrue(bool(newer_meta["incremental"]))

    def test_x_csv_adapter_requires_timestamp_and_text_columns(self) -> None:
        adapter = XCsvAdapter(
            settings=self.settings,
            name="bad csv",
            provenance="unit-test",
            raw_bytes=b"id,url\n1,https://x.com/example/status/1\n",
        )

        with self.assertRaises(RuntimeError):
            adapter.fetch_history()

    def test_x_csv_adapter_can_load_from_local_file_and_remote_url(self) -> None:
        csv_bytes = (
            "timestamp,text,author_handle,author_name\n"
            "2025-02-03T15:00:00-05:00,Trump and markets,macroalpha,Macro Alpha\n"
        ).encode("utf-8")
        local_path = Path(self.temp_dir.name) / "sample.csv"
        local_path.write_bytes(csv_bytes)

        local_adapter = XCsvAdapter.from_local_file(self.settings, str(local_path), name="local")
        local_posts, _ = local_adapter.fetch_history()
        self.assertEqual(local_adapter.provenance, f"file:{local_path}")
        self.assertEqual(len(local_posts), 1)

        response = _FakeResponse(content=csv_bytes)
        with mock.patch("trump_workbench.ingestion.requests.get", return_value=response) as mocked_get:
            remote_adapter = XCsvAdapter.from_remote_url(self.settings, "https://example.com/posts.csv", name="remote")

        remote_posts, _ = remote_adapter.fetch_history()
        self.assertEqual(len(remote_posts), 1)
        mocked_get.assert_called_once_with("https://example.com/posts.csv", headers=HTTP_HEADERS, timeout=45)

    def test_ingestion_service_combines_refreshes_and_handles_empty_adapter_lists(self) -> None:
        posts = self._normalized_frame()
        service = IngestionService()

        combined, manifest = service.run_refresh([_DummyAdapter("first", posts), _DummyAdapter("second", posts)])
        empty_combined, empty_manifest = service.run_refresh([])
        incremental, incremental_manifest = service.run_incremental_refresh(
            [_DummyAdapter("first", posts)],
            pd.Timestamp("2025-02-03 14:30:00", tz="America/New_York"),
        )
        empty_incremental, empty_incremental_manifest = service.run_incremental_refresh([], None)

        self.assertEqual(len(combined), 1)
        self.assertEqual(len(manifest), 2)
        self.assertListEqual(list(empty_combined.columns), NORMALIZED_POST_COLUMNS)
        self.assertTrue(empty_manifest.empty)
        self.assertEqual(len(incremental), 1)
        self.assertEqual(len(incremental_manifest), 1)
        self.assertListEqual(list(empty_incremental.columns), NORMALIZED_POST_COLUMNS)
        self.assertTrue(empty_incremental_manifest.empty)


if __name__ == "__main__":
    unittest.main()
