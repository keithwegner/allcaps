from __future__ import annotations

import unittest

import pandas as pd

from trump_workbench.utils import clean_text, infer_mentions_trump, parse_timestamp_to_eastern


class UtilsTests(unittest.TestCase):
    def test_clean_text_strips_html_and_mojibake(self) -> None:
        raw = "Hello<br>world &amp; team ‚Äî test"
        self.assertEqual(clean_text(raw), "Hello world & team — test")

    def test_clean_text_marks_media_only(self) -> None:
        self.assertEqual(clean_text("", media_hint="image.jpg"), "[media-only post]")

    def test_parse_timestamp_to_eastern_normalizes_timezone(self) -> None:
        ts = parse_timestamp_to_eastern("2025-02-01T15:00:00Z")
        self.assertEqual(ts.tz.zone, "America/New_York")

    def test_infer_mentions_trump(self) -> None:
        self.assertTrue(infer_mentions_trump("Markets are reacting to Trump again"))
        self.assertFalse(infer_mentions_trump("Completely unrelated post"))


if __name__ == "__main__":
    unittest.main()
