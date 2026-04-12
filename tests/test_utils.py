from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from trump_workbench.config import EASTERN
from trump_workbench.utils import (
    business_minutes_until_close,
    clean_text,
    ensure_tz_naive_date,
    first_matching_column,
    fmt_pct,
    fmt_score,
    infer_author_is_trump,
    infer_mentions_trump,
    normalize_boolean,
    normalize_column_lookup,
    parse_timestamp_to_eastern,
    read_csv_bytes,
    safe_float,
    stable_text_id,
    truncate_text,
)


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

    def test_truncate_text_short_and_long(self) -> None:
        self.assertEqual(truncate_text("short text", max_chars=50), "short text")
        self.assertEqual(truncate_text("1234567890", max_chars=6), "12345…")

    def test_parse_timestamp_to_eastern_handles_invalid_and_naive_values(self) -> None:
        self.assertTrue(pd.isna(parse_timestamp_to_eastern(None)))
        self.assertTrue(pd.isna(parse_timestamp_to_eastern("")))
        self.assertTrue(pd.isna(parse_timestamp_to_eastern("not-a-real-timestamp")))

        localized = parse_timestamp_to_eastern("2025-02-01 15:00:00")
        self.assertFalse(pd.isna(localized))
        self.assertEqual(localized.tz.zone, EASTERN)

    def test_normalize_boolean_handles_empty_boolean_and_string_inputs(self) -> None:
        empty = normalize_boolean(pd.Series(dtype=object))
        self.assertTrue(empty.empty)

        boolean_series = pd.Series([True, False], dtype=bool)
        normalized_boolean = normalize_boolean(boolean_series)
        self.assertEqual(normalized_boolean.tolist(), [True, False])

        mixed = pd.Series(["yes", "No", "1", "0", "maybe", None], dtype=object)
        normalized_mixed = normalize_boolean(mixed, default=True)
        self.assertEqual(normalized_mixed.tolist(), [True, False, True, False, True, False])

    def test_read_csv_bytes_supports_fallback_encoding(self) -> None:
        latin1_bytes = "name\ncafé\n".encode("latin-1")
        parsed = read_csv_bytes(latin1_bytes)
        self.assertEqual(parsed.iloc[0]["name"], "café")

    def test_read_csv_bytes_raises_when_all_parsers_fail(self) -> None:
        with mock.patch("trump_workbench.utils.pd.read_csv", side_effect=ValueError("bad csv")):
            with self.assertRaises(RuntimeError):
                read_csv_bytes(b"irrelevant")

    def test_lookup_and_author_helpers(self) -> None:
        lookup = normalize_column_lookup(["Tweet URL", "Created At", "screen_name"])
        self.assertEqual(first_matching_column(lookup, ["tweet_url", "url"]), "Tweet URL")
        self.assertEqual(first_matching_column(lookup, ["created_at"]), "Created At")
        self.assertIsNone(first_matching_column(lookup, ["missing_column"]))

        self.assertTrue(infer_author_is_trump("@realDonaldTrump", "Someone Else"))
        self.assertTrue(infer_author_is_trump("other", "Donald Trump"))
        self.assertFalse(infer_author_is_trump("macroalpha", "Macro Alpha"))

    def test_formatting_and_numeric_helpers(self) -> None:
        stable_a = stable_text_id("alpha", 1)
        stable_b = stable_text_id("alpha", 1)
        self.assertEqual(stable_a, stable_b)
        self.assertEqual(len(stable_a), 16)

        aware = pd.Timestamp("2025-02-03 14:00:00+00:00")
        naive = ensure_tz_naive_date(aware)
        self.assertIsNone(naive.tzinfo)
        self.assertEqual(str(naive.date()), "2025-02-03")

        self.assertEqual(safe_float("3.5"), 3.5)
        self.assertEqual(safe_float("oops", default=7.0), 7.0)
        self.assertEqual(safe_float(float("nan"), default=2.0), 2.0)

        self.assertEqual(fmt_pct(0.1234), "12.34%")
        self.assertEqual(fmt_pct(float("nan")), "n/a")
        self.assertEqual(fmt_score(0.1234), "+0.1234")
        self.assertEqual(fmt_score(float("nan")), "n/a")

    def test_business_minutes_until_close_clamps_at_zero(self) -> None:
        before_close = business_minutes_until_close(pd.Timestamp("2025-02-03 15:15:00", tz=EASTERN))
        after_close = business_minutes_until_close(pd.Timestamp("2025-02-03 17:00:00", tz=EASTERN))

        self.assertEqual(before_close, 45.0)
        self.assertEqual(after_close, 0.0)


if __name__ == "__main__":
    unittest.main()
