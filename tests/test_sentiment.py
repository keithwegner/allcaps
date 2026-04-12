from __future__ import annotations

import builtins
import unittest
from unittest import mock

import pandas as pd

from trump_workbench import sentiment


class _WorkingAnalyzer:
    def polarity_scores(self, text: str) -> dict[str, float]:
        return {"compound": 0.42 if "great" in text else -0.1}


class _BrokenAnalyzer:
    def polarity_scores(self, text: str) -> dict[str, float]:
        raise RuntimeError("boom")


class SentimentTests(unittest.TestCase):
    def tearDown(self) -> None:
        sentiment.get_sentiment_backend.cache_clear()

    def test_get_sentiment_backend_falls_back_when_vader_import_fails(self) -> None:
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            if str(name).startswith("vaderSentiment"):
                raise ImportError("disabled for test")
            return original_import(name, globals, locals, fromlist, level)

        sentiment.get_sentiment_backend.cache_clear()
        with mock.patch("builtins.__import__", side_effect=fake_import):
            backend = sentiment.get_sentiment_backend()

        self.assertTrue(bool(backend["fallback"]))
        self.assertIsNone(backend["analyzer"])

    def test_tokenize_and_fallback_sentiment_scoring(self) -> None:
        tokens = sentiment.tokenize_for_sentiment("Very good, markets! Can't lose.")
        self.assertEqual(tokens, ["very", "good", "markets", "can't", "lose"])

        plain_positive = sentiment.fallback_sentiment_score("good growth")
        boosted_positive = sentiment.fallback_sentiment_score("very good growth")
        negated_positive = sentiment.fallback_sentiment_score("not good growth")
        negative = sentiment.fallback_sentiment_score("tariff risk war")

        self.assertGreater(plain_positive, 0.0)
        self.assertGreater(boosted_positive, 0.0)
        self.assertNotEqual(boosted_positive, plain_positive)
        self.assertLess(negated_positive, plain_positive)
        self.assertLess(negative, 0.0)
        self.assertEqual(sentiment.fallback_sentiment_score(""), 0.0)

    def test_score_post_sentiment_uses_backend_and_falls_back_on_error(self) -> None:
        with mock.patch(
            "trump_workbench.sentiment.get_sentiment_backend",
            return_value={"analyzer": _WorkingAnalyzer()},
        ):
            self.assertEqual(sentiment.score_post_sentiment("great market"), 0.42)

        fallback_value = sentiment.fallback_sentiment_score("great growth")
        with mock.patch(
            "trump_workbench.sentiment.get_sentiment_backend",
            return_value={"analyzer": _BrokenAnalyzer()},
        ):
            self.assertEqual(sentiment.score_post_sentiment("great growth"), fallback_value)

    def test_sentiment_label_and_add_sentiment_scores(self) -> None:
        self.assertEqual(sentiment.sentiment_label(float("nan")), "unknown")
        self.assertEqual(sentiment.sentiment_label(0.01), "neutral")
        self.assertEqual(sentiment.sentiment_label(0.2), "positive")
        self.assertEqual(sentiment.sentiment_label(-0.2), "negative")

        fallback_backend = {
            "name": "Fallback",
            "backend": "fallback_lexicon",
            "analyzer": None,
            "fallback": True,
        }

        with mock.patch("trump_workbench.sentiment.get_sentiment_backend", return_value=fallback_backend):
            empty_scored, empty_meta = sentiment.add_sentiment_scores(pd.DataFrame(columns=["cleaned_text"]))
            self.assertTrue(empty_scored.empty)
            self.assertEqual(empty_meta["backend"], "fallback_lexicon")

            scored, meta = sentiment.add_sentiment_scores(
                pd.DataFrame({"cleaned_text": ["great growth", "tariff risk", None]}),
            )

        self.assertEqual(meta["backend"], "fallback_lexicon")
        self.assertEqual(scored["sentiment_label"].tolist()[0], "positive")
        self.assertEqual(scored["sentiment_label"].tolist()[1], "negative")
        self.assertEqual(scored["sentiment_label"].tolist()[2], "neutral")


if __name__ == "__main__":
    unittest.main()
