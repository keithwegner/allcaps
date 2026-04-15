from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.config import AppSettings
from trump_workbench.enrichment import LLMEnrichmentService, SEMANTIC_SCHEMA_VERSION
from trump_workbench.storage import DuckDBStore


class FakeHostedProvider:
    provider_name = "hosted-test"

    def enrich_narrative(self, *, semantic_key: str, text: str, sentiment_label: str) -> dict[str, object]:
        return {
            "semantic_key": semantic_key,
            "semantic_topic": "markets",
            "semantic_policy_bucket": "economy",
            "semantic_stance": "positive",
            "semantic_market_relevance": 0.95,
            "semantic_urgency": 0.45,
            "semantic_primary_asset": "QQQ",
            "semantic_asset_targets": "QQQ,XLK",
            "semantic_confidence": 0.88,
            "semantic_summary": "Hosted provider market narrative.",
            "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        }


class FailingHostedProvider:
    provider_name = "hosted-failing"

    def enrich_narrative(self, *, semantic_key: str, text: str, sentiment_label: str) -> dict[str, object]:
        raise RuntimeError("provider unavailable")


class EnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(settings)
        self.service = LLMEnrichmentService(self.store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_enrich_posts_handles_empty_input(self) -> None:
        enriched = self.service.enrich_posts(pd.DataFrame(columns=["post_id", "cleaned_text", "sentiment_label"]), enabled=True)
        self.assertTrue(enriched.empty)
        self.assertIn("semantic_cache_hit", enriched.columns)
        self.assertIn("semantic_summary", enriched.columns)

    def test_heuristic_enrichment_populates_narrative_fields(self) -> None:
        posts = pd.DataFrame(
            [
                {
                    "post_id": "p-1",
                    "cleaned_text": "Trump says Nvidia and the Nasdaq will keep winning as AI spending surges.",
                    "sentiment_label": "positive",
                },
            ],
        )

        enriched = self.service.enrich_posts(posts, enabled=True)

        row = enriched.iloc[0]
        self.assertEqual(row["semantic_topic"], "markets")
        self.assertEqual(row["semantic_policy_bucket"], "economy")
        self.assertEqual(row["semantic_primary_asset"], "NVDA")
        self.assertIn("NVDA", row["semantic_asset_targets"])
        self.assertGreater(float(row["semantic_confidence"]), 0.0)
        self.assertEqual(row["semantic_schema_version"], SEMANTIC_SCHEMA_VERSION)
        self.assertIn("focus NVDA", row["semantic_summary"])

    def test_hosted_provider_is_used_and_cached(self) -> None:
        service = LLMEnrichmentService(self.store, provider=FakeHostedProvider())
        posts = pd.DataFrame(
            [
                {
                    "post_id": "p-2",
                    "cleaned_text": "Tariff headlines are hitting tech again.",
                    "sentiment_label": "negative",
                },
            ],
        )

        first = service.enrich_posts(posts, enabled=True)
        second = service.enrich_posts(posts, enabled=True)

        self.assertEqual(first.iloc[0]["semantic_provider"], "hosted-test")
        self.assertEqual(first.iloc[0]["semantic_primary_asset"], "QQQ")
        self.assertFalse(bool(first.iloc[0]["semantic_cache_hit"]))
        self.assertTrue(bool(second.iloc[0]["semantic_cache_hit"]))

    def test_failing_hosted_provider_falls_back_to_heuristics(self) -> None:
        service = LLMEnrichmentService(self.store, provider=FailingHostedProvider())
        posts = pd.DataFrame(
            [
                {
                    "post_id": "p-3",
                    "cleaned_text": "Breaking tariff warning raises pressure on energy and banks!",
                    "sentiment_label": "negative",
                },
            ],
        )

        enriched = service.enrich_posts(posts, enabled=True)

        self.assertEqual(enriched.iloc[0]["semantic_provider"], "heuristic-fallback")
        self.assertIn("XLE", enriched.iloc[0]["semantic_asset_targets"])

    def test_legacy_cache_rows_rebuild_lazily(self) -> None:
        legacy_cache = pd.DataFrame(
            [
                {
                    "semantic_key": "legacy-1",
                    "semantic_topic": "trade",
                    "semantic_policy_bucket": "trade",
                    "semantic_stance": "negative",
                    "semantic_market_relevance": 0.7,
                    "semantic_urgency": 0.4,
                    "semantic_provider": "heuristic-cache",
                },
            ],
        )
        self.store.save_frame("semantic_cache", legacy_cache)
        posts = pd.DataFrame(
            [
                {
                    "post_id": "legacy-1",
                    "cleaned_text": "Tariff headlines keep hitting the broad market.",
                    "sentiment_label": "negative",
                },
            ],
        )

        enriched = self.service.enrich_posts(posts, enabled=True)

        self.assertFalse(bool(enriched.iloc[0]["semantic_cache_hit"]))
        self.assertEqual(enriched.iloc[0]["semantic_schema_version"], SEMANTIC_SCHEMA_VERSION)
        self.assertTrue(bool(enriched.iloc[0]["semantic_summary"]))


if __name__ == "__main__":
    unittest.main()
