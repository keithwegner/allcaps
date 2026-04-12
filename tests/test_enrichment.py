from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.config import AppSettings
from trump_workbench.enrichment import LLMEnrichmentService
from trump_workbench.storage import DuckDBStore


class EnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.service = LLMEnrichmentService(DuckDBStore(settings))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_enrich_posts_handles_empty_input(self) -> None:
        enriched = self.service.enrich_posts(pd.DataFrame(columns=["post_id", "cleaned_text", "sentiment_label"]), enabled=True)
        self.assertTrue(enriched.empty)
        self.assertIn("semantic_cache_hit", enriched.columns)


if __name__ == "__main__":
    unittest.main()
