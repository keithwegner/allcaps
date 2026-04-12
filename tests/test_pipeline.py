from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.backtesting import BacktestService
from trump_workbench.config import AppSettings
from trump_workbench.discovery import DiscoveryService
from trump_workbench.enrichment import LLMEnrichmentService
from trump_workbench.experiments import ExperimentStore
from trump_workbench.features import FeatureService
from trump_workbench.ingestion import XCsvAdapter
from trump_workbench.modeling import ModelService
from trump_workbench.storage import DuckDBStore
from trump_workbench.contracts import ModelRunConfig


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(self.settings)
        self.discovery = DiscoveryService()
        self.enrichment = LLMEnrichmentService(self.store)
        self.features = FeatureService(self.enrichment)
        self.model_service = ModelService()
        self.backtests = BacktestService(self.model_service)
        self.experiments = ExperimentStore(self.store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _build_market(self) -> pd.DataFrame:
        dates = pd.bdate_range("2025-02-03", periods=140)
        opens = []
        closes = []
        volumes = []
        price = 100.0
        for idx, _ in enumerate(dates):
            daily_signal = 0.004 if idx % 4 in (0, 1) else -0.002
            open_price = price
            close_price = open_price * (1.0 + daily_signal)
            opens.append(open_price)
            closes.append(close_price)
            volumes.append(1_000_000 + idx * 1_000)
            price = close_price
        return pd.DataFrame(
            {
                "trade_date": dates,
                "open": opens,
                "high": [value * 1.01 for value in closes],
                "low": [value * 0.99 for value in opens],
                "close": closes,
                "volume": volumes,
            },
        )

    def _build_x_csv(self) -> bytes:
        rows = ["timestamp,text,url,is_retweet,author_handle,author_name,author_id,mentions_trump,replies_count,reblogs_count,favourites_count"]
        dates = pd.bdate_range("2025-02-03", periods=140)
        for idx, date in enumerate(dates[:-1]):
            positive = (idx + 1) % 4 in (0, 1)
            text = "Trump growth market rally" if positive else "Trump tariff risk pressure"
            handle = "macroalpha" if positive else "policybeta"
            row = ",".join(
                [
                    f"{date:%Y-%m-%d}T15:30:00-05:00",
                    text,
                    f"https://x.com/{handle}/status/{1000 + idx}",
                    "false",
                    handle,
                    handle.title(),
                    handle,
                    "true",
                    str(10 + idx % 5),
                    str(20 + idx % 7),
                    str(50 + idx % 11),
                ],
            )
            rows.append(row)
        return "\n".join(rows).encode("utf-8")

    def test_enrichment_cache_and_end_to_end_backtest(self) -> None:
        adapter = XCsvAdapter(
            settings=self.settings,
            name="synthetic-x",
            provenance="unit-test",
            raw_bytes=self._build_x_csv(),
        )
        posts, _ = adapter.fetch_history()
        tracked, _ = self.discovery.refresh_accounts(posts, pd.DataFrame(), as_of=posts["post_timestamp"].max(), auto_include_top_n=2)
        market = self._build_market()

        first_enriched = self.enrichment.enrich_posts(posts, enabled=True)
        second_enriched = self.enrichment.enrich_posts(posts, enabled=True)
        self.assertTrue((second_enriched["semantic_cache_hit"].astype(bool)).any())

        first_dataset = self.features.build_session_dataset(first_enriched, market, tracked, feature_version="v1", llm_enabled=True)

        config = ModelRunConfig(
            run_name="unit-test-run",
            llm_enabled=True,
            train_window=60,
            validation_window=20,
            test_window=20,
            step_size=20,
            threshold_grid=(0.0, 0.001, 0.002),
            minimum_signal_grid=(1, 2),
            account_weight_grid=(0.5, 1.0),
        )
        run, artifacts = self.backtests.run_walk_forward(config, first_dataset)
        self.assertGreater(len(artifacts["trades"]), 0)
        self.assertIn("robust_score", run.metrics)

        saved = self.experiments.save_run(
            run=run,
            config=artifacts["config"],
            trades=artifacts["trades"],
            predictions=artifacts["predictions"],
            windows=artifacts["windows"],
            importance=artifacts["importance"],
            model_artifact=artifacts["model_artifact"],
        )
        self.assertTrue(saved.model_path.exists())
        loaded = self.experiments.load_run(run.run_id)
        self.assertIsNotNone(loaded)


if __name__ == "__main__":
    unittest.main()
