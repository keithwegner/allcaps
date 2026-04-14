from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.backtesting import BacktestService
from trump_workbench.config import AppSettings
from trump_workbench.discovery import DiscoveryService
from trump_workbench.enrichment import LLMEnrichmentService
from trump_workbench.explanations import build_account_attribution, build_post_attribution
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

    def _build_asset_market(self, symbols: list[str]) -> pd.DataFrame:
        base = self._build_market()
        frames: list[pd.DataFrame] = []
        for offset, symbol in enumerate(symbols, start=1):
            frame = base.copy()
            scale = 1.0 + offset * 0.08
            frame["symbol"] = symbol
            frame["open"] = frame["open"] * scale
            frame["high"] = frame["high"] * scale
            frame["low"] = frame["low"] * scale
            frame["close"] = frame["close"] * scale
            frame["volume"] = frame["volume"] + offset * 10_000
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

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

        prepared_posts = self.features.prepare_session_posts(first_enriched, market, tracked, llm_enabled=True)
        first_dataset = self.features.build_session_dataset(
            first_enriched,
            market,
            tracked,
            feature_version="v1",
            llm_enabled=True,
            prepared_posts=prepared_posts,
        )

        config = ModelRunConfig(
            run_name="unit-test-run",
            target_asset="SPY",
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
        self.assertEqual(run.target_asset, "SPY")
        self.assertIn("robust_score", run.metrics)
        self.assertTrue(artifacts["leakage_audit"]["overall_pass"])
        self.assertIn("always_long_spy", artifacts["benchmarks"]["benchmark_name"].tolist())
        self.assertFalse(artifacts["feature_contributions"].empty)

        post_attribution = build_post_attribution(prepared_posts)
        account_attribution = build_account_attribution(post_attribution)

        saved = self.experiments.save_run(
            run=run,
            config=artifacts["config"],
            trades=artifacts["trades"],
            predictions=artifacts["predictions"],
            windows=artifacts["windows"],
            importance=artifacts["importance"],
            model_artifact=artifacts["model_artifact"],
            feature_contributions=artifacts["feature_contributions"],
            post_attribution=post_attribution,
            account_attribution=account_attribution,
            benchmarks=artifacts["benchmarks"],
            diagnostics=artifacts["diagnostics"],
            benchmark_curves=artifacts["benchmark_curves"],
            leakage_audit=artifacts["leakage_audit"],
        )
        self.assertTrue(saved.model_path.exists())
        loaded = self.experiments.load_run(run.run_id)
        self.assertIsNotNone(loaded)
        self.assertFalse(loaded["feature_contributions"].empty)
        self.assertFalse(loaded["benchmarks"].empty)
        self.assertTrue(bool(loaded["leakage_audit"].get("overall_pass", False)))

        replay_session = pd.Timestamp(first_dataset.loc[first_dataset["target_available"]].iloc[95]["signal_session_date"])
        replay_bundle = self.backtests.build_historical_replay(
            config,
            first_dataset,
            replay_session_date=replay_session,
            deployment_params=run.selected_params,
        )
        replay_prediction = replay_bundle["prediction"].iloc[0]
        self.assertEqual(pd.Timestamp(replay_prediction["signal_session_date"]), replay_session)
        self.assertLess(pd.Timestamp(replay_bundle["history_end"]), replay_session)
        self.assertEqual(
            replay_bundle["training_rows_used"],
            int((first_dataset["signal_session_date"] < replay_session).loc[first_dataset["target_available"]].sum()),
        )
        self.assertFalse(replay_bundle["feature_contributions"].empty)
        self.assertFalse(bool(replay_prediction["future_training_leakage"]))

    def test_asset_target_pipeline_supports_non_spy_runs(self) -> None:
        adapter = XCsvAdapter(
            settings=self.settings,
            name="synthetic-x",
            provenance="unit-test",
            raw_bytes=self._build_x_csv(),
        )
        posts, _ = adapter.fetch_history()
        posts["raw_text"] = [
            "Trump says Nasdaq and big tech are leading the market higher." if idx % 2 == 0 else
            "Trump trade pressure is weighing on energy and financials."
            for idx in range(len(posts))
        ]
        posts["cleaned_text"] = posts["raw_text"]
        tracked, _ = self.discovery.refresh_accounts(posts, pd.DataFrame(), as_of=posts["post_timestamp"].max(), auto_include_top_n=2)
        spy_market = self._build_market()
        asset_market = self._build_asset_market(["QQQ"])
        asset_universe = pd.DataFrame(
            [
                {
                    "symbol": "QQQ",
                    "display_name": "Invesco QQQ Trust",
                    "asset_type": "etf",
                    "source": "core_etf",
                },
            ],
        )

        prepared_posts = self.features.prepare_session_posts(posts, spy_market, tracked, llm_enabled=True)
        asset_post_mappings = self.features.build_asset_post_mappings(
            prepared_posts,
            asset_universe,
            llm_enabled=True,
        )
        self.assertFalse(asset_post_mappings.loc[asset_post_mappings["asset_symbol"] == "QQQ"].empty)

        asset_dataset = self.features.build_asset_session_dataset(
            asset_post_mappings=asset_post_mappings,
            asset_market=asset_market,
            feature_version="asset-v1",
            llm_enabled=True,
            asset_universe=asset_universe,
        )
        qqq_dataset = asset_dataset.loc[asset_dataset["asset_symbol"] == "QQQ"].copy()
        qqq_dataset["target_asset"] = "QQQ"
        self.assertFalse(qqq_dataset.empty)

        config = ModelRunConfig(
            run_name="qqq-target-run",
            target_asset="QQQ",
            feature_version="asset-v1",
            llm_enabled=True,
            train_window=60,
            validation_window=20,
            test_window=20,
            step_size=20,
            threshold_grid=(0.0, 0.001, 0.002),
            minimum_signal_grid=(1, 2),
            account_weight_grid=(0.5, 1.0),
        )
        run, artifacts = self.backtests.run_walk_forward(config, qqq_dataset)
        self.assertEqual(run.target_asset, "QQQ")
        self.assertIn("always_long_qqq", artifacts["benchmarks"]["benchmark_name"].tolist())
        self.assertTrue((artifacts["predictions"]["target_asset"].astype(str) == "QQQ").all())

        post_attribution = build_post_attribution(
            asset_post_mappings.loc[asset_post_mappings["asset_symbol"] == "QQQ"].copy(),
        )
        account_attribution = build_account_attribution(post_attribution)
        self.experiments.save_run(
            run=run,
            config=artifacts["config"],
            trades=artifacts["trades"],
            predictions=artifacts["predictions"],
            windows=artifacts["windows"],
            importance=artifacts["importance"],
            model_artifact=artifacts["model_artifact"],
            feature_contributions=artifacts["feature_contributions"],
            post_attribution=post_attribution,
            account_attribution=account_attribution,
            benchmarks=artifacts["benchmarks"],
            diagnostics=artifacts["diagnostics"],
            benchmark_curves=artifacts["benchmark_curves"],
            leakage_audit=artifacts["leakage_audit"],
        )
        loaded = self.experiments.load_run(run.run_id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["run"]["target_asset"], "QQQ")
        latest_qqq = self.experiments.load_latest_model_artifact(target_asset="QQQ")
        self.assertIsNotNone(latest_qqq)
        assert latest_qqq is not None
        self.assertEqual(latest_qqq[0].metadata["target_asset"], "QQQ")


if __name__ == "__main__":
    unittest.main()
