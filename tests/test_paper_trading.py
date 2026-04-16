from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.config import AppSettings
from trump_workbench.contracts import LiveMonitorConfig
from trump_workbench.paper_trading import PaperTradingService, ensure_paper_benchmark_curve_frame, ensure_paper_decision_journal_frame, ensure_paper_equity_curve_frame, ensure_paper_portfolio_registry_frame, ensure_paper_trade_ledger_frame
from trump_workbench.storage import DuckDBStore


class PaperTradingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(self.settings)
        self.paper_service = PaperTradingService(self.store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _live_config(run_id: str = "portfolio-1", variant: str = "pooled", fallback_mode: str = "SPY") -> LiveMonitorConfig:
        return LiveMonitorConfig(
            mode="portfolio_run",
            fallback_mode=fallback_mode,
            portfolio_run_id=run_id,
            portfolio_run_name=f"Run {run_id}",
            deployment_variant=variant,
        )

    @staticmethod
    def _snapshot_rows(
        generated_at: pd.Timestamp,
        signal_session_date: str = "2026-04-15",
        next_session_date: str = "2026-04-16",
        next_session_open_ts: str = "2026-04-16 13:30:00+00:00",
        spy_score: float = 0.005,
        qqq_score: float = 0.010,
        spy_threshold: float = 0.001,
        qqq_threshold: float = 0.001,
        spy_posts: int = 5,
        qqq_posts: int = 5,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "generated_at": generated_at,
                    "variant_name": "pooled",
                    "signal_session_date": pd.Timestamp(signal_session_date, tz="UTC"),
                    "next_session_date": pd.Timestamp(next_session_date, tz="UTC"),
                    "asset_symbol": "SPY",
                    "run_id": "portfolio-1",
                    "run_name": "Joint run",
                    "feature_version": "asset-v1",
                    "model_version": "portfolio-model",
                    "expected_return_score": spy_score,
                    "confidence": 0.6,
                    "threshold": spy_threshold,
                    "min_post_count": 1,
                    "post_count": spy_posts,
                    "target_available": False,
                    "tradeable": True,
                    "next_session_open": 500.0,
                    "next_session_close": 510.0,
                    "next_session_open_ts": pd.Timestamp(next_session_open_ts),
                },
                {
                    "generated_at": generated_at,
                    "variant_name": "pooled",
                    "signal_session_date": pd.Timestamp(signal_session_date, tz="UTC"),
                    "next_session_date": pd.Timestamp(next_session_date, tz="UTC"),
                    "asset_symbol": "QQQ",
                    "run_id": "portfolio-1",
                    "run_name": "Joint run",
                    "feature_version": "asset-v1",
                    "model_version": "portfolio-model",
                    "expected_return_score": qqq_score,
                    "confidence": 0.7,
                    "threshold": qqq_threshold,
                    "min_post_count": 1,
                    "post_count": qqq_posts,
                    "target_available": False,
                    "tradeable": True,
                    "next_session_open": 400.0,
                    "next_session_close": 420.0,
                    "next_session_open_ts": pd.Timestamp(next_session_open_ts),
                },
            ],
        )

    def test_capture_uses_latest_snapshot_before_open_cutoff(self) -> None:
        config = self.paper_service.build_config(
            live_config=self._live_config(),
            portfolio_run_name="Joint run",
            transaction_cost_bps=2.0,
            enabled=True,
            now=pd.Timestamp("2026-04-15 20:00:00+00:00"),
        )
        pre_open_early = self._snapshot_rows(
            generated_at=pd.Timestamp("2026-04-16 11:00:00+00:00"),
            spy_score=0.006,
            qqq_score=0.009,
        )
        pre_open_late = self._snapshot_rows(
            generated_at=pd.Timestamp("2026-04-16 13:00:00+00:00"),
            spy_score=0.008,
            qqq_score=0.012,
        )
        post_open = self._snapshot_rows(
            generated_at=pd.Timestamp("2026-04-16 14:00:00+00:00"),
            spy_score=0.020,
            qqq_score=0.003,
        )
        self.store.save_frame(
            "live_asset_snapshots",
            pd.concat([pre_open_early, pre_open_late, post_open], ignore_index=True),
            metadata={"row_count": 6},
        )

        journal = self.paper_service.capture_authoritative_decisions(
            config,
            as_of=pd.Timestamp("2026-04-16 15:00:00+00:00"),
        )

        portfolio_rows = journal.loc[journal["paper_portfolio_id"].astype(str) == config.paper_portfolio_id].copy()
        self.assertEqual(len(portfolio_rows), 1)
        self.assertEqual(str(portfolio_rows.iloc[0]["winning_asset"]), "QQQ")
        self.assertEqual(pd.Timestamp(portfolio_rows.iloc[0]["generated_at"]), pd.Timestamp("2026-04-16 13:00:00+00:00"))

    def test_flat_decision_creates_no_trade_but_builds_curves(self) -> None:
        config = self.paper_service.build_config(
            live_config=self._live_config(fallback_mode="FLAT"),
            portfolio_run_name="Joint run",
            transaction_cost_bps=2.0,
            enabled=True,
            now=pd.Timestamp("2026-04-15 20:00:00+00:00"),
        )
        flat_rows = self._snapshot_rows(
            generated_at=pd.Timestamp("2026-04-16 12:00:00+00:00"),
            spy_score=0.0001,
            qqq_score=0.0002,
            spy_threshold=0.005,
            qqq_threshold=0.005,
            spy_posts=1,
            qqq_posts=1,
        )
        self.store.save_frame("live_asset_snapshots", flat_rows, metadata={"row_count": 2})
        self.store.save_frame(
            "spy_daily",
            pd.DataFrame([{"trade_date": pd.Timestamp("2026-04-16"), "open": 500.0, "close": 510.0}]),
            metadata={"row_count": 1},
        )

        self.paper_service.process_live_history(config, as_of=pd.Timestamp("2026-04-16 16:00:00+00:00"))

        journal = ensure_paper_decision_journal_frame(self.store.read_frame("paper_decision_journal"))
        trades = ensure_paper_trade_ledger_frame(self.store.read_frame("paper_trade_ledger"))
        equity = ensure_paper_equity_curve_frame(self.store.read_frame("paper_equity_curve"))
        benchmark = ensure_paper_benchmark_curve_frame(self.store.read_frame("paper_benchmark_curve"))

        portfolio_journal = journal.loc[journal["paper_portfolio_id"].astype(str) == config.paper_portfolio_id]
        self.assertEqual(len(portfolio_journal), 1)
        self.assertEqual(str(portfolio_journal.iloc[0]["settlement_status"]), "flat")
        self.assertTrue(trades.empty)
        self.assertEqual(len(equity), 1)
        self.assertAlmostEqual(float(equity.iloc[0]["equity"]), 100000.0)
        self.assertEqual(len(benchmark), 1)
        self.assertGreater(float(benchmark.iloc[0]["equity"]), 100000.0)

    def test_long_trade_settlement_uses_open_close_and_transaction_costs(self) -> None:
        config = self.paper_service.build_config(
            live_config=self._live_config(),
            portfolio_run_name="Joint run",
            transaction_cost_bps=10.0,
            enabled=True,
            now=pd.Timestamp("2026-04-15 20:00:00+00:00"),
        )
        rows = self._snapshot_rows(
            generated_at=pd.Timestamp("2026-04-16 12:00:00+00:00"),
            spy_score=0.002,
            qqq_score=0.015,
        )
        self.store.save_frame("live_asset_snapshots", rows, metadata={"row_count": 2})
        self.store.save_frame(
            "asset_daily",
            pd.DataFrame(
                [
                    {"symbol": "QQQ", "trade_date": pd.Timestamp("2026-04-16"), "open": 400.0, "close": 420.0},
                    {"symbol": "SPY", "trade_date": pd.Timestamp("2026-04-16"), "open": 500.0, "close": 510.0},
                ],
            ),
            metadata={"row_count": 2},
        )

        self.paper_service.process_live_history(config, as_of=pd.Timestamp("2026-04-16 16:00:00+00:00"))

        trades = ensure_paper_trade_ledger_frame(self.store.read_frame("paper_trade_ledger"))
        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertEqual(str(trade["asset_symbol"]), "QQQ")
        self.assertAlmostEqual(float(trade["gross_return"]), 0.05)
        self.assertAlmostEqual(float(trade["net_return"]), 0.048)
        self.assertAlmostEqual(float(trade["ending_equity"]), 104800.0)

    def test_switching_live_config_archives_previous_paper_portfolio(self) -> None:
        first = self.paper_service.upsert_current_for_live_config(
            live_config=self._live_config(run_id="portfolio-1", variant="pooled"),
            portfolio_run_name="Run 1",
            transaction_cost_bps=2.0,
            starting_cash=100000.0,
            enabled=True,
            now=pd.Timestamp("2026-04-15 20:00:00+00:00"),
        )
        second = self.paper_service.upsert_current_for_live_config(
            live_config=self._live_config(run_id="portfolio-2", variant="per_asset"),
            portfolio_run_name="Run 2",
            transaction_cost_bps=5.0,
            starting_cash=125000.0,
            enabled=True,
            now=pd.Timestamp("2026-04-16 20:00:00+00:00"),
        )

        registry = ensure_paper_portfolio_registry_frame(self.store.read_frame("paper_portfolio_registry"))

        self.assertEqual(len(registry), 2)
        first_row = registry.loc[registry["paper_portfolio_id"].astype(str) == first.paper_portfolio_id].iloc[0]
        second_row = registry.loc[registry["paper_portfolio_id"].astype(str) == second.paper_portfolio_id].iloc[0]
        self.assertTrue(pd.notna(first_row["archived_at"]))
        self.assertTrue(pd.isna(second_row["archived_at"]))
        self.assertEqual(float(second_row["starting_cash"]), 125000.0)


if __name__ == "__main__":
    unittest.main()
