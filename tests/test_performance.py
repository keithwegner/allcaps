from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.config import AppSettings
from trump_workbench.paper_trading import (
    PAPER_BENCHMARK_CURVE_COLUMNS,
    PAPER_DECISION_JOURNAL_COLUMNS,
    PAPER_EQUITY_CURVE_COLUMNS,
    PAPER_PORTFOLIO_REGISTRY_COLUMNS,
    PAPER_TRADE_LEDGER_COLUMNS,
)
from trump_workbench.performance import (
    PERFORMANCE_DIAGNOSTIC_COLUMNS,
    PerformanceObservatoryService,
    build_performance_summary,
    build_score_outcome_frame,
    ensure_performance_diagnostic_frame,
)
from trump_workbench.storage import DuckDBStore


class PerformanceObservatoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(self.settings)
        self.service = PerformanceObservatoryService(self.store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _registry(portfolio_ids: list[str] | None = None) -> pd.DataFrame:
        ids = portfolio_ids or ["paper-active"]
        rows = []
        for idx, paper_id in enumerate(ids):
            rows.append(
                {
                    "paper_portfolio_id": paper_id,
                    "portfolio_run_id": f"run-{idx + 1}",
                    "portfolio_run_name": f"Run {idx + 1}",
                    "deployment_variant": "per_asset",
                    "fallback_mode": "SPY",
                    "transaction_cost_bps": 2.0,
                    "starting_cash": 100000.0,
                    "enabled": idx == 0,
                    "created_at": pd.Timestamp("2026-04-01 12:00:00", tz="UTC") + pd.Timedelta(days=idx),
                    "archived_at": pd.NaT if idx == 0 else pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
                },
            )
        return pd.DataFrame(rows, columns=PAPER_PORTFOLIO_REGISTRY_COLUMNS)

    @staticmethod
    def _journal(paper_id: str, count: int, fallback_every: int = 0) -> pd.DataFrame:
        rows = []
        for idx in range(count):
            signal_date = pd.Timestamp("2026-04-01", tz="UTC") + pd.Timedelta(days=idx)
            fallback = bool(fallback_every and idx % fallback_every == 0)
            rows.append(
                {
                    "paper_portfolio_id": paper_id,
                    "generated_at": signal_date + pd.Timedelta(hours=12),
                    "signal_session_date": signal_date,
                    "next_session_date": signal_date + pd.Timedelta(days=1),
                    "decision_cutoff_ts": signal_date + pd.Timedelta(days=1, hours=13, minutes=30),
                    "portfolio_run_id": "run-1",
                    "portfolio_run_name": "Run 1",
                    "deployment_variant": "per_asset",
                    "winning_asset": "SPY" if fallback else "QQQ",
                    "winning_run_id": "run-1",
                    "decision_source": "fallback" if fallback else "eligible",
                    "fallback_mode": "SPY",
                    "stance": "LONG",
                    "winner_score": 0.01 + idx * 0.001,
                    "runner_up_asset": "SPY" if not fallback else "QQQ",
                    "runner_up_score": 0.005,
                    "eligible_asset_count": 1 if not fallback else 0,
                    "settlement_status": "settled",
                    "settled_at": signal_date + pd.Timedelta(days=1, hours=21),
                },
            )
        return pd.DataFrame(rows, columns=PAPER_DECISION_JOURNAL_COLUMNS)

    @staticmethod
    def _trades(paper_id: str, returns: list[float]) -> pd.DataFrame:
        rows = []
        equity = 100000.0
        for idx, net_return in enumerate(returns):
            signal_date = pd.Timestamp("2026-04-01", tz="UTC") + pd.Timedelta(days=idx)
            starting_equity = equity
            equity = equity * (1.0 + net_return)
            rows.append(
                {
                    "paper_portfolio_id": paper_id,
                    "signal_session_date": signal_date,
                    "next_session_date": signal_date + pd.Timedelta(days=1),
                    "asset_symbol": "QQQ",
                    "run_id": "run-1",
                    "decision_source": "eligible",
                    "stance": "LONG",
                    "next_session_open": 100.0,
                    "next_session_close": 100.0 * (1.0 + net_return),
                    "gross_return": net_return + 0.0004,
                    "net_return": net_return,
                    "benchmark_return": 0.001,
                    "transaction_cost_bps": 2.0,
                    "starting_equity": starting_equity,
                    "ending_equity": equity,
                    "settled_at": signal_date + pd.Timedelta(days=1, hours=21),
                },
            )
        return pd.DataFrame(rows, columns=PAPER_TRADE_LEDGER_COLUMNS)

    @staticmethod
    def _equity(paper_id: str, returns: list[float], benchmark: bool = False) -> pd.DataFrame:
        rows = []
        equity = 100000.0
        for idx, return_pct in enumerate(returns):
            signal_date = pd.Timestamp("2026-04-01", tz="UTC") + pd.Timedelta(days=idx)
            equity = equity * (1.0 + return_pct)
            row = {
                "paper_portfolio_id": paper_id,
                "signal_session_date": signal_date,
                "next_session_date": signal_date + pd.Timedelta(days=1),
                "equity": equity,
                "return_pct": return_pct,
                "settled_at": signal_date + pd.Timedelta(days=1, hours=21),
            }
            if benchmark:
                row["benchmark_name"] = "always_long_spy"
            rows.append(row)
        columns = PAPER_BENCHMARK_CURVE_COLUMNS if benchmark else PAPER_EQUITY_CURVE_COLUMNS
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _live_snapshots() -> pd.DataFrame:
        rows = []
        generated_times = pd.date_range("2026-04-01 12:00:00", periods=14, freq="h", tz="UTC")
        for idx, generated_at in enumerate(generated_times):
            for asset_symbol in ["SPY", "QQQ"]:
                rows.append(
                    {
                        "generated_at": generated_at,
                        "variant_name": "per_asset",
                        "asset_symbol": asset_symbol,
                        "run_id": "run-1",
                        "expected_return_score": 0.01 + idx * 0.0001 + (0.001 if asset_symbol == "QQQ" else 0.0),
                        "confidence": 0.55 + idx * 0.001,
                        "post_count": 3 + idx % 4,
                    },
                )
        return pd.DataFrame(rows)

    def _save_frames(
        self,
        *,
        registry: pd.DataFrame | None = None,
        journal: pd.DataFrame | None = None,
        trades: pd.DataFrame | None = None,
        equity: pd.DataFrame | None = None,
        benchmark: pd.DataFrame | None = None,
        live_assets: pd.DataFrame | None = None,
    ) -> None:
        self.store.save_frame("paper_portfolio_registry", registry if registry is not None else self._registry())
        self.store.save_frame("paper_decision_journal", journal if journal is not None else pd.DataFrame(columns=PAPER_DECISION_JOURNAL_COLUMNS))
        self.store.save_frame("paper_trade_ledger", trades if trades is not None else pd.DataFrame(columns=PAPER_TRADE_LEDGER_COLUMNS))
        self.store.save_frame("paper_equity_curve", equity if equity is not None else pd.DataFrame(columns=PAPER_EQUITY_CURVE_COLUMNS))
        self.store.save_frame("paper_benchmark_curve", benchmark if benchmark is not None else pd.DataFrame(columns=PAPER_BENCHMARK_CURVE_COLUMNS))
        self.store.save_frame("live_asset_snapshots", live_assets if live_assets is not None else pd.DataFrame())

    def test_empty_paper_and_live_history_returns_insufficient_data_warnings(self) -> None:
        self._save_frames()

        diagnostics = self.service.evaluate_paper_portfolio("paper-active", generated_at=pd.Timestamp("2026-04-20", tz="UTC"))

        self.assertIn("warn", diagnostics["severity"].tolist())
        settled_row = diagnostics.loc[diagnostics["metric_name"] == "settled_trade_count"].iloc[0]
        drift_row = diagnostics.loc[diagnostics["metric_name"] == "live_score_history"].iloc[0]
        self.assertEqual(settled_row["severity"], "warn")
        self.assertEqual(drift_row["severity"], "warn")

    def test_settled_trades_compute_portfolio_summary_metrics(self) -> None:
        returns = [0.02, -0.01, 0.015, 0.005, 0.01]
        benchmark_returns = [0.002, 0.001, 0.001, 0.001, 0.001]
        self._save_frames(
            journal=self._journal("paper-active", len(returns), fallback_every=3),
            trades=self._trades("paper-active", returns),
            equity=self._equity("paper-active", returns),
            benchmark=self._equity("paper-active", benchmark_returns, benchmark=True),
            live_assets=self._live_snapshots(),
        )
        diagnostics = self.service.evaluate_paper_portfolio("paper-active", generated_at=pd.Timestamp("2026-04-20", tz="UTC"))

        summary = build_performance_summary(
            diagnostics=diagnostics,
            registry=self.store.read_frame("paper_portfolio_registry"),
            journal=self.store.read_frame("paper_decision_journal"),
            trades=self.store.read_frame("paper_trade_ledger"),
            equity=self.store.read_frame("paper_equity_curve"),
            benchmark=self.store.read_frame("paper_benchmark_curve"),
            paper_portfolio_id="paper-active",
        )

        self.assertEqual(summary["trade_count"], 5)
        self.assertAlmostEqual(summary["win_rate"], 0.8)
        self.assertGreater(summary["total_return"], summary["benchmark_return"])
        self.assertGreater(summary["alpha"], 0.0)
        self.assertLessEqual(summary["max_drawdown"], 0.0)

    def test_score_outcome_correlation_uses_settled_long_trades(self) -> None:
        returns = [0.001 + idx * 0.001 for idx in range(10)]
        self._save_frames(
            journal=self._journal("paper-active", len(returns)),
            trades=self._trades("paper-active", returns),
            equity=self._equity("paper-active", returns),
            benchmark=self._equity("paper-active", [0.001] * len(returns), benchmark=True),
            live_assets=self._live_snapshots(),
        )

        diagnostics = self.service.evaluate_paper_portfolio("paper-active", generated_at=pd.Timestamp("2026-04-20", tz="UTC"))
        correlation_row = diagnostics.loc[diagnostics["metric_name"] == "score_outcome_correlation"].iloc[0]
        score_outcomes = build_score_outcome_frame(
            self.store.read_frame("paper_decision_journal"),
            self.store.read_frame("paper_trade_ledger"),
            "paper-active",
        )

        self.assertEqual(len(score_outcomes), 10)
        self.assertEqual(correlation_row["severity"], "ok")
        self.assertGreater(float(correlation_row["observed_value"]), 0.9)

    def test_archived_portfolio_selection_does_not_mix_rows(self) -> None:
        registry = self._registry(["paper-active", "paper-archived"])
        journal = pd.concat([self._journal("paper-active", 1), self._journal("paper-archived", 1)], ignore_index=True)
        trades = pd.concat(
            [self._trades("paper-active", [0.02]), self._trades("paper-archived", [-0.05])],
            ignore_index=True,
        )
        equity = pd.concat(
            [self._equity("paper-active", [0.02]), self._equity("paper-archived", [-0.05])],
            ignore_index=True,
        )
        benchmark = pd.concat(
            [
                self._equity("paper-active", [0.001], benchmark=True),
                self._equity("paper-archived", [0.001], benchmark=True),
            ],
            ignore_index=True,
        )
        self._save_frames(registry=registry, journal=journal, trades=trades, equity=equity, benchmark=benchmark)

        active_summary = self.service.build_summary(
            "paper-active",
            diagnostics=self.service.evaluate_paper_portfolio("paper-active"),
        )
        archived_summary = self.service.build_summary(
            "paper-archived",
            diagnostics=self.service.evaluate_paper_portfolio("paper-archived"),
        )

        self.assertGreater(active_summary["total_return"], 0.0)
        self.assertLess(archived_summary["total_return"], 0.0)
        self.assertEqual(active_summary["trade_count"], 1)
        self.assertEqual(archived_summary["trade_count"], 1)

    def test_persist_snapshot_overwrites_latest_and_appends_history(self) -> None:
        self._save_frames(live_assets=self._live_snapshots())

        first = self.service.persist_snapshot("paper-active", generated_at=pd.Timestamp("2026-04-20 12:00", tz="UTC"))
        second = self.service.persist_snapshot("paper-active", generated_at=pd.Timestamp("2026-04-21 12:00", tz="UTC"))

        latest = ensure_performance_diagnostic_frame(self.store.read_frame("model_performance_latest"))
        history = ensure_performance_diagnostic_frame(self.store.read_frame("model_performance_history"))
        self.assertEqual(set(latest["snapshot_id"]), set(second["snapshot_id"]))
        self.assertIn(str(first.iloc[0]["snapshot_id"]), set(history["snapshot_id"].astype(str)))
        self.assertIn(str(second.iloc[0]["snapshot_id"]), set(history["snapshot_id"].astype(str)))

    def test_ensure_performance_diagnostic_frame_fills_missing_columns(self) -> None:
        frame = ensure_performance_diagnostic_frame(
            pd.DataFrame(
                [
                    {
                        "snapshot_id": "s1",
                        "severity": "WARN",
                        "observed_value": "1.5",
                    },
                ],
            ),
        )

        self.assertEqual(frame.columns.tolist(), PERFORMANCE_DIAGNOSTIC_COLUMNS)
        self.assertEqual(frame.iloc[0]["severity"], "warn")
        self.assertAlmostEqual(float(frame.iloc[0]["observed_value"]), 1.5)


if __name__ == "__main__":
    unittest.main()

