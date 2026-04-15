from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trump_workbench.config import AppSettings, EASTERN
from trump_workbench.health import (
    HEALTH_CHECK_COLUMNS,
    DataHealthService,
    create_refresh_id,
)
from trump_workbench.storage import DuckDBStore


class DataHealthServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = AppSettings(base_dir=Path(self.temp_dir.name))
        self.store = DuckDBStore(self.settings)
        self.service = DataHealthService()
        self.generated_at = pd.Timestamp("2026-04-15 15:00:00", tz="UTC")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _complete_datasets(self) -> dict[str, pd.DataFrame]:
        base_ts = pd.Timestamp("2026-04-15 10:00:00", tz=EASTERN)
        return {
            "normalized_posts": pd.DataFrame(
                [
                    {
                        "source_platform": "X",
                        "source_type": "x_csv",
                        "author_account_id": "acct-a",
                        "author_handle": "macroa",
                        "author_display_name": "Macro A",
                        "author_is_trump": False,
                        "post_id": "post-1",
                        "post_url": "https://x.com/macroa/status/1",
                        "post_timestamp": base_ts,
                        "raw_text": "Trump growth",
                        "cleaned_text": "Trump growth",
                        "is_reshare": False,
                        "has_media": False,
                        "replies_count": 1,
                        "reblogs_count": 2,
                        "favourites_count": 3,
                        "mentions_trump": True,
                        "source_provenance": "unit-test",
                        "engagement_score": 6.0,
                        "sentiment_score": 0.2,
                        "sentiment_label": "positive",
                    },
                ],
            ),
            "source_manifests": pd.DataFrame(
                [
                    {
                        "source": "Truth Social archive",
                        "provenance": "unit-test",
                        "post_count": 1,
                        "coverage_start": base_ts,
                        "coverage_end": base_ts,
                        "status": "ok",
                        "detail": "",
                    },
                ],
            ),
            "sp500_daily": pd.DataFrame([{"trade_date": pd.Timestamp("2026-04-15"), "close": 6100.0}]),
            "spy_daily": pd.DataFrame(
                [
                    {
                        "trade_date": pd.Timestamp("2026-04-15"),
                        "open": 600.0,
                        "high": 605.0,
                        "low": 598.0,
                        "close": 603.0,
                        "volume": 1_000_000,
                    },
                ],
            ),
            "asset_universe": pd.DataFrame(
                [
                    {"symbol": "SPY", "display_name": "SPY", "asset_type": "etf", "source": "core_etf", "is_default": True, "is_watchlist": False},
                    {"symbol": "QQQ", "display_name": "QQQ", "asset_type": "etf", "source": "core_etf", "is_default": True, "is_watchlist": False},
                ],
            ),
            "asset_daily": pd.DataFrame(
                [
                    {"symbol": "SPY", "trade_date": pd.Timestamp("2026-04-15"), "open": 600.0, "high": 605.0, "low": 598.0, "close": 603.0, "volume": 1_000_000},
                    {"symbol": "QQQ", "trade_date": pd.Timestamp("2026-04-15"), "open": 500.0, "high": 506.0, "low": 498.0, "close": 504.0, "volume": 500_000},
                ],
            ),
            "asset_intraday": pd.DataFrame(
                [
                    {"symbol": "SPY", "timestamp": base_ts, "open": 600.0, "high": 601.0, "low": 599.0, "close": 600.5, "volume": 1000, "interval": "5m"},
                    {"symbol": "QQQ", "timestamp": base_ts, "open": 500.0, "high": 501.0, "low": 499.0, "close": 500.5, "volume": 900, "interval": "5m"},
                ],
            ),
            "asset_market_manifest": pd.DataFrame(
                [
                    {"symbol": "SPY", "dataset_kind": "daily", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15"), "end_at": pd.Timestamp("2026-04-15"), "detail": ""},
                    {"symbol": "SPY", "dataset_kind": "intraday_5m", "row_count": 1, "status": "ok", "start_at": base_ts, "end_at": base_ts, "detail": ""},
                    {"symbol": "QQQ", "dataset_kind": "daily", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15"), "end_at": pd.Timestamp("2026-04-15"), "detail": ""},
                    {"symbol": "QQQ", "dataset_kind": "intraday_5m", "row_count": 1, "status": "ok", "start_at": base_ts, "end_at": base_ts, "detail": ""},
                ],
            ),
            "tracked_accounts": pd.DataFrame(
                [
                    {
                        "version_id": "v1",
                        "account_id": "acct-a",
                        "handle": "macroa",
                        "display_name": "Macro A",
                        "source_platform": "X",
                        "discovery_score": 1.0,
                        "status": "active",
                        "first_seen_at": pd.Timestamp("2026-04-10"),
                        "last_seen_at": pd.Timestamp("2026-04-15"),
                        "effective_from": pd.Timestamp("2026-04-15"),
                        "effective_to": pd.NaT,
                        "auto_included": True,
                        "provenance": "unit-test",
                        "mention_count": 1,
                        "engagement_mean": 10.0,
                        "active_days": 1,
                    },
                ],
            ),
            "asset_post_mappings": pd.DataFrame(
                [
                    {
                        "asset_symbol": "QQQ",
                        "post_id": "post-1",
                        "session_date": pd.Timestamp("2026-04-15"),
                        "asset_relevance_score": 0.6,
                        "mapping_reason": "same-session",
                    },
                ],
            ),
            "asset_session_features": pd.DataFrame(
                [
                    {
                        "asset_symbol": "QQQ",
                        "signal_session_date": pd.Timestamp("2026-04-15"),
                        "post_count": 1,
                        "target_next_session_return": 0.01,
                        "target_available": True,
                    },
                ],
            ),
        }

    def _dataset_registry(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows = []
        for dataset_name, frame in datasets.items():
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "parquet_path": f"/tmp/{dataset_name}.parquet",
                    "row_count": int(len(frame)),
                    "updated_at": self.generated_at - pd.Timedelta(hours=1),
                    "metadata_json": "{}",
                },
            )
        return pd.DataFrame(rows)

    def _history_rows(self, scope_kind: str, scope_key: str, check_name: str, observed_values: list[float]) -> pd.DataFrame:
        rows = []
        for idx, value in enumerate(observed_values):
            ts = self.generated_at - pd.Timedelta(days=idx + 1)
            rows.append(
                {
                    "snapshot_id": f"snap-{idx}",
                    "generated_at": ts,
                    "refresh_id": f"refresh-{idx}",
                    "scope_kind": scope_kind,
                    "scope_key": scope_key,
                    "check_name": check_name,
                    "severity": "ok",
                    "observed_value": float(value),
                    "baseline_value": pd.NA,
                    "detail": "",
                },
            )
        return pd.DataFrame(rows, columns=HEALTH_CHECK_COLUMNS)

    def test_missing_required_dataset_is_severe(self) -> None:
        datasets = self._complete_datasets()
        datasets["normalized_posts"] = pd.DataFrame()

        health = self.service.evaluate(
            datasets=datasets,
            dataset_registry=self._dataset_registry(datasets),
            history=pd.DataFrame(columns=HEALTH_CHECK_COLUMNS),
            refresh_id="refresh-1",
            generated_at=self.generated_at,
        )

        row = health.loc[
            (health["scope_key"] == "normalized_posts")
            & (health["check_name"] == "dataset_presence")
        ].iloc[0]
        self.assertEqual(row["severity"], "severe")

    def test_missing_required_columns_is_severe(self) -> None:
        datasets = self._complete_datasets()
        datasets["asset_daily"] = pd.DataFrame(
            [{"symbol": "SPY", "trade_date": pd.Timestamp("2026-04-15"), "close": 603.0}],
        )

        health = self.service.evaluate(
            datasets=datasets,
            dataset_registry=self._dataset_registry(datasets),
            history=pd.DataFrame(columns=HEALTH_CHECK_COLUMNS),
            refresh_id="refresh-2",
            generated_at=self.generated_at,
        )

        row = health.loc[
            (health["scope_key"] == "asset_daily")
            & (health["check_name"] == "required_columns")
        ].iloc[0]
        self.assertEqual(row["severity"], "severe")
        self.assertIn("open", str(row["detail"]))

    def test_manifest_errors_are_severe(self) -> None:
        datasets = self._complete_datasets()
        datasets["source_manifests"].loc[0, "status"] = "error"
        datasets["source_manifests"].loc[0, "detail"] = "remote CSV timeout"
        datasets["asset_market_manifest"].loc[1, "status"] = "error"
        datasets["asset_market_manifest"].loc[1, "detail"] = "intraday download failed"

        health = self.service.evaluate(
            datasets=datasets,
            dataset_registry=self._dataset_registry(datasets),
            history=pd.DataFrame(columns=HEALTH_CHECK_COLUMNS),
            refresh_id="refresh-3",
            generated_at=self.generated_at,
        )

        source_row = health.loc[
            (health["scope_kind"] == "source_manifest")
            & (health["check_name"] == "manifest_status")
        ].iloc[0]
        market_row = health.loc[
            (health["scope_key"] == "SPY:intraday_5m")
            & (health["check_name"] == "manifest_status")
        ].iloc[0]
        self.assertEqual(source_row["severity"], "severe")
        self.assertEqual(market_row["severity"], "severe")

    def test_duplicate_rate_warn_and_severe_thresholds(self) -> None:
        datasets = self._complete_datasets()
        normalized_rows = []
        for idx in range(100):
            normalized_rows.append(
                {
                    **datasets["normalized_posts"].iloc[0].to_dict(),
                    "post_id": f"post-{idx}",
                    "post_url": f"https://x.com/macroa/status/{idx}",
                },
            )
        normalized_rows.append({**normalized_rows[0]})
        datasets["normalized_posts"] = pd.DataFrame(normalized_rows)

        daily_rows = []
        for idx in range(100):
            daily_rows.append(
                {
                    "symbol": "SPY",
                    "trade_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1_000,
                },
            )
        daily_rows.extend([daily_rows[0], daily_rows[1], daily_rows[2]])
        datasets["asset_daily"] = pd.DataFrame(daily_rows)

        health = self.service.evaluate(
            datasets=datasets,
            dataset_registry=self._dataset_registry(datasets),
            history=pd.DataFrame(columns=HEALTH_CHECK_COLUMNS),
            refresh_id="refresh-4",
            generated_at=self.generated_at,
        )

        post_row = health.loc[
            (health["scope_key"] == "normalized_posts")
            & (health["check_name"] == "duplicate_rate")
        ].iloc[0]
        daily_row = health.loc[
            (health["scope_key"] == "asset_daily")
            & (health["check_name"] == "duplicate_rate")
        ].iloc[0]
        self.assertEqual(post_row["severity"], "warn")
        self.assertEqual(daily_row["severity"], "severe")

    def test_row_count_anomaly_thresholds_and_intraday_lag(self) -> None:
        datasets = self._complete_datasets()
        datasets["normalized_posts"] = pd.concat(
            [datasets["normalized_posts"]] * 160,
            ignore_index=True,
        ).assign(post_id=lambda df: [f"post-{idx}" for idx in range(len(df))])
        datasets["asset_intraday"] = pd.DataFrame(
            [
                {"symbol": "SPY", "timestamp": pd.Timestamp("2026-04-15 10:00:00", tz=EASTERN), "open": 600.0, "high": 601.0, "low": 599.0, "close": 600.5, "volume": 1000, "interval": "5m"},
                {"symbol": "QQQ", "timestamp": pd.Timestamp("2026-04-15 09:20:00", tz=EASTERN), "open": 500.0, "high": 501.0, "low": 499.0, "close": 500.5, "volume": 900, "interval": "5m"},
                {"symbol": "XLK", "timestamp": pd.Timestamp("2026-04-15 07:50:00", tz=EASTERN), "open": 200.0, "high": 201.0, "low": 199.0, "close": 200.5, "volume": 700, "interval": "5m"},
            ],
        )
        datasets["asset_market_manifest"] = pd.DataFrame(
            [
                {"symbol": "SPY", "dataset_kind": "daily", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15"), "end_at": pd.Timestamp("2026-04-15"), "detail": ""},
                {"symbol": "SPY", "dataset_kind": "intraday_5m", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15 10:00:00", tz=EASTERN), "end_at": pd.Timestamp("2026-04-15 10:00:00", tz=EASTERN), "detail": ""},
                {"symbol": "QQQ", "dataset_kind": "daily", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15"), "end_at": pd.Timestamp("2026-04-15"), "detail": ""},
                {"symbol": "QQQ", "dataset_kind": "intraday_5m", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15 09:20:00", tz=EASTERN), "end_at": pd.Timestamp("2026-04-15 09:20:00", tz=EASTERN), "detail": ""},
                {"symbol": "XLK", "dataset_kind": "intraday_5m", "row_count": 1, "status": "ok", "start_at": pd.Timestamp("2026-04-15 07:50:00", tz=EASTERN), "end_at": pd.Timestamp("2026-04-15 07:50:00", tz=EASTERN), "detail": ""},
            ],
        )
        history = self._history_rows("dataset", "normalized_posts", "row_count_anomaly", [100.0] * 10)

        health = self.service.evaluate(
            datasets=datasets,
            dataset_registry=self._dataset_registry(datasets),
            history=history,
            refresh_id="refresh-5",
            generated_at=self.generated_at,
        )

        anomaly_row = health.loc[
            (health["scope_key"] == "normalized_posts")
            & (health["check_name"] == "row_count_anomaly")
        ].iloc[0]
        qqq_lag = health.loc[
            (health["scope_key"] == "QQQ")
            & (health["check_name"] == "intraday_lag_minutes")
        ].iloc[0]
        xlk_lag = health.loc[
            (health["scope_key"] == "XLK")
            & (health["check_name"] == "intraday_lag_minutes")
        ].iloc[0]
        self.assertEqual(anomaly_row["severity"], "warn")
        self.assertEqual(qqq_lag["severity"], "warn")
        self.assertEqual(xlk_lag["severity"], "severe")

    def test_persist_snapshot_round_trips_latest_and_history(self) -> None:
        datasets = self._complete_datasets()
        for dataset_name, frame in datasets.items():
            self.store.save_frame(dataset_name, frame, metadata={"row_count": int(len(frame))})

        first_refresh_id = create_refresh_id("full", self.generated_at)
        first_latest = self.service.persist_snapshot(self.store, refresh_id=first_refresh_id, generated_at=self.generated_at)
        second_refresh_id = create_refresh_id("incremental", self.generated_at + pd.Timedelta(minutes=5))
        second_latest = self.service.persist_snapshot(
            self.store,
            refresh_id=second_refresh_id,
            generated_at=self.generated_at + pd.Timedelta(minutes=5),
        )

        latest = self.store.read_frame("data_health_latest")
        history = self.store.read_frame("data_health_history")

        self.assertFalse(first_latest.empty)
        self.assertFalse(second_latest.empty)
        self.assertEqual(str(latest.iloc[0]["refresh_id"]), second_refresh_id)
        self.assertGreaterEqual(history["snapshot_id"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
