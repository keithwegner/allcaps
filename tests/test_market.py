from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import pandas as pd

from trump_workbench.market import (
    ASSET_DAILY_COLUMNS,
    ASSET_INTRADAY_COLUMNS,
    build_asset_universe,
    build_watchlist_frame,
    normalize_symbols,
    MarketDataService,
)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class MarketDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = MarketDataService()

    def test_load_sp500_daily_uses_fred_when_available(self) -> None:
        fred_df = pd.DataFrame(
            {
                "DATE": pd.to_datetime(["2025-02-03", "2025-02-04", "2025-02-05"]),
                "SP500": [6000.0, None, 6025.0],
            },
        )
        pandas_datareader = types.ModuleType("pandas_datareader")
        pandas_datareader.data = types.SimpleNamespace(DataReader=lambda *args, **kwargs: fred_df)

        with mock.patch.dict(sys.modules, {"pandas_datareader": pandas_datareader}):
            result = self.service.load_sp500_daily("2025-02-01", "2025-02-05")

        self.assertEqual(result["close"].tolist(), [6000.0, 6025.0])

    def test_load_sp500_daily_falls_back_to_yfinance(self) -> None:
        pandas_datareader = types.ModuleType("pandas_datareader")
        pandas_datareader.data = types.SimpleNamespace(DataReader=mock.Mock(side_effect=RuntimeError("fred down")))

        columns = pd.MultiIndex.from_tuples([("Close", "^GSPC")])
        yfinance_frame = pd.DataFrame(
            [[6100.0], [6150.0]],
            index=pd.to_datetime(["2025-02-03", "2025-02-04"]),
            columns=columns,
        )
        yfinance_frame.index.name = "Date"
        yfinance = types.ModuleType("yfinance")
        yfinance.download = mock.Mock(return_value=yfinance_frame)

        with mock.patch.dict(sys.modules, {"pandas_datareader": pandas_datareader, "yfinance": yfinance}):
            result = self.service.load_sp500_daily("2025-02-01", "2025-02-05")

        self.assertEqual(result["close"].tolist(), [6100.0, 6150.0])

    def test_load_sp500_daily_raises_after_all_sources_fail(self) -> None:
        pandas_datareader = types.ModuleType("pandas_datareader")
        pandas_datareader.data = types.SimpleNamespace(DataReader=mock.Mock(side_effect=RuntimeError("fred down")))
        yfinance = types.ModuleType("yfinance")
        yfinance.download = mock.Mock(side_effect=RuntimeError("yf down"))

        with mock.patch.dict(sys.modules, {"pandas_datareader": pandas_datareader, "yfinance": yfinance}):
            with self.assertRaises(RuntimeError):
                self.service.load_sp500_daily("2025-02-01", "2025-02-05")

    def test_load_spy_daily_handles_success_and_failure(self) -> None:
        columns = pd.MultiIndex.from_tuples(
            [
                ("Open", "SPY"),
                ("High", "SPY"),
                ("Low", "SPY"),
                ("Close", "SPY"),
                ("Volume", "SPY"),
            ],
        )
        yfinance_frame = pd.DataFrame(
            [[100.0, 101.0, 99.0, 100.5, 1_000_000]],
            index=pd.to_datetime(["2025-02-03"]),
            columns=columns,
        )
        yfinance_frame.index.name = "Date"
        yfinance = types.ModuleType("yfinance")
        yfinance.download = mock.Mock(return_value=yfinance_frame)

        with mock.patch.dict(sys.modules, {"yfinance": yfinance}):
            result = self.service.load_spy_daily("2025-02-01", "2025-02-05")

        self.assertEqual(result.iloc[0]["close"], 100.5)
        with mock.patch.dict(sys.modules, {"yfinance": yfinance}):
            generic = self.service.load_asset_daily("QQQ", "2025-02-01", "2025-02-05")
        self.assertEqual(generic.iloc[0]["symbol"], "QQQ")
        self.assertEqual(generic.iloc[0]["close"], 100.5)

        failing_yfinance = types.ModuleType("yfinance")
        failing_yfinance.download = mock.Mock(side_effect=RuntimeError("spy failed"))
        with mock.patch.dict(sys.modules, {"yfinance": failing_yfinance}):
            with self.assertRaises(RuntimeError):
                self.service.load_spy_daily("2025-02-01", "2025-02-05")

    def test_asset_universe_and_batch_market_loaders(self) -> None:
        self.assertEqual(normalize_symbols([" spy ", "QQQ", "spy", "", "MSFT"]), ["SPY", "QQQ", "MSFT"])
        watchlist = build_watchlist_frame(["msft", "nvda", "SPY"])
        self.assertEqual(watchlist["symbol"].tolist(), ["MSFT", "NVDA"])
        universe = build_asset_universe(["msft", "nvda"])
        self.assertIn("SPY", universe["symbol"].tolist())
        self.assertIn("MSFT", universe["symbol"].tolist())

        def fake_download(symbol: str, *args, **kwargs) -> pd.DataFrame:
            if kwargs.get("interval"):
                frame = pd.DataFrame(
                    {
                        "Open": [100.0, 101.0],
                        "High": [101.0, 102.0],
                        "Low": [99.5, 100.5],
                        "Close": [100.5, 101.5],
                        "Volume": [1000, 1100],
                    },
                    index=pd.to_datetime(["2025-02-03 09:30:00", "2025-02-03 09:35:00"]),
                )
                frame.index.name = "Datetime"
                return frame
            frame = pd.DataFrame(
                {
                    "Open": [100.0, 101.0],
                    "High": [101.0, 102.0],
                    "Low": [99.0, 100.0],
                    "Close": [100.5, 101.5],
                    "Volume": [1_000_000, 1_200_000],
                },
                index=pd.to_datetime(["2025-02-03", "2025-02-04"]),
            )
            frame.index.name = "Date"
            return frame

        yfinance = types.ModuleType("yfinance")
        yfinance.download = mock.Mock(side_effect=fake_download)
        with mock.patch.dict(sys.modules, {"yfinance": yfinance}):
            daily, daily_manifest = self.service.load_assets_daily(["SPY", "MSFT"], "2025-02-01", "2025-02-05")
            intraday, intraday_manifest = self.service.load_assets_intraday(["SPY", "MSFT"], interval="5m", lookback_days=5)

        self.assertListEqual(list(daily.columns), ASSET_DAILY_COLUMNS)
        self.assertListEqual(list(intraday.columns), ASSET_INTRADAY_COLUMNS)
        self.assertEqual(sorted(daily["symbol"].unique().tolist()), ["MSFT", "SPY"])
        self.assertEqual(sorted(intraday["symbol"].unique().tolist()), ["MSFT", "SPY"])
        self.assertTrue((daily_manifest["status"] == "ok").all())
        self.assertTrue((intraday_manifest["status"] == "ok").all())

    def test_load_spy_intraday_month_handles_csv_and_error_payloads(self) -> None:
        csv_text = (
            "timestamp,open,high,low,close,volume\n"
            "2025-02-03 09:30:00,100,101,99,100.5,1000\n"
            "2025-02-03 09:35:00,100.5,101.5,100,101,1200\n"
        )

        with mock.patch("trump_workbench.market.requests.get", return_value=_FakeResponse(csv_text)):
            intraday = self.service.load_spy_intraday_month("2025-02", "5min", "demo")

        self.assertEqual(len(intraday), 2)
        self.assertEqual(str(intraday.iloc[0]["timestamp"].tz), "America/New_York")

        with mock.patch("trump_workbench.market.requests.get", return_value=_FakeResponse('{"Error Message":"bad key"}')):
            with self.assertRaises(RuntimeError):
                self.service.load_spy_intraday_month("2025-02", "5min", "demo")

        with mock.patch("trump_workbench.market.requests.get", return_value=_FakeResponse("open,close\n1,2\n")):
            with self.assertRaises(RuntimeError):
                self.service.load_spy_intraday_month("2025-02", "5min", "demo")


if __name__ == "__main__":
    unittest.main()
