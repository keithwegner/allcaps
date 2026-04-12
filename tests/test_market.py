from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import pandas as pd

from trump_workbench.market import MarketDataService


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

        failing_yfinance = types.ModuleType("yfinance")
        failing_yfinance.download = mock.Mock(side_effect=RuntimeError("spy failed"))
        with mock.patch.dict(sys.modules, {"yfinance": failing_yfinance}):
            with self.assertRaises(RuntimeError):
                self.service.load_spy_daily("2025-02-01", "2025-02-05")

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
