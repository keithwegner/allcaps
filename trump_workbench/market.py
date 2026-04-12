from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
import requests

from .config import EASTERN

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TrumpTradingWorkbench/2.0; +https://example.invalid)",
}


class MarketDataService:
    def load_sp500_daily(self, start: str, end: str) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        errors: list[str] = []

        try:
            from pandas_datareader import data as pdr  # type: ignore

            df = pdr.DataReader("SP500", "fred", start_ts, end_ts).reset_index()
            df = df.rename(columns={"DATE": "trade_date", "SP500": "close"})
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("trade_date").reset_index(drop=True)
            if not df.empty:
                return df[["trade_date", "close"]]
            errors.append("FRED returned an empty dataframe.")
        except Exception as exc:
            errors.append(f"FRED load failed: {exc}")

        try:
            import yfinance as yf  # type: ignore

            df = yf.download(
                "^GSPC",
                start=start_ts.strftime("%Y-%m-%d"),
                end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index().rename(columns={"Date": "trade_date", "Close": "close"})
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("trade_date").reset_index(drop=True)
            if not df.empty:
                return df[["trade_date", "close"]]
            errors.append("yfinance returned an empty dataframe.")
        except Exception as exc:
            errors.append(f"yfinance load failed: {exc}")

        raise RuntimeError("Unable to load daily S&P 500 data. " + " | ".join(errors))

    def load_spy_daily(self, start: str, end: str) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        try:
            import yfinance as yf  # type: ignore

            df = yf.download(
                "SPY",
                start=start_ts.strftime("%Y-%m-%d"),
                end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to load SPY daily data: {exc}") from exc

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={"Date": "trade_date"})
        df.columns = [str(column).lower() for column in df.columns]
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        keep = ["trade_date", "open", "high", "low", "close", "volume"]
        for column in keep[1:]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["open", "close"]).sort_values("trade_date").reset_index(drop=True)
        return df[keep]

    def load_spy_intraday_month(
        self,
        month_str: str,
        interval: str,
        api_key: str,
    ) -> pd.DataFrame:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": "SPY",
            "interval": interval,
            "month": month_str,
            "outputsize": "full",
            "extended_hours": "false",
            "datatype": "csv",
            "apikey": api_key,
        }
        response = requests.get(ALPHA_VANTAGE_URL, params=params, headers=HTTP_HEADERS, timeout=60)
        response.raise_for_status()
        text = response.text.strip()

        if text.startswith("{"):
            payload = json.loads(text)
            message = payload.get("Error Message") or payload.get("Note") or payload.get("Information") or str(payload)
            raise RuntimeError(message)

        df = pd.read_csv(io.StringIO(text))
        if "timestamp" not in df.columns:
            raise RuntimeError("Alpha Vantage intraday response did not include a timestamp column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()
        df["timestamp"] = df["timestamp"].map(
            lambda ts: ts.tz_localize(EASTERN) if ts.tzinfo is None else ts.tz_convert(EASTERN),
        )
        for column in ["open", "high", "low", "close", "volume"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df.sort_values("timestamp").reset_index(drop=True)
