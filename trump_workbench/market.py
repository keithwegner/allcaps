from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
import requests

from .config import DEFAULT_ETF_SYMBOLS, EASTERN

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TrumpTradingWorkbench/2.0; +https://example.invalid)",
}
WATCHLIST_COLUMNS = ["symbol", "display_name", "asset_type", "source"]
ASSET_UNIVERSE_COLUMNS = ["symbol", "display_name", "asset_type", "source", "is_default", "is_watchlist"]
ASSET_DAILY_COLUMNS = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
ASSET_INTRADAY_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "interval"]
MARKET_MANIFEST_COLUMNS = ["symbol", "dataset_kind", "row_count", "status", "start_at", "end_at", "detail"]


def normalize_symbols(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        symbol = str(value or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def build_watchlist_frame(symbols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    rows = [
        {
            "symbol": symbol,
            "display_name": symbol,
            "asset_type": "equity",
            "source": "watchlist",
        }
        for symbol in normalize_symbols(symbols)
        if symbol not in DEFAULT_ETF_SYMBOLS
    ]
    return pd.DataFrame(rows, columns=WATCHLIST_COLUMNS)


def build_asset_universe(watchlist_symbols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol in DEFAULT_ETF_SYMBOLS:
        rows.append(
            {
                "symbol": symbol,
                "display_name": symbol,
                "asset_type": "etf",
                "source": "core_etf",
                "is_default": True,
                "is_watchlist": False,
            },
        )
    for symbol in normalize_symbols(watchlist_symbols):
        if symbol in DEFAULT_ETF_SYMBOLS:
            continue
        rows.append(
            {
                "symbol": symbol,
                "display_name": symbol,
                "asset_type": "equity",
                "source": "watchlist",
                "is_default": False,
                "is_watchlist": True,
            },
        )
    return pd.DataFrame(rows, columns=ASSET_UNIVERSE_COLUMNS)


def _standardize_ohlcv_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "trade_date"})
    df.columns = [str(column).lower() for column in df.columns]
    if "trade_date" not in df.columns:
        raise RuntimeError(f"{symbol} daily data returned no trade_date column.")
    keep = ["trade_date", "open", "high", "low", "close", "volume"]
    for column in keep[1:]:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["trade_date", "open", "close"]).sort_values("trade_date").reset_index(drop=True)
    df.insert(0, "symbol", symbol)
    return df[ASSET_DAILY_COLUMNS]


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
        return self.load_asset_daily("SPY", start, end).drop(columns=["symbol"])

    def load_asset_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        try:
            import yfinance as yf  # type: ignore

            df = yf.download(
                symbol,
                start=start_ts.strftime("%Y-%m-%d"),
                end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to load {symbol} daily data: {exc}") from exc
        return _standardize_ohlcv_frame(df, symbol)

    def load_assets_daily(self, symbols: list[str] | tuple[str, ...], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[pd.DataFrame] = []
        manifest_rows: list[dict[str, Any]] = []
        for symbol in normalize_symbols(symbols):
            try:
                daily = self.load_asset_daily(symbol, start, end)
                rows.append(daily)
                manifest_rows.append(
                    {
                        "symbol": symbol,
                        "dataset_kind": "daily",
                        "row_count": int(len(daily)),
                        "status": "ok",
                        "start_at": daily["trade_date"].min() if not daily.empty else pd.NaT,
                        "end_at": daily["trade_date"].max() if not daily.empty else pd.NaT,
                        "detail": "",
                    },
                )
            except Exception as exc:
                manifest_rows.append(
                    {
                        "symbol": symbol,
                        "dataset_kind": "daily",
                        "row_count": 0,
                        "status": "error",
                        "start_at": pd.NaT,
                        "end_at": pd.NaT,
                        "detail": str(exc),
                    },
                )
        daily_frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=ASSET_DAILY_COLUMNS)
        manifest = pd.DataFrame(manifest_rows, columns=MARKET_MANIFEST_COLUMNS)
        return daily_frame, manifest

    def load_asset_intraday(
        self,
        symbol: str,
        interval: str = "5m",
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        period_days = max(1, min(int(lookback_days), 60))
        try:
            import yfinance as yf  # type: ignore

            df = yf.download(
                symbol,
                period=f"{period_days}d",
                interval=interval,
                progress=False,
                auto_adjust=False,
                prepost=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to load {symbol} intraday data: {exc}") from exc

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={"Datetime": "timestamp", "Date": "timestamp"})
        df.columns = [str(column).lower() for column in df.columns]
        if "timestamp" not in df.columns:
            raise RuntimeError(f"{symbol} intraday data did not include a timestamp column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()
        df["timestamp"] = df["timestamp"].map(
            lambda ts: ts.tz_localize(EASTERN) if ts.tzinfo is None else ts.tz_convert(EASTERN),
        )
        for column in ["open", "high", "low", "close", "volume"]:
            if column not in df.columns:
                df[column] = pd.NA
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["open", "close"]).sort_values("timestamp").reset_index(drop=True)
        df.insert(0, "symbol", symbol)
        df["interval"] = interval
        return df[ASSET_INTRADAY_COLUMNS]

    def load_assets_intraday(
        self,
        symbols: list[str] | tuple[str, ...],
        interval: str = "5m",
        lookback_days: int = 30,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[pd.DataFrame] = []
        manifest_rows: list[dict[str, Any]] = []
        for symbol in normalize_symbols(symbols):
            try:
                intraday = self.load_asset_intraday(symbol, interval=interval, lookback_days=lookback_days)
                rows.append(intraday)
                manifest_rows.append(
                    {
                        "symbol": symbol,
                        "dataset_kind": f"intraday_{interval}",
                        "row_count": int(len(intraday)),
                        "status": "ok",
                        "start_at": intraday["timestamp"].min() if not intraday.empty else pd.NaT,
                        "end_at": intraday["timestamp"].max() if not intraday.empty else pd.NaT,
                        "detail": "",
                    },
                )
            except Exception as exc:
                manifest_rows.append(
                    {
                        "symbol": symbol,
                        "dataset_kind": f"intraday_{interval}",
                        "row_count": 0,
                        "status": "error",
                        "start_at": pd.NaT,
                        "end_at": pd.NaT,
                        "detail": str(exc),
                    },
                )
        intraday_frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=ASSET_INTRADAY_COLUMNS)
        manifest = pd.DataFrame(manifest_rows, columns=MARKET_MANIFEST_COLUMNS)
        return intraday_frame, manifest

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
