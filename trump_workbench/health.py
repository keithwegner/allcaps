from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import EASTERN
from .contracts import NORMALIZED_POST_COLUMNS, TRACKED_ACCOUNT_COLUMNS
from .features import ASSET_POST_MAPPING_COLUMNS
from .market import ASSET_DAILY_COLUMNS, ASSET_INTRADAY_COLUMNS, ASSET_UNIVERSE_COLUMNS, MARKET_MANIFEST_COLUMNS
from .storage import DuckDBStore
from .utils import stable_text_id

HEALTH_CHECK_COLUMNS = [
    "snapshot_id",
    "generated_at",
    "refresh_id",
    "scope_kind",
    "scope_key",
    "check_name",
    "severity",
    "observed_value",
    "baseline_value",
    "detail",
]

REFRESH_HISTORY_COLUMNS = [
    "refresh_id",
    "started_at",
    "completed_at",
    "refresh_mode",
    "status",
    "error_message",
]

SEVERITY_ORDER = {"ok": 0, "warn": 1, "severe": 2}
REQUIRED_DATASET_COLUMNS: dict[str, list[str]] = {
    "normalized_posts": NORMALIZED_POST_COLUMNS,
    "source_manifests": ["source", "post_count"],
    "sp500_daily": ["trade_date", "close"],
    "spy_daily": ["trade_date", "open", "close"],
    "asset_universe": ASSET_UNIVERSE_COLUMNS,
    "asset_daily": ASSET_DAILY_COLUMNS,
    "asset_intraday": ASSET_INTRADAY_COLUMNS,
    "asset_market_manifest": MARKET_MANIFEST_COLUMNS,
    "tracked_accounts": TRACKED_ACCOUNT_COLUMNS,
    "asset_post_mappings": [
        "asset_symbol",
        "post_id",
        "session_date",
        "asset_relevance_score",
        "mapping_reason",
    ],
    "asset_session_features": [
        "asset_symbol",
        "signal_session_date",
        "post_count",
        "target_next_session_return",
        "target_available",
    ],
}
REQUIRED_DATASETS = tuple(REQUIRED_DATASET_COLUMNS.keys())
FRESHNESS_THRESHOLDS_HOURS = {
    "asset_intraday": (6.0, 24.0),
    "asset_market_manifest": (6.0, 24.0),
}
DEFAULT_FRESHNESS_THRESHOLDS = (24.0, 72.0)
DUPLICATE_KEY_COLUMNS = {
    "normalized_posts": ["post_id"],
    "asset_daily": ["symbol", "trade_date"],
    "asset_intraday": ["symbol", "timestamp", "interval"],
}


def _empty_health_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=HEALTH_CHECK_COLUMNS)


def _empty_refresh_history_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REFRESH_HISTORY_COLUMNS)


def _coerce_utc_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _ensure_health_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_health_frame()
    out = frame.copy()
    for column in HEALTH_CHECK_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out["generated_at"] = pd.to_datetime(out["generated_at"], errors="coerce", utc=True)
    out["observed_value"] = pd.to_numeric(out["observed_value"], errors="coerce")
    out["baseline_value"] = pd.to_numeric(out["baseline_value"], errors="coerce")
    out["severity"] = out["severity"].fillna("ok").astype(str).str.lower()
    return out[HEALTH_CHECK_COLUMNS].copy()


def ensure_refresh_history_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_refresh_history_frame()
    out = frame.copy()
    for column in REFRESH_HISTORY_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out["started_at"] = pd.to_datetime(out["started_at"], errors="coerce", utc=True)
    out["completed_at"] = pd.to_datetime(out["completed_at"], errors="coerce", utc=True)
    out["refresh_mode"] = out["refresh_mode"].fillna("").astype(str)
    out["status"] = out["status"].fillna("").astype(str)
    out["error_message"] = out["error_message"].fillna("").astype(str)
    return out[REFRESH_HISTORY_COLUMNS].copy()


def create_refresh_id(refresh_mode: str, started_at: pd.Timestamp) -> str:
    return stable_text_id(refresh_mode, _coerce_utc_timestamp(started_at).isoformat())


def make_refresh_history_frame(
    refresh_id: str,
    refresh_mode: str,
    status: str,
    started_at: pd.Timestamp,
    completed_at: pd.Timestamp,
    error_message: str = "",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "refresh_id": str(refresh_id),
                "started_at": _coerce_utc_timestamp(started_at),
                "completed_at": _coerce_utc_timestamp(completed_at),
                "refresh_mode": str(refresh_mode),
                "status": str(status),
                "error_message": str(error_message or ""),
            },
        ],
        columns=REFRESH_HISTORY_COLUMNS,
    )


def build_health_summary(latest: pd.DataFrame, refresh_history: pd.DataFrame) -> dict[str, Any]:
    checks = _ensure_health_frame(latest)
    history = ensure_refresh_history_frame(refresh_history)
    if checks.empty:
        overall_severity = "ok"
        severe_count = 0
        warn_count = 0
        anomaly_count = 0
    else:
        severity_scores = checks["severity"].map(SEVERITY_ORDER).fillna(0)
        overall_score = int(severity_scores.max()) if not severity_scores.empty else 0
        overall_severity = next(
            severity
            for severity, score in SEVERITY_ORDER.items()
            if score == overall_score
        )
        severe_count = int((checks["severity"] == "severe").sum())
        warn_count = int((checks["severity"] == "warn").sum())
        anomaly_count = int(
            checks.loc[
                checks["check_name"].astype(str).str.contains("anomaly", case=False, na=False)
                & (checks["severity"] != "ok")
            ].shape[0],
        )

    last_refresh_status = "n/a"
    last_refresh_at = pd.NaT
    last_refresh_mode = ""
    if not history.empty:
        order_col = "completed_at" if history["completed_at"].notna().any() else "started_at"
        last_row = history.sort_values(order_col).iloc[-1]
        last_refresh_status = str(last_row.get("status", "n/a") or "n/a")
        last_refresh_at = last_row.get(order_col, pd.NaT)
        last_refresh_mode = str(last_row.get("refresh_mode", "") or "")

    return {
        "overall_severity": overall_severity,
        "severe_count": severe_count,
        "warn_count": warn_count,
        "anomaly_count": anomaly_count,
        "last_refresh_status": last_refresh_status,
        "last_refresh_at": last_refresh_at,
        "last_refresh_mode": last_refresh_mode,
    }


def build_health_trend_frame(history: pd.DataFrame) -> pd.DataFrame:
    checks = _ensure_health_frame(history)
    if checks.empty:
        return pd.DataFrame(columns=["snapshot_id", "generated_at", "ok_count", "warn_count", "severe_count", "overall_severity"])
    grouped = (
        checks.groupby(["snapshot_id", "generated_at", "severity"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["ok", "warn", "severe"]:
        if column not in grouped.columns:
            grouped[column] = 0
    grouped = grouped.rename(columns={"ok": "ok_count", "warn": "warn_count", "severe": "severe_count"})

    def resolve_overall(row: pd.Series) -> str:
        if int(row.get("severe_count", 0)) > 0:
            return "severe"
        if int(row.get("warn_count", 0)) > 0:
            return "warn"
        return "ok"

    grouped["overall_severity"] = grouped.apply(resolve_overall, axis=1)
    return grouped.sort_values("generated_at").reset_index(drop=True)


class DataHealthService:
    def __init__(self, history_window: int = 10) -> None:
        self.history_window = history_window

    def evaluate_store(
        self,
        store: DuckDBStore,
        refresh_id: str = "",
        generated_at: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        datasets = {
            dataset_name: store.read_frame(dataset_name)
            for dataset_name in REQUIRED_DATASETS
        }
        return self.evaluate(
            datasets=datasets,
            dataset_registry=store.dataset_registry(),
            history=store.read_frame("data_health_history"),
            refresh_id=refresh_id,
            generated_at=generated_at,
        )

    def persist_snapshot(
        self,
        store: DuckDBStore,
        refresh_id: str,
        generated_at: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        latest = self.evaluate_store(store, refresh_id=refresh_id, generated_at=generated_at)
        snapshot_id = str(latest.iloc[0]["snapshot_id"]) if not latest.empty else ""
        store.save_frame(
            "data_health_latest",
            latest,
            metadata={"row_count": int(len(latest)), "snapshot_id": snapshot_id},
        )
        store.append_frame(
            "data_health_history",
            latest,
            dedupe_on=["snapshot_id", "scope_kind", "scope_key", "check_name"],
            metadata={"snapshot_id": snapshot_id},
        )
        return latest

    def evaluate(
        self,
        datasets: dict[str, pd.DataFrame],
        dataset_registry: pd.DataFrame,
        history: pd.DataFrame | None = None,
        refresh_id: str = "",
        generated_at: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        generated_ts = _coerce_utc_timestamp(generated_at)
        if pd.isna(generated_ts):
            generated_ts = pd.Timestamp.now(tz="UTC")
        snapshot_id = stable_text_id(refresh_id or "ad-hoc", generated_ts.isoformat())
        checks_history = _ensure_health_frame(history)
        registry = dataset_registry.copy() if dataset_registry is not None else pd.DataFrame()
        if not registry.empty:
            if "dataset_name" not in registry.columns:
                registry["dataset_name"] = ""
            registry["dataset_name"] = registry["dataset_name"].astype(str)
            registry["updated_at"] = registry["updated_at"].map(_coerce_utc_timestamp)

        rows: list[dict[str, Any]] = []

        def add_row(
            scope_kind: str,
            scope_key: str,
            check_name: str,
            severity: str,
            observed_value: float | int | None,
            baseline_value: float | int | None = None,
            detail: str = "",
        ) -> None:
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "generated_at": generated_ts,
                    "refresh_id": str(refresh_id),
                    "scope_kind": str(scope_kind),
                    "scope_key": str(scope_key),
                    "check_name": str(check_name),
                    "severity": str(severity),
                    "observed_value": float(observed_value) if observed_value is not None and pd.notna(observed_value) else np.nan,
                    "baseline_value": float(baseline_value) if baseline_value is not None and pd.notna(baseline_value) else np.nan,
                    "detail": str(detail or ""),
                },
            )

        for dataset_name, required_columns in REQUIRED_DATASET_COLUMNS.items():
            frame = datasets.get(dataset_name, pd.DataFrame())
            row_count = int(len(frame))
            add_row(
                "dataset",
                dataset_name,
                "dataset_presence",
                "severe" if row_count == 0 else "ok",
                row_count,
                detail="Dataset missing or empty." if row_count == 0 else "",
            )
            if row_count == 0:
                continue
            missing_columns = [column for column in required_columns if column not in frame.columns]
            add_row(
                "dataset",
                dataset_name,
                "required_columns",
                "severe" if missing_columns else "ok",
                len(missing_columns),
                detail=", ".join(missing_columns),
            )

        source_manifests = datasets.get("source_manifests", pd.DataFrame())
        if not source_manifests.empty and "status" in source_manifests.columns:
            for idx, row in source_manifests.iterrows():
                scope_key = str(row.get("source") or row.get("provenance") or f"source_{idx}")
                status = str(row.get("status", "ok") or "ok").lower()
                add_row(
                    "source_manifest",
                    scope_key,
                    "manifest_status",
                    "severe" if status == "error" else "ok",
                    row.get("post_count", 0),
                    detail=str(row.get("detail", "") or row.get("provenance", "") or ""),
                )

        asset_market_manifest = datasets.get("asset_market_manifest", pd.DataFrame())
        if not asset_market_manifest.empty and {"symbol", "dataset_kind", "status"}.issubset(asset_market_manifest.columns):
            for _, row in asset_market_manifest.iterrows():
                scope_key = f"{str(row.get('symbol', '')).upper()}:{row.get('dataset_kind', '')}"
                status = str(row.get("status", "ok") or "ok").lower()
                add_row(
                    "asset_market",
                    scope_key,
                    "manifest_status",
                    "severe" if status == "error" else "ok",
                    row.get("row_count", 0),
                    detail=str(row.get("detail", "") or ""),
                )

        for dataset_name in REQUIRED_DATASETS:
            frame = datasets.get(dataset_name, pd.DataFrame())
            if frame.empty:
                continue
            warn_hours, severe_hours = FRESHNESS_THRESHOLDS_HOURS.get(dataset_name, DEFAULT_FRESHNESS_THRESHOLDS)
            registry_row = registry.loc[registry["dataset_name"] == dataset_name].tail(1) if not registry.empty else pd.DataFrame()
            if registry_row.empty or pd.isna(registry_row.iloc[0].get("updated_at", pd.NaT)):
                add_row(
                    "dataset",
                    dataset_name,
                    "freshness_hours",
                    "severe",
                    np.nan,
                    severe_hours,
                    detail="Dataset registry entry or updated_at timestamp is missing.",
                )
                continue
            updated_at = registry_row.iloc[0]["updated_at"]
            age_hours = float((generated_ts - updated_at).total_seconds() / 3600.0)
            severity = "ok"
            if age_hours > severe_hours:
                severity = "severe"
            elif age_hours > warn_hours:
                severity = "warn"
            add_row(
                "dataset",
                dataset_name,
                "freshness_hours",
                severity,
                age_hours,
                warn_hours,
                detail=f"Last updated at {updated_at.tz_convert(EASTERN).isoformat()}",
            )

        for dataset_name, subset in DUPLICATE_KEY_COLUMNS.items():
            frame = datasets.get(dataset_name, pd.DataFrame())
            if frame.empty or not set(subset).issubset(frame.columns):
                continue
            duplicate_rate = float(frame.duplicated(subset=subset, keep="last").mean())
            severity = "ok"
            if duplicate_rate > 0.02:
                severity = "severe"
            elif duplicate_rate > 0.005:
                severity = "warn"
            add_row(
                "dataset",
                dataset_name,
                "duplicate_rate",
                severity,
                duplicate_rate,
                0.005,
                detail=f"Duplicate keys checked on {', '.join(subset)}",
            )

        asset_universe = datasets.get("asset_universe", pd.DataFrame())
        asset_daily = datasets.get("asset_daily", pd.DataFrame())
        if not asset_universe.empty and "symbol" in asset_universe.columns:
            daily_counts = (
                asset_daily.groupby("symbol").size().astype(int).to_dict()
                if not asset_daily.empty and "symbol" in asset_daily.columns
                else {}
            )
            for symbol in asset_universe["symbol"].astype(str).str.upper().tolist():
                count = int(daily_counts.get(symbol, 0))
                add_row(
                    "asset_daily",
                    symbol,
                    "daily_coverage",
                    "severe" if count == 0 else "ok",
                    count,
                    1.0,
                    detail="No daily OHLCV rows were stored for this symbol." if count == 0 else "",
                )

        asset_intraday = datasets.get("asset_intraday", pd.DataFrame())
        if not asset_intraday.empty and {"symbol", "timestamp"}.issubset(asset_intraday.columns):
            intraday = asset_intraday.copy()
            intraday["timestamp"] = pd.to_datetime(intraday["timestamp"], errors="coerce", utc=True)
            intraday = intraday.dropna(subset=["timestamp"]).copy()
            if not intraday.empty:
                global_latest = intraday["timestamp"].max()
                for symbol, group in intraday.groupby(intraday["symbol"].astype(str).str.upper()):
                    symbol_latest = group["timestamp"].max()
                    lag_minutes = float((global_latest - symbol_latest).total_seconds() / 60.0)
                    severity = "ok"
                    if lag_minutes > 120.0:
                        severity = "severe"
                    elif lag_minutes > 30.0:
                        severity = "warn"
                    add_row(
                        "asset_intraday",
                        str(symbol),
                        "intraday_lag_minutes",
                        severity,
                        lag_minutes,
                        30.0,
                        detail=f"Latest intraday row at {symbol_latest.tz_convert(EASTERN).isoformat()}",
                    )

        for dataset_name in REQUIRED_DATASETS:
            frame = datasets.get(dataset_name, pd.DataFrame())
            row_count = int(len(frame))
            baseline = self._history_median(checks_history, "dataset", dataset_name, "row_count_anomaly")
            severity, detail = self._ratio_severity(row_count, baseline)
            add_row("dataset", dataset_name, "row_count_anomaly", severity, row_count, baseline, detail)

        if not asset_market_manifest.empty and {"symbol", "dataset_kind", "row_count"}.issubset(asset_market_manifest.columns):
            for _, row in asset_market_manifest.iterrows():
                scope_key = f"{str(row.get('symbol', '')).upper()}:{row.get('dataset_kind', '')}"
                row_count = float(row.get("row_count", 0) or 0)
                baseline = self._history_median(checks_history, "asset_market", scope_key, "market_coverage_anomaly")
                severity, detail = self._ratio_severity(row_count, baseline)
                add_row("asset_market", scope_key, "market_coverage_anomaly", severity, row_count, baseline, detail)

        health = pd.DataFrame(rows, columns=HEALTH_CHECK_COLUMNS)
        if health.empty:
            return _empty_health_frame()
        health["severity_rank"] = health["severity"].map(SEVERITY_ORDER).fillna(0)
        health = health.sort_values(
            ["severity_rank", "scope_kind", "scope_key", "check_name"],
            ascending=[False, True, True, True],
        ).drop(columns=["severity_rank"])
        return health.reset_index(drop=True)

    def _history_median(
        self,
        history: pd.DataFrame,
        scope_kind: str,
        scope_key: str,
        check_name: str,
    ) -> float | None:
        if history.empty:
            return None
        relevant = history.loc[
            (history["scope_kind"].astype(str) == str(scope_kind))
            & (history["scope_key"].astype(str) == str(scope_key))
            & (history["check_name"].astype(str) == str(check_name))
        ].copy()
        if relevant.empty:
            return None
        relevant = relevant.sort_values("generated_at", ascending=False).head(self.history_window)
        values = pd.to_numeric(relevant["observed_value"], errors="coerce").dropna()
        if values.empty:
            return None
        return float(values.median())

    @staticmethod
    def _ratio_severity(current_value: float | int, baseline_value: float | None) -> tuple[str, str]:
        if baseline_value is None or pd.isna(baseline_value) or float(baseline_value) <= 0.0:
            return "ok", ""
        ratio = float(current_value) / float(baseline_value)
        if ratio < 0.5 or ratio > 2.0:
            return "severe", f"Current value is {ratio:.2f}x the recent median baseline."
        if ratio < 0.75 or ratio > 1.5:
            return "warn", f"Current value is {ratio:.2f}x the recent median baseline."
        return "ok", f"Current value is {ratio:.2f}x the recent median baseline."
