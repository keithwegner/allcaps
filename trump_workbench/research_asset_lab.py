from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .config import AppSettings, DEFAULT_ETF_SYMBOLS
from .market import ASSET_INTRADAY_COLUMNS
from .research import (
    build_asset_comparison_chart,
    build_asset_comparison_frame,
    build_event_study_chart,
    build_event_study_frame,
    build_intraday_comparison_chart,
    build_intraday_comparison_frame,
    filter_posts,
    make_asset_mapping_table,
    make_asset_session_table,
    truncate_text,
)
from .research_workspace import (
    _figure_json,
    _frame_records,
    _public_filter_payload,
    _resolve_filters,
    detect_source_mode,
)
from .storage import DuckDBStore

TABLE_LIMIT = 250
DEFAULT_INTRADAY_BEFORE_MINUTES = 120
DEFAULT_INTRADAY_AFTER_MINUTES = 240


def _to_records(frame: pd.DataFrame, limit: int | None = TABLE_LIMIT) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return _frame_records(frame.head(limit) if limit is not None else frame)


def _normalize_mode(value: str | None) -> str:
    return "price" if str(value or "").strip().lower() == "price" else "normalized"


def _coerce_positive_int(value: int | None, default: int, minimum: int = 1, maximum: int = 390) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _asset_options(asset_universe: pd.DataFrame, asset_daily: pd.DataFrame) -> list[dict[str, Any]]:
    daily_symbols = (
        set(asset_daily["symbol"].dropna().astype(str).str.upper().tolist())
        if not asset_daily.empty and "symbol" in asset_daily.columns
        else set()
    )
    options: list[dict[str, Any]] = []
    if not asset_universe.empty and "symbol" in asset_universe.columns:
        universe = asset_universe.copy()
        universe["symbol"] = universe["symbol"].astype(str).str.upper()
        universe = universe.loc[universe["symbol"] != "SPY"].copy()
        for _, row in universe.iterrows():
            symbol = str(row.get("symbol", "") or "").upper()
            if not symbol:
                continue
            source = str(row.get("source", "") or "")
            is_watchlist = bool(row.get("is_watchlist", False)) or source == "watchlist"
            display_name = str(row.get("display_name", "") or symbol)
            options.append(
                {
                    "symbol": symbol,
                    "label": f"{symbol} - {display_name}",
                    "source": source,
                    "is_watchlist": is_watchlist,
                    "has_daily": symbol in daily_symbols if daily_symbols else True,
                },
            )
    elif daily_symbols:
        for symbol in sorted(symbol for symbol in daily_symbols if symbol != "SPY"):
            options.append(
                {
                    "symbol": symbol,
                    "label": symbol,
                    "source": "asset_daily",
                    "is_watchlist": False,
                    "has_daily": True,
                },
            )
    return sorted(options, key=lambda item: (not bool(item["is_watchlist"]), str(item["symbol"])))


def _select_asset(asset_options: list[dict[str, Any]], selected_asset: str | None) -> str:
    symbols = [str(option["symbol"]).upper() for option in asset_options]
    requested = str(selected_asset or "").strip().upper()
    if requested in symbols:
        return requested
    for option in asset_options:
        if bool(option.get("is_watchlist")):
            return str(option["symbol"]).upper()
    return symbols[0] if symbols else ""


def _benchmark_options(selected_asset: str) -> list[str]:
    selected = str(selected_asset).upper()
    return ["None", *[symbol for symbol in DEFAULT_ETF_SYMBOLS if symbol not in {"SPY", selected}]]


def _select_benchmark(selected_asset: str, benchmark_symbol: str | None) -> str:
    requested = str(benchmark_symbol or "").strip().upper()
    if requested in {"", "NONE", "NULL"}:
        return "None"
    options = _benchmark_options(selected_asset)
    return requested if requested in options else "None"


def _normalize_daily_market(asset_daily: pd.DataFrame, date_start: pd.Timestamp, date_end: pd.Timestamp) -> pd.DataFrame:
    if asset_daily.empty:
        return asset_daily.copy()
    if not {"symbol", "trade_date"}.issubset(asset_daily.columns):
        return asset_daily.head(0).copy()
    frame = asset_daily.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    return frame.loc[
        (frame["trade_date"] >= pd.Timestamp(date_start).normalize())
        & (frame["trade_date"] <= pd.Timestamp(date_end).normalize())
    ].copy()


def _normalize_asset_features(
    asset_session_features: pd.DataFrame,
    selected_asset: str,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    allowed_sessions: set[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    if asset_session_features.empty:
        return asset_session_features.copy()
    if not {"asset_symbol", "signal_session_date"}.issubset(asset_session_features.columns):
        return asset_session_features.head(0).copy()
    frame = asset_session_features.copy()
    frame["asset_symbol"] = frame["asset_symbol"].astype(str).str.upper()
    frame["signal_session_date"] = pd.to_datetime(frame["signal_session_date"], errors="coerce").dt.normalize()
    frame = frame.loc[
        (frame["asset_symbol"] == str(selected_asset).upper())
        & (frame["signal_session_date"] >= pd.Timestamp(date_start).normalize())
        & (frame["signal_session_date"] <= pd.Timestamp(date_end).normalize())
    ].copy()
    if allowed_sessions is not None:
        frame = frame.loc[frame["signal_session_date"].isin(allowed_sessions)].copy()
    return frame


def _filter_asset_mappings(
    asset_post_mappings: pd.DataFrame,
    filtered_posts: pd.DataFrame,
    selected_asset: str,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
) -> pd.DataFrame:
    if asset_post_mappings.empty:
        return asset_post_mappings.copy()
    if not {"asset_symbol", "session_date"}.issubset(asset_post_mappings.columns):
        return asset_post_mappings.head(0).copy()
    frame = asset_post_mappings.copy()
    frame["asset_symbol"] = frame["asset_symbol"].astype(str).str.upper()
    frame["session_date"] = pd.to_datetime(frame["session_date"], errors="coerce").dt.normalize()
    frame = frame.loc[
        (frame["asset_symbol"] == str(selected_asset).upper())
        & (frame["session_date"] >= pd.Timestamp(date_start).normalize())
        & (frame["session_date"] <= pd.Timestamp(date_end).normalize())
    ].copy()
    if "post_id" in frame.columns and "post_id" in filtered_posts.columns:
        allowed_post_ids = set(filtered_posts["post_id"].dropna().astype(str).tolist())
        frame = frame.loc[frame["post_id"].astype(str).isin(allowed_post_ids)].copy()
    if "reaction_anchor_ts" not in frame.columns and "post_timestamp" in frame.columns:
        frame["reaction_anchor_ts"] = frame["post_timestamp"]
    return frame


def _normalize_intraday_frame(intraday: pd.DataFrame, timezone: str) -> pd.DataFrame:
    if intraday.empty:
        return pd.DataFrame(columns=ASSET_INTRADAY_COLUMNS)
    if not {"symbol", "timestamp"}.issubset(intraday.columns):
        return pd.DataFrame(columns=ASSET_INTRADAY_COLUMNS)
    frame = intraday.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if getattr(timestamps.dt, "tz", None) is None:
        timestamps = timestamps.dt.tz_localize(timezone, nonexistent="NaT", ambiguous="NaT")
    else:
        timestamps = timestamps.dt.tz_convert(timezone)
    frame["timestamp"] = timestamps
    return frame.dropna(subset=["timestamp"]).copy()


def _anchor_timestamp(value: Any, timezone: str) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    if getattr(timestamp, "tzinfo", None) is None:
        timestamp = timestamp.tz_localize(timezone, nonexistent="NaT", ambiguous="NaT")
    else:
        timestamp = timestamp.tz_convert(timezone)
    return None if pd.isna(timestamp) else pd.Timestamp(timestamp)


def _intraday_coverage(intraday: pd.DataFrame, required_symbols: set[str]) -> pd.DataFrame:
    if intraday.empty:
        return pd.DataFrame(columns=["symbol", "bars", "first_timestamp", "last_timestamp", "covered_dates"])
    frame = intraday.loc[intraday["symbol"].isin(required_symbols)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "bars", "first_timestamp", "last_timestamp", "covered_dates"])
    frame["session_date"] = frame["timestamp"].dt.date
    return (
        frame.groupby("symbol", as_index=False)
        .agg(
            bars=("timestamp", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            covered_dates=("session_date", lambda values: ", ".join(sorted({str(value) for value in values}))),
        )
        .sort_values("symbol")
        .reset_index(drop=True)
    )


def _intraday_anchor_options(
    mappings: pd.DataFrame,
    intraday: pd.DataFrame,
    required_symbols: set[str],
    timezone: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame, str]:
    if mappings.empty:
        return [], mappings.head(0).copy(), "No mapped posts are available for the selected asset and filters."
    if intraday.empty:
        return [], mappings.head(0).copy(), "No stored intraday data is available yet."

    intraday_dates = {
        symbol: set(
            intraday.loc[intraday["symbol"] == symbol, "timestamp"]
            .dropna()
            .dt.date
            .tolist()
        )
        for symbol in required_symbols
    }
    eligible_sessions = {
        session_date
        for session_date in pd.to_datetime(mappings["session_date"], errors="coerce").dt.date.dropna().unique().tolist()
        if all(session_date in intraday_dates.get(symbol, set()) for symbol in required_symbols)
    }
    if not eligible_sessions:
        return [], mappings.head(0).copy(), (
            "No recent intraday coverage overlaps mapped sessions for "
            f"{', '.join(sorted(required_symbols))}."
        )

    eligible = mappings.loc[
        pd.to_datetime(mappings["session_date"], errors="coerce").dt.date.isin(eligible_sessions)
    ].copy()
    eligible = eligible.sort_values(["session_date", "post_timestamp"], ascending=[False, True]).reset_index(drop=True)
    options: list[dict[str, Any]] = []
    for index, row in eligible.iterrows():
        anchor_id = str(row.get("post_id", "") or f"row-{index}")
        timestamp = _anchor_timestamp(row.get("reaction_anchor_ts", row.get("post_timestamp")), timezone)
        if timestamp is None:
            continue
        author = str(row.get("author_handle", "") or row.get("author_display_name", "") or "unknown")
        text = truncate_text(str(row.get("cleaned_text", "") or ""), max_chars=96)
        session_date = pd.Timestamp(row["session_date"]).date().isoformat()
        options.append(
            {
                "anchor_id": anchor_id,
                "session_date": session_date,
                "label": f"{timestamp:%Y-%m-%d %H:%M} ET | @{author} | {text}",
                "post_timestamp": timestamp.isoformat(),
            },
        )
    return options, eligible, "" if options else "Mapped posts exist, but none have a usable intraday anchor timestamp."


def _select_anchor(
    eligible_mappings: pd.DataFrame,
    anchor_options: list[dict[str, Any]],
    intraday_session_date: str | None,
    intraday_anchor_post_id: str | None,
) -> tuple[dict[str, Any] | None, pd.Series | None]:
    if eligible_mappings.empty or not anchor_options:
        return None, None
    option_lookup = {str(option["anchor_id"]): option for option in anchor_options}
    requested_anchor = str(intraday_anchor_post_id or "").strip()
    requested_session = str(intraday_session_date or "").strip()
    selected_option = option_lookup.get(requested_anchor)
    if selected_option is None and requested_session:
        selected_option = next((option for option in anchor_options if str(option["session_date"]) == requested_session), None)
    selected_option = selected_option or anchor_options[0]
    selected_anchor_id = str(selected_option["anchor_id"])
    for _, row in eligible_mappings.iterrows():
        row_anchor_id = str(row.get("post_id", "") or "")
        if row_anchor_id == selected_anchor_id:
            return selected_option, row
    try:
        fallback_index = int(selected_anchor_id.replace("row-", ""))
        if fallback_index in eligible_mappings.index:
            return selected_option, eligible_mappings.loc[fallback_index]
    except ValueError:
        pass
    return selected_option, eligible_mappings.iloc[0]


def build_research_asset_lab(
    *,
    settings: AppSettings,
    store: DuckDBStore,
    date_start: str | None = None,
    date_end: str | None = None,
    platforms: Iterable[str] | None = None,
    include_reshares: bool | None = None,
    tracked_only: bool | None = None,
    trump_authored_only: bool | None = None,
    keyword: str | None = None,
    selected_asset: str | None = None,
    comparison_mode: str | None = None,
    benchmark_symbol: str | None = None,
    pre_sessions: int | None = None,
    post_sessions: int | None = None,
    intraday_session_date: str | None = None,
    intraday_anchor_post_id: str | None = None,
    before_minutes: int | None = None,
    after_minutes: int | None = None,
) -> dict[str, Any]:
    posts = store.read_frame("normalized_posts")
    asset_universe = store.read_frame("asset_universe")
    asset_daily = store.read_frame("asset_daily")
    asset_intraday = store.read_frame("asset_intraday")
    asset_post_mappings = store.read_frame("asset_post_mappings")
    asset_session_features = store.read_frame("asset_session_features")

    source_mode = detect_source_mode(posts)
    filters = _resolve_filters(
        settings,
        posts,
        date_start=date_start,
        date_end=date_end,
        platforms=platforms,
        include_reshares=include_reshares,
        tracked_only=tracked_only,
        trump_authored_only=trump_authored_only,
        keyword=keyword,
    )
    public_filters = _public_filter_payload(filters)
    options = _asset_options(asset_universe, asset_daily)
    resolved_asset = _select_asset(options, selected_asset)
    mode = _normalize_mode(comparison_mode)
    resolved_benchmark = _select_benchmark(resolved_asset, benchmark_symbol)
    benchmark_value = None if resolved_benchmark == "None" else resolved_benchmark
    resolved_pre_sessions = _coerce_positive_int(pre_sessions, 3, minimum=1, maximum=10)
    resolved_post_sessions = _coerce_positive_int(post_sessions, 5, minimum=1, maximum=10)
    resolved_before_minutes = _coerce_positive_int(before_minutes, DEFAULT_INTRADAY_BEFORE_MINUTES, minimum=30, maximum=390)
    resolved_after_minutes = _coerce_positive_int(after_minutes, DEFAULT_INTRADAY_AFTER_MINUTES, minimum=30, maximum=780)

    empty_comparison = build_asset_comparison_chart(pd.DataFrame(), resolved_asset or "asset", mode)
    empty_event = build_event_study_chart(pd.DataFrame(), resolved_asset or "asset")
    empty_intraday = build_intraday_comparison_chart(pd.DataFrame(), resolved_asset or "asset", pd.Timestamp.now(tz=settings.timezone))
    if not resolved_asset or asset_daily.empty:
        message = "Refresh datasets and add at least one non-SPY tracked asset to use the Asset Lab."
        return {
            "ready": False,
            "message": message,
            "source_mode": source_mode,
            "filters": public_filters,
            "controls": {
                "selected_asset": resolved_asset,
                "comparison_mode": mode,
                "benchmark_symbol": resolved_benchmark,
                "pre_sessions": resolved_pre_sessions,
                "post_sessions": resolved_post_sessions,
                "before_minutes": resolved_before_minutes,
                "after_minutes": resolved_after_minutes,
                "intraday_session_date": intraday_session_date or "",
                "intraday_anchor_post_id": intraday_anchor_post_id or "",
            },
            "asset_options": options,
            "benchmark_options": _benchmark_options(resolved_asset),
            "headline_metrics": {},
            "charts": {
                "asset_comparison": _figure_json(empty_comparison),
                "event_study": _figure_json(empty_event),
                "intraday_reaction": _figure_json(empty_intraday),
            },
            "asset_session_rows": [],
            "asset_mapping_rows": [],
            "comparison_rows": [],
            "event_study_rows": [],
            "intraday_anchor_options": [],
            "intraday_coverage": [],
            "intraday_rows": [],
            "intraday_message": message,
        }

    filtered_posts = filter_posts(
        posts=posts,
        date_start=filters["date_start"],
        date_end=filters["date_end"],
        include_reshares=filters["include_reshares"],
        platforms=filters["platforms"],
        keyword=filters["keyword"],
        tracked_only=filters["tracked_only"],
        trump_authored_only=filters["trump_authored_only"],
    )
    daily_market = _normalize_daily_market(asset_daily, filters["date_start"], filters["date_end"])
    selected_mappings = _filter_asset_mappings(
        asset_post_mappings,
        filtered_posts,
        resolved_asset,
        filters["date_start"],
        filters["date_end"],
    )
    allowed_sessions = (
        set(pd.to_datetime(selected_mappings["session_date"], errors="coerce").dropna().dt.normalize().tolist())
        if not selected_mappings.empty and "session_date" in selected_mappings.columns
        else None
    )
    selected_features = _normalize_asset_features(
        asset_session_features,
        resolved_asset,
        filters["date_start"],
        filters["date_end"],
        allowed_sessions=allowed_sessions,
    )
    comparison = build_asset_comparison_frame(
        asset_market=daily_market,
        selected_symbol=resolved_asset,
        benchmark_symbol=benchmark_value,
        date_start=filters["date_start"],
        date_end=filters["date_end"],
    )
    event_study = build_event_study_frame(
        asset_market=daily_market,
        asset_session_features=selected_features,
        selected_symbol=resolved_asset,
        benchmark_symbol=benchmark_value,
        pre_sessions=resolved_pre_sessions,
        post_sessions=resolved_post_sessions,
    )
    comparison_chart = build_asset_comparison_chart(comparison, resolved_asset, mode)
    event_chart = build_event_study_chart(event_study, resolved_asset)
    asset_session_table = make_asset_session_table(selected_features, resolved_asset)
    asset_mapping_table = make_asset_mapping_table(selected_mappings, resolved_asset)

    intraday_frame = _normalize_intraday_frame(asset_intraday, settings.timezone)
    required_intraday_symbols = {"SPY", resolved_asset}
    if benchmark_value:
        required_intraday_symbols.add(benchmark_value)
    anchor_options, eligible_mappings, intraday_message = _intraday_anchor_options(
        selected_mappings,
        intraday_frame,
        required_intraday_symbols,
        settings.timezone,
    )
    selected_anchor, selected_anchor_row = _select_anchor(
        eligible_mappings,
        anchor_options,
        intraday_session_date,
        intraday_anchor_post_id,
    )
    intraday_comparison = pd.DataFrame()
    intraday_chart = empty_intraday
    if selected_anchor_row is not None:
        anchor_ts = _anchor_timestamp(
            selected_anchor_row.get("reaction_anchor_ts", selected_anchor_row.get("post_timestamp")),
            settings.timezone,
        )
        if anchor_ts is not None:
            intraday_comparison = build_intraday_comparison_frame(
                intraday_frame=intraday_frame,
                selected_symbol=resolved_asset,
                anchor_ts=anchor_ts,
                before_minutes=resolved_before_minutes,
                after_minutes=resolved_after_minutes,
                benchmark_symbol=benchmark_value,
            )
            intraday_chart = build_intraday_comparison_chart(intraday_comparison, resolved_asset, anchor_ts)
            if intraday_comparison.empty and not intraday_message:
                intraday_message = "No intraday comparison rows were returned for the selected window."
    coverage = _intraday_coverage(intraday_frame, required_intraday_symbols)

    sessions_in_range = int(len(comparison))
    spy_move = float(comparison["spy_normalized_return"].iloc[-1]) if not comparison.empty and "spy_normalized_return" in comparison.columns else None
    asset_move = float(comparison["asset_normalized_return"].iloc[-1]) if not comparison.empty and "asset_normalized_return" in comparison.columns else None
    headline_metrics = {
        "sessions_in_range": sessions_in_range,
        "spy_move": spy_move,
        "asset_move": asset_move,
        "asset_vs_spy_spread": float(asset_move - spy_move) if asset_move is not None and spy_move is not None else None,
        "mapped_post_count": int(len(selected_mappings)),
        "asset_session_count": int(len(selected_features)),
        "event_count": int(event_study["event_count"].max()) if not event_study.empty and "event_count" in event_study.columns else 0,
        "intraday_bars": int(len(intraday_comparison)),
    }

    return {
        "ready": True,
        "message": "",
        "source_mode": source_mode,
        "filters": public_filters,
        "controls": {
            "selected_asset": resolved_asset,
            "comparison_mode": mode,
            "benchmark_symbol": resolved_benchmark,
            "pre_sessions": resolved_pre_sessions,
            "post_sessions": resolved_post_sessions,
            "before_minutes": resolved_before_minutes,
            "after_minutes": resolved_after_minutes,
            "intraday_session_date": selected_anchor.get("session_date", "") if selected_anchor else "",
            "intraday_anchor_post_id": selected_anchor.get("anchor_id", "") if selected_anchor else "",
        },
        "asset_options": options,
        "benchmark_options": _benchmark_options(resolved_asset),
        "headline_metrics": headline_metrics,
        "charts": {
            "asset_comparison": _figure_json(comparison_chart),
            "event_study": _figure_json(event_chart),
            "intraday_reaction": _figure_json(intraday_chart),
        },
        "asset_session_rows": _to_records(asset_session_table.sort_values("trade_date", ascending=False) if not asset_session_table.empty else asset_session_table),
        "asset_mapping_rows": _to_records(asset_mapping_table),
        "comparison_rows": _to_records(comparison, limit=TABLE_LIMIT),
        "event_study_rows": _to_records(event_study, limit=TABLE_LIMIT),
        "intraday_anchor_options": anchor_options[:TABLE_LIMIT],
        "intraday_coverage": _to_records(coverage, limit=None),
        "intraday_rows": _to_records(intraday_comparison, limit=TABLE_LIMIT),
        "intraday_message": intraday_message,
    }
