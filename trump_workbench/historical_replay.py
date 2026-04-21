from __future__ import annotations

from typing import Any

import pandas as pd

from .backtesting import BacktestService
from .contracts import ModelRunConfig
from .experiments import ExperimentStore
from .explanations import build_account_attribution, build_post_attribution
from .features import FeatureService
from .model_training import build_model_target_bundle
from .storage import DuckDBStore

REPLAY_MIN_HISTORY_ROWS = 20


def _normalize_session_date(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value).normalize()
    except Exception:
        return None


def _filter_for_session(frame: pd.DataFrame, session_date: pd.Timestamp | None, column: str = "signal_session_date") -> pd.DataFrame:
    if frame.empty or session_date is None or column not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    dates = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    return frame.loc[dates == session_date].copy()


def _frame_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    if limit is not None:
        out = out.head(max(0, int(limit)))
    return out.to_dict(orient="records")


def _bundle_to_run_config(run_bundle: dict[str, Any]) -> ModelRunConfig:
    config = run_bundle.get("config", {}) or {}
    run_meta = run_bundle.get("run", {}) or {}
    return ModelRunConfig(
        run_name=str(config.get("run_name") or run_meta.get("run_name") or "historical-replay"),
        target_asset=str(config.get("target_asset") or run_meta.get("target_asset") or "SPY"),
        feature_version=str(config.get("feature_version", "v1")),
        llm_enabled=bool(config.get("llm_enabled", False)),
        train_window=int(config.get("train_window", 90) or 90),
        validation_window=int(config.get("validation_window", 30) or 30),
        test_window=int(config.get("test_window", 30) or 30),
        step_size=int(config.get("step_size", 30) or 30),
        threshold_grid=tuple(float(value) for value in config.get("threshold_grid", [0.0, 0.001, 0.0025, 0.005])),
        minimum_signal_grid=tuple(int(value) for value in config.get("minimum_signal_grid", [1, 2, 3])),
        account_weight_grid=tuple(float(value) for value in config.get("account_weight_grid", [0.5, 1.0, 1.5])),
        ridge_alpha=float(config.get("ridge_alpha", 1.0) or 1.0),
        transaction_cost_bps=float(config.get("transaction_cost_bps", 2.0) or 2.0),
        start_date=config.get("start_date"),
        end_date=config.get("end_date"),
        notes=str(config.get("notes", "")),
    )


def _eligible_replay_sessions(feature_rows: pd.DataFrame, min_history_rows: int = REPLAY_MIN_HISTORY_ROWS) -> pd.DataFrame:
    if feature_rows.empty:
        return pd.DataFrame()
    eligible = feature_rows.sort_values("signal_session_date").reset_index(drop=True).copy()
    if "target_available" not in eligible.columns:
        eligible["target_available"] = False
    eligible["history_rows_available"] = eligible["target_available"].fillna(False).astype(int).cumsum().shift(fill_value=0)
    eligible = eligible.loc[eligible["history_rows_available"] >= min_history_rows].copy()
    return eligible.reset_index(drop=True)


def _replay_option_label(row: pd.Series) -> str:
    session_date = pd.Timestamp(row["signal_session_date"])
    return (
        f"{session_date:%Y-%m-%d} | posts {int(row.get('post_count', 0))} | "
        f"prior train rows {int(row.get('history_rows_available', 0))}"
    )


def _build_replay_comparison_frame(replay_row: pd.Series, full_history_row: pd.Series | None) -> pd.DataFrame:
    rows = [
        {"metric": "Replay score", "value": float(replay_row.get("expected_return_score", 0.0))},
        {"metric": "Replay confidence", "value": float(replay_row.get("prediction_confidence", 0.0))},
        {"metric": "Replay threshold", "value": float(replay_row.get("deployment_threshold", 0.0))},
        {"metric": "Replay min post count", "value": int(replay_row.get("deployment_min_post_count", 1))},
        {"metric": "Training rows used", "value": int(replay_row.get("training_rows_used", 0))},
    ]
    actual = replay_row.get("target_next_session_return")
    if pd.notna(actual):
        rows.append({"metric": "Actual next-session return", "value": float(actual)})
    if full_history_row is not None:
        full_score = float(full_history_row.get("expected_return_score", 0.0) or 0.0)
        replay_score = float(replay_row.get("expected_return_score", 0.0) or 0.0)
        rows.extend(
            [
                {"metric": "Full-history score", "value": full_score},
                {"metric": "Replay vs full-history drift", "value": replay_score - full_score},
                {"metric": "Full-history confidence", "value": float(full_history_row.get("prediction_confidence", 0.0) or 0.0)},
            ],
        )
    return pd.DataFrame(rows)


def _run_options(runs: pd.DataFrame) -> list[dict[str, Any]]:
    if runs.empty:
        return []
    out = runs.copy()
    if "run_type" not in out.columns:
        out["run_type"] = "asset_model"
    out["run_type"] = out["run_type"].fillna("asset_model").replace("", "asset_model")
    if "allocator_mode" not in out.columns:
        out["allocator_mode"] = ""
    rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        metrics = row.get("metrics_json") or {}
        rows.append(
            {
                "run_id": str(row.get("run_id", "") or ""),
                "run_name": str(row.get("run_name", "") or ""),
                "target_asset": str(row.get("target_asset", "SPY") or "SPY").upper(),
                "run_type": str(row.get("run_type", "asset_model") or "asset_model"),
                "allocator_mode": str(row.get("allocator_mode", "") or ""),
                "created_at": row.get("created_at"),
                "robust_score": float(metrics.get("robust_score", 0.0) or 0.0),
                "total_return": float(metrics.get("total_return", 0.0) or 0.0),
            },
        )
    return rows


def _asset_model_run_options(runs: pd.DataFrame) -> list[dict[str, Any]]:
    return [row for row in _run_options(runs) if row["run_type"] == "asset_model"]


def _empty_replay_payload(message: str, *, runs: pd.DataFrame, selected_run_id: str = "") -> dict[str, Any]:
    asset_runs = _asset_model_run_options(runs)
    return {
        "ready": False,
        "message": message,
        "selected_run_id": selected_run_id,
        "selected_session_date": "",
        "min_history_rows": REPLAY_MIN_HISTORY_ROWS,
        "run_options": asset_runs,
        "sessions": [],
        "summary": {
            "saved_run_count": int(len(runs)),
            "asset_model_run_count": int(len(asset_runs)),
            "eligible_session_count": 0,
        },
    }


def _select_asset_run(runs: pd.DataFrame, run_id: str | None) -> tuple[str, str | None]:
    asset_runs = _asset_model_run_options(runs)
    if not asset_runs:
        return "", "Historical replay currently supports asset-model runs only."
    asset_run_ids = {row["run_id"] for row in asset_runs}
    if run_id:
        if run_id not in asset_run_ids:
            return "", "Selected replay run is unavailable or is not an asset-model run."
        return run_id, None
    return str(asset_runs[0]["run_id"]), None


def _build_feature_rows_for_run(
    *,
    store: DuckDBStore,
    feature_service: FeatureService,
    run_bundle: dict[str, Any],
) -> tuple[ModelRunConfig, pd.DataFrame, pd.DataFrame]:
    posts = store.read_frame("normalized_posts")
    spy = store.read_frame("spy_daily")
    tracked_accounts = store.read_frame("tracked_accounts")
    if posts.empty or spy.empty:
        raise RuntimeError("Refresh datasets first so replay can rebuild historical features.")
    run_config = _bundle_to_run_config(run_bundle)
    feature_rows, attribution_posts = build_model_target_bundle(
        store=store,
        feature_service=feature_service,
        posts=posts,
        spy_market=spy,
        tracked_accounts=tracked_accounts,
        llm_enabled=run_config.llm_enabled,
        target_asset=run_config.target_asset,
        feature_version=run_config.feature_version,
    )
    if run_config.start_date:
        feature_rows = feature_rows.loc[feature_rows["signal_session_date"] >= pd.Timestamp(run_config.start_date)].copy()
    if run_config.end_date:
        feature_rows = feature_rows.loc[feature_rows["signal_session_date"] <= pd.Timestamp(run_config.end_date)].copy()
    return run_config, feature_rows.reset_index(drop=True), attribution_posts.reset_index(drop=True)


def build_historical_replay_payload(
    *,
    store: DuckDBStore,
    experiment_store: ExperimentStore,
    feature_service: FeatureService,
    run_id: str | None = None,
) -> dict[str, Any]:
    runs = experiment_store.list_runs()
    if runs.empty:
        return _empty_replay_payload("Save at least one asset-model run first so Historical Replay has a template configuration to follow.", runs=runs)

    selected_run_id, selection_error = _select_asset_run(runs, run_id)
    if selection_error:
        return _empty_replay_payload(selection_error, runs=runs, selected_run_id=selected_run_id)

    loaded = experiment_store.load_run(selected_run_id)
    if loaded is None:
        return _empty_replay_payload("The selected replay template could not be loaded.", runs=runs, selected_run_id=selected_run_id)

    try:
        _, feature_rows, _ = _build_feature_rows_for_run(
            store=store,
            feature_service=feature_service,
            run_bundle=loaded,
        )
    except RuntimeError as exc:
        return _empty_replay_payload(str(exc), runs=runs, selected_run_id=selected_run_id)

    eligible_sessions = _eligible_replay_sessions(feature_rows)
    if eligible_sessions.empty:
        return _empty_replay_payload(
            "No replay sessions are eligible yet. Historical replay needs at least 20 earlier target-available sessions.",
            runs=runs,
            selected_run_id=selected_run_id,
        )

    replay_choices = eligible_sessions.sort_values("signal_session_date", ascending=False).reset_index(drop=True)
    sessions = []
    for _, row in replay_choices.iterrows():
        session_date = pd.Timestamp(row["signal_session_date"]).normalize()
        sessions.append(
            {
                "value": f"{session_date:%Y-%m-%d}",
                "label": _replay_option_label(row),
                "signal_session_date": session_date,
                "post_count": int(row.get("post_count", 0) or 0),
                "history_rows_available": int(row.get("history_rows_available", 0) or 0),
            },
        )

    asset_runs = _asset_model_run_options(runs)
    return {
        "ready": True,
        "message": "",
        "selected_run_id": selected_run_id,
        "selected_session_date": str(sessions[0]["value"]) if sessions else "",
        "min_history_rows": REPLAY_MIN_HISTORY_ROWS,
        "run_options": asset_runs,
        "sessions": sessions,
        "summary": {
            "saved_run_count": int(len(runs)),
            "asset_model_run_count": int(len(asset_runs)),
            "eligible_session_count": int(len(sessions)),
        },
    }


def build_historical_replay_session_payload(
    *,
    store: DuckDBStore,
    experiment_store: ExperimentStore,
    feature_service: FeatureService,
    backtest_service: BacktestService,
    run_id: str,
    signal_session_date: str,
) -> dict[str, Any]:
    runs = experiment_store.list_runs()
    selected_run_id, selection_error = _select_asset_run(runs, run_id)
    if selection_error:
        return {"ready": False, "message": selection_error, "run_id": run_id, "signal_session_date": signal_session_date}

    loaded = experiment_store.load_run(selected_run_id)
    if loaded is None:
        return {"ready": False, "message": "The selected replay template could not be loaded.", "run_id": run_id, "signal_session_date": signal_session_date}

    try:
        run_config, feature_rows, attribution_posts = _build_feature_rows_for_run(
            store=store,
            feature_service=feature_service,
            run_bundle=loaded,
        )
    except RuntimeError as exc:
        return {"ready": False, "message": str(exc), "run_id": run_id, "signal_session_date": signal_session_date}

    replay_date = _normalize_session_date(signal_session_date)
    eligible_sessions = _eligible_replay_sessions(feature_rows)
    eligible_dates = pd.to_datetime(eligible_sessions.get("signal_session_date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dt.normalize()
    if replay_date is None or replay_date not in set(eligible_dates.dropna().tolist()):
        return {
            "ready": False,
            "message": "Selected replay session is not eligible. Historical replay needs at least 20 earlier target-available sessions.",
            "run_id": run_id,
            "signal_session_date": signal_session_date,
        }

    try:
        replay = backtest_service.build_historical_replay(
            run_config=run_config,
            feature_rows=feature_rows,
            replay_session_date=replay_date,
            deployment_params=loaded["selected_params"],
        )
    except RuntimeError as exc:
        return {"ready": False, "message": str(exc), "run_id": run_id, "signal_session_date": signal_session_date}

    replay_prediction = replay["prediction"].iloc[0]
    replay_feature_contributions = replay["feature_contributions"]
    if "session_date" in attribution_posts.columns:
        session_post_source = attribution_posts.loc[
            pd.to_datetime(attribution_posts["session_date"], errors="coerce").dt.normalize() == replay_date
        ].copy()
    else:
        session_post_source = pd.DataFrame()
    session_post_attribution = build_post_attribution(session_post_source)
    session_account_attribution = build_account_attribution(session_post_attribution)

    full_history_match = _filter_for_session(loaded["predictions"], replay_date)
    full_history_row = full_history_match.iloc[0] if not full_history_match.empty else None
    comparison_frame = _build_replay_comparison_frame(replay_prediction, full_history_row)

    prediction_records = _frame_records(replay["prediction"])
    prediction_record = prediction_records[0] if prediction_records else {}
    return {
        "ready": True,
        "message": "",
        "run_id": selected_run_id,
        "run_name": str((loaded.get("run") or {}).get("run_name", selected_run_id)),
        "target_asset": str(run_config.target_asset).upper(),
        "signal_session_date": replay_date,
        "metrics": {
            "target_asset": str(run_config.target_asset).upper(),
            "replay_session": replay_date,
            "replay_score": float(replay_prediction.get("expected_return_score", 0.0) or 0.0),
            "replay_confidence": float(replay_prediction.get("prediction_confidence", 0.0) or 0.0),
            "suggested_stance": str(replay_prediction.get("suggested_stance", "")),
            "training_rows_used": int(replay.get("training_rows_used", 0) or 0),
            "history_start": replay.get("history_start"),
            "history_end": replay.get("history_end"),
            "actual_next_session_return": replay_prediction.get("target_next_session_return"),
            "full_history_score": None if full_history_row is None else float(full_history_row.get("expected_return_score", 0.0) or 0.0),
            "replay_vs_full_history_drift": None
            if full_history_row is None
            else float(replay_prediction.get("expected_return_score", 0.0) or 0.0) - float(full_history_row.get("expected_return_score", 0.0) or 0.0),
        },
        "metadata": {
            "template_run_id": selected_run_id,
            "history_start": replay.get("history_start"),
            "history_end": replay.get("history_end"),
            "training_rows_used": int(replay.get("training_rows_used", 0) or 0),
            "deployment_params": replay.get("deployment_params", {}),
            "future_training_leakage": bool(prediction_record.get("future_training_leakage", False)),
            "full_history_comparison_available": full_history_row is not None,
        },
        "prediction": prediction_records,
        "comparison_rows": _frame_records(comparison_frame),
        "feature_importance": _frame_records(replay["importance"], limit=25),
        "feature_contributions": _frame_records(replay_feature_contributions),
        "post_attribution": _frame_records(session_post_attribution),
        "account_attribution": _frame_records(session_account_attribution),
    }
