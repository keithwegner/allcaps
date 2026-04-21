from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import uuid
from typing import Any

import pandas as pd

from .backtesting import BacktestService
from .contracts import ModelRunConfig, PortfolioRunConfig
from .experiments import ExperimentStore
from .explanations import build_account_attribution, build_post_attribution
from .features import FeatureService
from .market import normalize_symbols
from .modeling import NARRATIVE_FEATURE_MODES, SUPPORTED_PORTFOLIO_MODEL_FAMILIES
from .scheduler import acquire_refresh_lock, release_refresh_lock
from .storage import DuckDBStore

MODEL_TRAINING_JOB_COLUMNS = [
    "job_id",
    "workflow_mode",
    "status",
    "started_at",
    "completed_at",
    "error_message",
    "run_id",
    "run_name",
    "run_type",
    "allocator_mode",
    "target_asset",
    "selected_symbols",
    "summary",
]

VALID_MODEL_WORKFLOWS = {"single_asset", "saved_run_portfolio", "joint_portfolio"}


@dataclass(frozen=True)
class ModelTrainingRequest:
    workflow_mode: str
    run_name: str = ""
    target_asset: str = "SPY"
    feature_version: str = ""
    llm_enabled: bool = False
    train_window: int = 90
    validation_window: int = 30
    test_window: int = 30
    step_size: int = 30
    transaction_cost_bps: float = 2.0
    ridge_alpha: float = 1.0
    threshold_grid: str | list[float] = "0,0.001,0.0025,0.005"
    minimum_signal_grid: str | list[int] = "1,2,3"
    account_weight_grid: str | list[float] = "0.5,1.0,1.5"
    fallback_mode: str = "SPY"
    component_run_ids: tuple[str, ...] = ()
    selected_symbols: tuple[str, ...] = ()
    topology_variants: tuple[str, ...] = ()
    model_families: tuple[str, ...] = ()
    narrative_feature_modes: tuple[str, ...] = ()


def ensure_model_training_jobs_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=MODEL_TRAINING_JOB_COLUMNS)
    out = frame.copy()
    for column in MODEL_TRAINING_JOB_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out["started_at"] = pd.to_datetime(out["started_at"], errors="coerce", utc=True)
    out["completed_at"] = pd.to_datetime(out["completed_at"], errors="coerce", utc=True)
    for column in ["job_id", "workflow_mode", "status", "error_message", "run_id", "run_name", "run_type", "allocator_mode", "target_asset", "selected_symbols", "summary"]:
        out[column] = out[column].fillna("").astype(str)
    return out[MODEL_TRAINING_JOB_COLUMNS].copy()


def _frame_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    if limit is not None:
        out = out.tail(max(0, int(limit)))
    return out.to_dict(orient="records")


def _json_summary(payload: dict[str, Any] | None = None) -> str:
    return json.dumps(payload or {}, default=str, sort_keys=True)


def _parse_grid(value: str | list[Any] | tuple[Any, ...], cast: type[float] | type[int]) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        parts = [item for item in value if item is not None and str(item).strip() != ""]
    else:
        parts = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if not parts:
        raise ValueError("Grid values cannot be empty.")
    if cast is int:
        return tuple(int(part) for part in parts)
    return tuple(float(part) for part in parts)


def _job_row(
    *,
    job_id: str,
    workflow_mode: str,
    status: str,
    started_at: pd.Timestamp,
    completed_at: pd.Timestamp | pd.NaT = pd.NaT,
    error_message: str = "",
    run_id: str = "",
    run_name: str = "",
    run_type: str = "",
    allocator_mode: str = "",
    target_asset: str = "",
    selected_symbols: list[str] | tuple[str, ...] | str = "",
    summary: dict[str, Any] | None = None,
) -> pd.DataFrame:
    symbols = selected_symbols if isinstance(selected_symbols, str) else ",".join(str(symbol).upper() for symbol in selected_symbols)
    return pd.DataFrame(
        [
            {
                "job_id": str(job_id),
                "workflow_mode": str(workflow_mode),
                "status": str(status),
                "started_at": pd.Timestamp(started_at),
                "completed_at": completed_at,
                "error_message": str(error_message or ""),
                "run_id": str(run_id or ""),
                "run_name": str(run_name or ""),
                "run_type": str(run_type or ""),
                "allocator_mode": str(allocator_mode or ""),
                "target_asset": str(target_asset or ""),
                "selected_symbols": symbols,
                "summary": _json_summary(summary),
            },
        ],
        columns=MODEL_TRAINING_JOB_COLUMNS,
    )


def save_model_training_job(store: DuckDBStore, row: pd.DataFrame) -> pd.DataFrame:
    current = ensure_model_training_jobs_frame(store.read_frame("model_training_jobs"))
    combined = pd.concat([current, ensure_model_training_jobs_frame(row)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["job_id"], keep="last").reset_index(drop=True)
    store.save_frame(
        "model_training_jobs",
        combined,
        metadata={
            "row_count": int(len(combined)),
            "latest_status": str(combined.iloc[-1]["status"]) if not combined.empty else "",
        },
    )
    return combined


def _target_asset_options(store: DuckDBStore) -> list[dict[str, str]]:
    asset_universe = store.read_frame("asset_universe")
    if asset_universe.empty or "symbol" not in asset_universe.columns:
        return [{"symbol": "SPY", "label": "SPY"}]
    symbols = normalize_symbols(asset_universe["symbol"].astype(str).tolist())
    ordered = ["SPY", *[symbol for symbol in symbols if symbol != "SPY"]]
    rows: list[dict[str, str]] = []
    for symbol in ordered:
        match = asset_universe.loc[asset_universe["symbol"].astype(str).str.upper() == symbol]
        display_name = str(match.iloc[0].get("display_name", symbol) or symbol) if not match.empty else symbol
        rows.append({"symbol": symbol, "label": f"{symbol} - {display_name}" if display_name != symbol else symbol})
    return rows


def _normalize_runs_frame(runs: pd.DataFrame) -> pd.DataFrame:
    normalized = runs.copy()
    if normalized.empty:
        return normalized
    if "target_asset" not in normalized.columns:
        normalized["target_asset"] = "SPY"
    normalized["target_asset"] = normalized["target_asset"].fillna("SPY").replace("", "SPY").astype(str).str.upper()
    if "run_type" not in normalized.columns:
        normalized["run_type"] = "asset_model"
    normalized["run_type"] = normalized["run_type"].fillna("asset_model").replace("", "asset_model").astype(str)
    if "allocator_mode" not in normalized.columns:
        normalized["allocator_mode"] = ""
    normalized["allocator_mode"] = normalized["allocator_mode"].fillna("").astype(str)
    return normalized


def _run_options(runs: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = _normalize_runs_frame(runs)
    if normalized.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in normalized.iterrows():
        metrics = row.get("metrics_json") or {}
        selected = row.get("selected_params_json") or {}
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
                "deployment_variant": str(selected.get("deployment_variant", "") or ""),
            },
        )
    return rows


def build_model_training_payload(
    *,
    store: DuckDBStore,
    experiment_store: ExperimentStore,
    active_job_id: str = "",
) -> dict[str, Any]:
    posts = store.read_frame("normalized_posts")
    spy = store.read_frame("spy_daily")
    asset_session_features = store.read_frame("asset_session_features")
    runs = _normalize_runs_frame(experiment_store.list_runs())
    jobs = ensure_model_training_jobs_frame(store.read_frame("model_training_jobs"))

    feature_versions = ["asset-v1"]
    available_symbols: list[str] = []
    if not asset_session_features.empty:
        if "feature_version" in asset_session_features.columns:
            feature_versions = sorted(asset_session_features["feature_version"].dropna().astype(str).unique().tolist()) or feature_versions
        if "asset_symbol" in asset_session_features.columns:
            available_symbols = sorted(asset_session_features["asset_symbol"].dropna().astype(str).str.upper().unique().tolist())
            if "SPY" in available_symbols:
                available_symbols = ["SPY", *[symbol for symbol in available_symbols if symbol != "SPY"]]

    run_options = _run_options(runs)
    asset_model_runs = [row for row in run_options if row["run_type"] == "asset_model"]
    spy_runs = [row for row in asset_model_runs if row["target_asset"] == "SPY"]
    readiness = {
        "single_asset": [] if not posts.empty and not spy.empty else ["Refresh datasets first so normalized posts and SPY market data are available."],
        "saved_run_portfolio": [] if spy_runs else ["Save at least one SPY asset-model run before building a saved-run portfolio allocator."],
        "joint_portfolio": [] if not asset_session_features.empty else ["Refresh datasets first so asset-session features are available."],
    }
    latest_active = jobs.loc[jobs["status"].isin(["queued", "running"])].tail(1)
    return {
        "admin": {
            "write_requires_unlock": True,
        },
        "status": {
            "ready": not any(readiness.values()),
            "active_job_id": str(active_job_id or (latest_active.iloc[0]["job_id"] if not latest_active.empty else "")),
            "readiness_errors": readiness,
        },
        "defaults": {
            "single_asset": {
                "run_name": "baseline-research-run",
                "target_asset": "SPY",
                "feature_version": "v1",
                "threshold_grid": "0,0.001,0.0025,0.005",
                "minimum_signal_grid": "1,2,3",
                "account_weight_grid": "0.5,1.0,1.5",
            },
            "saved_run_portfolio": {"run_name": "portfolio-allocator-run", "fallback_mode": "SPY"},
            "joint_portfolio": {
                "run_name": "joint-portfolio-run",
                "fallback_mode": "SPY",
                "feature_version": feature_versions[0],
                "selected_symbols": [symbol for symbol in ["SPY", "QQQ", "NVDA"] if symbol in available_symbols] or available_symbols[:3],
                "topology_variants": ["per_asset", "pooled"],
                "model_families": ["ridge", "elastic_net", "hist_gradient_boosting_regressor"],
                "narrative_feature_modes": ["baseline"],
            },
        },
        "asset_options": _target_asset_options(store),
        "feature_versions": feature_versions,
        "asset_session_symbols": available_symbols,
        "narrative_feature_modes": list(NARRATIVE_FEATURE_MODES),
        "model_families": list(SUPPORTED_PORTFOLIO_MODEL_FAMILIES),
        "topology_variants": ["per_asset", "pooled"],
        "run_options": run_options,
        "asset_model_runs": asset_model_runs,
        "recent_jobs": _frame_records(jobs, limit=50),
    }


def build_model_target_bundle(
    *,
    store: DuckDBStore,
    feature_service: FeatureService,
    posts: pd.DataFrame,
    spy_market: pd.DataFrame,
    tracked_accounts: pd.DataFrame,
    llm_enabled: bool,
    target_asset: str,
    feature_version: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_target = str(target_asset).upper()
    resolved_feature_version = feature_version or ("v1" if normalized_target == "SPY" else "asset-v1")
    asset_universe = store.read_frame("asset_universe")
    asset_daily = store.read_frame("asset_daily")
    prepared_posts = feature_service.prepare_session_posts(
        posts=posts,
        market_calendar=spy_market,
        tracked_accounts=tracked_accounts,
        llm_enabled=llm_enabled,
    )

    if normalized_target == "SPY":
        feature_rows = feature_service.build_session_dataset(
            posts=posts,
            spy_market=spy_market,
            tracked_accounts=tracked_accounts,
            feature_version=resolved_feature_version,
            llm_enabled=llm_enabled,
            prepared_posts=prepared_posts,
        )
        feature_rows["target_asset"] = "SPY"
        feature_rows["target_asset_display_name"] = "SPDR S&P 500 ETF Trust"
        return feature_rows.reset_index(drop=True), prepared_posts.reset_index(drop=True)

    if asset_universe.empty or asset_daily.empty:
        raise RuntimeError("Refresh datasets first so non-SPY target assets have market and universe data available.")

    asset_post_mappings = feature_service.build_asset_post_mappings(
        prepared_posts=prepared_posts,
        asset_universe=asset_universe,
        llm_enabled=llm_enabled,
    )
    asset_feature_rows = feature_service.build_asset_session_dataset(
        asset_post_mappings=asset_post_mappings,
        asset_market=asset_daily,
        feature_version=resolved_feature_version,
        llm_enabled=llm_enabled,
        asset_universe=asset_universe,
    )
    feature_rows = asset_feature_rows.loc[
        asset_feature_rows["asset_symbol"].astype(str).str.upper() == normalized_target
    ].copy()
    if feature_rows.empty:
        raise RuntimeError(f"No session feature rows were available for target asset `{normalized_target}`.")
    asset_rows = asset_universe.loc[asset_universe["symbol"].astype(str).str.upper() == normalized_target]
    display_name = str(asset_rows.iloc[0].get("display_name", normalized_target)) if not asset_rows.empty else normalized_target
    feature_rows["target_asset"] = normalized_target
    feature_rows["target_asset_display_name"] = display_name
    attribution_posts = asset_post_mappings.loc[
        asset_post_mappings["asset_symbol"].astype(str).str.upper() == normalized_target
    ].copy()
    attribution_posts["target_asset"] = normalized_target
    return feature_rows.reset_index(drop=True), attribution_posts.reset_index(drop=True)


def _save_single_asset_run(
    *,
    store: DuckDBStore,
    experiment_store: ExperimentStore,
    feature_service: FeatureService,
    backtest_service: BacktestService,
    request: ModelTrainingRequest,
) -> tuple[str, dict[str, Any]]:
    posts = store.read_frame("normalized_posts")
    spy = store.read_frame("spy_daily")
    tracked_accounts = store.read_frame("tracked_accounts")
    if posts.empty or spy.empty:
        raise RuntimeError("Refresh datasets first so the modeling pipeline has normalized posts and SPY market data.")
    target_asset = str(request.target_asset or "SPY").upper()
    feature_version = request.feature_version or ("v1" if target_asset == "SPY" else "asset-v1")
    feature_rows, attribution_posts = build_model_target_bundle(
        store=store,
        feature_service=feature_service,
        posts=posts,
        spy_market=spy,
        tracked_accounts=tracked_accounts,
        llm_enabled=bool(request.llm_enabled),
        target_asset=target_asset,
        feature_version=feature_version,
    )
    store.save_frame(
        "session_features_latest",
        feature_rows,
        metadata={
            "llm_enabled": bool(request.llm_enabled),
            "row_count": int(len(feature_rows)),
            "target_asset": target_asset,
            "feature_version": feature_version,
        },
    )
    config = ModelRunConfig(
        run_name=request.run_name or "baseline-research-run",
        target_asset=target_asset,
        feature_version=feature_version,
        llm_enabled=bool(request.llm_enabled),
        train_window=int(request.train_window),
        validation_window=int(request.validation_window),
        test_window=int(request.test_window),
        step_size=int(request.step_size),
        threshold_grid=_parse_grid(request.threshold_grid, float),
        minimum_signal_grid=_parse_grid(request.minimum_signal_grid, int),
        account_weight_grid=_parse_grid(request.account_weight_grid, float),
        ridge_alpha=float(request.ridge_alpha),
        transaction_cost_bps=float(request.transaction_cost_bps),
    )
    run, artifacts = backtest_service.run_walk_forward(config, feature_rows)
    post_attribution = build_post_attribution(attribution_posts)
    account_attribution = build_account_attribution(post_attribution)
    predicted_sessions = {
        session_date
        for session_date in pd.to_datetime(
            artifacts["predictions"].get("signal_session_date", pd.Series(dtype="datetime64[ns]")),
            errors="coerce",
        ).dropna().dt.normalize().tolist()
    }
    if predicted_sessions and not post_attribution.empty:
        post_attribution = post_attribution.loc[
            pd.to_datetime(post_attribution["signal_session_date"], errors="coerce").dt.normalize().isin(predicted_sessions)
        ].reset_index(drop=True)
    if predicted_sessions and not account_attribution.empty:
        account_attribution = account_attribution.loc[
            pd.to_datetime(account_attribution["signal_session_date"], errors="coerce").dt.normalize().isin(predicted_sessions)
        ].reset_index(drop=True)
    experiment_store.save_run(
        run=run,
        config=artifacts["config"],
        trades=artifacts["trades"],
        predictions=artifacts["predictions"],
        windows=artifacts["windows"],
        importance=artifacts["importance"],
        model_artifact=artifacts["model_artifact"],
        feature_contributions=artifacts["feature_contributions"],
        post_attribution=post_attribution,
        account_attribution=account_attribution,
        benchmarks=artifacts["benchmarks"],
        diagnostics=artifacts["diagnostics"],
        benchmark_curves=artifacts["benchmark_curves"],
        leakage_audit=artifacts["leakage_audit"],
    )
    return run.run_id, {"run": run.to_dict(), "metrics": run.metrics}


def _asset_for_bundle(run_id: str, bundle: dict[str, Any]) -> str:
    return str(
        (bundle.get("config") or {}).get("target_asset")
        or (bundle.get("run") or {}).get("target_asset")
        or run_id,
    ).upper()


def _save_saved_run_portfolio(
    *,
    experiment_store: ExperimentStore,
    backtest_service: BacktestService,
    request: ModelTrainingRequest,
) -> tuple[str, dict[str, Any]]:
    component_ids = [str(run_id) for run_id in request.component_run_ids if str(run_id)]
    if not component_ids:
        raise RuntimeError("Select at least one saved SPY run before building a portfolio allocator.")
    component_bundles = {
        run_id: bundle
        for run_id in component_ids
        if (bundle := experiment_store.load_run(run_id)) is not None
    }
    missing = [run_id for run_id in component_ids if run_id not in component_bundles]
    if missing:
        raise RuntimeError(f"Component runs are missing: {', '.join(missing)}")
    assets = [_asset_for_bundle(run_id, bundle) for run_id, bundle in component_bundles.items()]
    if "SPY" not in assets:
        raise RuntimeError("A saved-run portfolio allocator requires exactly one SPY component run.")
    if len(assets) != len(set(assets)):
        raise RuntimeError("A saved-run portfolio allocator can include at most one run per asset.")
    portfolio_config = PortfolioRunConfig(
        run_name=request.run_name or "portfolio-allocator-run",
        allocator_mode="saved_runs",
        fallback_mode=str(request.fallback_mode or "SPY").upper(),
        transaction_cost_bps=float(request.transaction_cost_bps),
        component_run_ids=tuple(component_ids),
        universe_symbols=tuple(assets),
    )
    run, artifacts = backtest_service.run_saved_run_allocator(portfolio_config, component_bundles)
    experiment_store.save_portfolio_run(
        run=run,
        config=artifacts["config"],
        trades=artifacts["trades"],
        decision_history=artifacts["predictions"],
        candidate_predictions=artifacts["candidate_predictions"],
        component_summary=artifacts["windows"],
        benchmarks=artifacts["benchmarks"],
        benchmark_curves=artifacts["benchmark_curves"],
        diagnostics=artifacts["diagnostics"],
        leakage_audit=artifacts["leakage_audit"],
    )
    return run.run_id, {"run": run.to_dict(), "metrics": run.metrics, "selected_symbols": assets}


def _save_joint_portfolio(
    *,
    store: DuckDBStore,
    experiment_store: ExperimentStore,
    backtest_service: BacktestService,
    request: ModelTrainingRequest,
) -> tuple[str, dict[str, Any]]:
    asset_session_features = store.read_frame("asset_session_features")
    if asset_session_features.empty:
        raise RuntimeError("Refresh datasets first so the asset-session feature dataset is available.")
    selected_symbols = normalize_symbols(list(request.selected_symbols))
    if len(selected_symbols) < 2:
        raise RuntimeError("Select at least two symbols for a joint portfolio run.")
    fallback_mode = str(request.fallback_mode or "SPY").upper()
    if fallback_mode == "SPY" and "SPY" not in selected_symbols:
        raise RuntimeError("Include SPY in the selected symbols when fallback mode is SPY.")
    topology_variants = tuple(str(value) for value in (request.topology_variants or ("per_asset", "pooled")))
    model_families = tuple(str(value) for value in (request.model_families or SUPPORTED_PORTFOLIO_MODEL_FAMILIES))
    narrative_modes = tuple(str(value) for value in (request.narrative_feature_modes or ("baseline",)))
    if not topology_variants:
        raise RuntimeError("Select at least one topology variant.")
    if not model_families:
        raise RuntimeError("Select at least one model family.")
    if not narrative_modes:
        raise RuntimeError("Select at least one narrative feature mode.")
    portfolio_config = PortfolioRunConfig(
        run_name=request.run_name or "joint-portfolio-run",
        allocator_mode="joint_model",
        fallback_mode=fallback_mode,
        transaction_cost_bps=float(request.transaction_cost_bps),
        universe_symbols=tuple(selected_symbols),
        selected_symbols=tuple(selected_symbols),
        llm_enabled=bool(request.llm_enabled),
        feature_version=request.feature_version or "asset-v1",
        train_window=int(request.train_window),
        validation_window=int(request.validation_window),
        test_window=int(request.test_window),
        step_size=int(request.step_size),
        threshold_grid=_parse_grid(request.threshold_grid, float),
        minimum_signal_grid=_parse_grid(request.minimum_signal_grid, int),
        account_weight_grid=_parse_grid(request.account_weight_grid, float),
        model_families=model_families,
        topology_variants=topology_variants,
        narrative_feature_modes=narrative_modes,
    )
    run, artifacts = backtest_service.run_joint_model_allocator(portfolio_config, asset_session_features)
    experiment_store.save_portfolio_run(
        run=run,
        config=artifacts["config"],
        trades=artifacts["trades"],
        decision_history=artifacts["predictions"],
        candidate_predictions=artifacts["candidate_predictions"],
        component_summary=artifacts["windows"],
        benchmarks=artifacts["benchmarks"],
        benchmark_curves=artifacts["benchmark_curves"],
        diagnostics=artifacts["diagnostics"],
        leakage_audit=artifacts["leakage_audit"],
        variant_summary=artifacts["variant_summary"],
        portfolio_model_bundle=artifacts["portfolio_model_bundle"],
        importance=artifacts["importance"],
    )
    return run.run_id, {"run": run.to_dict(), "metrics": run.metrics, "selected_symbols": selected_symbols}


def run_model_training_job(
    *,
    store: DuckDBStore,
    experiment_store: ExperimentStore,
    feature_service: FeatureService,
    backtest_service: BacktestService,
    request: ModelTrainingRequest,
    job_id: str,
    lock_fd: int | None,
) -> None:
    started_at = pd.Timestamp.now(tz="UTC")
    workflow_mode = str(request.workflow_mode or "").strip().lower()
    try:
        save_model_training_job(
            store,
            _job_row(job_id=job_id, workflow_mode=workflow_mode, status="running", started_at=started_at),
        )
        if workflow_mode == "single_asset":
            run_id, summary = _save_single_asset_run(
                store=store,
                experiment_store=experiment_store,
                feature_service=feature_service,
                backtest_service=backtest_service,
                request=request,
            )
        elif workflow_mode == "saved_run_portfolio":
            run_id, summary = _save_saved_run_portfolio(
                experiment_store=experiment_store,
                backtest_service=backtest_service,
                request=request,
            )
        elif workflow_mode == "joint_portfolio":
            run_id, summary = _save_joint_portfolio(
                store=store,
                experiment_store=experiment_store,
                backtest_service=backtest_service,
                request=request,
            )
        else:
            raise RuntimeError(f"Unsupported model training workflow: {request.workflow_mode}")

        run_payload = summary.get("run", {}) if isinstance(summary, dict) else {}
        save_model_training_job(
            store,
            _job_row(
                job_id=job_id,
                workflow_mode=workflow_mode,
                status="success",
                started_at=started_at,
                completed_at=pd.Timestamp.now(tz="UTC"),
                run_id=run_id,
                run_name=str(run_payload.get("run_name", request.run_name) or request.run_name),
                run_type=str(run_payload.get("run_type", "asset_model") or "asset_model"),
                allocator_mode=str(run_payload.get("allocator_mode", "") or ""),
                target_asset=str(run_payload.get("target_asset", request.target_asset) or request.target_asset),
                selected_symbols=summary.get("selected_symbols", request.selected_symbols) if isinstance(summary, dict) else request.selected_symbols,
                summary=summary,
            ),
        )
    except Exception as exc:
        save_model_training_job(
            store,
            _job_row(
                job_id=job_id,
                workflow_mode=workflow_mode,
                status="error",
                started_at=started_at,
                completed_at=pd.Timestamp.now(tz="UTC"),
                error_message=str(exc),
                run_name=request.run_name,
                target_asset=request.target_asset,
                selected_symbols=request.selected_symbols,
            ),
        )
    finally:
        release_refresh_lock(store.settings, lock_fd)


def submit_model_training_job(
    *,
    store: DuckDBStore,
    experiment_store: ExperimentStore,
    feature_service: FeatureService,
    backtest_service: BacktestService,
    request: ModelTrainingRequest,
    run_inline: bool = False,
) -> tuple[str, list[str]]:
    workflow_mode = str(request.workflow_mode or "").strip().lower()
    if workflow_mode not in VALID_MODEL_WORKFLOWS:
        return "", [f"Unsupported model training workflow: {request.workflow_mode}"]

    lock_fd = acquire_refresh_lock(store.settings)
    if lock_fd is None:
        return "", ["A dataset refresh, scheduler cycle, or model training job is already running."]

    job_id = f"model-training-{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    save_model_training_job(
        store,
        _job_row(
            job_id=job_id,
            workflow_mode=workflow_mode,
            status="queued",
            started_at=pd.Timestamp.now(tz="UTC"),
            run_name=request.run_name,
            target_asset=request.target_asset,
            selected_symbols=request.selected_symbols,
        ),
    )
    kwargs = {
        "store": store,
        "experiment_store": experiment_store,
        "feature_service": feature_service,
        "backtest_service": backtest_service,
        "request": request,
        "job_id": job_id,
        "lock_fd": lock_fd,
    }
    if run_inline:
        run_model_training_job(**kwargs)
    else:
        threading.Thread(target=run_model_training_job, kwargs=kwargs, daemon=True).start()
    return job_id, []
