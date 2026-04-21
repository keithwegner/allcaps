from __future__ import annotations

from typing import Any

import pandas as pd

from .contracts import LiveMonitorConfig, PaperPortfolioConfig
from .experiments import ExperimentStore
from .live_monitor import build_live_portfolio_run_state, seed_live_monitor_config, validate_live_monitor_config
from .modeling import ModelService
from .paper_trading import (
    PaperTradingService,
    ensure_paper_benchmark_curve_frame,
    ensure_paper_decision_journal_frame,
    ensure_paper_equity_curve_frame,
    ensure_paper_portfolio_registry_frame,
    ensure_paper_trade_ledger_frame,
    paper_config_matches_live,
)
from .performance import PerformanceObservatoryService
from .storage import DuckDBStore


def _records(frame: pd.DataFrame, limit: int | None = None, tail: bool = False) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    if limit is not None:
        out = out.tail(limit) if tail else out.head(limit)
    return out.to_dict(orient="records")


def _joint_portfolio_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    normalized = runs.copy()
    for column, default in [("run_type", "asset_model"), ("allocator_mode", "")]:
        if column not in normalized.columns:
            normalized[column] = default
        normalized[column] = normalized[column].fillna(default).astype(str)
    return normalized.loc[
        (normalized["run_type"] == "portfolio_allocator")
        & (normalized["allocator_mode"] == "joint_model")
    ].copy()


def _run_options(joint_runs: pd.DataFrame) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for _, row in joint_runs.iterrows():
        selected = row.get("selected_params_json", {}) or {}
        metrics = row.get("metrics_json", {}) or {}
        options.append(
            {
                "run_id": str(row.get("run_id", "") or ""),
                "run_name": str(row.get("run_name", "") or ""),
                "created_at": row.get("created_at"),
                "fallback_mode": str(selected.get("fallback_mode", "SPY") or "SPY").upper(),
                "deployment_variant": str(selected.get("deployment_variant", "") or ""),
                "deployment_narrative_feature_mode": str(selected.get("deployment_narrative_feature_mode", "") or ""),
                "selected_symbols": ", ".join(str(symbol).upper() for symbol in selected.get("selected_symbols", []) or []),
                "transaction_cost_bps": float(selected.get("transaction_cost_bps", 0.0) or 0.0),
                "robust_score": metrics.get("robust_score"),
                "total_return": metrics.get("total_return"),
            },
        )
    return options


def _selected_run_row(joint_runs: pd.DataFrame, run_id: str) -> pd.Series | None:
    matches = joint_runs.loc[joint_runs["run_id"].astype(str) == str(run_id)].copy()
    if matches.empty:
        return None
    return matches.iloc[0]


def build_live_config_from_run(
    runs: pd.DataFrame,
    portfolio_run_id: str,
    fallback_mode: str,
) -> tuple[LiveMonitorConfig | None, list[str]]:
    normalized_fallback = str(fallback_mode or "SPY").upper()
    joint_runs = _joint_portfolio_runs(runs)
    selected_row = _selected_run_row(joint_runs, portfolio_run_id)
    if selected_row is None:
        return None, [f"Portfolio run `{portfolio_run_id}` is not available."]
    selected = selected_row.get("selected_params_json", {}) or {}
    config = LiveMonitorConfig(
        mode="portfolio_run",
        fallback_mode=normalized_fallback,
        portfolio_run_id=str(portfolio_run_id),
        portfolio_run_name=str(selected_row.get("run_name", portfolio_run_id) or portfolio_run_id),
        deployment_variant=str(selected.get("deployment_variant", "") or ""),
    )
    return config, validate_live_monitor_config(config, runs)


def _active_run_context(runs: pd.DataFrame, config: LiveMonitorConfig | None) -> dict[str, Any]:
    if config is None:
        return {"run_name": "", "transaction_cost_bps": 0.0}
    joint_runs = _joint_portfolio_runs(runs)
    row = _selected_run_row(joint_runs, str(config.portfolio_run_id or ""))
    if row is None:
        return {"run_name": str(config.portfolio_run_name or config.portfolio_run_id or ""), "transaction_cost_bps": 0.0}
    selected = row.get("selected_params_json", {}) or {}
    return {
        "run_name": str(row.get("run_name", config.portfolio_run_name or config.portfolio_run_id) or config.portfolio_run_name or config.portfolio_run_id),
        "transaction_cost_bps": float(selected.get("transaction_cost_bps", 0.0) or 0.0),
    }


def _paper_payload(store: DuckDBStore, paper_service: PaperTradingService, config: LiveMonitorConfig | None) -> dict[str, Any]:
    registry = ensure_paper_portfolio_registry_frame(store.read_frame("paper_portfolio_registry"))
    current_config = paper_service.load_current_config()
    active_config = current_config if paper_config_matches_live(current_config, config) else None
    journal = ensure_paper_decision_journal_frame(store.read_frame("paper_decision_journal"))
    trades = ensure_paper_trade_ledger_frame(store.read_frame("paper_trade_ledger"))
    equity = ensure_paper_equity_curve_frame(store.read_frame("paper_equity_curve"))
    benchmark = ensure_paper_benchmark_curve_frame(store.read_frame("paper_benchmark_curve"))
    active_id = str(active_config.paper_portfolio_id) if active_config is not None else ""
    return {
        "current_config": current_config.to_dict() if current_config is not None else None,
        "active_config": active_config.to_dict() if active_config is not None else None,
        "portfolios": _records(registry.sort_values("created_at", ascending=False) if not registry.empty else registry, limit=50),
        "decision_journal": _records(journal.loc[journal["paper_portfolio_id"].astype(str) == active_id].sort_values("signal_session_date") if active_id else pd.DataFrame(), limit=25, tail=True),
        "trade_ledger": _records(trades.loc[trades["paper_portfolio_id"].astype(str) == active_id].sort_values("signal_session_date") if active_id else pd.DataFrame(), limit=25, tail=True),
        "equity_curve": _records(equity.loc[equity["paper_portfolio_id"].astype(str) == active_id].sort_values("signal_session_date") if active_id else pd.DataFrame()),
        "benchmark_curve": _records(benchmark.loc[benchmark["paper_portfolio_id"].astype(str) == active_id].sort_values("signal_session_date") if active_id else pd.DataFrame()),
    }


def build_live_ops_payload(
    *,
    store: DuckDBStore,
    experiment_store: ExperimentStore,
    model_service: ModelService,
    paper_service: PaperTradingService,
    performance_service: PerformanceObservatoryService | None = None,
    generated_at: pd.Timestamp | None = None,
    capture_result: dict[str, Any] | None = None,
    public_mode: bool = False,
) -> dict[str, Any]:
    runs = experiment_store.list_runs()
    joint_runs = _joint_portfolio_runs(runs)
    saved_config = experiment_store.load_live_monitor_config()
    saved_is_portfolio = saved_config is not None and str(saved_config.mode or "portfolio_run") == "portfolio_run"
    active_config = saved_config if saved_is_portfolio else None
    seeded_config = seed_live_monitor_config(runs)
    config_errors = validate_live_monitor_config(active_config, runs) if active_config is not None else []

    board = pd.DataFrame()
    decision = pd.DataFrame()
    warnings: list[str] = []
    if active_config is not None and not config_errors:
        board, decision, _explanations, warnings = build_live_portfolio_run_state(
            store=store,
            model_service=model_service,
            experiment_store=experiment_store,
            config=active_config,
            generated_at=generated_at or pd.Timestamp.now(tz="UTC").floor("s"),
        )

    asset_history = store.read_frame("live_asset_snapshots")
    decision_history = store.read_frame("live_decision_snapshots")
    if active_config is not None:
        if not asset_history.empty and "run_id" in asset_history.columns:
            asset_history = asset_history.loc[asset_history["run_id"].astype(str) == str(active_config.portfolio_run_id)].copy()
        if not asset_history.empty and "variant_name" in asset_history.columns and active_config.deployment_variant:
            asset_history = asset_history.loc[asset_history["variant_name"].astype(str) == str(active_config.deployment_variant)].copy()
        if not decision_history.empty and "portfolio_run_id" in decision_history.columns:
            decision_history = decision_history.loc[decision_history["portfolio_run_id"].astype(str) == str(active_config.portfolio_run_id)].copy()
        if not decision_history.empty and "deployment_variant" in decision_history.columns and active_config.deployment_variant:
            decision_history = decision_history.loc[decision_history["deployment_variant"].astype(str) == str(active_config.deployment_variant)].copy()

    return {
        "configured": bool(active_config is not None and not config_errors),
        "errors": config_errors if active_config is not None else ["No live monitor config has been saved yet."],
        "warnings": warnings,
        "admin": {
            "mode": "public" if public_mode else "private",
            "write_requires_unlock": True,
            "capture_scope": "stored_data_only",
        },
        "current_config": active_config.to_dict() if active_config is not None else None,
        "seeded_config": seeded_config.to_dict() if seeded_config is not None else None,
        "run_options": _run_options(joint_runs),
        "decision": _records(decision)[0] if not decision.empty else None,
        "board": _records(board),
        "asset_history": _records(asset_history.sort_values("generated_at") if not asset_history.empty and "generated_at" in asset_history.columns else asset_history, limit=250, tail=True),
        "decision_history": _records(decision_history.sort_values("generated_at") if not decision_history.empty and "generated_at" in decision_history.columns else decision_history, limit=50, tail=True),
        "paper": _paper_payload(store, paper_service, active_config),
        "capture_result": capture_result or {"persisted_assets": 0, "persisted_decisions": 0, "captured": 0, "settled": 0, "performance_persisted": False},
    }


def run_live_capture(
    *,
    store: DuckDBStore,
    experiment_store: ExperimentStore,
    model_service: ModelService,
    paper_service: PaperTradingService,
    performance_service: PerformanceObservatoryService | None = None,
    generated_at: pd.Timestamp | None = None,
) -> dict[str, Any]:
    runs = experiment_store.list_runs()
    live_config = experiment_store.load_live_monitor_config()
    errors = validate_live_monitor_config(live_config, runs)
    if errors or live_config is None or str(live_config.mode or "portfolio_run") != "portfolio_run":
        return {"errors": errors or ["Save a portfolio-run live config before capturing."], "persisted_assets": 0, "persisted_decisions": 0, "captured": 0, "settled": 0, "performance_persisted": False}

    snapshot_time = pd.Timestamp.now(tz="UTC").floor("s") if generated_at is None else pd.Timestamp(generated_at).floor("s")
    board, decision, _explanations, warnings = build_live_portfolio_run_state(
        store=store,
        model_service=model_service,
        experiment_store=experiment_store,
        config=live_config,
        generated_at=snapshot_time,
    )
    result: dict[str, Any] = {
        "generated_at": snapshot_time,
        "warnings": warnings,
        "persisted_assets": 0,
        "persisted_decisions": 0,
        "captured": 0,
        "settled": 0,
        "performance_persisted": False,
    }
    if board.empty or decision.empty:
        return result

    experiment_store.save_live_asset_snapshots(board)
    experiment_store.save_live_decision_snapshots(decision)
    result["persisted_assets"] = int(len(board))
    result["persisted_decisions"] = int(len(decision))

    paper_config = paper_service.load_current_config()
    if paper_config_matches_live(paper_config, live_config):
        paper_update = paper_service.process_live_history(paper_config, as_of=snapshot_time)
        result.update(paper_update)
        if performance_service is not None and isinstance(paper_config, PaperPortfolioConfig):
            performance_service.persist_snapshot(paper_config.paper_portfolio_id, generated_at=snapshot_time)
            result["performance_persisted"] = True
    return result


def apply_paper_action(
    *,
    store: DuckDBStore,
    experiment_store: ExperimentStore,
    paper_service: PaperTradingService,
    action: str,
    starting_cash: float | None = None,
    now: pd.Timestamp | None = None,
) -> tuple[PaperPortfolioConfig | None, list[str]]:
    live_config = experiment_store.load_live_monitor_config()
    runs = experiment_store.list_runs()
    errors = validate_live_monitor_config(live_config, runs)
    if errors or live_config is None or str(live_config.mode or "portfolio_run") != "portfolio_run":
        return None, errors or ["Save a portfolio-run live config before changing paper trading."]

    context = _active_run_context(runs, live_config)
    current = paper_service.load_current_config()
    active = current if paper_config_matches_live(current, live_config) else None
    normalized_action = str(action or "").strip().lower()
    timestamp = pd.Timestamp.now(tz="UTC").floor("s") if now is None else pd.Timestamp(now).floor("s")

    if normalized_action == "enable":
        return (
            paper_service.upsert_current_for_live_config(
                live_config=live_config,
                portfolio_run_name=context["run_name"],
                transaction_cost_bps=float(context["transaction_cost_bps"]),
                starting_cash=float(starting_cash or 100000.0),
                enabled=True,
                reset=False,
                now=timestamp,
            ),
            [],
        )
    if normalized_action == "reset":
        return (
            paper_service.upsert_current_for_live_config(
                live_config=live_config,
                portfolio_run_name=context["run_name"],
                transaction_cost_bps=float(context["transaction_cost_bps"]),
                starting_cash=float(starting_cash or (active.starting_cash if active is not None else 100000.0)),
                enabled=True,
                reset=True,
                now=timestamp,
            ),
            [],
        )
    if active is None:
        return None, ["No active paper portfolio matches the pinned live config."]
    if normalized_action == "disable":
        return (
            paper_service.upsert_current_for_live_config(
                live_config=live_config,
                portfolio_run_name=context["run_name"],
                transaction_cost_bps=float(active.transaction_cost_bps),
                starting_cash=float(active.starting_cash),
                enabled=False,
                reset=False,
                now=timestamp,
            ),
            [],
        )
    if normalized_action == "archive":
        return paper_service.archive_current_config(now=timestamp), []
    return None, ["Paper action must be one of: enable, disable, reset, archive."]
