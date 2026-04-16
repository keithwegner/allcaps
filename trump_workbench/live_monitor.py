from __future__ import annotations

from typing import Any

import pandas as pd

from .contracts import LiveMonitorConfig, LiveMonitorPinnedRun
from .explanations import build_account_attribution, build_post_attribution
from .modeling import LinearModelArtifact, ModelService, add_asset_indicator_columns
from .portfolio import PORTFOLIO_CANDIDATE_COLUMNS, PORTFOLIO_DECISION_COLUMNS, VALID_FALLBACK_MODES, rank_portfolio_candidates

LIVE_MONITOR_CONFIG_PATH = "live_monitor/config.json"
LIVE_ASSET_SNAPSHOT_COLUMNS = [
    "generated_at",
    "variant_name",
    *PORTFOLIO_CANDIDATE_COLUMNS,
]
LIVE_DECISION_SNAPSHOT_COLUMNS = [
    "generated_at",
    "deployment_variant",
    *PORTFOLIO_DECISION_COLUMNS,
]


def seed_live_monitor_config(runs: pd.DataFrame) -> LiveMonitorConfig | None:
    if runs.empty or "run_id" not in runs.columns:
        return None
    normalized = runs.copy()
    if "created_at" in normalized.columns:
        normalized = normalized.sort_values("created_at", ascending=False)
    if "run_type" not in normalized.columns:
        normalized["run_type"] = "asset_model"
    if "allocator_mode" not in normalized.columns:
        normalized["allocator_mode"] = ""
    portfolio_runs = normalized.loc[
        (normalized["run_type"].astype(str) == "portfolio_allocator")
        & (normalized["allocator_mode"].astype(str) == "joint_model")
    ].copy()
    if not portfolio_runs.empty:
        row = portfolio_runs.iloc[0]
        selected = row.get("selected_params_json", {}) or {}
        return LiveMonitorConfig(
            mode="portfolio_run",
            fallback_mode=str(selected.get("fallback_mode", "SPY") or "SPY").upper(),
            portfolio_run_id=str(row.get("run_id", "") or ""),
            portfolio_run_name=str(row.get("run_name", "") or ""),
            deployment_variant=str(selected.get("deployment_variant", "") or ""),
        )

    if "target_asset" not in normalized.columns:
        normalized["target_asset"] = "SPY"
    normalized["target_asset"] = normalized["target_asset"].fillna("SPY").astype(str).str.upper()
    asset_model_runs = normalized.loc[normalized["run_type"].astype(str) == "asset_model"].copy()
    if asset_model_runs.empty:
        return None
    pinned_runs: list[LiveMonitorPinnedRun] = []
    for asset_symbol, group in asset_model_runs.groupby("target_asset", sort=False):
        row = group.iloc[0]
        pinned_runs.append(
            LiveMonitorPinnedRun(
                asset_symbol=str(asset_symbol).upper(),
                run_id=str(row.get("run_id", "") or ""),
                run_name=str(row.get("run_name", "") or ""),
            ),
        )
    pinned_runs = sorted(pinned_runs, key=lambda item: (item.asset_symbol != "SPY", item.asset_symbol))
    return LiveMonitorConfig(mode="asset_model_set", fallback_mode="SPY", pinned_runs=pinned_runs)


def validate_live_monitor_config(config: LiveMonitorConfig | None, runs: pd.DataFrame) -> list[str]:
    if config is None:
        return ["Save a pinned live portfolio configuration before using the decision console."]

    errors: list[str] = []
    fallback_mode = str(config.fallback_mode or "SPY").upper()
    if fallback_mode not in VALID_FALLBACK_MODES:
        errors.append("Fallback mode must be `SPY` or `FLAT`.")

    mode = str(config.mode or "portfolio_run")
    if mode == "portfolio_run":
        portfolio_runs = runs.copy()
        if portfolio_runs.empty:
            errors.append("Save at least one joint portfolio run before configuring the live console.")
            return errors
        if "run_type" not in portfolio_runs.columns:
            portfolio_runs["run_type"] = "asset_model"
        if "allocator_mode" not in portfolio_runs.columns:
            portfolio_runs["allocator_mode"] = ""
        portfolio_runs = portfolio_runs.loc[
            (portfolio_runs["run_type"].astype(str) == "portfolio_allocator")
            & (portfolio_runs["allocator_mode"].astype(str) == "joint_model")
        ].copy()
        if portfolio_runs.empty:
            errors.append("No joint portfolio runs are available yet.")
            return errors
        portfolio_run_id = str(config.portfolio_run_id or "")
        if not portfolio_run_id:
            errors.append("A saved joint portfolio run must be selected.")
            return errors
        portfolio_lookup = portfolio_runs.set_index("run_id", drop=False)
        if portfolio_run_id not in portfolio_lookup.index:
            errors.append(f"Portfolio run `{portfolio_run_id}` is not available anymore.")
            return errors
        return errors

    if not config.pinned_runs:
        errors.append("At least one pinned run is required.")
        return errors

    run_lookup = runs.copy()
    if not run_lookup.empty:
        if "target_asset" not in run_lookup.columns:
            run_lookup["target_asset"] = "SPY"
        run_lookup["target_asset"] = run_lookup["target_asset"].fillna("SPY").astype(str).str.upper()
        run_lookup = run_lookup.set_index("run_id", drop=False)

    seen_assets: set[str] = set()
    spy_count = 0
    for pinned in config.pinned_runs:
        asset_symbol = str(pinned.asset_symbol or "").upper()
        if not asset_symbol:
            errors.append("Pinned runs must include an asset symbol.")
            continue
        if asset_symbol in seen_assets:
            errors.append(f"Only one pinned run is allowed for `{asset_symbol}`.")
            continue
        seen_assets.add(asset_symbol)
        if asset_symbol == "SPY":
            spy_count += 1

        run_id = str(pinned.run_id or "")
        if not run_id:
            errors.append(f"`{asset_symbol}` must have a saved run selected.")
            continue
        if run_lookup.empty or run_id not in run_lookup.index:
            errors.append(f"Pinned run `{run_id}` for `{asset_symbol}` is not available anymore.")
            continue
        row_asset = str(run_lookup.loc[run_id, "target_asset"] or "SPY").upper()
        if row_asset != asset_symbol:
            errors.append(f"Pinned run `{run_id}` targets `{row_asset}`, not `{asset_symbol}`.")

    if spy_count != 1:
        errors.append("Exactly one pinned `SPY` run is required.")
    return errors


def rank_live_asset_snapshots(snapshots: pd.DataFrame, fallback_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if snapshots.empty:
        return (
            pd.DataFrame(columns=LIVE_ASSET_SNAPSHOT_COLUMNS),
            pd.DataFrame(columns=LIVE_DECISION_SNAPSHOT_COLUMNS),
        )

    board = snapshots.copy()
    board["generated_at"] = pd.to_datetime(board["generated_at"], errors="coerce")
    ranked_board, decision = rank_portfolio_candidates(board, fallback_mode=fallback_mode, require_tradeable=False)
    for column in LIVE_ASSET_SNAPSHOT_COLUMNS:
        if column not in ranked_board.columns:
            ranked_board[column] = pd.NA
    for column in LIVE_DECISION_SNAPSHOT_COLUMNS:
        if column not in decision.columns:
            decision[column] = pd.NA
    return ranked_board[LIVE_ASSET_SNAPSHOT_COLUMNS].copy(), decision[LIVE_DECISION_SNAPSHOT_COLUMNS].copy()


def _normalize_session_date(value: Any) -> pd.Timestamp | None:
    session_date = pd.to_datetime(value, errors="coerce")
    if pd.isna(session_date):
        return None
    return pd.Timestamp(session_date).normalize()


def _apply_live_account_weight(feature_rows: pd.DataFrame, account_weight: float) -> pd.DataFrame:
    adjusted = feature_rows.copy()
    for column in ["tracked_weighted_mentions", "tracked_weighted_engagement", "tracked_account_post_count"]:
        if column in adjusted.columns:
            adjusted[column] = adjusted[column] * float(account_weight)
    return adjusted


def _portfolio_live_model_artifact(payload: dict[str, Any]) -> LinearModelArtifact:
    return LinearModelArtifact.from_dict(payload)


def _predict_portfolio_live_variant(
    model_service: ModelService,
    feature_rows: pd.DataFrame,
    variant_payload: dict[str, Any],
    selected_symbols: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    topology = str(variant_payload.get("topology", "per_asset") or "per_asset")
    models_payload = variant_payload.get("models", {}) or {}
    predictions: list[pd.DataFrame] = []
    explanations: dict[str, dict[str, Any]] = {}
    if topology == "pooled":
        pooled_model = models_payload.get("pooled")
        if not isinstance(pooled_model, dict):
            return pd.DataFrame(), {}
        artifact = _portfolio_live_model_artifact(pooled_model)
        pooled_rows = add_asset_indicator_columns(feature_rows, selected_symbols)
        scored = model_service.predict(artifact, pooled_rows)
        predictions.append(scored)
        for asset_symbol in selected_symbols:
            asset_rows = scored.loc[scored["asset_symbol"].astype(str).str.upper() == asset_symbol].copy()
            if asset_rows.empty:
                continue
            explanations[asset_symbol] = {
                "artifact": artifact,
                "prediction_frame": asset_rows,
                "feature_contributions": model_service.explain_predictions(artifact, asset_rows),
            }
    else:
        for asset_symbol in selected_symbols:
            model_payload = models_payload.get(asset_symbol)
            if not isinstance(model_payload, dict):
                continue
            artifact = _portfolio_live_model_artifact(model_payload)
            asset_rows = feature_rows.loc[feature_rows["asset_symbol"].astype(str).str.upper() == asset_symbol].copy()
            if asset_rows.empty:
                continue
            scored = model_service.predict(artifact, asset_rows)
            predictions.append(scored)
            explanations[asset_symbol] = {
                "artifact": artifact,
                "prediction_frame": scored,
                "feature_contributions": model_service.explain_predictions(artifact, scored),
            }
    if not predictions:
        return pd.DataFrame(), {}
    combined = pd.concat(predictions, ignore_index=True).sort_values(["signal_session_date", "asset_symbol"]).reset_index(drop=True)
    return combined, explanations


def build_live_portfolio_run_state(
    store: Any,
    model_service: ModelService,
    experiment_store: Any,
    config: LiveMonitorConfig,
    generated_at: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    explanation_lookup: dict[str, dict[str, Any]] = {}
    snapshot_time = pd.Timestamp.utcnow().floor("s") if generated_at is None else pd.Timestamp(generated_at)
    run_id = str(config.portfolio_run_id or "")
    if not run_id:
        return pd.DataFrame(columns=LIVE_ASSET_SNAPSHOT_COLUMNS), pd.DataFrame(columns=LIVE_DECISION_SNAPSHOT_COLUMNS), explanation_lookup, ["Select and save a joint portfolio run first."]

    run_bundle = experiment_store.load_run(run_id)
    if run_bundle is None:
        return pd.DataFrame(columns=LIVE_ASSET_SNAPSHOT_COLUMNS), pd.DataFrame(columns=LIVE_DECISION_SNAPSHOT_COLUMNS), explanation_lookup, [f"Saved portfolio run `{run_id}` could not be loaded."]

    portfolio_bundle = run_bundle.get("portfolio_model_bundle", {}) or {}
    deployment_variant = str(config.deployment_variant or portfolio_bundle.get("deployment_variant") or "")
    variant_payload = (portfolio_bundle.get("variants", {}) or {}).get(deployment_variant, {})
    if not variant_payload:
        return pd.DataFrame(columns=LIVE_ASSET_SNAPSHOT_COLUMNS), pd.DataFrame(columns=LIVE_DECISION_SNAPSHOT_COLUMNS), explanation_lookup, [f"Deployment variant `{deployment_variant}` is not available in the pinned portfolio run."]

    selected_symbols = [str(symbol).upper() for symbol in (variant_payload.get("selected_symbols") or portfolio_bundle.get("selected_symbols") or [])]
    asset_session_features = store.read_frame("asset_session_features")
    asset_post_mappings = store.read_frame("asset_post_mappings")
    if asset_session_features.empty:
        return pd.DataFrame(columns=LIVE_ASSET_SNAPSHOT_COLUMNS), pd.DataFrame(columns=LIVE_DECISION_SNAPSHOT_COLUMNS), explanation_lookup, ["Refresh datasets first so `asset_session_features` is available."]
    live_rows = asset_session_features.copy()
    live_rows["asset_symbol"] = live_rows["asset_symbol"].astype(str).str.upper()
    if selected_symbols:
        live_rows = live_rows.loc[live_rows["asset_symbol"].isin(selected_symbols)].copy()
    feature_version = str(variant_payload.get("feature_version") or portfolio_bundle.get("feature_version") or "")
    if feature_version and "feature_version" in live_rows.columns:
        matching = live_rows.loc[live_rows["feature_version"].astype(str) == feature_version].copy()
        if not matching.empty:
            live_rows = matching
    llm_enabled = bool(variant_payload.get("llm_enabled", portfolio_bundle.get("llm_enabled", False)))
    if "llm_enabled" in live_rows.columns:
        matching = live_rows.loc[live_rows["llm_enabled"].fillna(False).astype(bool) == llm_enabled].copy()
        if not matching.empty:
            live_rows = matching
    if live_rows.empty:
        return pd.DataFrame(columns=LIVE_ASSET_SNAPSHOT_COLUMNS), pd.DataFrame(columns=LIVE_DECISION_SNAPSHOT_COLUMNS), explanation_lookup, ["No live asset-session rows matched the pinned portfolio run."]

    latest_rows = (
        live_rows.sort_values(["asset_symbol", "signal_session_date"])
        .groupby("asset_symbol", as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    weighted_rows = _apply_live_account_weight(latest_rows, float(variant_payload.get("account_weight", 1.0) or 1.0))
    scored_rows, explanation_models = _predict_portfolio_live_variant(
        model_service=model_service,
        feature_rows=weighted_rows,
        variant_payload=variant_payload,
        selected_symbols=selected_symbols,
    )
    if scored_rows.empty:
        return pd.DataFrame(columns=LIVE_ASSET_SNAPSHOT_COLUMNS), pd.DataFrame(columns=LIVE_DECISION_SNAPSHOT_COLUMNS), explanation_lookup, ["The pinned portfolio models could not score any live rows."]

    threshold = float(variant_payload.get("threshold", 0.0) or 0.0)
    min_post_count = int(variant_payload.get("min_post_count", 1) or 1)
    run_name = str(config.portfolio_run_name or run_bundle.get("run", {}).get("run_name", run_id) or run_id)
    snapshot_rows: list[dict[str, Any]] = []
    for _, row in scored_rows.sort_values("signal_session_date").iterrows():
        asset_symbol = str(row.get("asset_symbol", "") or "").upper()
        snapshot_rows.append(
            {
                "generated_at": snapshot_time,
                "variant_name": deployment_variant,
                "signal_session_date": row.get("signal_session_date"),
                "next_session_date": row.get("next_session_date"),
                "asset_symbol": asset_symbol,
                "run_id": run_id,
                "run_name": run_name,
                "feature_version": str(row.get("feature_version", feature_version) or feature_version),
                "model_version": str(row.get("model_version", "") or ""),
                "expected_return_score": float(row.get("expected_return_score", 0.0) or 0.0),
                "confidence": float(row.get("prediction_confidence", 0.0) or 0.0),
                "threshold": threshold,
                "min_post_count": min_post_count,
                "post_count": int(row.get("post_count", 0) or 0),
                "target_available": bool(row.get("target_available", False)),
                "tradeable": bool(row.get("tradeable", row.get("target_available", False))),
                "next_session_open": row.get("next_session_open"),
                "next_session_close": row.get("next_session_close"),
                "next_session_open_ts": row.get("next_session_open_ts"),
            },
        )
        session_date = _normalize_session_date(row.get("signal_session_date"))
        if asset_post_mappings.empty or session_date is None:
            post_attribution = pd.DataFrame()
        else:
            post_rows = asset_post_mappings.copy()
            post_rows["asset_symbol"] = post_rows["asset_symbol"].astype(str).str.upper()
            session_mask = pd.to_datetime(post_rows["session_date"], errors="coerce").dt.normalize() == session_date
            post_rows = post_rows.loc[(post_rows["asset_symbol"] == asset_symbol) & session_mask].copy()
            post_attribution = build_post_attribution(post_rows)
        account_attribution = build_account_attribution(post_attribution)
        explanation_payload = explanation_models.get(asset_symbol, {})
        explanation_lookup[asset_symbol] = {
            "prediction_row": row,
            "feature_contributions": explanation_payload.get("feature_contributions", pd.DataFrame()),
            "post_attribution": post_attribution,
            "account_attribution": account_attribution,
            "variant_name": deployment_variant,
        }

    snapshots = pd.DataFrame(snapshot_rows)
    if snapshots.empty:
        return pd.DataFrame(columns=LIVE_ASSET_SNAPSHOT_COLUMNS), pd.DataFrame(columns=LIVE_DECISION_SNAPSHOT_COLUMNS), explanation_lookup, ["No live portfolio snapshots were produced."]

    ranked_board, decision = rank_live_asset_snapshots(snapshots, config.fallback_mode)
    decision["deployment_variant"] = deployment_variant
    decision["portfolio_run_id"] = run_id
    decision["portfolio_run_name"] = run_name
    for asset_symbol, payload in explanation_lookup.items():
        board_match = ranked_board.loc[ranked_board["asset_symbol"].astype(str) == asset_symbol]
        if not board_match.empty:
            payload["board_row"] = board_match.iloc[0]
    return ranked_board, decision, explanation_lookup, warnings
