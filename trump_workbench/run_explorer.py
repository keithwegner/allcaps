from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .experiments import ExperimentStore
from .modeling import classify_feature_family


HEAVY_TABLE_LIMIT = 100
SESSION_TABLE_LIMIT = 12


def json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def frame_records(frame: pd.DataFrame, limit: int | None = None, tail: bool = False) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    if limit is not None:
        out = out.tail(limit) if tail else out.head(limit)
    return [json_safe(record) for record in out.to_dict(orient="records")]


def _plotly_json(figure: go.Figure | None) -> dict[str, Any] | None:
    if figure is None:
        return None
    return json_safe(figure.to_plotly_json())


def normalize_session_date(value: Any) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.tz_localize(None)
    return pd.Timestamp(timestamp).normalize()


def filter_for_session(frame: pd.DataFrame, session_date: pd.Timestamp | None, column: str = "signal_session_date") -> pd.DataFrame:
    if frame.empty or session_date is None or column not in frame.columns:
        return frame.head(0).copy()
    values = pd.to_datetime(frame[column], errors="coerce")
    if getattr(values.dt, "tz", None) is not None:
        values = values.dt.tz_localize(None)
    return frame.loc[values.dt.normalize() == session_date].copy()


def prediction_option_label(row: pd.Series) -> str:
    session_date = normalize_session_date(row.get("signal_session_date"))
    date_label = f"{session_date:%Y-%m-%d}" if session_date is not None else "unknown session"
    score = float(row.get("expected_return_score", 0.0) or 0.0)
    post_count = int(row.get("post_count", 0) or 0)
    return f"{date_label} | score {score:+.3%} | posts {post_count}"


def portfolio_decision_option_label(row: pd.Series) -> str:
    session_date = normalize_session_date(row.get("signal_session_date"))
    date_label = f"{session_date:%Y-%m-%d}" if session_date is not None else "unknown session"
    winning_asset = str(row.get("winning_asset", "") or "FLAT")
    winner_score = float(row.get("winner_score", 0.0) or 0.0)
    runner_up = str(row.get("runner_up_asset", "") or "n/a")
    return f"{date_label} | winner {winning_asset} | score {winner_score:+.3%} | runner-up {runner_up}"


def _format_compare_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, float):
        return round(value, 6)
    return value


def comparison_settings(run_bundle: dict[str, Any]) -> dict[str, Any]:
    config = run_bundle.get("config", {}) or {}
    selected = run_bundle.get("selected_params", {}) or {}
    artifact = run_bundle.get("model_artifact")
    run_meta = run_bundle.get("run", {}) or {}
    portfolio_bundle = run_bundle.get("portfolio_model_bundle", {}) or {}
    deployment_variant = str(
        selected.get("deployment_variant")
        or config.get("deployment_variant")
        or run_meta.get("deployment_variant")
        or portfolio_bundle.get("deployment_variant")
        or "",
    )
    variant_payload = (portfolio_bundle.get("variants", {}) or {}).get(deployment_variant, {})
    metadata = getattr(artifact, "metadata", {}) if artifact is not None else {}
    return {
        "run_type": str(run_meta.get("run_type") or metadata.get("run_type", "asset_model")),
        "allocator_mode": str(config.get("allocator_mode") or run_meta.get("allocator_mode") or metadata.get("allocator_mode", "")),
        "target_asset": str(config.get("target_asset") or run_meta.get("target_asset") or metadata.get("target_asset", "SPY")),
        "feature_version": config.get("feature_version", "v1"),
        "llm_enabled": bool(config.get("llm_enabled", metadata.get("llm_enabled", False))),
        "train_window": config.get("train_window"),
        "validation_window": config.get("validation_window"),
        "test_window": config.get("test_window"),
        "step_size": config.get("step_size"),
        "ridge_alpha": config.get("ridge_alpha"),
        "transaction_cost_bps": config.get("transaction_cost_bps"),
        "threshold_grid": tuple(config.get("threshold_grid", [])),
        "minimum_signal_grid": tuple(config.get("minimum_signal_grid", [])),
        "account_weight_grid": tuple(config.get("account_weight_grid", [])),
        "fallback_mode": str(config.get("fallback_mode") or run_meta.get("fallback_mode") or selected.get("fallback_mode") or "").upper(),
        "component_run_ids": tuple(selected.get("component_run_ids", config.get("component_run_ids", run_meta.get("component_run_ids", [])))),
        "universe_symbols": tuple(selected.get("universe_symbols", config.get("universe_symbols", run_meta.get("universe_symbols", [])))),
        "selected_symbols": tuple(selected.get("selected_symbols", config.get("selected_symbols", run_meta.get("selected_symbols", [])))),
        "deployment_variant": deployment_variant,
        "deployment_topology": str(selected.get("deployment_topology") or variant_payload.get("topology") or deployment_variant or ""),
        "deployment_narrative_feature_mode": str(
            selected.get("deployment_narrative_feature_mode")
            or variant_payload.get("narrative_feature_mode")
            or "unspecified",
        ),
        "topology_variants": tuple(config.get("topology_variants", run_meta.get("topology_variants", []))),
        "narrative_feature_modes": tuple(
            config.get("narrative_feature_modes", run_meta.get("narrative_feature_modes", portfolio_bundle.get("narrative_feature_modes", []))),
        ),
        "model_families": tuple(config.get("model_families", run_meta.get("model_families", []))),
        "deploy_threshold": selected.get("threshold"),
        "deploy_min_post_count": selected.get("min_post_count"),
        "deploy_account_weight": selected.get("account_weight"),
    }


def bundle_feature_names(run_bundle: dict[str, Any], variant_name: str | None = None) -> list[str]:
    run_type = comparison_settings(run_bundle).get("run_type", "asset_model")
    artifact = run_bundle.get("model_artifact")
    if run_type == "asset_model" and artifact is not None:
        return list(getattr(artifact, "feature_names", []))

    portfolio_bundle = run_bundle.get("portfolio_model_bundle", {}) or {}
    deployment_variant = str(
        variant_name
        or portfolio_bundle.get("deployment_variant")
        or comparison_settings(run_bundle).get("deployment_variant")
        or "",
    )
    variant_payload = (portfolio_bundle.get("variants", {}) or {}).get(deployment_variant, {})
    models_payload = variant_payload.get("models", {}) or {}
    feature_names: set[str] = set()
    for artifact_payload in models_payload.values():
        if isinstance(artifact_payload, dict):
            feature_names.update(str(name) for name in artifact_payload.get("feature_names", []) if name)
    return sorted(feature_names)


def build_metric_comparison_table(base_run_id: str, run_bundles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if base_run_id not in run_bundles:
        return pd.DataFrame()
    base_metrics = run_bundles[base_run_id].get("metrics", {}) or {}
    rows: list[dict[str, Any]] = []
    for run_id, bundle in run_bundles.items():
        metrics = bundle.get("metrics", {}) or {}
        settings = comparison_settings(bundle)
        feature_names = bundle_feature_names(bundle)
        rows.append(
            {
                "run_id": run_id,
                "run_name": bundle.get("run", {}).get("run_name", run_id),
                "run_type": settings.get("run_type", "asset_model"),
                "allocator_mode": settings.get("allocator_mode", ""),
                "target_asset": settings.get("target_asset", "SPY"),
                "deployment_variant": settings.get("deployment_variant", ""),
                "deployment_narrative_feature_mode": settings.get("deployment_narrative_feature_mode", ""),
                "total_return": metrics.get("total_return", 0.0),
                "sharpe": metrics.get("sharpe", 0.0),
                "sortino": metrics.get("sortino", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "robust_score": metrics.get("robust_score", 0.0),
                "trade_count": metrics.get("trade_count", 0.0),
                "feature_count": len(feature_names),
                "delta_total_return_vs_base": metrics.get("total_return", 0.0) - base_metrics.get("total_return", 0.0),
                "delta_sharpe_vs_base": metrics.get("sharpe", 0.0) - base_metrics.get("sharpe", 0.0),
                "delta_robust_score_vs_base": metrics.get("robust_score", 0.0) - base_metrics.get("robust_score", 0.0),
            },
        )
    return pd.DataFrame(rows).sort_values("delta_robust_score_vs_base", ascending=False).reset_index(drop=True)


def build_setting_diff_table(base_run_id: str, run_bundles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if base_run_id not in run_bundles:
        return pd.DataFrame()
    settings_by_run = {run_id: comparison_settings(bundle) for run_id, bundle in run_bundles.items()}
    base_settings = settings_by_run[base_run_id]
    diff_rows: list[dict[str, Any]] = []
    for key in sorted({setting for settings in settings_by_run.values() for setting in settings}):
        row = {"setting": key}
        base_value = base_settings.get(key)
        is_different = False
        for run_id, settings in settings_by_run.items():
            value = settings.get(key)
            row[run_id] = _format_compare_value(value)
            if value != base_value:
                is_different = True
        if is_different:
            diff_rows.append(row)
    return pd.DataFrame(diff_rows)


def build_feature_diff_table(base_run_id: str, run_bundles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if base_run_id not in run_bundles:
        return pd.DataFrame()
    base_features = set(bundle_feature_names(run_bundles[base_run_id]))
    rows: list[dict[str, Any]] = []
    for run_id, bundle in run_bundles.items():
        feature_names = bundle_feature_names(bundle)
        families = Counter(classify_feature_family(feature_name) for feature_name in feature_names)
        features = set(feature_names)
        unique_vs_base = sorted(features - base_features)
        omitted_vs_base = sorted(base_features - features)
        rows.append(
            {
                "run_id": run_id,
                "run_name": bundle.get("run", {}).get("run_name", run_id),
                "target_asset": comparison_settings(bundle).get("target_asset", "SPY"),
                "deployment_variant": comparison_settings(bundle).get("deployment_variant", ""),
                "deployment_narrative_feature_mode": comparison_settings(bundle).get("deployment_narrative_feature_mode", ""),
                "feature_count": len(feature_names),
                "semantic_features": families.get("semantic", 0),
                "policy_features": families.get("policy", 0),
                "market_context_features": families.get("market_context", 0),
                "social_sentiment_features": families.get("social_sentiment", 0),
                "activity_features": families.get("activity", 0),
                "account_structure_features": families.get("account_structure", 0),
                "unique_vs_base_count": len(unique_vs_base),
                "omitted_vs_base_count": len(omitted_vs_base),
                "unique_vs_base": ", ".join(unique_vs_base[:6]),
                "omitted_vs_base": ", ".join(omitted_vs_base[:6]),
            },
        )
    return pd.DataFrame(rows).sort_values(["unique_vs_base_count", "feature_count"], ascending=[False, False]).reset_index(drop=True)


def variant_summary_with_narrative_defaults(variant_summary: pd.DataFrame) -> pd.DataFrame:
    normalized = variant_summary.copy()
    if normalized.empty:
        return normalized
    if "variant_name" not in normalized.columns:
        normalized["variant_name"] = ""
    if "topology" not in normalized.columns:
        normalized["topology"] = normalized["variant_name"].astype(str)
    if "narrative_feature_mode" not in normalized.columns:
        normalized["narrative_feature_mode"] = "unspecified"
    normalized["topology"] = normalized["topology"].replace("", pd.NA).fillna(normalized["variant_name"].astype(str))
    normalized["narrative_feature_mode"] = normalized["narrative_feature_mode"].replace("", "unspecified").fillna("unspecified")
    return normalized


def build_narrative_lift_table(variant_summary: pd.DataFrame) -> pd.DataFrame:
    summary = variant_summary_with_narrative_defaults(variant_summary)
    if summary.empty or "baseline" not in set(summary["narrative_feature_mode"].astype(str)):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    metric_pairs = [
        ("validation_robust_score", "validation_robust_lift"),
        ("validation_total_return", "validation_return_lift"),
        ("test_robust_score", "test_robust_lift"),
        ("test_total_return", "test_return_lift"),
    ]
    baseline_rows = summary.loc[summary["narrative_feature_mode"].astype(str) == "baseline"].copy()
    for _, row in summary.iterrows():
        mode = str(row.get("narrative_feature_mode", "unspecified") or "unspecified")
        if mode in {"baseline", "unspecified"}:
            continue
        topology = str(row.get("topology", "") or "")
        baseline = baseline_rows.loc[baseline_rows["topology"].astype(str) == topology]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        lift_row = {
            "variant_name": row.get("variant_name", ""),
            "topology": topology,
            "narrative_feature_mode": mode,
            "baseline_variant": baseline_row.get("variant_name", ""),
        }
        for metric_column, output_column in metric_pairs:
            current_value = pd.to_numeric(pd.Series([row.get(metric_column)]), errors="coerce").iloc[0]
            baseline_value = pd.to_numeric(pd.Series([baseline_row.get(metric_column)]), errors="coerce").iloc[0]
            lift_row[output_column] = (
                float(current_value - baseline_value)
                if pd.notna(current_value) and pd.notna(baseline_value)
                else np.nan
            )
        rows.append(lift_row)
    return pd.DataFrame(rows).sort_values("validation_robust_lift", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def build_feature_family_summary(
    run_bundle: dict[str, Any],
    variant_name: str | None = None,
    importance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    importance_frame = pd.DataFrame() if importance is None else importance.copy()
    if not importance_frame.empty and "feature_name" in importance_frame.columns:
        if variant_name and "variant_name" in importance_frame.columns:
            filtered = importance_frame.loc[importance_frame["variant_name"].astype(str) == variant_name].copy()
            if not filtered.empty:
                importance_frame = filtered
        importance_frame["feature_family"] = importance_frame["feature_name"].astype(str).map(classify_feature_family)
        value_column = "importance" if "importance" in importance_frame.columns else "abs_coefficient"
        if value_column not in importance_frame.columns:
            value_column = None
        rows: list[dict[str, Any]] = []
        for family, family_rows in importance_frame.groupby("feature_family", sort=True):
            top_features = family_rows.copy()
            if value_column is not None:
                top_features[value_column] = pd.to_numeric(top_features[value_column], errors="coerce").fillna(0.0)
                top_features = top_features.sort_values(value_column, ascending=False)
            rows.append(
                {
                    "feature_family": family,
                    "feature_count": int(family_rows["feature_name"].nunique()),
                    "total_importance": float(pd.to_numeric(family_rows[value_column], errors="coerce").fillna(0.0).sum()) if value_column else np.nan,
                    "top_features": ", ".join(top_features["feature_name"].dropna().astype(str).head(5).tolist()),
                },
            )
        return pd.DataFrame(rows).sort_values(["total_importance", "feature_count"], ascending=[False, False]).reset_index(drop=True)

    feature_names = bundle_feature_names(run_bundle, variant_name=variant_name)
    families = Counter(classify_feature_family(feature_name) for feature_name in feature_names)
    rows = [
        {
            "feature_family": family,
            "feature_count": int(count),
            "total_importance": np.nan,
            "top_features": ", ".join(
                feature_name for feature_name in feature_names if classify_feature_family(feature_name) == family
            ),
        }
        for family, count in families.items()
    ]
    return pd.DataFrame(rows).sort_values("feature_count", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def build_benchmark_delta_table(base_run_id: str, run_bundles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if base_run_id not in run_bundles:
        return pd.DataFrame()
    base_benchmarks = run_bundles[base_run_id].get("benchmarks", pd.DataFrame())
    if base_benchmarks.empty or "benchmark_name" not in base_benchmarks.columns:
        return pd.DataFrame()
    base_lookup = base_benchmarks.set_index("benchmark_name")
    base_strategy = float(base_lookup.loc["strategy", "total_return"]) if "strategy" in base_lookup.index and "total_return" in base_lookup.columns else np.nan
    rows: list[dict[str, Any]] = []
    for run_id, bundle in run_bundles.items():
        benchmarks = bundle.get("benchmarks", pd.DataFrame())
        if benchmarks.empty or "benchmark_name" not in benchmarks.columns:
            continue
        target_asset = comparison_settings(bundle).get("target_asset", "SPY")
        lookup = benchmarks.set_index("benchmark_name")
        strategy_return = float(lookup.loc["strategy", "total_return"]) if "strategy" in lookup.index and "total_return" in lookup.columns else np.nan
        for benchmark_name in sorted(set(base_lookup.index).union(set(lookup.index))):
            current_total = float(lookup.loc[benchmark_name, "total_return"]) if benchmark_name in lookup.index and "total_return" in lookup.columns else np.nan
            base_total = float(base_lookup.loc[benchmark_name, "total_return"]) if benchmark_name in base_lookup.index and "total_return" in base_lookup.columns else np.nan
            current_edge = strategy_return - current_total if pd.notna(strategy_return) and pd.notna(current_total) else np.nan
            base_edge = base_strategy - base_total if pd.notna(base_strategy) and pd.notna(base_total) else np.nan
            rows.append(
                {
                    "run_id": run_id,
                    "target_asset": target_asset,
                    "benchmark_name": benchmark_name,
                    "total_return": current_total,
                    "delta_total_return_vs_base": current_total - base_total if pd.notna(current_total) and pd.notna(base_total) else np.nan,
                    "edge_vs_strategy": current_edge,
                    "delta_edge_vs_base": current_edge - base_edge if pd.notna(current_edge) and pd.notna(base_edge) else np.nan,
                },
            )
    return pd.DataFrame(rows).sort_values(["benchmark_name", "delta_edge_vs_base"], ascending=[True, False]).reset_index(drop=True)


def summarize_run_changes(base_run_id: str, run_bundles: dict[str, dict[str, Any]]) -> list[str]:
    if base_run_id not in run_bundles:
        return []
    base_metrics = run_bundles[base_run_id].get("metrics", {}) or {}
    base_settings = comparison_settings(run_bundles[base_run_id])
    base_features = set(bundle_feature_names(run_bundles[base_run_id]))
    notes: list[str] = []
    for run_id, bundle in run_bundles.items():
        if run_id == base_run_id:
            continue
        metrics = bundle.get("metrics", {}) or {}
        settings = comparison_settings(bundle)
        features = set(bundle_feature_names(bundle))
        parts: list[str] = [
            f"robust score {metrics.get('robust_score', 0.0) - base_metrics.get('robust_score', 0.0):+.3f}",
            f"total return {metrics.get('total_return', 0.0) - base_metrics.get('total_return', 0.0):+.2%}",
        ]
        if settings.get("llm_enabled") != base_settings.get("llm_enabled"):
            parts.append(f"LLM {'on' if settings.get('llm_enabled') else 'off'} vs {'on' if base_settings.get('llm_enabled') else 'off'}")
        if settings.get("run_type") != base_settings.get("run_type"):
            parts.append(f"run type {base_settings.get('run_type')} -> {settings.get('run_type')}")
        if settings.get("allocator_mode") != base_settings.get("allocator_mode"):
            parts.append(f"allocator {base_settings.get('allocator_mode') or 'n/a'} -> {settings.get('allocator_mode') or 'n/a'}")
        if settings.get("target_asset") != base_settings.get("target_asset"):
            parts.append(f"target asset {base_settings.get('target_asset')} -> {settings.get('target_asset')}")
        if settings.get("fallback_mode") != base_settings.get("fallback_mode"):
            parts.append(f"fallback {base_settings.get('fallback_mode') or 'n/a'} -> {settings.get('fallback_mode') or 'n/a'}")
        if settings.get("deploy_threshold") != base_settings.get("deploy_threshold"):
            parts.append(f"threshold {base_settings.get('deploy_threshold')} -> {settings.get('deploy_threshold')}")
        if settings.get("deploy_min_post_count") != base_settings.get("deploy_min_post_count"):
            parts.append(f"min posts {base_settings.get('deploy_min_post_count')} -> {settings.get('deploy_min_post_count')}")
        if settings.get("deploy_account_weight") != base_settings.get("deploy_account_weight"):
            parts.append(f"account weight {base_settings.get('deploy_account_weight')} -> {settings.get('deploy_account_weight')}")
        if settings.get("deployment_variant") != base_settings.get("deployment_variant"):
            parts.append(f"deployment variant {base_settings.get('deployment_variant') or 'n/a'} -> {settings.get('deployment_variant') or 'n/a'}")
        if settings.get("deployment_narrative_feature_mode") != base_settings.get("deployment_narrative_feature_mode"):
            parts.append(
                "narrative mode "
                f"{base_settings.get('deployment_narrative_feature_mode') or 'n/a'} -> "
                f"{settings.get('deployment_narrative_feature_mode') or 'n/a'}",
            )
        unique_vs_base = sorted(features - base_features)
        omitted_vs_base = sorted(base_features - features)
        if unique_vs_base:
            parts.append(f"{len(unique_vs_base)} added features ({', '.join(unique_vs_base[:3])})")
        if omitted_vs_base:
            parts.append(f"{len(omitted_vs_base)} removed features ({', '.join(omitted_vs_base[:3])})")
        notes.append(f"{run_id}: " + "; ".join(parts) + ".")
    return notes


def _filter_variant(frame: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    if frame.empty or not variant_name or "variant_name" not in frame.columns:
        return frame.copy()
    filtered = frame.loc[frame["variant_name"].astype(str) == variant_name].copy()
    return filtered if not filtered.empty else frame.copy()


def _sort_by_date(frame: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame.copy()
    out = frame.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    return out.sort_values(column, ascending=ascending).reset_index(drop=True)


def _equity_figure(trades: pd.DataFrame, title: str) -> dict[str, Any] | None:
    if trades.empty:
        return None
    x_column = "next_session_date" if "next_session_date" in trades.columns else "signal_session_date"
    y_column = next((column for column in ["equity_curve", "ending_equity", "equity"] if column in trades.columns), None)
    if x_column not in trades.columns or y_column is None:
        return None
    curve = _sort_by_date(trades[[x_column, y_column]].dropna(), x_column)
    if curve.empty:
        return None
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=curve[x_column], y=curve[y_column], mode="lines", name="Strategy equity"))
    figure.update_layout(title=title, xaxis_title="Trade date", yaxis_title="Equity", margin={"l": 30, "r": 20, "t": 55, "b": 35})
    return _plotly_json(figure)


def _benchmark_figure(curves: pd.DataFrame) -> dict[str, Any] | None:
    if curves.empty:
        return None
    x_column = "next_session_date" if "next_session_date" in curves.columns else "signal_session_date"
    if x_column not in curves.columns:
        return None
    figure = go.Figure()
    numeric_columns = [
        column for column in curves.columns
        if column not in {x_column, "variant_name"} and pd.api.types.is_numeric_dtype(curves[column])
    ]
    if not numeric_columns:
        return None
    sorted_curves = _sort_by_date(curves, x_column)
    for column in numeric_columns:
        figure.add_trace(go.Scatter(x=sorted_curves[x_column], y=sorted_curves[column], mode="lines", name=column))
    figure.update_layout(title="Strategy vs. benchmark equity curves", xaxis_title="Trade date", yaxis_title="Equity", margin={"l": 30, "r": 20, "t": 55, "b": 35})
    return _plotly_json(figure)


def _diagnostics_figure(diagnostics: pd.DataFrame, run_type: str) -> dict[str, Any] | None:
    if diagnostics.empty:
        return None
    figure = go.Figure()
    if run_type == "portfolio_allocator" and {"signal_session_date", "winner_score"}.issubset(diagnostics.columns):
        rows = _sort_by_date(diagnostics, "signal_session_date")
        figure.add_trace(go.Scatter(x=rows["signal_session_date"], y=rows["winner_score"], mode="lines+markers", name="Winner score"))
        if "winner_gap_vs_runner_up" in rows.columns:
            figure.add_trace(go.Scatter(x=rows["signal_session_date"], y=rows["winner_gap_vs_runner_up"], mode="lines", name="Gap vs runner-up"))
        figure.update_layout(title="Portfolio allocator diagnostics", xaxis_title="Signal session", yaxis_title="Score")
        return _plotly_json(figure)
    if {"expected_return_score", "actual_next_session_return"}.issubset(diagnostics.columns):
        figure.add_trace(
            go.Scatter(
                x=diagnostics["expected_return_score"],
                y=diagnostics["actual_next_session_return"],
                mode="markers",
                marker={"size": 8, "opacity": 0.65},
                name="Predictions",
            ),
        )
        figure.update_layout(
            title="Prediction diagnostics: expected vs actual next-session return",
            xaxis_title="Expected return score",
            yaxis_title="Actual next-session return",
        )
        return _plotly_json(figure)
    return None


def _session_options(frame: pd.DataFrame, run_type: str) -> list[dict[str, Any]]:
    if frame.empty or "signal_session_date" not in frame.columns:
        return []
    rows = _sort_by_date(frame, "signal_session_date", ascending=False)
    options: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        session_date = normalize_session_date(row.get("signal_session_date"))
        if session_date is None:
            continue
        label = portfolio_decision_option_label(row) if run_type == "portfolio_allocator" else prediction_option_label(row)
        options.append({"value": f"{session_date:%Y-%m-%d}", "label": label})
    seen: set[str] = set()
    unique_options: list[dict[str, Any]] = []
    for option in options:
        if option["value"] not in seen:
            unique_options.append(option)
            seen.add(option["value"])
    return unique_options[:250]


def _asset_session_payload(
    prediction_rows: pd.DataFrame,
    feature_contributions: pd.DataFrame,
    post_attribution: pd.DataFrame,
    account_attribution: pd.DataFrame,
    session_date: pd.Timestamp | None,
) -> dict[str, Any]:
    selected_prediction = filter_for_session(prediction_rows, session_date)
    selected_prediction = selected_prediction.head(1)
    session_contributions = filter_for_session(feature_contributions, session_date)
    session_posts = filter_for_session(post_attribution, session_date)
    session_accounts = filter_for_session(account_attribution, session_date)
    if not session_contributions.empty and "contribution" in session_contributions.columns:
        session_contributions = session_contributions.assign(
            abs_contribution=pd.to_numeric(session_contributions["contribution"], errors="coerce").abs(),
        ).sort_values("abs_contribution", ascending=False)
    if not session_posts.empty and "post_signal_score" in session_posts.columns:
        session_posts = session_posts.assign(
            abs_post_signal_score=pd.to_numeric(session_posts["post_signal_score"], errors="coerce").abs(),
        ).sort_values("abs_post_signal_score", ascending=False)
    if not session_accounts.empty and "net_post_signal" in session_accounts.columns:
        session_accounts = session_accounts.assign(
            abs_net_post_signal=pd.to_numeric(session_accounts["net_post_signal"], errors="coerce").abs(),
        ).sort_values("abs_net_post_signal", ascending=False)
    return {
        "prediction": frame_records(selected_prediction, limit=1),
        "feature_contributions": frame_records(session_contributions, limit=SESSION_TABLE_LIMIT),
        "post_attribution": frame_records(session_posts, limit=SESSION_TABLE_LIMIT),
        "account_attribution": frame_records(session_accounts, limit=SESSION_TABLE_LIMIT),
        "candidates": [],
    }


def _portfolio_session_payload(
    decision_rows: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    session_date: pd.Timestamp | None,
) -> dict[str, Any]:
    selected_decision = filter_for_session(decision_rows, session_date).head(1)
    session_candidates = filter_for_session(candidate_predictions, session_date)
    if not session_candidates.empty and "expected_return_score" in session_candidates.columns:
        sort_columns = [column for column in ["is_winner", "qualifies", "expected_return_score"] if column in session_candidates.columns]
        ascending = [False, False, False][: len(sort_columns)]
        session_candidates = session_candidates.sort_values(sort_columns, ascending=ascending)
        if {"expected_return_score", "threshold"}.issubset(session_candidates.columns):
            session_candidates["threshold_gap"] = (
                pd.to_numeric(session_candidates["expected_return_score"], errors="coerce")
                - pd.to_numeric(session_candidates["threshold"], errors="coerce")
            )
    return {
        "decision": frame_records(selected_decision, limit=1),
        "candidates": frame_records(session_candidates, limit=SESSION_TABLE_LIMIT),
        "prediction": [],
        "feature_contributions": [],
        "post_attribution": [],
        "account_attribution": [],
    }


def _model_artifact_summary(run_bundle: dict[str, Any], variant_name: str) -> dict[str, Any]:
    artifact = run_bundle.get("model_artifact")
    feature_names = bundle_feature_names(run_bundle, variant_name=variant_name)
    if artifact is None:
        return {"feature_count": len(feature_names), "feature_names": feature_names[:25]}
    return {
        "model_version": getattr(artifact, "model_version", ""),
        "train_rows": getattr(artifact, "train_rows", None),
        "residual_std": getattr(artifact, "residual_std", None),
        "feature_count": len(feature_names),
        "feature_names": feature_names[:25],
        "metadata": getattr(artifact, "metadata", {}),
    }


def build_run_detail_payload(
    experiment_store: ExperimentStore,
    run_id: str,
    *,
    variant_name: str | None = None,
    session_date: str | None = None,
) -> dict[str, Any]:
    run_bundle = experiment_store.load_run(run_id)
    if run_bundle is None:
        return {
            "found": False,
            "run_id": run_id,
            "errors": [f"Run {run_id} could not be loaded."],
            "run": {},
            "settings": {},
            "metrics": {},
            "selected_params": {},
            "model_artifact": {},
            "charts": {},
            "tables": {},
            "row_counts": {},
            "session_options": [],
            "selected_session": {},
            "leakage_audit": {},
        }

    settings = comparison_settings(run_bundle)
    run_type = str(settings.get("run_type", "asset_model"))
    active_variant = str(variant_name or settings.get("deployment_variant") or "")

    trades = _filter_variant(run_bundle.get("trades", pd.DataFrame()), active_variant)
    predictions = _filter_variant(run_bundle.get("predictions", pd.DataFrame()), active_variant)
    windows = _filter_variant(run_bundle.get("windows", pd.DataFrame()), active_variant)
    importance = _filter_variant(run_bundle.get("importance", pd.DataFrame()), active_variant)
    benchmarks = _filter_variant(run_bundle.get("benchmarks", pd.DataFrame()), active_variant)
    diagnostics = _filter_variant(run_bundle.get("diagnostics", pd.DataFrame()), active_variant)
    benchmark_curves = _filter_variant(run_bundle.get("benchmark_curves", pd.DataFrame()), active_variant)
    candidate_predictions = _filter_variant(run_bundle.get("candidate_predictions", pd.DataFrame()), active_variant)
    variant_summary = variant_summary_with_narrative_defaults(run_bundle.get("variant_summary", pd.DataFrame()))
    narrative_lift = build_narrative_lift_table(variant_summary)
    family_summary = build_feature_family_summary(run_bundle, variant_name=active_variant, importance=importance)

    options = _session_options(predictions, run_type)
    selected_date = normalize_session_date(session_date) if session_date else None
    if selected_date is None and options:
        selected_date = normalize_session_date(options[0]["value"])
    if run_type == "portfolio_allocator":
        selected_session = _portfolio_session_payload(predictions, candidate_predictions, selected_date)
    else:
        selected_session = _asset_session_payload(
            predictions,
            run_bundle.get("feature_contributions", pd.DataFrame()),
            run_bundle.get("post_attribution", pd.DataFrame()),
            run_bundle.get("account_attribution", pd.DataFrame()),
            selected_date,
        )
    selected_session["session_date"] = f"{selected_date:%Y-%m-%d}" if selected_date is not None else ""

    row_counts = {
        "trades": int(len(trades)),
        "predictions": int(len(predictions)),
        "windows": int(len(windows)),
        "importance": int(len(importance)),
        "benchmarks": int(len(benchmarks)),
        "diagnostics": int(len(diagnostics)),
        "benchmark_curves": int(len(benchmark_curves)),
        "candidate_predictions": int(len(candidate_predictions)),
        "variant_summary": int(len(variant_summary)),
    }
    diagnostics_table = diagnostics.copy()
    if "absolute_error" in diagnostics_table.columns:
        diagnostics_table = diagnostics_table.sort_values("absolute_error", ascending=False)
    else:
        diagnostics_table = _sort_by_date(diagnostics_table, "signal_session_date", ascending=False)

    return json_safe(
        {
            "found": True,
            "run_id": run_id,
            "errors": [],
            "run": run_bundle.get("run", {}),
            "settings": settings,
            "metrics": run_bundle.get("metrics", {}) or {},
            "selected_params": run_bundle.get("selected_params", {}) or {},
            "model_artifact": _model_artifact_summary(run_bundle, active_variant),
            "charts": {
                "equity": _equity_figure(trades, "Walk-forward out-of-sample equity curve"),
                "benchmarks": _benchmark_figure(benchmark_curves),
                "diagnostics": _diagnostics_figure(diagnostics, run_type),
            },
            "tables": {
                "benchmarks": frame_records(benchmarks, limit=HEAVY_TABLE_LIMIT),
                "variant_summary": frame_records(variant_summary, limit=HEAVY_TABLE_LIMIT),
                "narrative_lift": frame_records(narrative_lift, limit=HEAVY_TABLE_LIMIT),
                "feature_family_summary": frame_records(family_summary, limit=HEAVY_TABLE_LIMIT),
                "windows": frame_records(windows, limit=HEAVY_TABLE_LIMIT),
                "feature_importance": frame_records(importance, limit=50),
                "diagnostics": frame_records(diagnostics_table, limit=HEAVY_TABLE_LIMIT),
                "trades": frame_records(_sort_by_date(trades, "next_session_date", ascending=False), limit=HEAVY_TABLE_LIMIT),
                "candidate_predictions": frame_records(candidate_predictions, limit=HEAVY_TABLE_LIMIT),
            },
            "row_counts": row_counts,
            "session_options": options,
            "selected_session": selected_session,
            "leakage_audit": run_bundle.get("leakage_audit", {}) or {},
        },
    )


def _comparison_equity_figure(run_bundles: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    figure = go.Figure()
    trace_count = 0
    for run_id, bundle in run_bundles.items():
        settings = comparison_settings(bundle)
        trades = _filter_variant(bundle.get("trades", pd.DataFrame()), str(settings.get("deployment_variant") or ""))
        if trades.empty:
            continue
        x_column = "next_session_date" if "next_session_date" in trades.columns else "signal_session_date"
        y_column = next((column for column in ["equity_curve", "ending_equity", "equity"] if column in trades.columns), None)
        if x_column not in trades.columns or y_column is None:
            continue
        rows = _sort_by_date(trades[[x_column, y_column]].dropna(), x_column)
        if rows.empty:
            continue
        figure.add_trace(go.Scatter(x=rows[x_column], y=rows[y_column], mode="lines", name=run_id))
        trace_count += 1
    if trace_count == 0:
        return None
    figure.update_layout(title="Selected run equity curves", xaxis_title="Trade date", yaxis_title="Equity", margin={"l": 30, "r": 20, "t": 55, "b": 35})
    return _plotly_json(figure)


def build_run_comparison_payload(
    experiment_store: ExperimentStore,
    run_ids: list[str] | tuple[str, ...] | None,
    *,
    base_run_id: str | None = None,
) -> dict[str, Any]:
    requested = [str(run_id) for run_id in (run_ids or []) if str(run_id).strip()]
    run_bundles: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for run_id in requested:
        bundle = experiment_store.load_run(run_id)
        if bundle is None:
            missing.append(run_id)
        else:
            run_bundles[run_id] = bundle

    if not run_bundles:
        return {
            "ready": False,
            "base_run_id": base_run_id or "",
            "run_ids": requested,
            "missing_run_ids": missing,
            "scorecard": [],
            "setting_diffs": [],
            "feature_diffs": [],
            "benchmark_deltas": [],
            "change_notes": [],
            "charts": {},
        }

    resolved_base = str(base_run_id or "")
    if resolved_base not in run_bundles:
        resolved_base = next(iter(run_bundles))

    return json_safe(
        {
            "ready": len(run_bundles) >= 2,
            "base_run_id": resolved_base,
            "run_ids": list(run_bundles.keys()),
            "missing_run_ids": missing,
            "scorecard": frame_records(build_metric_comparison_table(resolved_base, run_bundles)),
            "setting_diffs": frame_records(build_setting_diff_table(resolved_base, run_bundles)),
            "feature_diffs": frame_records(build_feature_diff_table(resolved_base, run_bundles)),
            "benchmark_deltas": frame_records(build_benchmark_delta_table(resolved_base, run_bundles)),
            "change_notes": summarize_run_changes(resolved_base, run_bundles),
            "charts": {"equity": _comparison_equity_figure(run_bundles)},
        },
    )
