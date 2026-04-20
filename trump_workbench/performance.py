from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .paper_trading import (
    ensure_paper_benchmark_curve_frame,
    ensure_paper_decision_journal_frame,
    ensure_paper_equity_curve_frame,
    ensure_paper_portfolio_registry_frame,
    ensure_paper_trade_ledger_frame,
)
from .utils import stable_text_id

PERFORMANCE_DIAGNOSTIC_COLUMNS = [
    "snapshot_id",
    "generated_at",
    "paper_portfolio_id",
    "portfolio_run_id",
    "deployment_variant",
    "scope_kind",
    "scope_key",
    "metric_name",
    "severity",
    "observed_value",
    "baseline_value",
    "detail",
]

PERFORMANCE_SEVERITY_ORDER = {"ok": 0, "warn": 1, "severe": 2}


def _empty_performance_diagnostic_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PERFORMANCE_DIAGNOSTIC_COLUMNS)


def _coerce_utc_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def ensure_performance_diagnostic_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_performance_diagnostic_frame()
    out = frame.copy()
    for column in PERFORMANCE_DIAGNOSTIC_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out["generated_at"] = pd.to_datetime(out["generated_at"], errors="coerce", utc=True)
    out["severity"] = out["severity"].fillna("ok").astype(str).str.lower()
    out["observed_value"] = pd.to_numeric(out["observed_value"], errors="coerce")
    out["baseline_value"] = pd.to_numeric(out["baseline_value"], errors="coerce")
    return out[PERFORMANCE_DIAGNOSTIC_COLUMNS].copy()


def _severity_from_thresholds(
    observed_value: float | None,
    *,
    warn_below: float | None = None,
    severe_below: float | None = None,
    warn_above: float | None = None,
    severe_above: float | None = None,
) -> str:
    if observed_value is None or pd.isna(observed_value):
        return "warn"
    value = float(observed_value)
    if severe_below is not None and value < severe_below:
        return "severe"
    if severe_above is not None and value > severe_above:
        return "severe"
    if warn_below is not None and value < warn_below:
        return "warn"
    if warn_above is not None and value > warn_above:
        return "warn"
    return "ok"


def _max_drawdown(equity: pd.Series) -> float:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    if values.empty:
        return 0.0
    running_max = values.cummax()
    drawdowns = values / running_max - 1.0
    return float(drawdowns.min())


def _series_return(curve: pd.DataFrame, starting_cash: float) -> float:
    if curve.empty or starting_cash == 0:
        return 0.0
    ordered = curve.sort_values("next_session_date")
    latest_equity = pd.to_numeric(ordered["equity"], errors="coerce").dropna()
    if latest_equity.empty:
        return 0.0
    return float(latest_equity.iloc[-1] / float(starting_cash) - 1.0)


def _score_outcome_join(journal: pd.DataFrame, trades: pd.DataFrame, paper_portfolio_id: str) -> pd.DataFrame:
    selected_journal = journal.loc[journal["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
    selected_trades = trades.loc[trades["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
    if selected_journal.empty or selected_trades.empty:
        return pd.DataFrame(columns=["signal_session_date", "winning_asset", "winner_score", "net_return"])
    selected_journal["signal_session_key"] = pd.to_datetime(
        selected_journal["signal_session_date"],
        errors="coerce",
        utc=True,
    ).dt.normalize()
    selected_trades["signal_session_key"] = pd.to_datetime(
        selected_trades["signal_session_date"],
        errors="coerce",
        utc=True,
    ).dt.normalize()
    joined = selected_journal.merge(
        selected_trades[["signal_session_key", "asset_symbol", "net_return"]],
        on="signal_session_key",
        how="inner",
    )
    if joined.empty:
        return pd.DataFrame(columns=["signal_session_date", "winning_asset", "winner_score", "net_return"])
    joined["winner_score"] = pd.to_numeric(joined["winner_score"], errors="coerce")
    joined["net_return"] = pd.to_numeric(joined["net_return"], errors="coerce")
    joined = joined.dropna(subset=["winner_score", "net_return"]).copy()
    return joined[
        ["signal_session_date", "winning_asset", "asset_symbol", "winner_score", "net_return"]
    ].reset_index(drop=True)


def build_equity_comparison_frame(
    equity: pd.DataFrame,
    benchmark: pd.DataFrame,
    paper_portfolio_id: str,
) -> pd.DataFrame:
    selected_equity = ensure_paper_equity_curve_frame(equity)
    selected_benchmark = ensure_paper_benchmark_curve_frame(benchmark)
    selected_equity = selected_equity.loc[selected_equity["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
    selected_benchmark = selected_benchmark.loc[
        (selected_benchmark["paper_portfolio_id"].astype(str) == str(paper_portfolio_id))
        & (selected_benchmark["benchmark_name"].astype(str) == "always_long_spy")
    ].copy()
    if selected_equity.empty and selected_benchmark.empty:
        return pd.DataFrame(columns=["next_session_date", "paper_portfolio_equity", "spy_benchmark_equity"])
    left = selected_equity[["next_session_date", "equity"]].rename(columns={"equity": "paper_portfolio_equity"})
    right = selected_benchmark[["next_session_date", "equity"]].rename(columns={"equity": "spy_benchmark_equity"})
    merged = left.merge(right, on="next_session_date", how="outer").sort_values("next_session_date").reset_index(drop=True)
    return merged


def build_rolling_return_frame(
    trades: pd.DataFrame,
    paper_portfolio_id: str,
    window: int = 5,
) -> pd.DataFrame:
    selected_trades = ensure_paper_trade_ledger_frame(trades)
    selected_trades = selected_trades.loc[selected_trades["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
    if selected_trades.empty:
        return pd.DataFrame(columns=["next_session_date", "net_return", "rolling_net_return"])
    selected_trades = selected_trades.sort_values("next_session_date").reset_index(drop=True)
    selected_trades["net_return"] = pd.to_numeric(selected_trades["net_return"], errors="coerce").fillna(0.0)
    selected_trades["rolling_net_return"] = selected_trades["net_return"].rolling(max(1, int(window)), min_periods=1).mean()
    return selected_trades[["next_session_date", "net_return", "rolling_net_return"]].copy()


def build_score_outcome_frame(
    journal: pd.DataFrame,
    trades: pd.DataFrame,
    paper_portfolio_id: str,
) -> pd.DataFrame:
    return _score_outcome_join(
        ensure_paper_decision_journal_frame(journal),
        ensure_paper_trade_ledger_frame(trades),
        paper_portfolio_id,
    )


def build_score_bucket_outcome_frame(
    journal: pd.DataFrame,
    trades: pd.DataFrame,
    paper_portfolio_id: str,
) -> pd.DataFrame:
    scored = build_score_outcome_frame(journal, trades, paper_portfolio_id)
    if scored.empty:
        return pd.DataFrame(columns=["score_bucket", "trade_count", "mean_net_return", "win_rate"])
    try:
        scored["score_bucket"] = pd.qcut(
            scored["winner_score"],
            q=min(3, int(scored["winner_score"].nunique())),
            labels=["low", "medium", "high"][: min(3, int(scored["winner_score"].nunique()))],
            duplicates="drop",
        ).astype(str)
    except ValueError:
        scored["score_bucket"] = "all"
    grouped = (
        scored.groupby("score_bucket", dropna=False)
        .agg(
            trade_count=("net_return", "size"),
            mean_net_return=("net_return", "mean"),
            win_rate=("net_return", lambda series: float((series > 0).mean()) if len(series) else 0.0),
        )
        .reset_index()
    )
    return grouped


def build_winner_distribution_frame(
    journal: pd.DataFrame,
    paper_portfolio_id: str,
) -> pd.DataFrame:
    selected_journal = ensure_paper_decision_journal_frame(journal)
    selected_journal = selected_journal.loc[selected_journal["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
    if selected_journal.empty:
        return pd.DataFrame(columns=["winning_asset", "decision_count"])
    selected_journal["winning_asset"] = selected_journal["winning_asset"].fillna("").astype(str).replace({"": "FLAT"})
    counts = selected_journal.groupby("winning_asset").size().reset_index(name="decision_count")
    return counts.sort_values("decision_count", ascending=False).reset_index(drop=True)


def build_live_score_drift_frame(
    live_asset_snapshots: pd.DataFrame,
    portfolio_run_id: str,
    deployment_variant: str,
    recent_window: int = 10,
    baseline_window: int = 30,
) -> pd.DataFrame:
    if live_asset_snapshots.empty:
        return pd.DataFrame(columns=["asset_symbol", "metric_name", "recent_mean", "baseline_mean", "baseline_std", "z_score"])
    snapshots = live_asset_snapshots.copy()
    snapshots["run_id"] = snapshots.get("run_id", "").astype(str)
    snapshots = snapshots.loc[snapshots["run_id"] == str(portfolio_run_id)].copy()
    if "variant_name" in snapshots.columns and str(deployment_variant or ""):
        snapshots = snapshots.loc[snapshots["variant_name"].astype(str) == str(deployment_variant)].copy()
    if snapshots.empty or "asset_symbol" not in snapshots.columns:
        return pd.DataFrame(columns=["asset_symbol", "metric_name", "recent_mean", "baseline_mean", "baseline_std", "z_score"])
    snapshots["generated_at"] = pd.to_datetime(snapshots["generated_at"], errors="coerce", utc=True)
    snapshots = snapshots.dropna(subset=["generated_at"]).sort_values("generated_at").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for asset_symbol, group in snapshots.groupby(snapshots["asset_symbol"].astype(str).str.upper()):
        ordered = group.sort_values("generated_at")
        for metric_name in ["expected_return_score", "confidence", "post_count"]:
            if metric_name not in ordered.columns:
                continue
            values = pd.to_numeric(ordered[metric_name], errors="coerce").dropna()
            if values.empty:
                continue
            recent = values.tail(max(1, int(recent_window)))
            prior = values.iloc[: max(0, len(values) - len(recent))].tail(max(1, int(baseline_window)))
            recent_mean = float(recent.mean())
            baseline_mean = float(prior.mean()) if not prior.empty else np.nan
            baseline_std = float(prior.std(ddof=0)) if len(prior) > 1 else np.nan
            scale = max(
                abs(baseline_std) if pd.notna(baseline_std) else 0.0,
                abs(baseline_mean) * 0.1 if pd.notna(baseline_mean) else 0.0,
                1e-6,
            )
            z_score = abs(recent_mean - baseline_mean) / scale if pd.notna(baseline_mean) else np.nan
            rows.append(
                {
                    "asset_symbol": asset_symbol,
                    "metric_name": metric_name,
                    "recent_mean": recent_mean,
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std,
                    "z_score": z_score,
                },
            )
    return pd.DataFrame(rows)


def build_performance_summary(
    diagnostics: pd.DataFrame,
    registry: pd.DataFrame,
    journal: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    benchmark: pd.DataFrame,
    paper_portfolio_id: str,
) -> dict[str, Any]:
    checks = ensure_performance_diagnostic_frame(diagnostics)
    selected_registry = ensure_paper_portfolio_registry_frame(registry)
    selected_journal = ensure_paper_decision_journal_frame(journal)
    selected_trades = ensure_paper_trade_ledger_frame(trades)
    selected_equity = ensure_paper_equity_curve_frame(equity)
    selected_benchmark = ensure_paper_benchmark_curve_frame(benchmark)

    selected_registry = selected_registry.loc[selected_registry["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
    selected_journal = selected_journal.loc[selected_journal["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
    selected_trades = selected_trades.loc[selected_trades["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
    selected_equity = selected_equity.loc[selected_equity["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
    selected_benchmark = selected_benchmark.loc[selected_benchmark["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
    starting_cash = float(selected_registry.iloc[0]["starting_cash"]) if not selected_registry.empty else 100000.0

    severity_score = int(checks["severity"].map(PERFORMANCE_SEVERITY_ORDER).fillna(0).max()) if not checks.empty else 0
    overall_severity = next(
        severity for severity, score in PERFORMANCE_SEVERITY_ORDER.items() if score == severity_score
    )
    total_return = _series_return(selected_equity, starting_cash)
    benchmark_return = _series_return(selected_benchmark, starting_cash)
    trade_count = int(len(selected_trades))
    win_rate = float((pd.to_numeric(selected_trades["net_return"], errors="coerce") > 0).mean()) if trade_count else 0.0
    pending_decisions = int((selected_journal["settlement_status"].astype(str) == "pending").sum()) if not selected_journal.empty else 0
    fallback_rate = float((selected_journal["decision_source"].astype(str) == "fallback").mean()) if not selected_journal.empty else 0.0
    score_outcomes = _score_outcome_join(selected_journal, selected_trades, paper_portfolio_id)
    score_correlation = (
        float(score_outcomes["winner_score"].corr(score_outcomes["net_return"]))
        if len(score_outcomes) >= 2
        else np.nan
    )
    max_drawdown = _max_drawdown(selected_equity["equity"]) if not selected_equity.empty else 0.0
    return {
        "overall_severity": overall_severity,
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "alpha": total_return - benchmark_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "pending_decisions": pending_decisions,
        "fallback_rate": fallback_rate,
        "score_outcome_correlation": score_correlation,
    }


class PerformanceObservatoryService:
    def __init__(self, store: Any, recent_window: int = 10, baseline_window: int = 30) -> None:
        self.store = store
        self.recent_window = recent_window
        self.baseline_window = baseline_window

    def evaluate_paper_portfolio(
        self,
        paper_portfolio_id: str,
        generated_at: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        generated_ts = _coerce_utc_timestamp(generated_at)
        if pd.isna(generated_ts):
            generated_ts = pd.Timestamp.now(tz="UTC")
        snapshot_id = stable_text_id("performance", paper_portfolio_id, generated_ts.isoformat())

        registry = ensure_paper_portfolio_registry_frame(self.store.read_frame("paper_portfolio_registry"))
        journal = ensure_paper_decision_journal_frame(self.store.read_frame("paper_decision_journal"))
        trades = ensure_paper_trade_ledger_frame(self.store.read_frame("paper_trade_ledger"))
        equity = ensure_paper_equity_curve_frame(self.store.read_frame("paper_equity_curve"))
        benchmark = ensure_paper_benchmark_curve_frame(self.store.read_frame("paper_benchmark_curve"))
        live_assets = self.store.read_frame("live_asset_snapshots")

        registry_row = registry.loc[registry["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
        portfolio_run_id = str(registry_row.iloc[0]["portfolio_run_id"]) if not registry_row.empty else ""
        deployment_variant = str(registry_row.iloc[0]["deployment_variant"]) if not registry_row.empty else ""
        selected_journal = journal.loc[journal["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
        selected_trades = trades.loc[trades["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
        selected_equity = equity.loc[equity["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
        selected_benchmark = benchmark.loc[benchmark["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
        starting_cash = float(registry_row.iloc[0]["starting_cash"]) if not registry_row.empty else 100000.0

        rows: list[dict[str, object]] = []

        def add_row(
            scope_kind: str,
            scope_key: str,
            metric_name: str,
            severity: str,
            observed_value: float | int | None,
            baseline_value: float | int | None = None,
            detail: str = "",
        ) -> None:
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "generated_at": generated_ts,
                    "paper_portfolio_id": str(paper_portfolio_id),
                    "portfolio_run_id": portfolio_run_id,
                    "deployment_variant": deployment_variant,
                    "scope_kind": str(scope_kind),
                    "scope_key": str(scope_key),
                    "metric_name": str(metric_name),
                    "severity": str(severity),
                    "observed_value": float(observed_value) if observed_value is not None and pd.notna(observed_value) else np.nan,
                    "baseline_value": float(baseline_value) if baseline_value is not None and pd.notna(baseline_value) else np.nan,
                    "detail": str(detail or ""),
                },
            )

        if registry_row.empty:
            add_row(
                "portfolio",
                str(paper_portfolio_id),
                "portfolio_registry",
                "warn",
                0,
                1,
                "No paper portfolio registry row exists for this portfolio id.",
            )
            return ensure_performance_diagnostic_frame(pd.DataFrame(rows, columns=PERFORMANCE_DIAGNOSTIC_COLUMNS))

        settled_count = int(len(selected_trades))
        add_row(
            "portfolio",
            str(paper_portfolio_id),
            "settled_trade_count",
            "warn" if settled_count < 5 else "ok",
            settled_count,
            5,
            "At least five settled trades are needed before outcome diagnostics are reliable." if settled_count < 5 else "",
        )

        total_return = _series_return(selected_equity, starting_cash)
        benchmark_return = _series_return(selected_benchmark, starting_cash)
        alpha = total_return - benchmark_return
        alpha_severity = "ok" if settled_count < 5 else _severity_from_thresholds(alpha, warn_below=-0.02, severe_below=-0.05)
        add_row("portfolio", str(paper_portfolio_id), "strategy_total_return", "ok", total_return, 0.0)
        add_row("portfolio", "always_long_spy", "spy_benchmark_return", "ok", benchmark_return, 0.0)
        add_row(
            "portfolio",
            str(paper_portfolio_id),
            "alpha_vs_spy",
            alpha_severity,
            alpha,
            0.0,
            "Alpha thresholds activate after at least five settled trades." if settled_count < 5 else "",
        )

        drawdown = _max_drawdown(selected_equity["equity"]) if not selected_equity.empty else 0.0
        add_row(
            "portfolio",
            str(paper_portfolio_id),
            "max_drawdown",
            _severity_from_thresholds(drawdown, warn_below=-0.05, severe_below=-0.10),
            drawdown,
            -0.05,
        )
        win_rate = float((pd.to_numeric(selected_trades["net_return"], errors="coerce") > 0).mean()) if settled_count else 0.0
        avg_net_return = float(pd.to_numeric(selected_trades["net_return"], errors="coerce").mean()) if settled_count else 0.0
        add_row("portfolio", str(paper_portfolio_id), "win_rate", "ok", win_rate, 0.5)
        add_row("portfolio", str(paper_portfolio_id), "average_net_return", "ok", avg_net_return, 0.0)

        pending_count = int((selected_journal["settlement_status"].astype(str) == "pending").sum()) if not selected_journal.empty else 0
        decision_count = int(len(selected_journal))
        fallback_rate = float((selected_journal["decision_source"].astype(str) == "fallback").mean()) if decision_count else 0.0
        flat_rate = (
            float((selected_journal["stance"].astype(str).str.upper() == "FLAT").mean())
            if decision_count
            else 0.0
        )
        eligible_rate = float((selected_journal["decision_source"].astype(str) == "eligible").mean()) if decision_count else 0.0
        fallback_severity = "ok" if decision_count < 10 else _severity_from_thresholds(fallback_rate, warn_above=0.50, severe_above=0.75)
        add_row("decision_quality", str(paper_portfolio_id), "pending_decision_count", "ok", pending_count, 0)
        add_row(
            "decision_quality",
            str(paper_portfolio_id),
            "fallback_decision_rate",
            fallback_severity,
            fallback_rate,
            0.50,
            "Fallback-rate thresholds activate after at least ten decisions." if decision_count < 10 else "",
        )
        add_row("decision_quality", str(paper_portfolio_id), "flat_decision_rate", "ok", flat_rate, 0.0)
        add_row("decision_quality", str(paper_portfolio_id), "eligible_decision_rate", "ok", eligible_rate, 1.0)

        score_outcomes = _score_outcome_join(journal, trades, paper_portfolio_id)
        if len(score_outcomes) >= 2:
            correlation = float(score_outcomes["winner_score"].corr(score_outcomes["net_return"]))
        else:
            correlation = np.nan
        if len(score_outcomes) < 10:
            corr_severity = "warn"
            corr_detail = "At least ten settled scored trades are needed for score/outcome correlation."
        elif len(score_outcomes) >= 20 and pd.notna(correlation) and correlation < -0.25:
            corr_severity = "severe"
            corr_detail = ""
        elif pd.notna(correlation) and correlation < 0.0:
            corr_severity = "warn"
            corr_detail = ""
        else:
            corr_severity = "ok"
            corr_detail = ""
        add_row(
            "decision_quality",
            str(paper_portfolio_id),
            "score_outcome_correlation",
            corr_severity,
            correlation,
            0.0,
            corr_detail,
        )

        drift = build_live_score_drift_frame(
            live_asset_snapshots=live_assets,
            portfolio_run_id=portfolio_run_id,
            deployment_variant=deployment_variant,
            recent_window=self.recent_window,
            baseline_window=self.baseline_window,
        )
        if drift.empty:
            add_row(
                "drift",
                "live_asset_snapshots",
                "live_score_history",
                "warn",
                0,
                self.recent_window + 1,
                "No live snapshot history is available for drift analysis yet.",
            )
        else:
            for _, row in drift.iterrows():
                z_score = row.get("z_score")
                if pd.isna(z_score):
                    severity = "warn"
                    detail = "No prior baseline window is available for this asset metric yet."
                elif float(z_score) >= 3.0:
                    severity = "severe"
                    detail = "Recent value moved more than roughly 3x baseline dispersion."
                elif float(z_score) >= 2.0:
                    severity = "warn"
                    detail = "Recent value moved more than roughly 2x baseline dispersion."
                else:
                    severity = "ok"
                    detail = ""
                add_row(
                    "drift",
                    str(row["asset_symbol"]),
                    f"{row['metric_name']}_drift_z",
                    severity,
                    z_score,
                    2.0,
                    detail,
                )

        winner_distribution = build_winner_distribution_frame(journal, paper_portfolio_id)
        if not winner_distribution.empty:
            top_share = float(winner_distribution["decision_count"].iloc[0] / winner_distribution["decision_count"].sum())
            add_row(
                "drift",
                str(winner_distribution["winning_asset"].iloc[0]),
                "winner_concentration_rate",
                _severity_from_thresholds(top_share, warn_above=0.75, severe_above=0.90) if decision_count >= 10 else "ok",
                top_share,
                0.75,
                "Winner concentration thresholds activate after at least ten decisions." if decision_count < 10 else "",
            )

        diagnostics = ensure_performance_diagnostic_frame(pd.DataFrame(rows, columns=PERFORMANCE_DIAGNOSTIC_COLUMNS))
        if diagnostics.empty:
            return diagnostics
        diagnostics["severity_rank"] = diagnostics["severity"].map(PERFORMANCE_SEVERITY_ORDER).fillna(0)
        diagnostics = diagnostics.sort_values(
            ["severity_rank", "scope_kind", "scope_key", "metric_name"],
            ascending=[False, True, True, True],
        ).drop(columns=["severity_rank"])
        return diagnostics.reset_index(drop=True)

    def persist_snapshot(
        self,
        paper_portfolio_id: str,
        generated_at: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        latest = self.evaluate_paper_portfolio(paper_portfolio_id, generated_at=generated_at)
        snapshot_id = str(latest.iloc[0]["snapshot_id"]) if not latest.empty else ""
        self.store.save_frame(
            "model_performance_latest",
            latest,
            metadata={"row_count": int(len(latest)), "snapshot_id": snapshot_id},
        )
        self.store.append_frame(
            "model_performance_history",
            latest,
            dedupe_on=["snapshot_id", "paper_portfolio_id", "scope_kind", "scope_key", "metric_name"],
            metadata={"snapshot_id": snapshot_id},
        )
        return latest

    def load_latest_for_portfolio(self, paper_portfolio_id: str) -> pd.DataFrame:
        latest = ensure_performance_diagnostic_frame(self.store.read_frame("model_performance_latest"))
        filtered = latest.loc[latest["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
        if not filtered.empty:
            return filtered.reset_index(drop=True)
        history = ensure_performance_diagnostic_frame(self.store.read_frame("model_performance_history"))
        filtered = history.loc[history["paper_portfolio_id"].astype(str) == str(paper_portfolio_id)].copy()
        if filtered.empty:
            return _empty_performance_diagnostic_frame()
        latest_snapshot_time = filtered["generated_at"].max()
        latest_snapshot = filtered.loc[filtered["generated_at"] == latest_snapshot_time].copy()
        return latest_snapshot.reset_index(drop=True)

    def build_summary(self, paper_portfolio_id: str, diagnostics: pd.DataFrame | None = None) -> dict[str, Any]:
        diagnostics_frame = (
            ensure_performance_diagnostic_frame(diagnostics)
            if diagnostics is not None
            else self.load_latest_for_portfolio(paper_portfolio_id)
        )
        return build_performance_summary(
            diagnostics=diagnostics_frame,
            registry=self.store.read_frame("paper_portfolio_registry"),
            journal=self.store.read_frame("paper_decision_journal"),
            trades=self.store.read_frame("paper_trade_ledger"),
            equity=self.store.read_frame("paper_equity_curve"),
            benchmark=self.store.read_frame("paper_benchmark_curve"),
            paper_portfolio_id=paper_portfolio_id,
        )

