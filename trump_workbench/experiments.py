from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import BacktestRun, LinearModelArtifact, LiveMonitorConfig, SavedRunArtifacts
from .live_monitor import LIVE_MONITOR_CONFIG_PATH
from .storage import DuckDBStore


class ExperimentStore:
    def __init__(self, store: DuckDBStore) -> None:
        self.store = store

    def save_run(
        self,
        run: BacktestRun,
        config: dict[str, Any],
        trades: pd.DataFrame,
        predictions: pd.DataFrame,
        windows: pd.DataFrame,
        importance: pd.DataFrame,
        model_artifact: dict[str, Any],
        feature_contributions: pd.DataFrame,
        post_attribution: pd.DataFrame,
        account_attribution: pd.DataFrame,
        benchmarks: pd.DataFrame,
        diagnostics: pd.DataFrame,
        benchmark_curves: pd.DataFrame,
        leakage_audit: dict[str, Any],
    ) -> SavedRunArtifacts:
        run_dir = self.store.artifact_path("runs", run.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_path = run_dir / "summary.json"
        trades_path = run_dir / "trades.parquet"
        predictions_path = run_dir / "predictions.parquet"
        windows_path = run_dir / "windows.parquet"
        importance_path = run_dir / "importance.parquet"
        model_path = run_dir / "model.json"
        feature_contributions_path = run_dir / "feature_contributions.parquet"
        post_attribution_path = run_dir / "post_attribution.parquet"
        account_attribution_path = run_dir / "account_attribution.parquet"
        benchmarks_path = run_dir / "benchmarks.parquet"
        diagnostics_path = run_dir / "diagnostics.parquet"
        benchmark_curves_path = run_dir / "benchmark_curves.parquet"
        leakage_audit_path = run_dir / "leakage_audit.json"

        summary_payload = {
            "run": run.to_dict(),
            "config": config,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        trades.to_parquet(trades_path, index=False)
        predictions.to_parquet(predictions_path, index=False)
        windows.to_parquet(windows_path, index=False)
        importance.to_parquet(importance_path, index=False)
        model_path.write_text(json.dumps(model_artifact, indent=2, default=str), encoding="utf-8")
        feature_contributions.to_parquet(feature_contributions_path, index=False)
        post_attribution.to_parquet(post_attribution_path, index=False)
        account_attribution.to_parquet(account_attribution_path, index=False)
        benchmarks.to_parquet(benchmarks_path, index=False)
        diagnostics.to_parquet(diagnostics_path, index=False)
        benchmark_curves.to_parquet(benchmark_curves_path, index=False)
        leakage_audit_path.write_text(json.dumps(leakage_audit, indent=2, default=str), encoding="utf-8")

        self.store.save_run_record(
            run_id=run.run_id,
            run_name=run.run_name,
            target_asset=run.target_asset,
            run_type=run.run_type,
            allocator_mode=run.allocator_mode,
            config_hash=run.config_hash,
            metrics=run.metrics,
            selected_params=run.selected_params,
            artifact_paths={
                "summary_path": str(summary_path),
                "trades_path": str(trades_path),
                "predictions_path": str(predictions_path),
                "windows_path": str(windows_path),
                "importance_path": str(importance_path),
                "model_path": str(model_path),
            },
        )
        return SavedRunArtifacts(
            summary_path=summary_path,
            trades_path=trades_path,
            predictions_path=predictions_path,
            windows_path=windows_path,
            importance_path=importance_path,
            model_path=model_path,
            feature_contributions_path=feature_contributions_path,
            post_attribution_path=post_attribution_path,
            account_attribution_path=account_attribution_path,
            benchmarks_path=benchmarks_path,
            diagnostics_path=diagnostics_path,
            benchmark_curves_path=benchmark_curves_path,
            leakage_audit_path=leakage_audit_path,
        )

    def save_portfolio_run(
        self,
        run: BacktestRun,
        config: dict[str, Any],
        trades: pd.DataFrame,
        decision_history: pd.DataFrame,
        candidate_predictions: pd.DataFrame,
        component_summary: pd.DataFrame,
        benchmarks: pd.DataFrame,
        benchmark_curves: pd.DataFrame,
        diagnostics: pd.DataFrame,
        leakage_audit: dict[str, Any],
    ) -> SavedRunArtifacts:
        run_dir = self.store.artifact_path("runs", run.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_path = run_dir / "summary.json"
        trades_path = run_dir / "portfolio_trades.parquet"
        predictions_path = run_dir / "decision_history.parquet"
        windows_path = run_dir / "components.parquet"
        importance_path = run_dir / "importance.parquet"
        model_path = run_dir / "model.json"
        candidate_predictions_path = run_dir / "candidate_predictions.parquet"
        benchmarks_path = run_dir / "portfolio_benchmarks.parquet"
        diagnostics_path = run_dir / "decision_diagnostics.parquet"
        benchmark_curves_path = run_dir / "portfolio_benchmark_curves.parquet"
        leakage_audit_path = run_dir / "leakage_audit.json"
        feature_contributions_path = run_dir / "feature_contributions.parquet"
        post_attribution_path = run_dir / "post_attribution.parquet"
        account_attribution_path = run_dir / "account_attribution.parquet"

        summary_payload = {
            "run": run.to_dict(),
            "config": config,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
        trades.to_parquet(trades_path, index=False)
        decision_history.to_parquet(predictions_path, index=False)
        component_summary.to_parquet(windows_path, index=False)
        pd.DataFrame().to_parquet(importance_path, index=False)
        candidate_predictions.to_parquet(candidate_predictions_path, index=False)
        benchmarks.to_parquet(benchmarks_path, index=False)
        diagnostics.to_parquet(diagnostics_path, index=False)
        benchmark_curves.to_parquet(benchmark_curves_path, index=False)
        pd.DataFrame().to_parquet(feature_contributions_path, index=False)
        pd.DataFrame().to_parquet(post_attribution_path, index=False)
        pd.DataFrame().to_parquet(account_attribution_path, index=False)
        model_path.write_text(
            json.dumps(
                {
                    "model_version": f"{run.run_name}-portfolio",
                    "feature_names": [],
                    "intercept": 0.0,
                    "coefficients": [],
                    "means": [],
                    "stds": [],
                    "residual_std": 0.0,
                    "train_rows": int(len(candidate_predictions)),
                    "metadata": {
                        "run_type": run.run_type,
                        "allocator_mode": run.allocator_mode,
                        "target_asset": run.target_asset,
                    },
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        leakage_audit_path.write_text(json.dumps(leakage_audit, indent=2, default=str), encoding="utf-8")

        self.store.save_run_record(
            run_id=run.run_id,
            run_name=run.run_name,
            target_asset=run.target_asset,
            run_type=run.run_type,
            allocator_mode=run.allocator_mode,
            config_hash=run.config_hash,
            metrics=run.metrics,
            selected_params=run.selected_params,
            artifact_paths={
                "summary_path": str(summary_path),
                "trades_path": str(trades_path),
                "predictions_path": str(predictions_path),
                "windows_path": str(windows_path),
                "importance_path": str(importance_path),
                "model_path": str(model_path),
            },
        )
        return SavedRunArtifacts(
            summary_path=summary_path,
            trades_path=trades_path,
            predictions_path=predictions_path,
            windows_path=windows_path,
            importance_path=importance_path,
            model_path=model_path,
            feature_contributions_path=feature_contributions_path,
            post_attribution_path=post_attribution_path,
            account_attribution_path=account_attribution_path,
            benchmarks_path=benchmarks_path,
            diagnostics_path=diagnostics_path,
            benchmark_curves_path=benchmark_curves_path,
            leakage_audit_path=leakage_audit_path,
            candidate_predictions_path=candidate_predictions_path,
        )

    def list_runs(self) -> pd.DataFrame:
        return self.store.list_run_records()

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self.list_runs()
        if rows.empty:
            return None
        row = rows.loc[rows["run_id"] == run_id]
        if row.empty:
            return None
        record = row.iloc[0]
        summary_payload = self._read_json(Path(record["summary_path"]))
        return {
            "summary": Path(record["summary_path"]),
            "run": summary_payload.get("run", {}),
            "config": summary_payload.get("config", {}),
            "trades": pd.read_parquet(record["trades_path"]),
            "predictions": pd.read_parquet(record["predictions_path"]),
            "windows": pd.read_parquet(record["windows_path"]),
            "importance": pd.read_parquet(record["importance_path"]),
            "model_artifact": LinearModelArtifact.from_dict(self._read_json(Path(record["model_path"]))),
            "feature_contributions": self._read_optional_parquet(Path(record["summary_path"]).parent / "feature_contributions.parquet"),
            "post_attribution": self._read_optional_parquet(Path(record["summary_path"]).parent / "post_attribution.parquet"),
            "account_attribution": self._read_optional_parquet(Path(record["summary_path"]).parent / "account_attribution.parquet"),
            "metrics": record["metrics_json"],
            "selected_params": record["selected_params_json"],
            "candidate_predictions": self._read_optional_parquet(Path(record["summary_path"]).parent / "candidate_predictions.parquet"),
            "benchmarks": self._read_optional_parquet_paths(
                [
                    Path(record["summary_path"]).parent / "benchmarks.parquet",
                    Path(record["summary_path"]).parent / "portfolio_benchmarks.parquet",
                ],
            ),
            "diagnostics": self._read_optional_parquet_paths(
                [
                    Path(record["summary_path"]).parent / "diagnostics.parquet",
                    Path(record["summary_path"]).parent / "decision_diagnostics.parquet",
                ],
            ),
            "benchmark_curves": self._read_optional_parquet_paths(
                [
                    Path(record["summary_path"]).parent / "benchmark_curves.parquet",
                    Path(record["summary_path"]).parent / "portfolio_benchmark_curves.parquet",
                ],
            ),
            "leakage_audit": self._read_optional_json(Path(record["summary_path"]).parent / "leakage_audit.json"),
        }

    def load_latest_model_artifact(self, target_asset: str | None = "SPY") -> tuple[LinearModelArtifact, dict[str, Any]] | None:
        runs = self.list_runs()
        if runs.empty:
            return None
        normalized_target = str(target_asset).upper() if target_asset else None
        for _, row in runs.iterrows():
            summary_payload = self._read_json(Path(row["summary_path"]))
            config = summary_payload.get("config", {}) or {}
            run_meta = summary_payload.get("run", {}) or {}
            run_type = str(run_meta.get("run_type") or row.get("run_type") or "asset_model")
            if run_type != "asset_model":
                continue
            row_target = str(
                config.get("target_asset")
                or run_meta.get("target_asset")
                or row.get("target_asset")
                or "SPY",
            ).upper()
            if normalized_target is not None and row_target != normalized_target:
                continue
            artifact = LinearModelArtifact.from_dict(self._read_json(Path(row["model_path"])))
            return artifact, row["selected_params_json"]
        return None

    def save_live_monitor_config(self, config: LiveMonitorConfig) -> Path:
        return self.store.save_json_artifact(LIVE_MONITOR_CONFIG_PATH, config.to_dict())

    def load_live_monitor_config(self) -> LiveMonitorConfig | None:
        payload = self.store.read_json_artifact(LIVE_MONITOR_CONFIG_PATH)
        if payload is None:
            return None
        return LiveMonitorConfig.from_dict(payload)

    def save_prediction_snapshots(self, snapshots: pd.DataFrame) -> None:
        if snapshots.empty:
            return
        normalized = snapshots.copy()
        if "target_asset" not in normalized.columns:
            normalized["target_asset"] = "SPY"
        self.store.append_frame(
            "prediction_snapshots",
            normalized,
            dedupe_on=["signal_session_date", "generated_at", "target_asset"],
            metadata={"dataset": "prediction_snapshots"},
        )

    def save_live_asset_snapshots(self, snapshots: pd.DataFrame) -> None:
        if snapshots.empty:
            return
        self.store.append_frame(
            "live_asset_snapshots",
            snapshots.copy(),
            dedupe_on=["generated_at", "asset_symbol", "run_id"],
            metadata={"dataset": "live_asset_snapshots"},
        )

    def save_live_decision_snapshots(self, snapshots: pd.DataFrame) -> None:
        if snapshots.empty:
            return
        self.store.append_frame(
            "live_decision_snapshots",
            snapshots.copy(),
            dedupe_on=["generated_at", "winning_asset", "winning_run_id"],
            metadata={"dataset": "live_decision_snapshots"},
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_optional_parquet(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    @staticmethod
    def _read_optional_parquet_paths(paths: list[Path]) -> pd.DataFrame:
        for path in paths:
            if path.exists():
                return pd.read_parquet(path)
        return pd.DataFrame()

    @staticmethod
    def _read_optional_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
