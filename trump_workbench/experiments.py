from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import BacktestRun, LinearModelArtifact, SavedRunArtifacts
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
    ) -> SavedRunArtifacts:
        run_dir = self.store.artifact_path("runs", run.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_path = run_dir / "summary.json"
        trades_path = run_dir / "trades.parquet"
        predictions_path = run_dir / "predictions.parquet"
        windows_path = run_dir / "windows.parquet"
        importance_path = run_dir / "importance.parquet"
        model_path = run_dir / "model.json"

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

        self.store.save_run_record(
            run_id=run.run_id,
            run_name=run.run_name,
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
        return {
            "summary": Path(record["summary_path"]),
            "trades": pd.read_parquet(record["trades_path"]),
            "predictions": pd.read_parquet(record["predictions_path"]),
            "windows": pd.read_parquet(record["windows_path"]),
            "importance": pd.read_parquet(record["importance_path"]),
            "model_artifact": LinearModelArtifact.from_dict(self._read_json(Path(record["model_path"]))),
            "metrics": record["metrics_json"],
            "selected_params": record["selected_params_json"],
        }

    def load_latest_model_artifact(self) -> tuple[LinearModelArtifact, dict[str, Any]] | None:
        runs = self.list_runs()
        if runs.empty:
            return None
        row = runs.iloc[0]
        artifact = LinearModelArtifact.from_dict(self._read_json(Path(row["model_path"])))
        return artifact, row["selected_params_json"]

    def save_prediction_snapshots(self, snapshots: pd.DataFrame) -> None:
        if snapshots.empty:
            return
        self.store.append_frame(
            "prediction_snapshots",
            snapshots,
            dedupe_on=["signal_session_date", "generated_at"],
            metadata={"dataset": "prediction_snapshots"},
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))
