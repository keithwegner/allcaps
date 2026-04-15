from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppSettings


class DuckDBStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - exercised in runtime setup
            raise RuntimeError(
                "duckdb is required for the workbench. Install it with `pip install duckdb`.",
            ) from exc
        self._duckdb = duckdb
        self._init_db()

    def _connect(self):
        return self._duckdb.connect(str(self.settings.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists dataset_registry (
                    dataset_name varchar primary key,
                    parquet_path varchar,
                    row_count bigint,
                    updated_at timestamp,
                    metadata_json varchar
                )
                """,
            )
            conn.execute(
                """
                create table if not exists experiment_runs (
                    run_id varchar primary key,
                    run_name varchar,
                    target_asset varchar,
                    run_type varchar,
                    allocator_mode varchar,
                    created_at timestamp,
                    config_hash varchar,
                    metrics_json varchar,
                    selected_params_json varchar,
                    summary_path varchar,
                    trades_path varchar,
                    predictions_path varchar,
                    windows_path varchar,
                    importance_path varchar,
                    model_path varchar
                )
                """,
            )
            conn.execute("alter table experiment_runs add column if not exists target_asset varchar")
            conn.execute("alter table experiment_runs add column if not exists run_type varchar")
            conn.execute("alter table experiment_runs add column if not exists allocator_mode varchar")

    def dataset_path(self, dataset_name: str) -> Path:
        return self.settings.lake_dir / f"{dataset_name}.parquet"

    def artifact_path(self, *parts: str) -> Path:
        return self.settings.artifact_dir.joinpath(*parts)

    def save_frame(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = self.dataset_path(dataset_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        with self._connect() as conn:
            conn.execute("delete from dataset_registry where dataset_name = ?", [dataset_name])
            conn.execute(
                """
                insert into dataset_registry
                (dataset_name, parquet_path, row_count, updated_at, metadata_json)
                values (?, ?, ?, current_timestamp, ?)
                """,
                [
                    dataset_name,
                    str(path),
                    int(len(df)),
                    json.dumps(metadata or {}),
                ],
            )
        return path

    def append_frame(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        dedupe_on: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        existing = self.read_frame(dataset_name)
        frames = [existing, df] if not existing.empty else [df]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if dedupe_on:
            combined = combined.drop_duplicates(subset=dedupe_on, keep="last")
        return self.save_frame(dataset_name, combined.reset_index(drop=True), metadata=metadata)

    def read_frame(self, dataset_name: str) -> pd.DataFrame:
        path = self.dataset_path(dataset_name)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def dataset_registry(self) -> pd.DataFrame:
        with self._connect() as conn:
            return conn.execute(
                """
                select dataset_name, parquet_path, row_count, updated_at, metadata_json
                from dataset_registry
                order by dataset_name
                """,
            ).fetchdf()

    def save_json_artifact(self, relative_path: str, payload: dict[str, Any]) -> Path:
        path = self.artifact_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def read_json_artifact(self, relative_path: str) -> dict[str, Any] | None:
        path = self.artifact_path(relative_path)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_run_record(
        self,
        run_id: str,
        run_name: str,
        target_asset: str,
        run_type: str,
        allocator_mode: str,
        config_hash: str,
        metrics: dict[str, Any],
        selected_params: dict[str, Any],
        artifact_paths: dict[str, str],
    ) -> None:
        with self._connect() as conn:
            conn.execute("delete from experiment_runs where run_id = ?", [run_id])
            conn.execute(
                """
                insert into experiment_runs
                (run_id, run_name, target_asset, run_type, allocator_mode, created_at, config_hash, metrics_json, selected_params_json,
                 summary_path, trades_path, predictions_path, windows_path, importance_path, model_path)
                values (?, ?, ?, ?, ?, current_timestamp, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    run_name,
                    target_asset,
                    run_type,
                    allocator_mode,
                    config_hash,
                    json.dumps(metrics, default=str),
                    json.dumps(selected_params, default=str),
                    artifact_paths.get("summary_path", ""),
                    artifact_paths.get("trades_path", ""),
                    artifact_paths.get("predictions_path", ""),
                    artifact_paths.get("windows_path", ""),
                    artifact_paths.get("importance_path", ""),
                    artifact_paths.get("model_path", ""),
                ],
            )

    def list_run_records(self) -> pd.DataFrame:
        with self._connect() as conn:
            df = conn.execute(
                """
                select *
                from experiment_runs
                order by created_at desc
                """,
            ).fetchdf()
        if df.empty:
            return df
        if "run_type" not in df.columns:
            df["run_type"] = "asset_model"
        else:
            df["run_type"] = df["run_type"].fillna("asset_model").replace("", "asset_model")
        if "allocator_mode" not in df.columns:
            df["allocator_mode"] = ""
        else:
            df["allocator_mode"] = df["allocator_mode"].fillna("")
        df["metrics_json"] = df["metrics_json"].map(json.loads)
        df["selected_params_json"] = df["selected_params_json"].map(json.loads)
        return df
