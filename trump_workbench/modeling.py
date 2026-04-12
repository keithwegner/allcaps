from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import LinearModelArtifact, ModelRunConfig

META_COLUMNS = {
    "signal_session_date",
    "next_session_date",
    "next_session_open",
    "next_session_close",
    "feature_version",
    "llm_enabled",
    "target_next_session_return",
    "target_available",
    "tradeable",
}


@dataclass
class LinearReturnModel:
    artifact: LinearModelArtifact

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        ridge_alpha: float,
        model_version: str,
        metadata: dict[str, Any],
    ) -> "LinearReturnModel":
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_clean = y.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        means = X_clean.mean(axis=0)
        stds = X_clean.std(axis=0, ddof=0).replace(0.0, 1.0)
        X_scaled = ((X_clean - means) / stds).clip(-8.0, 8.0)
        X_np = np.nan_to_num(X_scaled.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        y_np = np.nan_to_num(y_clean.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X_design = np.column_stack([np.ones(len(X_np)), X_np])
        identity = np.eye(X_design.shape[1])
        identity[0, 0] = 0.0
        with np.errstate(all="ignore"):
            gram = X_design.T @ X_design
            target = X_design.T @ y_np
            beta = np.linalg.pinv(gram + ridge_alpha * identity) @ target
            predictions = X_design @ beta
        beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
        residual_std = float(np.std(y_np - predictions, ddof=0))
        artifact = LinearModelArtifact(
            model_version=model_version,
            feature_names=list(X.columns),
            intercept=float(beta[0]),
            coefficients=[float(value) for value in beta[1:]],
            means=[float(value) for value in means.tolist()],
            stds=[float(value) for value in stds.tolist()],
            residual_std=residual_std,
            train_rows=int(len(X)),
            metadata=metadata,
        )
        return cls(artifact=artifact)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        means = np.asarray(self.artifact.means, dtype=float)
        stds = np.asarray(self.artifact.stds, dtype=float)
        coefs = np.asarray(self.artifact.coefficients, dtype=float)
        X_clean = X[self.artifact.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_np = np.nan_to_num(X_clean.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = np.clip((X_np - means) / stds, -8.0, 8.0)
        with np.errstate(all="ignore"):
            predictions = self.artifact.intercept + X_scaled @ coefs
        return np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)


class ModelService:
    def select_feature_columns(self, df: pd.DataFrame, llm_enabled: bool) -> list[str]:
        numeric_columns = [
            column
            for column in df.columns
            if column not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
        ]
        if not llm_enabled:
            numeric_columns = [column for column in numeric_columns if not column.startswith("semantic_") and not column.startswith("policy_")]
        return sorted(numeric_columns)

    def train(
        self,
        run_config: ModelRunConfig,
        feature_rows: pd.DataFrame,
        model_version: str,
    ) -> tuple[LinearModelArtifact, pd.DataFrame]:
        train_df = feature_rows.dropna(subset=["target_next_session_return"]).copy()
        if train_df.empty:
            raise RuntimeError("No trainable rows were available for the model.")
        feature_columns = self.select_feature_columns(train_df, llm_enabled=run_config.llm_enabled)
        if not feature_columns:
            raise RuntimeError("No numeric feature columns were available for the model.")
        model = LinearReturnModel.fit(
            X=train_df[feature_columns].fillna(0.0),
            y=train_df["target_next_session_return"].fillna(0.0),
            ridge_alpha=run_config.ridge_alpha,
            model_version=model_version,
            metadata={"llm_enabled": run_config.llm_enabled},
        )
        importance = pd.DataFrame(
            {
                "feature_name": feature_columns,
                "coefficient": model.artifact.coefficients,
            },
        )
        importance["abs_coefficient"] = importance["coefficient"].abs()
        importance = importance.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
        return model.artifact, importance

    def predict(
        self,
        artifact: LinearModelArtifact,
        feature_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        model = LinearReturnModel(artifact=artifact)
        output = feature_rows.copy()
        if output.empty:
            output["expected_return_score"] = pd.Series(dtype=float)
            output["prediction_confidence"] = pd.Series(dtype=float)
            return output
        for column in artifact.feature_names:
            if column not in output.columns:
                output[column] = 0.0
        predictions = model.predict(output[artifact.feature_names].fillna(0.0))
        output["expected_return_score"] = predictions
        output["prediction_confidence"] = 1.0 / (1.0 + artifact.residual_std * 100.0 + np.abs(predictions) * 25.0)
        output["model_version"] = artifact.model_version
        return output
