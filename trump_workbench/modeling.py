from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import LinearModelArtifact, ModelRunConfig

SUPPORTED_PORTFOLIO_MODEL_FAMILIES = (
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest_regressor",
    "hist_gradient_boosting_regressor",
)
LINEAR_MODEL_FAMILIES = {"custom_linear", "ridge", "lasso", "elastic_net"}
ASSET_INDICATOR_PREFIX = "asset_indicator__"

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

EXPLANATION_COLUMNS = [
    "signal_session_date",
    "next_session_date",
    "model_version",
    "expected_return_score",
    "prediction_confidence",
    "feature_name",
    "feature_family",
    "raw_value",
    "standardized_value",
    "coefficient",
    "contribution",
    "abs_contribution",
    "contribution_share",
]


def _import_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import ElasticNet, Lasso, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment/setup dependent
        raise RuntimeError(
            "scikit-learn is required for joint portfolio model search. Install it with `pip install scikit-learn`.",
        ) from exc
    return {
        "ElasticNet": ElasticNet,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "Lasso": Lasso,
        "Pipeline": Pipeline,
        "RandomForestRegressor": RandomForestRegressor,
        "Ridge": Ridge,
        "StandardScaler": StandardScaler,
        "permutation_importance": permutation_importance,
    }


def classify_feature_family(feature_name: str) -> str:
    if feature_name.startswith(ASSET_INDICATOR_PREFIX):
        return "asset_identity"
    if feature_name.startswith("semantic_"):
        return "semantic"
    if feature_name.startswith("policy_"):
        return "policy"
    if feature_name.startswith("prev_") or feature_name in {"session_return", "rolling_vol_5d", "close_vs_ma_5", "volume_z_5"}:
        return "market_context"
    if "sentiment" in feature_name:
        return "social_sentiment"
    if "engagement" in feature_name:
        return "social_engagement"
    if "tracked" in feature_name or "author" in feature_name:
        return "account_structure"
    if "post" in feature_name or "mention" in feature_name:
        return "activity"
    return "other"


def _feature_stats(X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return means, stds


def _serialize_estimator(estimator: Any) -> str:
    return base64.b64encode(pickle.dumps(estimator)).decode("ascii")


def _deserialize_estimator(payload: str) -> Any:
    return pickle.loads(base64.b64decode(payload.encode("ascii")))


def _build_importance_frame(
    feature_names: list[str],
    values: list[float],
    *,
    column_name: str = "coefficient",
    model_family: str,
) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature_name": feature_names,
            column_name: values,
        },
    )
    importance["abs_coefficient"] = pd.to_numeric(importance[column_name], errors="coerce").abs()
    importance["model_family"] = model_family
    return importance.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def add_asset_indicator_columns(frame: pd.DataFrame, symbols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    output = frame.copy()
    normalized = [str(symbol).upper() for symbol in symbols]
    asset_series = output.get("asset_symbol", pd.Series("", index=output.index)).astype(str).str.upper()
    for symbol in normalized:
        output[f"{ASSET_INDICATOR_PREFIX}{symbol.lower()}"] = (asset_series == symbol).astype(float)
    return output


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
        means, stds = _feature_stats(X_clean)
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
            model_family="custom_linear",
            feature_names=list(X.columns),
            intercept=float(beta[0]),
            coefficients=[float(value) for value in beta[1:]],
            means=[float(value) for value in means.tolist()],
            stds=[float(value) for value in stds.tolist()],
            residual_std=residual_std,
            train_rows=int(len(X)),
            metadata=metadata,
            feature_importances=[float(abs(value)) for value in beta[1:].tolist()],
            explanation_kind="linear_exact",
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
    def select_feature_columns(
        self,
        df: pd.DataFrame,
        llm_enabled: bool,
        extra_feature_columns: list[str] | None = None,
    ) -> list[str]:
        numeric_columns = [
            column
            for column in df.columns
            if column not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
        ]
        if not llm_enabled:
            numeric_columns = [
                column
                for column in numeric_columns
                if not column.startswith("semantic_") and not column.startswith("policy_")
            ]
        if extra_feature_columns:
            for column in extra_feature_columns:
                if column in df.columns and column not in numeric_columns:
                    numeric_columns.append(column)
        return sorted(set(numeric_columns))

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
            metadata={"llm_enabled": run_config.llm_enabled, "target_asset": run_config.target_asset},
        )
        importance = _build_importance_frame(
            feature_columns,
            model.artifact.coefficients,
            model_family=model.artifact.model_family,
        )
        return model.artifact, importance

    def train_with_family(
        self,
        feature_rows: pd.DataFrame,
        *,
        llm_enabled: bool,
        model_family: str,
        model_version: str,
        metadata: dict[str, Any],
        feature_columns: list[str] | None = None,
        regularization: float = 1.0,
        compute_importance: bool = True,
    ) -> tuple[LinearModelArtifact, pd.DataFrame]:
        train_df = feature_rows.dropna(subset=["target_next_session_return"]).copy()
        if train_df.empty:
            raise RuntimeError("No trainable rows were available for the model.")
        selected_columns = feature_columns or self.select_feature_columns(train_df, llm_enabled=llm_enabled)
        if not selected_columns:
            raise RuntimeError("No numeric feature columns were available for the model.")

        X_clean = train_df[selected_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_clean = train_df["target_next_session_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        means, stds = _feature_stats(X_clean)
        y_np = np.nan_to_num(y_clean.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        if model_family == "ridge":
            sklearn = _import_sklearn()
            pipeline = sklearn["Pipeline"](
                [
                    ("scaler", sklearn["StandardScaler"]()),
                    ("model", sklearn["Ridge"](alpha=max(float(regularization), 1e-6))),
                ],
            )
            pipeline.fit(X_clean, y_clean)
            scaler = pipeline.named_steps["scaler"]
            estimator = pipeline.named_steps["model"]
            predictions = pipeline.predict(X_clean)
            artifact = LinearModelArtifact(
                model_version=model_version,
                model_family=model_family,
                feature_names=selected_columns,
                intercept=float(estimator.intercept_),
                coefficients=[float(value) for value in estimator.coef_.tolist()],
                means=[float(value) for value in scaler.mean_.tolist()],
                stds=[float(value) for value in np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_).tolist()],
                residual_std=float(np.std(y_np - predictions, ddof=0)),
                train_rows=int(len(train_df)),
                metadata=metadata,
                feature_importances=[float(abs(value)) for value in estimator.coef_.tolist()],
                explanation_kind="linear_exact",
            )
            return artifact, _build_importance_frame(
                selected_columns,
                artifact.coefficients,
                model_family=model_family,
            )

        if model_family in {"lasso", "elastic_net"}:
            sklearn = _import_sklearn()
            estimator_cls = sklearn["Lasso"] if model_family == "lasso" else sklearn["ElasticNet"]
            estimator_kwargs: dict[str, Any] = {
                "alpha": max(float(regularization), 1e-4),
                "max_iter": 5000,
                "random_state": 42,
            }
            if model_family == "elastic_net":
                estimator_kwargs["l1_ratio"] = 0.5
            pipeline = sklearn["Pipeline"](
                [
                    ("scaler", sklearn["StandardScaler"]()),
                    ("model", estimator_cls(**estimator_kwargs)),
                ],
            )
            pipeline.fit(X_clean, y_clean)
            scaler = pipeline.named_steps["scaler"]
            estimator = pipeline.named_steps["model"]
            predictions = pipeline.predict(X_clean)
            artifact = LinearModelArtifact(
                model_version=model_version,
                model_family=model_family,
                feature_names=selected_columns,
                intercept=float(estimator.intercept_),
                coefficients=[float(value) for value in estimator.coef_.tolist()],
                means=[float(value) for value in scaler.mean_.tolist()],
                stds=[float(value) for value in np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_).tolist()],
                residual_std=float(np.std(y_np - predictions, ddof=0)),
                train_rows=int(len(train_df)),
                metadata=metadata,
                feature_importances=[float(abs(value)) for value in estimator.coef_.tolist()],
                explanation_kind="linear_exact",
            )
            return artifact, _build_importance_frame(
                selected_columns,
                artifact.coefficients,
                model_family=model_family,
            )

        sklearn = _import_sklearn()
        X_np = np.nan_to_num(X_clean.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if model_family == "random_forest_regressor":
            estimator = sklearn["RandomForestRegressor"](
                n_estimators=120,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            )
        elif model_family == "hist_gradient_boosting_regressor":
            estimator = sklearn["HistGradientBoostingRegressor"](
                learning_rate=0.05,
                max_depth=4,
                max_iter=160,
                random_state=42,
            )
        else:
            raise RuntimeError(f"Unsupported model family `{model_family}`.")

        estimator.fit(X_np, y_np)
        predictions = estimator.predict(X_np)
        importances: list[float]
        if hasattr(estimator, "feature_importances_"):
            importances = [float(value) for value in getattr(estimator, "feature_importances_").tolist()]
        elif compute_importance:
            sample_size = min(len(train_df), 128)
            sampled_X = X_np[:sample_size]
            sampled_y = y_np[:sample_size]
            permutation = sklearn["permutation_importance"](
                estimator,
                sampled_X,
                sampled_y,
                n_repeats=3,
                random_state=42,
            )
            importances = [float(max(value, 0.0)) for value in permutation.importances_mean.tolist()]
        else:
            importances = [0.0 for _ in selected_columns]
        artifact = LinearModelArtifact(
            model_version=model_version,
            model_family=model_family,
            feature_names=selected_columns,
            means=[float(value) for value in means.tolist()],
            stds=[float(value) for value in stds.tolist()],
            residual_std=float(np.std(y_np - predictions, ddof=0)),
            train_rows=int(len(train_df)),
            metadata=metadata,
            feature_importances=importances,
            serialized_estimator_b64=_serialize_estimator(estimator),
            explanation_kind="importance_proxy",
        )
        return artifact, _build_importance_frame(
            selected_columns,
            importances,
            model_family=model_family,
        )

    def predict(
        self,
        artifact: LinearModelArtifact,
        feature_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        output = feature_rows.copy()
        if output.empty:
            output["expected_return_score"] = pd.Series(dtype=float)
            output["prediction_confidence"] = pd.Series(dtype=float)
            output["model_version"] = pd.Series(dtype=str)
            return output
        for column in artifact.feature_names:
            if column not in output.columns:
                output[column] = 0.0
        predictions = self.predict_scores(artifact, output)
        output["expected_return_score"] = predictions
        output["prediction_confidence"] = 1.0 / (1.0 + artifact.residual_std * 100.0 + np.abs(predictions) * 25.0)
        output["model_version"] = artifact.model_version
        return output

    def predict_scores(
        self,
        artifact: LinearModelArtifact,
        feature_rows: pd.DataFrame,
    ) -> np.ndarray:
        for column in artifact.feature_names:
            if column not in feature_rows.columns:
                feature_rows[column] = 0.0
        X_clean = feature_rows[artifact.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if artifact.model_family in LINEAR_MODEL_FAMILIES and not artifact.serialized_estimator_b64:
            model = LinearReturnModel(artifact=artifact)
            return model.predict(X_clean)

        estimator = _deserialize_estimator(artifact.serialized_estimator_b64)
        X_np = np.nan_to_num(X_clean.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(all="ignore"):
            predictions = estimator.predict(X_np)
        return np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

    def explain_predictions(
        self,
        artifact: LinearModelArtifact,
        feature_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = feature_rows.copy()
        if rows.empty:
            return pd.DataFrame(columns=EXPLANATION_COLUMNS)
        if "expected_return_score" not in rows.columns or "prediction_confidence" not in rows.columns:
            rows = self.predict(artifact, rows)

        for column in artifact.feature_names:
            if column not in rows.columns:
                rows[column] = 0.0

        X_clean = rows[artifact.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_np = np.nan_to_num(X_clean.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        means = np.asarray(artifact.means, dtype=float) if artifact.means else np.zeros(len(artifact.feature_names))
        stds = np.asarray(artifact.stds, dtype=float) if artifact.stds else np.ones(len(artifact.feature_names))
        if len(means) != len(artifact.feature_names):
            means = np.zeros(len(artifact.feature_names))
        if len(stds) != len(artifact.feature_names):
            stds = np.ones(len(artifact.feature_names))
        stds = np.where(stds == 0.0, 1.0, stds)
        X_scaled = np.clip((X_np - means) / stds, -8.0, 8.0)

        if artifact.explanation_kind == "linear_exact" and len(artifact.coefficients) == len(artifact.feature_names):
            weights = np.asarray(artifact.coefficients, dtype=float)
            contributions = X_scaled * weights
        else:
            weights = np.asarray(
                artifact.feature_importances
                if len(artifact.feature_importances) == len(artifact.feature_names)
                else [1.0 for _ in artifact.feature_names],
                dtype=float,
            )
            contributions = X_scaled * weights

        raw_frame = pd.DataFrame(X_np, columns=artifact.feature_names)
        scaled_frame = pd.DataFrame(X_scaled, columns=artifact.feature_names)
        contribution_frame = pd.DataFrame(contributions, columns=artifact.feature_names)

        meta = rows[
            [
                "signal_session_date",
                "next_session_date",
                "model_version",
                "expected_return_score",
                "prediction_confidence",
            ]
        ].copy()
        meta["row_id"] = np.arange(len(meta))

        raw_long = raw_frame.assign(row_id=meta["row_id"]).melt(
            id_vars="row_id",
            var_name="feature_name",
            value_name="raw_value",
        )
        scaled_long = scaled_frame.assign(row_id=meta["row_id"]).melt(
            id_vars="row_id",
            var_name="feature_name",
            value_name="standardized_value",
        )
        contribution_long = contribution_frame.assign(row_id=meta["row_id"]).melt(
            id_vars="row_id",
            var_name="feature_name",
            value_name="contribution",
        )

        explanation = contribution_long.merge(raw_long, on=["row_id", "feature_name"]).merge(
            scaled_long,
            on=["row_id", "feature_name"],
        )
        explanation = explanation.merge(meta, on="row_id", how="left")
        weight_map = dict(zip(artifact.feature_names, weights.tolist()))
        explanation["coefficient"] = explanation["feature_name"].map(weight_map).astype(float)
        explanation["feature_family"] = explanation["feature_name"].map(classify_feature_family)
        explanation["abs_contribution"] = explanation["contribution"].abs()
        total_abs = explanation.groupby("row_id")["abs_contribution"].transform("sum").replace(0.0, np.nan)
        explanation["contribution_share"] = (explanation["abs_contribution"] / total_abs).fillna(0.0)
        explanation = explanation[EXPLANATION_COLUMNS].sort_values(
            ["signal_session_date", "abs_contribution", "feature_name"],
            ascending=[True, False, True],
        )
        return explanation.reset_index(drop=True)
