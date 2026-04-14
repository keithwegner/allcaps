from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .contracts import BacktestRun, ModelRunConfig
from .modeling import ModelService


def compute_metrics(returns: pd.Series, positions: pd.Series) -> dict[str, float]:
    returns = returns.fillna(0.0)
    positions = positions.fillna(0.0)
    if returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "exposure": 0.0,
            "trade_count": 0.0,
            "robust_score": 0.0,
        }
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    avg = float(returns.mean())
    std = float(returns.std(ddof=0))
    downside = returns.where(returns < 0.0, 0.0)
    downside_std = float(downside.std(ddof=0))
    annualized_return = float((1.0 + avg) ** 252 - 1.0) if avg > -1.0 else -1.0
    annualized_volatility = float(std * np.sqrt(252.0))
    sharpe = float(avg / std * np.sqrt(252.0)) if std > 0 else 0.0
    sortino = float(avg / downside_std * np.sqrt(252.0)) if downside_std > 0 else sharpe
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    trades = positions > 0
    trade_returns = returns.loc[trades]
    win_rate = float((trade_returns > 0).mean()) if not trade_returns.empty else 0.0
    exposure = float(positions.mean()) if not positions.empty else 0.0
    robust_score = float(sharpe + 0.5 * sortino + total_return * 4.0 - abs(max_drawdown) * 6.0)
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "exposure": exposure,
        "trade_count": float(int(trades.sum())),
        "robust_score": robust_score,
    }


def simulate_position_mask(
    frame: pd.DataFrame,
    signal_name: str,
    position_mask: pd.Series,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    trades = frame.copy()
    trades = trades.dropna(subset=["next_session_open", "next_session_close"]).copy()
    position_mask = position_mask.reindex(trades.index).fillna(False)
    trades["trade_taken"] = position_mask.astype(bool)
    trades["position"] = trades["trade_taken"].astype(float)
    trades["gross_return"] = (trades["next_session_close"] / trades["next_session_open"] - 1.0).fillna(0.0)
    round_trip_cost = (transaction_cost_bps / 10000.0) * 2.0
    trades["net_return"] = np.where(trades["trade_taken"], trades["gross_return"] - round_trip_cost, 0.0)
    trades["benchmark_return"] = trades["gross_return"].fillna(0.0)
    trades["equity_curve"] = (1.0 + trades["net_return"]).cumprod()
    trades["signal_name"] = signal_name
    metrics = compute_metrics(trades["net_return"], trades["position"])
    return trades.reset_index(drop=True), metrics


def simulate_long_flat(
    predictions: pd.DataFrame,
    threshold: float,
    min_post_count: int,
    transaction_cost_bps: float,
    account_weight: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    signal_mask = (
        (predictions["expected_return_score"] > threshold)
        & (predictions["post_count"] >= min_post_count)
        & predictions["target_available"].fillna(False)
    )
    trades, metrics = simulate_position_mask(
        predictions,
        signal_name="strategy",
        position_mask=signal_mask,
        transaction_cost_bps=transaction_cost_bps,
    )
    trades["threshold"] = threshold
    trades["min_post_count"] = min_post_count
    trades["account_weight"] = account_weight
    return trades, metrics


def build_leakage_audit(feature_rows: pd.DataFrame, window_rows: pd.DataFrame) -> dict[str, Any]:
    if feature_rows.empty:
        return {
            "overall_pass": True,
            "future_feature_timestamp_violations": 0,
            "next_session_order_violations": 0,
            "tradeable_without_target_violations": 0,
            "window_order_violations": 0,
        }
    next_date = pd.to_datetime(feature_rows["next_session_date"], errors="coerce")
    signal_date = pd.to_datetime(feature_rows["signal_session_date"], errors="coerce")
    next_open_ts = pd.to_datetime(feature_rows["next_session_open_ts"], errors="coerce", utc=True)
    feature_max_ts = pd.to_datetime(feature_rows["feature_source_max_ts"], errors="coerce", utc=True)
    future_feature_violations = int(
        (feature_max_ts.notna() & next_open_ts.notna() & (feature_max_ts >= next_open_ts)).sum(),
    )
    next_session_order_violations = int((next_date.notna() & signal_date.notna() & (next_date <= signal_date)).sum())
    tradeable_without_target_violations = int((feature_rows["tradeable"].fillna(False) & ~feature_rows["target_available"].fillna(False)).sum())
    window_order_violations = 0
    if not window_rows.empty:
        train_end = pd.to_datetime(window_rows["train_end"], errors="coerce")
        validation_start = pd.to_datetime(window_rows["validation_start"], errors="coerce")
        validation_end = pd.to_datetime(window_rows["validation_end"], errors="coerce")
        test_start = pd.to_datetime(window_rows["test_start"], errors="coerce")
        window_order_violations = int(((validation_start <= train_end) | (test_start <= validation_end)).sum())
    overall_pass = all(
        value == 0
        for value in [
            future_feature_violations,
            next_session_order_violations,
            tradeable_without_target_violations,
            window_order_violations,
        ]
    )
    return {
        "overall_pass": overall_pass,
        "future_feature_timestamp_violations": future_feature_violations,
        "next_session_order_violations": next_session_order_violations,
        "tradeable_without_target_violations": tradeable_without_target_violations,
        "window_order_violations": window_order_violations,
        "rows_audited": int(len(feature_rows)),
    }


def build_prediction_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    diagnostics = predictions.copy()
    if "target_asset" not in diagnostics.columns:
        diagnostics["target_asset"] = "SPY"
    diagnostics["actual_next_session_return"] = diagnostics["target_next_session_return"]
    diagnostics["prediction_error"] = diagnostics["expected_return_score"] - diagnostics["actual_next_session_return"]
    diagnostics["absolute_error"] = diagnostics["prediction_error"].abs()
    diagnostics["direction_correct"] = (
        np.sign(diagnostics["expected_return_score"].fillna(0.0))
        == np.sign(diagnostics["actual_next_session_return"].fillna(0.0))
    )
    keep = [
        "target_asset",
        "signal_session_date",
        "next_session_date",
        "expected_return_score",
        "actual_next_session_return",
        "prediction_error",
        "absolute_error",
        "direction_correct",
        "post_count",
        "trump_post_count",
        "tracked_account_post_count",
        "sentiment_avg",
        "feature_cutoff_before_next_open",
        "model_version",
    ]
    return diagnostics[keep].sort_values("signal_session_date").reset_index(drop=True)


def build_benchmark_suite(
    evaluation_rows: pd.DataFrame,
    strategy_trades: pd.DataFrame,
    strategy_metrics: dict[str, float],
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_inputs = evaluation_rows.dropna(subset=["next_session_open", "next_session_close"]).copy()
    target_asset = str(
        benchmark_inputs.get("target_asset", pd.Series(["SPY"])).iloc[0]
        if not benchmark_inputs.empty
        else "SPY",
    ).upper()
    strategies: list[tuple[str, pd.Series, dict[str, float] | None]] = [
        ("strategy", strategy_trades["trade_taken"].astype(bool), strategy_metrics),
        ("always_flat", pd.Series(False, index=benchmark_inputs.index), None),
        (f"always_long_{target_asset.lower()}", benchmark_inputs["target_available"].fillna(False), None),
        ("trump_only", (benchmark_inputs["trump_post_count"] > 0) & benchmark_inputs["target_available"].fillna(False), None),
        (
            "tracked_accounts_only",
            (benchmark_inputs["tracked_account_post_count"] > 0) & benchmark_inputs["target_available"].fillna(False),
            None,
        ),
    ]
    metric_rows: list[dict[str, Any]] = []
    curve = pd.DataFrame({"next_session_date": benchmark_inputs["next_session_date"].tolist()})
    strategy_total_return = strategy_metrics["total_return"]
    for name, mask, precomputed_metrics in strategies:
        trades, metrics = simulate_position_mask(
            benchmark_inputs,
            signal_name=name,
            position_mask=mask,
            transaction_cost_bps=transaction_cost_bps,
        )
        if precomputed_metrics is not None:
            metrics = precomputed_metrics
        row = {"benchmark_name": name, **metrics}
        row["target_asset"] = target_asset
        row["excess_total_return_vs_strategy"] = float(metrics["total_return"] - strategy_total_return)
        metric_rows.append(row)
        curve[name] = trades["equity_curve"].to_numpy()
    benchmarks = pd.DataFrame(metric_rows).sort_values("robust_score", ascending=False).reset_index(drop=True)
    return benchmarks, curve


class BacktestService:
    def __init__(self, model_service: ModelService) -> None:
        self.model_service = model_service

    def build_historical_replay(
        self,
        run_config: ModelRunConfig,
        feature_rows: pd.DataFrame,
        replay_session_date: pd.Timestamp,
        deployment_params: dict[str, Any],
    ) -> dict[str, Any]:
        replay_date = pd.Timestamp(replay_session_date).normalize()
        usable = feature_rows.copy()
        if "target_asset" not in usable.columns:
            usable["target_asset"] = run_config.target_asset
        if run_config.start_date:
            usable = usable.loc[usable["signal_session_date"] >= pd.Timestamp(run_config.start_date)]
        if run_config.end_date:
            usable = usable.loc[usable["signal_session_date"] <= pd.Timestamp(run_config.end_date)]
        usable = usable.sort_values("signal_session_date").reset_index(drop=True)
        signal_dates = pd.to_datetime(usable["signal_session_date"], errors="coerce").dt.normalize()
        replay_rows = usable.loc[signal_dates == replay_date].copy()
        if replay_rows.empty:
            raise RuntimeError("The selected replay session is not available in the session feature dataset.")

        history = usable.loc[(signal_dates < replay_date) & usable["target_available"].fillna(False)].copy()
        if len(history) < 20:
            raise RuntimeError("Historical replay needs at least 20 earlier target-available sessions to train a model.")

        account_weight = float(deployment_params.get("account_weight", 1.0) or 1.0)
        threshold = float(deployment_params.get("threshold", 0.0) or 0.0)
        min_post_count = int(deployment_params.get("min_post_count", 1) or 1)

        weighted_history = self._apply_account_weight(history, account_weight)
        weighted_replay = self._apply_account_weight(replay_rows, account_weight)
        artifact, importance = self.model_service.train(
            run_config=run_config,
            feature_rows=weighted_history,
            model_version=f"{run_config.run_name}-replay-{replay_date:%Y%m%d}",
        )
        replay_prediction = self.model_service.predict(artifact, weighted_replay).reset_index(drop=True)
        replay_prediction["deployment_threshold"] = threshold
        replay_prediction["deployment_min_post_count"] = min_post_count
        replay_prediction["deployment_account_weight"] = account_weight
        replay_prediction["historical_replay"] = True
        replay_prediction["training_rows_used"] = int(len(weighted_history))
        replay_prediction["history_start"] = history["signal_session_date"].min()
        replay_prediction["history_end"] = history["signal_session_date"].max()
        replay_prediction["suggested_stance"] = np.where(
            (replay_prediction["expected_return_score"] > threshold) & (replay_prediction["post_count"] >= min_post_count),
            f"LONG {run_config.target_asset.upper()} NEXT SESSION",
            "FLAT",
        )
        replay_prediction["future_training_leakage"] = False
        feature_contributions = self.model_service.explain_predictions(artifact, replay_prediction)
        return {
            "artifact": artifact,
            "importance": importance,
            "prediction": replay_prediction,
            "feature_contributions": feature_contributions,
            "training_rows_used": int(len(weighted_history)),
            "history_start": history["signal_session_date"].min(),
            "history_end": history["signal_session_date"].max(),
            "replay_session_date": replay_date,
            "deployment_params": {
                "threshold": threshold,
                "min_post_count": min_post_count,
                "account_weight": account_weight,
            },
        }

    def run_walk_forward(
        self,
        run_config: ModelRunConfig,
        feature_rows: pd.DataFrame,
    ) -> tuple[BacktestRun, dict[str, Any]]:
        usable = feature_rows.copy()
        if "target_asset" not in usable.columns:
            usable["target_asset"] = run_config.target_asset
        if run_config.start_date:
            usable = usable.loc[usable["signal_session_date"] >= pd.Timestamp(run_config.start_date)]
        if run_config.end_date:
            usable = usable.loc[usable["signal_session_date"] <= pd.Timestamp(run_config.end_date)]
        usable = usable.loc[usable["target_available"]].sort_values("signal_session_date").reset_index(drop=True)
        if usable.empty:
            raise RuntimeError("No feature rows with targets are available for walk-forward backtesting.")

        total_needed = run_config.train_window + run_config.validation_window + run_config.test_window
        if len(usable) < total_needed:
            train_window = max(20, int(len(usable) * 0.5))
            validation_window = max(10, int(len(usable) * 0.25))
            test_window = max(10, len(usable) - train_window - validation_window)
        else:
            train_window = run_config.train_window
            validation_window = run_config.validation_window
            test_window = run_config.test_window

        all_test_trades: list[pd.DataFrame] = []
        all_test_predictions: list[pd.DataFrame] = []
        window_rows: list[dict[str, Any]] = []
        selected_thresholds: list[float] = []
        selected_min_posts: list[int] = []
        selected_weights: list[float] = []
        final_importance = pd.DataFrame()

        last_start = len(usable) - (train_window + validation_window + test_window)
        starts = list(range(0, max(last_start, 0) + 1, max(run_config.step_size, 1)))
        if not starts:
            starts = [0]

        for window_id, start in enumerate(starts, start=1):
            train = usable.iloc[start : start + train_window].copy()
            validation = usable.iloc[start + train_window : start + train_window + validation_window].copy()
            test = usable.iloc[
                start + train_window + validation_window : start + train_window + validation_window + test_window
            ].copy()
            if train.empty or validation.empty or test.empty:
                continue

            best_params: dict[str, Any] | None = None
            best_score = -np.inf
            for account_weight in run_config.account_weight_grid:
                train_weighted = self._apply_account_weight(train, account_weight)
                validation_weighted = self._apply_account_weight(validation, account_weight)
                artifact, _ = self.model_service.train(
                    run_config=run_config,
                    feature_rows=train_weighted,
                    model_version=f"{run_config.run_name}-window-{window_id}",
                )
                validation_predictions = self.model_service.predict(artifact, validation_weighted)
                for threshold in run_config.threshold_grid:
                    for min_posts in run_config.minimum_signal_grid:
                        _, metrics = simulate_long_flat(
                            validation_predictions,
                            threshold=threshold,
                            min_post_count=min_posts,
                            transaction_cost_bps=run_config.transaction_cost_bps,
                            account_weight=account_weight,
                        )
                        if metrics["robust_score"] > best_score:
                            best_score = metrics["robust_score"]
                            best_params = {
                                "threshold": float(threshold),
                                "min_post_count": int(min_posts),
                                "account_weight": float(account_weight),
                            }

            if best_params is None:
                continue

            selected_thresholds.append(best_params["threshold"])
            selected_min_posts.append(best_params["min_post_count"])
            selected_weights.append(best_params["account_weight"])

            combined_train = pd.concat([train, validation], ignore_index=True)
            combined_train = self._apply_account_weight(combined_train, best_params["account_weight"])
            test_weighted = self._apply_account_weight(test, best_params["account_weight"])
            artifact, importance = self.model_service.train(
                run_config=run_config,
                feature_rows=combined_train,
                model_version=f"{run_config.run_name}-window-{window_id}-deployment",
            )
            final_importance = importance
            test_predictions = self.model_service.predict(artifact, test_weighted)
            test_predictions["window_id"] = window_id
            test_trades, test_metrics = simulate_long_flat(
                test_predictions,
                threshold=best_params["threshold"],
                min_post_count=best_params["min_post_count"],
                transaction_cost_bps=run_config.transaction_cost_bps,
                account_weight=best_params["account_weight"],
            )
            test_trades["window_id"] = window_id
            all_test_predictions.append(test_predictions)
            all_test_trades.append(test_trades)
            window_rows.append(
                {
                    "window_id": window_id,
                    "train_start": train["signal_session_date"].min(),
                    "train_end": train["signal_session_date"].max(),
                    "validation_start": validation["signal_session_date"].min(),
                    "validation_end": validation["signal_session_date"].max(),
                    "test_start": test["signal_session_date"].min(),
                    "test_end": test["signal_session_date"].max(),
                    **best_params,
                    **test_metrics,
                },
            )

        if not all_test_trades:
            raise RuntimeError("Walk-forward backtest could not produce any evaluation windows.")

        combined_test = pd.concat(all_test_trades, ignore_index=True)
        combined_predictions = pd.concat(all_test_predictions, ignore_index=True)
        combined_metrics = compute_metrics(combined_test["net_return"], combined_test["position"])
        deployment_params = {
            "threshold": float(np.median(selected_thresholds)) if selected_thresholds else float(run_config.threshold_grid[0]),
            "min_post_count": Counter(selected_min_posts).most_common(1)[0][0] if selected_min_posts else int(run_config.minimum_signal_grid[0]),
            "account_weight": float(np.median(selected_weights)) if selected_weights else float(run_config.account_weight_grid[0]),
        }

        final_training_rows = self._apply_account_weight(usable, deployment_params["account_weight"])
        final_model, final_importance = self.model_service.train(
            run_config=run_config,
            feature_rows=final_training_rows,
            model_version=f"{run_config.run_name}-final",
        )
        full_predictions = self.model_service.predict(
            final_model,
            self._apply_account_weight(feature_rows, deployment_params["account_weight"]),
        )
        full_predictions["deployment_threshold"] = deployment_params["threshold"]
        full_predictions["deployment_min_post_count"] = deployment_params["min_post_count"]
        feature_contributions = self.model_service.explain_predictions(final_model, full_predictions)

        window_df = pd.DataFrame(window_rows)
        diagnostics = build_prediction_diagnostics(combined_predictions)
        benchmarks, benchmark_curves = build_benchmark_suite(
            evaluation_rows=combined_predictions,
            strategy_trades=combined_test,
            strategy_metrics=combined_metrics,
            transaction_cost_bps=run_config.transaction_cost_bps,
        )
        leakage_audit = build_leakage_audit(feature_rows=combined_predictions, window_rows=window_df)

        config_hash = hashlib.sha1(str(run_config.to_dict()).encode("utf-8")).hexdigest()[:12]
        run_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{config_hash}"
        run = BacktestRun(
            run_id=run_id,
            run_name=run_config.run_name,
            target_asset=run_config.target_asset,
            config_hash=config_hash,
            train_window=train_window,
            validation_window=validation_window,
            test_window=test_window,
            metrics=combined_metrics,
            selected_params=deployment_params,
        )
        artifacts = {
            "run": asdict(run),
            "config": run_config.to_dict(),
            "trades": combined_test,
            "predictions": full_predictions,
            "windows": window_df,
            "importance": final_importance,
            "model_artifact": final_model.to_dict(),
            "feature_contributions": feature_contributions,
            "benchmarks": benchmarks,
            "diagnostics": diagnostics,
            "benchmark_curves": benchmark_curves,
            "leakage_audit": leakage_audit,
        }
        return run, artifacts

    @staticmethod
    def _apply_account_weight(df: pd.DataFrame, account_weight: float) -> pd.DataFrame:
        adjusted = df.copy()
        for column in ["tracked_weighted_mentions", "tracked_weighted_engagement", "tracked_account_post_count"]:
            if column in adjusted.columns:
                adjusted[column] = adjusted[column] * account_weight
        return adjusted
