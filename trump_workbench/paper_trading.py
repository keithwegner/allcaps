from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

import pandas as pd

from .config import EASTERN
from .contracts import LiveMonitorConfig, PaperDecisionRecord, PaperEquityPoint, PaperPortfolioConfig, PaperTradeRecord
from .portfolio import rank_portfolio_candidates

PAPER_PORTFOLIO_CONFIG_PATH = "live_monitor/paper_portfolio.json"
PAPER_PORTFOLIO_REGISTRY_COLUMNS = [
    "paper_portfolio_id",
    "portfolio_run_id",
    "portfolio_run_name",
    "deployment_variant",
    "fallback_mode",
    "transaction_cost_bps",
    "starting_cash",
    "enabled",
    "created_at",
    "archived_at",
]
PAPER_DECISION_JOURNAL_COLUMNS = [
    "paper_portfolio_id",
    "generated_at",
    "signal_session_date",
    "next_session_date",
    "decision_cutoff_ts",
    "portfolio_run_id",
    "portfolio_run_name",
    "deployment_variant",
    "winning_asset",
    "winning_run_id",
    "decision_source",
    "fallback_mode",
    "stance",
    "winner_score",
    "runner_up_asset",
    "runner_up_score",
    "eligible_asset_count",
    "settlement_status",
    "settled_at",
]
PAPER_TRADE_LEDGER_COLUMNS = [
    "paper_portfolio_id",
    "signal_session_date",
    "next_session_date",
    "asset_symbol",
    "run_id",
    "decision_source",
    "stance",
    "next_session_open",
    "next_session_close",
    "gross_return",
    "net_return",
    "benchmark_return",
    "transaction_cost_bps",
    "starting_equity",
    "ending_equity",
    "settled_at",
]
PAPER_EQUITY_CURVE_COLUMNS = [
    "paper_portfolio_id",
    "signal_session_date",
    "next_session_date",
    "equity",
    "return_pct",
    "settled_at",
]
PAPER_BENCHMARK_CURVE_COLUMNS = [
    "paper_portfolio_id",
    "benchmark_name",
    "signal_session_date",
    "next_session_date",
    "equity",
    "return_pct",
    "settled_at",
]


def ensure_paper_portfolio_registry_frame(frame: pd.DataFrame) -> pd.DataFrame:
    registry = frame.copy()
    for column in PAPER_PORTFOLIO_REGISTRY_COLUMNS:
        if column not in registry.columns:
            registry[column] = pd.NA
    for column in ["created_at", "archived_at"]:
        registry[column] = pd.to_datetime(registry[column], errors="coerce", utc=True)
    registry["enabled"] = pd.Series(registry["enabled"], dtype="boolean").fillna(False).astype(bool)
    registry["transaction_cost_bps"] = pd.to_numeric(registry["transaction_cost_bps"], errors="coerce").fillna(0.0)
    registry["starting_cash"] = pd.to_numeric(registry["starting_cash"], errors="coerce").fillna(100000.0)
    return registry[PAPER_PORTFOLIO_REGISTRY_COLUMNS].copy()


def ensure_paper_decision_journal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    journal = frame.copy()
    for column in PAPER_DECISION_JOURNAL_COLUMNS:
        if column not in journal.columns:
            journal[column] = pd.NA
    for column in ["generated_at", "signal_session_date", "next_session_date", "decision_cutoff_ts", "settled_at"]:
        journal[column] = pd.to_datetime(journal[column], errors="coerce", utc=True)
    for column in ["winner_score", "runner_up_score"]:
        journal[column] = pd.to_numeric(journal[column], errors="coerce").fillna(0.0)
    journal["eligible_asset_count"] = pd.to_numeric(journal["eligible_asset_count"], errors="coerce").fillna(0).astype(int)
    journal["settlement_status"] = journal["settlement_status"].fillna("pending").astype(str)
    return journal[PAPER_DECISION_JOURNAL_COLUMNS].copy()


def ensure_paper_trade_ledger_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ledger = frame.copy()
    for column in PAPER_TRADE_LEDGER_COLUMNS:
        if column not in ledger.columns:
            ledger[column] = pd.NA
    for column in ["signal_session_date", "next_session_date", "settled_at"]:
        ledger[column] = pd.to_datetime(ledger[column], errors="coerce", utc=True)
    for column in ["next_session_open", "next_session_close", "gross_return", "net_return", "benchmark_return", "transaction_cost_bps", "starting_equity", "ending_equity"]:
        ledger[column] = pd.to_numeric(ledger[column], errors="coerce")
    return ledger[PAPER_TRADE_LEDGER_COLUMNS].copy()


def ensure_paper_equity_curve_frame(frame: pd.DataFrame) -> pd.DataFrame:
    curve = frame.copy()
    for column in PAPER_EQUITY_CURVE_COLUMNS:
        if column not in curve.columns:
            curve[column] = pd.NA
    for column in ["signal_session_date", "next_session_date", "settled_at"]:
        curve[column] = pd.to_datetime(curve[column], errors="coerce", utc=True)
    for column in ["equity", "return_pct"]:
        curve[column] = pd.to_numeric(curve[column], errors="coerce")
    return curve[PAPER_EQUITY_CURVE_COLUMNS].copy()


def ensure_paper_benchmark_curve_frame(frame: pd.DataFrame) -> pd.DataFrame:
    curve = frame.copy()
    for column in PAPER_BENCHMARK_CURVE_COLUMNS:
        if column not in curve.columns:
            curve[column] = pd.NA
    for column in ["signal_session_date", "next_session_date", "settled_at"]:
        curve[column] = pd.to_datetime(curve[column], errors="coerce", utc=True)
    for column in ["equity", "return_pct"]:
        curve[column] = pd.to_numeric(curve[column], errors="coerce")
    curve["benchmark_name"] = curve["benchmark_name"].fillna("always_long_spy").astype(str)
    return curve[PAPER_BENCHMARK_CURVE_COLUMNS].copy()


def paper_config_matches_live(config: PaperPortfolioConfig | None, live_config: LiveMonitorConfig) -> bool:
    if config is None or config.is_archived:
        return False
    return (
        str(config.portfolio_run_id or "") == str(live_config.portfolio_run_id or "")
        and str(config.deployment_variant or "") == str(live_config.deployment_variant or "")
        and str(config.fallback_mode or "SPY").upper() == str(live_config.fallback_mode or "SPY").upper()
    )


class PaperTradingService:
    def __init__(self, store) -> None:
        self.store = store

    def load_current_config(self) -> PaperPortfolioConfig | None:
        payload = self.store.read_json_artifact(PAPER_PORTFOLIO_CONFIG_PATH)
        if payload is None:
            return None
        return PaperPortfolioConfig.from_dict(payload)

    def list_portfolios(self) -> pd.DataFrame:
        registry = ensure_paper_portfolio_registry_frame(self.store.read_frame("paper_portfolio_registry"))
        if registry.empty:
            return registry
        registry = registry.sort_values(["created_at", "paper_portfolio_id"], ascending=[False, False]).reset_index(drop=True)
        return registry

    def save_current_config(self, config: PaperPortfolioConfig) -> None:
        self.store.save_json_artifact(PAPER_PORTFOLIO_CONFIG_PATH, config.to_dict())
        registry = ensure_paper_portfolio_registry_frame(self.store.read_frame("paper_portfolio_registry"))
        registry = registry.loc[registry["paper_portfolio_id"].astype(str) != str(config.paper_portfolio_id)].copy()
        registry = pd.concat([registry, pd.DataFrame([asdict(config)])], ignore_index=True)
        registry = ensure_paper_portfolio_registry_frame(registry)
        registry = registry.sort_values(["created_at", "paper_portfolio_id"], ascending=[False, False]).reset_index(drop=True)
        self.store.save_frame(
            "paper_portfolio_registry",
            registry,
            metadata={"row_count": int(len(registry)), "active_count": int(registry["archived_at"].isna().sum())},
        )

    def build_config(
        self,
        live_config: LiveMonitorConfig,
        portfolio_run_name: str,
        transaction_cost_bps: float,
        starting_cash: float = 100000.0,
        enabled: bool = True,
        now: pd.Timestamp | None = None,
    ) -> PaperPortfolioConfig:
        created_at = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        if created_at.tzinfo is None:
            created_at = created_at.tz_localize("UTC")
        else:
            created_at = created_at.tz_convert("UTC")
        return PaperPortfolioConfig(
            paper_portfolio_id=f"paper-{uuid4().hex[:12]}",
            portfolio_run_id=str(live_config.portfolio_run_id or ""),
            portfolio_run_name=str(portfolio_run_name or live_config.portfolio_run_name or live_config.portfolio_run_id or ""),
            deployment_variant=str(live_config.deployment_variant or ""),
            fallback_mode=str(live_config.fallback_mode or "SPY").upper(),
            transaction_cost_bps=float(transaction_cost_bps or 0.0),
            starting_cash=float(starting_cash or 100000.0),
            enabled=bool(enabled),
            created_at=created_at.isoformat(),
            archived_at="",
        )

    def archive_current_config(self, now: pd.Timestamp | None = None) -> PaperPortfolioConfig | None:
        current = self.load_current_config()
        if current is None or current.is_archived:
            return current
        archived_at = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        if archived_at.tzinfo is None:
            archived_at = archived_at.tz_localize("UTC")
        else:
            archived_at = archived_at.tz_convert("UTC")
        archived = PaperPortfolioConfig(
            paper_portfolio_id=current.paper_portfolio_id,
            portfolio_run_id=current.portfolio_run_id,
            portfolio_run_name=current.portfolio_run_name,
            deployment_variant=current.deployment_variant,
            fallback_mode=current.fallback_mode,
            transaction_cost_bps=current.transaction_cost_bps,
            starting_cash=current.starting_cash,
            enabled=False,
            created_at=current.created_at,
            archived_at=archived_at.isoformat(),
        )
        self.save_current_config(archived)
        return archived

    def upsert_current_for_live_config(
        self,
        live_config: LiveMonitorConfig,
        portfolio_run_name: str,
        transaction_cost_bps: float,
        starting_cash: float = 100000.0,
        enabled: bool = True,
        reset: bool = False,
        now: pd.Timestamp | None = None,
    ) -> PaperPortfolioConfig:
        current = self.load_current_config()
        if reset or not paper_config_matches_live(current, live_config):
            if current is not None and not current.is_archived:
                self.archive_current_config(now=now)
            current = self.build_config(
                live_config=live_config,
                portfolio_run_name=portfolio_run_name,
                transaction_cost_bps=transaction_cost_bps,
                starting_cash=starting_cash,
                enabled=enabled,
                now=now,
            )
        else:
            current = PaperPortfolioConfig(
                paper_portfolio_id=current.paper_portfolio_id,
                portfolio_run_id=current.portfolio_run_id,
                portfolio_run_name=current.portfolio_run_name,
                deployment_variant=current.deployment_variant,
                fallback_mode=current.fallback_mode,
                transaction_cost_bps=current.transaction_cost_bps,
                starting_cash=current.starting_cash,
                enabled=bool(enabled),
                created_at=current.created_at,
                archived_at=current.archived_at,
            )
        self.save_current_config(current)
        return current

    def _replace_portfolio_rows(self, dataset_name: str, frame: pd.DataFrame, paper_portfolio_id: str) -> None:
        existing = self.store.read_frame(dataset_name)
        if "paper_portfolio_id" in existing.columns:
            existing = existing.loc[existing["paper_portfolio_id"].astype(str) != str(paper_portfolio_id)].copy()
        combined = pd.concat([existing, frame], ignore_index=True) if not existing.empty or not frame.empty else pd.DataFrame()
        self.store.save_frame(dataset_name, combined, metadata={"row_count": int(len(combined))})

    def _spy_price_lookup(self) -> pd.DataFrame:
        asset_daily = self.store.read_frame("asset_daily")
        if not asset_daily.empty:
            normalized = asset_daily.copy()
            normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
            spy_rows = normalized.loc[normalized["symbol"] == "SPY"].copy()
            if not spy_rows.empty:
                return spy_rows
        spy_daily = self.store.read_frame("spy_daily")
        if spy_daily.empty:
            return pd.DataFrame()
        spy_rows = spy_daily.copy()
        spy_rows["symbol"] = "SPY"
        return spy_rows

    @staticmethod
    def _price_lookup(frame: pd.DataFrame, symbol: str, trade_date: pd.Timestamp) -> tuple[float | None, float | None]:
        if frame.empty:
            return None, None
        normalized = frame.copy()
        if "symbol" not in normalized.columns:
            normalized["symbol"] = symbol
        normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
        normalized["trade_date"] = pd.to_datetime(normalized["trade_date"], errors="coerce").dt.tz_localize(None)
        rows = normalized.loc[
            (normalized["symbol"] == str(symbol).upper())
            & (normalized["trade_date"] == pd.Timestamp(trade_date).tz_localize(None) if pd.Timestamp(trade_date).tzinfo is not None else pd.Timestamp(trade_date))
        ].copy()
        if rows.empty:
            return None, None
        row = rows.iloc[-1]
        return (
            float(pd.to_numeric(row.get("open"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("open"), errors="coerce")) else None,
            float(pd.to_numeric(row.get("close"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("close"), errors="coerce")) else None,
        )

    @staticmethod
    def _fallback_cutoff(next_session_date: pd.Timestamp | None) -> pd.Timestamp | None:
        if next_session_date is None or pd.isna(next_session_date):
            return None
        local = pd.Timestamp(next_session_date)
        if local.tzinfo is not None:
            local = local.tz_convert(EASTERN)
        else:
            local = local.tz_localize(EASTERN)
        cutoff = pd.Timestamp(f"{local.date()} 09:30", tz=EASTERN)
        return cutoff.tz_convert("UTC")

    def capture_authoritative_decisions(
        self,
        config: PaperPortfolioConfig,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        journal = ensure_paper_decision_journal_frame(self.store.read_frame("paper_decision_journal"))
        if not config.enabled or config.is_archived:
            return journal

        live_assets = self.store.read_frame("live_asset_snapshots")
        if live_assets.empty:
            return journal
        board = live_assets.copy()
        board["run_id"] = board.get("run_id", "").astype(str)
        board = board.loc[board["run_id"] == str(config.portfolio_run_id)].copy()
        if "variant_name" in board.columns and str(config.deployment_variant or ""):
            board = board.loc[board["variant_name"].astype(str) == str(config.deployment_variant)].copy()
        if board.empty:
            return journal
        board["generated_at"] = pd.to_datetime(board["generated_at"], errors="coerce", utc=True)
        board["signal_session_date"] = pd.to_datetime(board["signal_session_date"], errors="coerce", utc=True)
        board["next_session_date"] = pd.to_datetime(board["next_session_date"], errors="coerce", utc=True)
        board["next_session_open_ts"] = pd.to_datetime(board.get("next_session_open_ts"), errors="coerce", utc=True)
        board = board.dropna(subset=["generated_at", "signal_session_date"]).copy()
        if board.empty:
            return journal

        current_ts = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
        if current_ts.tzinfo is None:
            current_ts = current_ts.tz_localize("UTC")
        else:
            current_ts = current_ts.tz_convert("UTC")

        existing_sessions = set(
            journal.loc[
                journal["paper_portfolio_id"].astype(str) == str(config.paper_portfolio_id),
                "signal_session_date",
            ]
            .dropna()
            .tolist(),
        )
        new_rows: list[dict[str, object]] = []
        grouped = board.groupby(board["signal_session_date"].dt.normalize(), sort=True)
        for _, session_rows in grouped:
            session_key = pd.Timestamp(session_rows["signal_session_date"].iloc[0]).normalize()
            if session_key.tzinfo is None:
                session_key = session_key.tz_localize("UTC")
            else:
                session_key = session_key.tz_convert("UTC")
            if session_key in existing_sessions:
                continue
            cutoff_candidates = session_rows["next_session_open_ts"].dropna()
            cutoff_ts = cutoff_candidates.min() if not cutoff_candidates.empty else self._fallback_cutoff(session_rows["next_session_date"].dropna().min() if not session_rows["next_session_date"].dropna().empty else None)
            if cutoff_ts is None or current_ts < cutoff_ts:
                continue
            eligible_snapshots = session_rows.loc[session_rows["generated_at"] < cutoff_ts].copy()
            if eligible_snapshots.empty:
                continue
            latest_generated_at = eligible_snapshots["generated_at"].max()
            latest_board = eligible_snapshots.loc[eligible_snapshots["generated_at"] == latest_generated_at].copy()
            ranked_board, decision = rank_portfolio_candidates(latest_board, fallback_mode=config.fallback_mode, require_tradeable=False)
            if decision.empty:
                continue
            decision_row = decision.iloc[0]
            record = PaperDecisionRecord(
                paper_portfolio_id=str(config.paper_portfolio_id),
                generated_at=pd.Timestamp(latest_generated_at),
                signal_session_date=pd.Timestamp(decision_row["signal_session_date"]),
                next_session_date=pd.Timestamp(decision_row["next_session_date"]) if pd.notna(decision_row["next_session_date"]) else None,
                decision_cutoff_ts=pd.Timestamp(cutoff_ts),
                portfolio_run_id=str(config.portfolio_run_id),
                portfolio_run_name=str(config.portfolio_run_name),
                deployment_variant=str(config.deployment_variant),
                winning_asset=str(decision_row.get("winning_asset", "") or ""),
                winning_run_id=str(decision_row.get("winning_run_id", "") or ""),
                decision_source=str(decision_row.get("decision_source", "") or ""),
                fallback_mode=str(decision_row.get("fallback_mode", config.fallback_mode) or config.fallback_mode),
                stance=str(decision_row.get("stance", "") or ""),
                winner_score=float(decision_row.get("winner_score", 0.0) or 0.0),
                runner_up_asset=str(decision_row.get("runner_up_asset", "") or ""),
                runner_up_score=float(decision_row.get("runner_up_score", 0.0) or 0.0),
                eligible_asset_count=int(decision_row.get("eligible_asset_count", 0) or 0),
                settlement_status="pending",
                settled_at=None,
            )
            new_rows.append(record.to_dict())

        if not new_rows:
            return journal
        updated = pd.concat([journal, pd.DataFrame(new_rows)], ignore_index=True)
        updated = ensure_paper_decision_journal_frame(updated)
        updated = updated.drop_duplicates(subset=["paper_portfolio_id", "signal_session_date"], keep="last")
        updated = updated.sort_values(["signal_session_date", "generated_at"]).reset_index(drop=True)
        self.store.save_frame(
            "paper_decision_journal",
            updated,
            metadata={"row_count": int(len(updated)), "portfolio_count": int(updated["paper_portfolio_id"].nunique()) if not updated.empty else 0},
        )
        return updated

    def settle_portfolio(
        self,
        config: PaperPortfolioConfig,
        as_of: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        journal = ensure_paper_decision_journal_frame(self.store.read_frame("paper_decision_journal"))
        if journal.empty:
            empty_trades = ensure_paper_trade_ledger_frame(pd.DataFrame())
            empty_curve = ensure_paper_equity_curve_frame(pd.DataFrame())
            empty_benchmark = ensure_paper_benchmark_curve_frame(pd.DataFrame())
            return journal, empty_trades, empty_curve, empty_benchmark

        current_ts = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
        if current_ts.tzinfo is None:
            current_ts = current_ts.tz_localize("UTC")
        else:
            current_ts = current_ts.tz_convert("UTC")

        asset_daily = self.store.read_frame("asset_daily")
        spy_daily = self._spy_price_lookup()
        portfolio_journal = journal.loc[journal["paper_portfolio_id"].astype(str) == str(config.paper_portfolio_id)].copy()
        if portfolio_journal.empty:
            empty_trades = ensure_paper_trade_ledger_frame(pd.DataFrame())
            empty_curve = ensure_paper_equity_curve_frame(pd.DataFrame())
            empty_benchmark = ensure_paper_benchmark_curve_frame(pd.DataFrame())
            return journal, empty_trades, empty_curve, empty_benchmark

        for idx, row in portfolio_journal.loc[portfolio_journal["settlement_status"].astype(str) == "pending"].iterrows():
            next_session_date = row.get("next_session_date")
            if pd.isna(next_session_date):
                continue
            benchmark_open, benchmark_close = self._price_lookup(spy_daily, "SPY", pd.Timestamp(next_session_date))
            if benchmark_open is None or benchmark_close is None:
                continue
            winning_asset = str(row.get("winning_asset", "") or "").upper()
            if not winning_asset:
                portfolio_journal.loc[idx, "settlement_status"] = "flat"
                portfolio_journal.loc[idx, "settled_at"] = current_ts
                continue
            trade_open, trade_close = self._price_lookup(asset_daily if not asset_daily.empty else spy_daily, winning_asset, pd.Timestamp(next_session_date))
            if trade_open is None or trade_close is None:
                continue
            portfolio_journal.loc[idx, "settlement_status"] = "settled"
            portfolio_journal.loc[idx, "settled_at"] = current_ts

        updated_journal = journal.loc[journal["paper_portfolio_id"].astype(str) != str(config.paper_portfolio_id)].copy()
        updated_journal = pd.concat([updated_journal, portfolio_journal], ignore_index=True)
        updated_journal = ensure_paper_decision_journal_frame(updated_journal)
        updated_journal = updated_journal.sort_values(["signal_session_date", "generated_at"]).reset_index(drop=True)
        self.store.save_frame(
            "paper_decision_journal",
            updated_journal,
            metadata={"row_count": int(len(updated_journal)), "portfolio_count": int(updated_journal["paper_portfolio_id"].nunique()) if not updated_journal.empty else 0},
        )

        settled_rows = portfolio_journal.loc[
            portfolio_journal["settlement_status"].astype(str).isin(["settled", "flat"])
        ].sort_values("signal_session_date").reset_index(drop=True)
        trade_rows: list[dict[str, object]] = []
        equity_rows: list[dict[str, object]] = []
        benchmark_rows: list[dict[str, object]] = []
        strategy_equity = float(config.starting_cash)
        benchmark_equity = float(config.starting_cash)
        round_trip_cost = (float(config.transaction_cost_bps) / 10000.0) * 2.0
        for _, row in settled_rows.iterrows():
            signal_session_date = pd.Timestamp(row["signal_session_date"])
            next_session_date = pd.Timestamp(row["next_session_date"]) if pd.notna(row["next_session_date"]) else None
            benchmark_open, benchmark_close = self._price_lookup(spy_daily, "SPY", next_session_date) if next_session_date is not None else (None, None)
            benchmark_return = 0.0
            if benchmark_open is not None and benchmark_close is not None and benchmark_open != 0:
                benchmark_return = float(benchmark_close / benchmark_open - 1.0)

            settled_at = pd.Timestamp(row["settled_at"]) if pd.notna(row["settled_at"]) else current_ts
            starting_equity = strategy_equity
            winning_asset = str(row.get("winning_asset", "") or "").upper()
            if winning_asset:
                trade_open, trade_close = self._price_lookup(asset_daily if not asset_daily.empty else spy_daily, winning_asset, next_session_date)
                if trade_open is not None and trade_close is not None and trade_open != 0:
                    gross_return = float(trade_close / trade_open - 1.0)
                    net_return = gross_return - round_trip_cost
                    ending_equity = starting_equity * (1.0 + net_return)
                    trade = PaperTradeRecord(
                        paper_portfolio_id=str(config.paper_portfolio_id),
                        signal_session_date=signal_session_date,
                        next_session_date=next_session_date,
                        asset_symbol=winning_asset,
                        run_id=str(row.get("winning_run_id", "") or ""),
                        decision_source=str(row.get("decision_source", "") or ""),
                        stance=str(row.get("stance", "") or ""),
                        next_session_open=float(trade_open),
                        next_session_close=float(trade_close),
                        gross_return=float(gross_return),
                        net_return=float(net_return),
                        benchmark_return=float(benchmark_return),
                        transaction_cost_bps=float(config.transaction_cost_bps),
                        starting_equity=float(starting_equity),
                        ending_equity=float(ending_equity),
                        settled_at=settled_at,
                    )
                    trade_rows.append(trade.to_dict())
                    strategy_equity = ending_equity
                else:
                    net_return = 0.0
            else:
                net_return = 0.0

            benchmark_equity = benchmark_equity * (1.0 + benchmark_return)
            equity_rows.append(
                PaperEquityPoint(
                    paper_portfolio_id=str(config.paper_portfolio_id),
                    signal_session_date=signal_session_date,
                    next_session_date=next_session_date,
                    equity=float(strategy_equity),
                    return_pct=float(net_return),
                    settled_at=settled_at,
                ).to_dict(),
            )
            benchmark_rows.append(
                {
                    "paper_portfolio_id": str(config.paper_portfolio_id),
                    "benchmark_name": "always_long_spy",
                    "signal_session_date": signal_session_date,
                    "next_session_date": next_session_date,
                    "equity": float(benchmark_equity),
                    "return_pct": float(benchmark_return),
                    "settled_at": settled_at,
                },
            )

        trades = ensure_paper_trade_ledger_frame(pd.DataFrame(trade_rows))
        equity_curve = ensure_paper_equity_curve_frame(pd.DataFrame(equity_rows))
        benchmark_curve = ensure_paper_benchmark_curve_frame(pd.DataFrame(benchmark_rows))
        self._replace_portfolio_rows("paper_trade_ledger", trades, config.paper_portfolio_id)
        self._replace_portfolio_rows("paper_equity_curve", equity_curve, config.paper_portfolio_id)
        self._replace_portfolio_rows("paper_benchmark_curve", benchmark_curve, config.paper_portfolio_id)
        return updated_journal, trades, equity_curve, benchmark_curve

    def process_live_history(
        self,
        config: PaperPortfolioConfig | None,
        as_of: pd.Timestamp | None = None,
    ) -> dict[str, int]:
        if config is None or not config.enabled or config.is_archived:
            return {"captured": 0, "settled": 0}
        before = ensure_paper_decision_journal_frame(self.store.read_frame("paper_decision_journal"))
        before_portfolio = before.loc[before["paper_portfolio_id"].astype(str) == str(config.paper_portfolio_id)].copy()
        before_count = int(len(before_portfolio))
        before_settled = int(before_portfolio["settlement_status"].astype(str).isin(["settled", "flat"]).sum()) if not before_portfolio.empty else 0
        self.capture_authoritative_decisions(config, as_of=as_of)
        updated_journal, _, _, _ = self.settle_portfolio(config, as_of=as_of)
        after_portfolio = updated_journal.loc[updated_journal["paper_portfolio_id"].astype(str) == str(config.paper_portfolio_id)].copy()
        after_count = int(len(after_portfolio))
        after_settled = int(after_portfolio["settlement_status"].astype(str).isin(["settled", "flat"]).sum()) if not after_portfolio.empty else 0
        return {
            "captured": max(0, after_count - before_count),
            "settled": max(0, after_settled - before_settled),
        }
