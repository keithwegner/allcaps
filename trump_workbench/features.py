from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_ETF_SYMBOLS, EASTERN
from .enrichment import LLMEnrichmentService, parse_semantic_asset_targets
from .utils import business_minutes_until_close, ensure_tz_naive_date, truncate_text

REGULAR_OPEN_MINUTE = 9 * 60 + 30
REGULAR_CLOSE_MINUTE = 16 * 60
ASSET_POST_MAPPING_COLUMNS = [
    "asset_symbol",
    "asset_display_name",
    "asset_type",
    "asset_source",
    "session_date",
    "post_id",
    "post_timestamp",
    "reaction_anchor_ts",
    "mapping_reason",
    "author_account_id",
    "author_handle",
    "author_display_name",
    "author_is_trump",
    "source_platform",
    "cleaned_text",
    "mentions_trump",
    "engagement_score",
    "sentiment_score",
    "sentiment_label",
    "semantic_topic",
    "semantic_policy_bucket",
    "semantic_stance",
    "semantic_market_relevance",
    "semantic_urgency",
    "semantic_primary_asset",
    "semantic_asset_targets",
    "semantic_confidence",
    "semantic_summary",
    "semantic_schema_version",
    "semantic_provider",
    "is_active_tracked_account",
    "tracked_discovery_score",
    "tracked_account_status",
    "rule_match_score",
    "semantic_match_score",
    "asset_relevance_score",
    "match_reasons",
    "match_rank",
    "is_primary_asset",
]
CORE_ASSET_ALIAS_TERMS = {
    "SPY": ["s&p 500", "s&p", "sp500", "broad market", "stock market"],
    "QQQ": ["nasdaq", "nasdaq 100", "big tech", "tech stocks"],
    "XLK": ["technology sector", "tech sector", "software", "hardware"],
    "XLF": ["financial sector", "financials", "banks", "banking"],
    "XLE": ["energy sector", "oil", "gas", "crude", "drilling"],
    "SMH": ["semiconductor", "semiconductors", "chip", "chips", "ai chip", "ai chips"],
}
SEMANTIC_TOPIC_ASSETS = {
    "markets": {"SPY", "QQQ", "XLK"},
    "trade": {"SPY", "XLE", "XLF"},
    "geopolitics": {"SPY", "XLE"},
    "immigration": {"SPY"},
    "judiciary": {"SPY"},
    "campaign": {"SPY", "QQQ"},
}
SEMANTIC_POLICY_ASSETS = {
    "economy": {"SPY", "QQQ", "XLF"},
    "trade": {"SPY", "XLE"},
    "foreign_policy": {"SPY", "XLE"},
    "immigration": {"SPY"},
    "legal": {"SPY"},
}
ALIAS_TOKEN_BLACKLIST = {
    "class",
    "common",
    "corp",
    "corporation",
    "company",
    "fund",
    "etf",
    "trust",
    "select",
    "sector",
    "spdr",
    "invesco",
    "vaneck",
}


def _alias_terms_for_asset(symbol: str, display_name: str) -> list[str]:
    lowered_symbol = symbol.lower()
    terms = {lowered_symbol}
    normalized_name = re.sub(r"[^a-z0-9&+ ]+", " ", str(display_name or "").lower()).strip()
    if normalized_name and normalized_name != lowered_symbol:
        terms.add(normalized_name)
        for token in normalized_name.split():
            if len(token) >= 4 and token not in ALIAS_TOKEN_BLACKLIST:
                terms.add(token)
    terms.update(CORE_ASSET_ALIAS_TERMS.get(symbol, []))
    return sorted(term for term in terms if term)


def _text_contains_term(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<!\\w){re.escape(term)}(?!\\w)", text))


def map_posts_to_trade_sessions(posts: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        result = posts.copy()
        result["session_date"] = pd.NaT
        result["reaction_anchor_ts"] = pd.NaT
        result["mapping_reason"] = ""
        return result

    market_dates = pd.DatetimeIndex(market["trade_date"].dropna().sort_values().unique())
    market_date_set = set(pd.Timestamp(d) for d in market_dates)

    session_dates: list[pd.Timestamp] = []
    anchor_times: list[pd.Timestamp | pd.NaT] = []
    reasons: list[str] = []

    for ts in posts["post_timestamp"]:
        local_ts = ts.tz_convert(EASTERN)
        local_date = pd.Timestamp(local_ts.date())
        minute_of_day = local_ts.hour * 60 + local_ts.minute
        next_day_idx = market_dates.searchsorted(local_date, side="right")

        if local_date in market_date_set and minute_of_day < REGULAR_CLOSE_MINUTE:
            session_date = local_date
            if minute_of_day < REGULAR_OPEN_MINUTE:
                anchor = pd.Timestamp(f"{local_date.date()} 09:30", tz=EASTERN)
                reason = "pre-market -> same session open"
            else:
                anchor = local_ts.floor("min")
                reason = "during regular hours -> same session"
        else:
            if next_day_idx >= len(market_dates):
                session_dates.append(pd.NaT)
                anchor_times.append(pd.NaT)
                reasons.append("no later market session in loaded range")
                continue
            session_date = pd.Timestamp(market_dates[next_day_idx])
            anchor = pd.Timestamp(f"{session_date.date()} 09:30", tz=EASTERN)
            if local_date in market_date_set:
                reason = "after close -> next session open"
            else:
                reason = "weekend/holiday -> next session open"

        session_dates.append(session_date)
        anchor_times.append(anchor)
        reasons.append(reason)

    mapped = posts.copy()
    mapped["session_date"] = session_dates
    mapped["reaction_anchor_ts"] = anchor_times
    mapped["mapping_reason"] = reasons
    return mapped.dropna(subset=["session_date"]).reset_index(drop=True)


class FeatureService:
    def __init__(self, enrichment_service: LLMEnrichmentService) -> None:
        self.enrichment_service = enrichment_service

    def prepare_session_posts(
        self,
        posts: pd.DataFrame,
        market_calendar: pd.DataFrame,
        tracked_accounts: pd.DataFrame,
        llm_enabled: bool,
    ) -> pd.DataFrame:
        if market_calendar.empty:
            return posts.head(0).copy()
        market = market_calendar.sort_values("trade_date").reset_index(drop=True).copy()
        mapped = map_posts_to_trade_sessions(posts, market[["trade_date"]])
        mapped = self.enrichment_service.enrich_posts(mapped, enabled=llm_enabled)
        mapped = self._flag_tracked_posts(mapped, tracked_accounts)
        return mapped

    def build_asset_post_mappings(
        self,
        prepared_posts: pd.DataFrame,
        asset_universe: pd.DataFrame,
        llm_enabled: bool,
    ) -> pd.DataFrame:
        if prepared_posts.empty or asset_universe.empty:
            return pd.DataFrame(columns=ASSET_POST_MAPPING_COLUMNS)

        assets = asset_universe.copy()
        if "display_name" not in assets.columns:
            assets["display_name"] = assets["symbol"]
        if "asset_type" not in assets.columns:
            assets["asset_type"] = np.where(assets["symbol"].isin(DEFAULT_ETF_SYMBOLS), "etf", "equity")
        if "source" not in assets.columns:
            assets["source"] = np.where(assets["symbol"].isin(DEFAULT_ETF_SYMBOLS), "core_etf", "watchlist")

        asset_terms = {
            str(row["symbol"]): _alias_terms_for_asset(str(row["symbol"]), str(row.get("display_name", row["symbol"])))
            for _, row in assets.iterrows()
        }

        mapping_rows: list[dict[str, Any]] = []
        for row_idx, (_, post) in enumerate(prepared_posts.iterrows(), start=1):
            text = str(post.get("cleaned_text", "") or "").lower()
            semantic_topic = str(post.get("semantic_topic", "") or "")
            semantic_policy = str(post.get("semantic_policy_bucket", "") or "")
            semantic_stance = str(post.get("semantic_stance", "") or "")
            market_relevance = float(post.get("semantic_market_relevance", 0.0) or 0.0)
            primary_asset = str(post.get("semantic_primary_asset", "") or "").upper()
            target_assets = set(parse_semantic_asset_targets(post.get("semantic_asset_targets", "")))
            post_id = str(post.get("post_id", "") or f"row-{row_idx}")
            candidates: list[dict[str, Any]] = []

            for _, asset in assets.iterrows():
                symbol = str(asset["symbol"])
                matched_terms = [term for term in asset_terms[symbol] if _text_contains_term(text, term)]
                rule_score = min(1.0, 0.45 + 0.15 * len(matched_terms) + (0.15 if symbol.lower() in matched_terms else 0.0)) if matched_terms else 0.0

                semantic_score = 0.0
                semantic_reasons: list[str] = []
                if llm_enabled:
                    has_explicit_semantic_asset = symbol in target_assets or (primary_asset and symbol == primary_asset)
                    if symbol in SEMANTIC_TOPIC_ASSETS.get(semantic_topic, set()):
                        semantic_score += 0.2 + market_relevance * 0.2
                        semantic_reasons.append(f"topic:{semantic_topic}")
                    if symbol in SEMANTIC_POLICY_ASSETS.get(semantic_policy, set()):
                        semantic_score += 0.15 + market_relevance * 0.15
                        semantic_reasons.append(f"policy:{semantic_policy}")
                    if symbol in target_assets:
                        semantic_score += 0.2 + market_relevance * 0.15
                        semantic_reasons.append(f"target:{symbol.lower()}")
                    if primary_asset and symbol == primary_asset:
                        semantic_score += 0.15
                        semantic_reasons.append("primary_asset")
                    if str(asset.get("asset_type", "")) == "equity" and rule_score > 0.0:
                        semantic_score += market_relevance * 0.2
                    if str(asset.get("asset_type", "")) == "equity" and rule_score <= 0.0 and not has_explicit_semantic_asset:
                        semantic_score = 0.0
                        semantic_reasons = []

                asset_relevance_score = min(1.0, rule_score + semantic_score)
                if asset_relevance_score <= 0.0:
                    continue

                reasons = [f"rule:{term}" for term in matched_terms]
                reasons.extend(semantic_reasons)
                candidates.append(
                    {
                        "asset_symbol": symbol,
                        "asset_display_name": str(asset.get("display_name", symbol)),
                        "asset_type": str(asset.get("asset_type", "")),
                        "asset_source": str(asset.get("source", "")),
                        "session_date": post["session_date"],
                        "post_id": post_id,
                        "post_timestamp": post["post_timestamp"],
                        "reaction_anchor_ts": post.get("reaction_anchor_ts", pd.NaT),
                        "mapping_reason": str(post.get("mapping_reason", "") or ""),
                        "author_account_id": post["author_account_id"],
                        "author_handle": post["author_handle"],
                        "author_display_name": post["author_display_name"],
                        "author_is_trump": bool(post["author_is_trump"]),
                        "source_platform": post["source_platform"],
                        "cleaned_text": post["cleaned_text"],
                        "mentions_trump": bool(post["mentions_trump"]),
                        "engagement_score": float(post["engagement_score"]),
                        "sentiment_score": float(post["sentiment_score"]),
                        "sentiment_label": str(post["sentiment_label"]),
                        "semantic_topic": semantic_topic,
                        "semantic_policy_bucket": semantic_policy,
                        "semantic_stance": semantic_stance,
                        "semantic_market_relevance": market_relevance,
                        "semantic_urgency": float(post.get("semantic_urgency", 0.0) or 0.0),
                        "semantic_primary_asset": primary_asset,
                        "semantic_asset_targets": ",".join(sorted(target_assets)),
                        "semantic_confidence": float(post.get("semantic_confidence", 0.0) or 0.0),
                        "semantic_summary": str(post.get("semantic_summary", "") or ""),
                        "semantic_schema_version": str(post.get("semantic_schema_version", "") or ""),
                        "semantic_provider": str(post.get("semantic_provider", "") or ""),
                        "is_active_tracked_account": bool(post.get("is_active_tracked_account", False)),
                        "tracked_discovery_score": float(post.get("tracked_discovery_score", 0.0) or 0.0),
                        "tracked_account_status": str(post.get("tracked_account_status", "none") or "none"),
                        "rule_match_score": float(rule_score),
                        "semantic_match_score": float(semantic_score),
                        "asset_relevance_score": float(asset_relevance_score),
                        "match_reasons": ", ".join(reasons),
                        "narrative_primary_match": primary_asset == symbol,
                    },
                )

            candidates = sorted(
                candidates,
                key=lambda item: (
                    item["narrative_primary_match"],
                    item["asset_relevance_score"],
                    item["rule_match_score"],
                    item["semantic_match_score"],
                    item["asset_symbol"] not in DEFAULT_ETF_SYMBOLS,
                ),
                reverse=True,
            )
            for match_rank, candidate in enumerate(candidates, start=1):
                candidate["match_rank"] = match_rank
                candidate["is_primary_asset"] = match_rank == 1
                mapping_rows.append(candidate)

        return pd.DataFrame(mapping_rows, columns=ASSET_POST_MAPPING_COLUMNS)

    def build_asset_session_dataset(
        self,
        asset_post_mappings: pd.DataFrame,
        asset_market: pd.DataFrame,
        feature_version: str,
        llm_enabled: bool,
        asset_universe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if asset_market.empty:
            return pd.DataFrame()

        mappings = asset_post_mappings.copy() if not asset_post_mappings.empty else pd.DataFrame(columns=ASSET_POST_MAPPING_COLUMNS)
        for column in ASSET_POST_MAPPING_COLUMNS:
            if column not in mappings.columns:
                mappings[column] = pd.Series(dtype=object)
        mappings["session_date"] = pd.to_datetime(mappings["session_date"], errors="coerce").dt.normalize()

        market = asset_market.sort_values(["symbol", "trade_date"]).reset_index(drop=True).copy()
        grouped_market = market.groupby("symbol", sort=False)
        market["session_return"] = grouped_market["close"].pct_change().fillna(0.0)
        market["prev_return_1d"] = grouped_market["close"].pct_change(1).fillna(0.0)
        market["prev_return_3d"] = grouped_market["close"].pct_change(3).fillna(0.0)
        market["prev_return_5d"] = grouped_market["close"].pct_change(5).fillna(0.0)
        market["rolling_vol_5d"] = grouped_market["close"].transform(lambda series: series.pct_change().rolling(5).std().fillna(0.0))
        market["close_ma_5"] = grouped_market["close"].transform(lambda series: series.rolling(5).mean().bfill().fillna(series))
        market["close_vs_ma_5"] = (market["close"] / market["close_ma_5"] - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        market["volume_ma_5"] = grouped_market["volume"].transform(lambda series: series.rolling(5).mean().bfill().fillna(series))
        market["volume_z_5"] = (
            (market["volume"] - market["volume_ma_5"])
            / grouped_market["volume"].transform(lambda series: series.rolling(5).std(ddof=0)).replace(0, np.nan)
        ).fillna(0.0)
        market["next_session_date"] = grouped_market["trade_date"].shift(-1)
        market["next_session_open"] = grouped_market["open"].shift(-1)
        market["next_session_close"] = grouped_market["close"].shift(-1)
        market["target_next_session_return"] = (market["next_session_close"] / market["next_session_open"] - 1.0).replace([np.inf, -np.inf], np.nan)
        market["target_available"] = market["next_session_open"].notna() & market["next_session_close"].notna()

        asset_lookup = (
            asset_universe.set_index("symbol").to_dict(orient="index")
            if asset_universe is not None and not asset_universe.empty and "symbol" in asset_universe.columns
            else {}
        )
        rows: list[dict[str, Any]] = []
        for _, session in market.iterrows():
            symbol = str(session["symbol"])
            session_date = pd.Timestamp(session["trade_date"])
            group = mappings.loc[
                (mappings["asset_symbol"] == symbol)
                & (mappings["session_date"] == session_date)
            ].copy()
            group = group.sort_values("post_timestamp").reset_index(drop=True)
            row = self._build_session_row(group, session, feature_version, llm_enabled)
            asset_meta = asset_lookup.get(symbol, {})
            row.update(
                {
                    "asset_symbol": symbol,
                    "asset_display_name": asset_meta.get("display_name", symbol),
                    "asset_type": asset_meta.get("asset_type", "equity"),
                    "asset_source": asset_meta.get("source", "watchlist"),
                    "rule_matched_post_count": int((group["rule_match_score"] > 0.0).sum()) if not group.empty else 0,
                    "semantic_matched_post_count": int((group["semantic_match_score"] > 0.0).sum()) if not group.empty else 0,
                    "primary_match_post_count": int(group["is_primary_asset"].fillna(False).sum()) if not group.empty else 0,
                    "asset_relevance_score_avg": float(group["asset_relevance_score"].mean()) if not group.empty else 0.0,
                    "asset_rule_match_score_avg": float(group["rule_match_score"].mean()) if not group.empty else 0.0,
                    "asset_semantic_match_score_avg": float(group["semantic_match_score"].mean()) if not group.empty else 0.0,
                },
            )
            rows.append(row)
        dataset = pd.DataFrame(rows).sort_values(["asset_symbol", "signal_session_date"]).reset_index(drop=True)
        if not dataset.empty and "target_available" in dataset.columns:
            dataset["tradeable"] = dataset["target_available"].fillna(False)
        return dataset

    def build_session_dataset(
        self,
        posts: pd.DataFrame,
        spy_market: pd.DataFrame,
        tracked_accounts: pd.DataFrame,
        feature_version: str,
        llm_enabled: bool,
        prepared_posts: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if spy_market.empty:
            return pd.DataFrame()

        market = spy_market.sort_values("trade_date").reset_index(drop=True).copy()
        mapped = (
            prepared_posts.copy()
            if prepared_posts is not None
            else self.prepare_session_posts(posts, market, tracked_accounts, llm_enabled)
        )

        market["session_return"] = market["close"].pct_change().fillna(0.0)
        market["prev_return_1d"] = market["close"].pct_change(1).fillna(0.0)
        market["prev_return_3d"] = market["close"].pct_change(3).fillna(0.0)
        market["prev_return_5d"] = market["close"].pct_change(5).fillna(0.0)
        market["rolling_vol_5d"] = market["close"].pct_change().rolling(5).std().fillna(0.0)
        market["close_ma_5"] = market["close"].rolling(5).mean().bfill().fillna(market["close"])
        market["close_vs_ma_5"] = (market["close"] / market["close_ma_5"] - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        market["volume_ma_5"] = market["volume"].rolling(5).mean().bfill().fillna(market["volume"])
        market["volume_z_5"] = ((market["volume"] - market["volume_ma_5"]) / market["volume"].rolling(5).std(ddof=0).replace(0, np.nan)).fillna(0.0)
        market["next_session_date"] = market["trade_date"].shift(-1)
        market["next_session_open"] = market["open"].shift(-1)
        market["next_session_close"] = market["close"].shift(-1)
        market["target_next_session_return"] = (market["next_session_close"] / market["next_session_open"] - 1.0).replace([np.inf, -np.inf], np.nan)
        market["target_available"] = market["next_session_open"].notna() & market["next_session_close"].notna()

        rows: list[dict[str, Any]] = []
        for _, session in market.iterrows():
            session_date = pd.Timestamp(session["trade_date"])
            group = mapped.loc[mapped["session_date"] == session_date].copy()
            group = group.sort_values("post_timestamp").reset_index(drop=True)
            rows.append(self._build_session_row(group, session, feature_version, llm_enabled))

        dataset = pd.DataFrame(rows).sort_values("signal_session_date").reset_index(drop=True)
        dataset["tradeable"] = dataset["target_available"].fillna(False)
        return dataset

    def _flag_tracked_posts(self, mapped: pd.DataFrame, tracked_accounts: pd.DataFrame) -> pd.DataFrame:
        if mapped.empty:
            mapped["is_active_tracked_account"] = pd.Series(dtype=bool)
            mapped["tracked_discovery_score"] = pd.Series(dtype=float)
            mapped["tracked_account_status"] = pd.Series(dtype=str)
            mapped["tracked_account_provenance"] = pd.Series(dtype=str)
            return mapped

        if tracked_accounts.empty:
            mapped["is_active_tracked_account"] = False
            mapped["tracked_discovery_score"] = 0.0
            mapped["tracked_account_status"] = "none"
            mapped["tracked_account_provenance"] = ""
            return mapped

        intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None, float, str, str]]] = {}
        for _, row in tracked_accounts.iterrows():
            account_id = str(row.get("account_id", ""))
            if not account_id:
                continue
            start = pd.to_datetime(row.get("effective_from"), errors="coerce")
            end = pd.to_datetime(row.get("effective_to"), errors="coerce")
            score = float(row.get("discovery_score", 0.0) or 0.0)
            status = str(row.get("status", "") or "")
            provenance = str(row.get("provenance", "") or "")
            if pd.isna(start):
                continue
            intervals.setdefault(account_id, []).append(
                (
                    ensure_tz_naive_date(start),
                    ensure_tz_naive_date(end) if pd.notna(end) else None,
                    score,
                    status,
                    provenance,
                ),
            )

        active_flags: list[bool] = []
        active_scores: list[float] = []
        active_statuses: list[str] = []
        active_provenance: list[str] = []
        for _, row in mapped.iterrows():
            account_id = str(row["author_account_id"])
            session_date = ensure_tz_naive_date(row["session_date"])
            intervals_for_account = intervals.get(account_id, [])
            score = 0.0
            active = False
            selected_status = "none"
            selected_provenance = ""
            for start, end, interval_score, status, provenance in intervals_for_account:
                if start <= session_date and (end is None or session_date < end):
                    if status == "suppressed":
                        active = False
                        score = 0.0
                        selected_status = status
                        selected_provenance = provenance
                        break
                    active = status in {"active", "pinned"}
                    score = max(score, interval_score)
                    selected_status = status
                    selected_provenance = provenance
            active_flags.append(active)
            active_scores.append(score)
            active_statuses.append(selected_status)
            active_provenance.append(selected_provenance)

        mapped = mapped.copy()
        mapped["is_active_tracked_account"] = active_flags
        mapped["tracked_discovery_score"] = active_scores
        mapped["tracked_account_status"] = active_statuses
        mapped["tracked_account_provenance"] = active_provenance
        return mapped

    def _build_session_row(
        self,
        group: pd.DataFrame,
        market_row: pd.Series,
        feature_version: str,
        llm_enabled: bool,
    ) -> dict[str, Any]:
        if group.empty:
            base = {
                "signal_session_date": market_row["trade_date"],
                "next_session_date": market_row["next_session_date"],
                "next_session_open": market_row["next_session_open"],
                "next_session_close": market_row["next_session_close"],
                "feature_version": feature_version,
                "llm_enabled": llm_enabled,
                "feature_source_min_ts": pd.NaT,
                "feature_source_max_ts": pd.NaT,
                "next_session_open_ts": pd.Timestamp(f"{pd.Timestamp(market_row['next_session_date']).date()} 09:30", tz=EASTERN) if pd.notna(market_row["next_session_date"]) else pd.NaT,
                "feature_cutoff_before_next_open": True,
                "has_posts": False,
                "post_count": 0,
                "trump_post_count": 0,
                "x_post_count": 0,
                "tracked_account_post_count": 0,
                "mention_post_count": 0,
                "positive_posts": 0,
                "neutral_posts": 0,
                "negative_posts": 0,
                "unique_author_count": 0,
                "active_tracked_author_count": 0,
                "top_author_share": 0.0,
                "tracked_weighted_mentions": 0.0,
                "tracked_weighted_engagement": 0.0,
                "total_engagement": 0.0,
                "avg_engagement": 0.0,
                "minutes_to_close_min": 0.0,
                "sentiment_open": 0.0,
                "sentiment_high": 0.0,
                "sentiment_low": 0.0,
                "sentiment_close": 0.0,
                "sentiment_avg": 0.0,
                "sentiment_std": 0.0,
                "sentiment_range": 0.0,
                "semantic_market_relevance_avg": 0.0,
                "semantic_urgency_avg": 0.0,
                "semantic_topic_markets": 0.0,
                "semantic_topic_trade": 0.0,
                "semantic_topic_geopolitics": 0.0,
                "semantic_topic_immigration": 0.0,
                "semantic_topic_judiciary": 0.0,
                "semantic_topic_campaign": 0.0,
                "semantic_topic_other": 0.0,
                "policy_economy": 0.0,
                "policy_trade": 0.0,
                "policy_foreign_policy": 0.0,
                "policy_immigration": 0.0,
                "policy_legal": 0.0,
                "policy_other": 0.0,
                "session_return": market_row["session_return"],
                "prev_return_1d": market_row["prev_return_1d"],
                "prev_return_3d": market_row["prev_return_3d"],
                "prev_return_5d": market_row["prev_return_5d"],
                "rolling_vol_5d": market_row["rolling_vol_5d"],
                "close_vs_ma_5": market_row["close_vs_ma_5"],
                "volume_z_5": market_row["volume_z_5"],
                "target_next_session_return": market_row["target_next_session_return"],
                "target_available": market_row["target_available"],
            }
            return base

        first_row = group.iloc[0]
        last_row = group.iloc[-1]
        author_counts = group["author_account_id"].value_counts()
        topic_counts = group["semantic_topic"].value_counts(normalize=True)
        policy_counts = group["semantic_policy_bucket"].value_counts(normalize=True)

        row = {
            "signal_session_date": market_row["trade_date"],
            "next_session_date": market_row["next_session_date"],
            "next_session_open": market_row["next_session_open"],
            "next_session_close": market_row["next_session_close"],
            "feature_version": feature_version,
            "llm_enabled": llm_enabled,
            "feature_source_min_ts": group["post_timestamp"].min(),
            "feature_source_max_ts": group["post_timestamp"].max(),
            "next_session_open_ts": pd.Timestamp(f"{pd.Timestamp(market_row['next_session_date']).date()} 09:30", tz=EASTERN) if pd.notna(market_row["next_session_date"]) else pd.NaT,
            "feature_cutoff_before_next_open": True,
            "has_posts": True,
            "post_count": int(len(group)),
            "trump_post_count": int(group["author_is_trump"].sum()),
            "x_post_count": int((group["source_platform"] == "X").sum()),
            "tracked_account_post_count": int(group["is_active_tracked_account"].sum()),
            "mention_post_count": int(group["mentions_trump"].sum()),
            "positive_posts": int((group["sentiment_label"] == "positive").sum()),
            "neutral_posts": int((group["sentiment_label"] == "neutral").sum()),
            "negative_posts": int((group["sentiment_label"] == "negative").sum()),
            "unique_author_count": int(group["author_account_id"].nunique()),
            "active_tracked_author_count": int(group.loc[group["is_active_tracked_account"], "author_account_id"].nunique()),
            "top_author_share": float(author_counts.iloc[0] / len(group)) if not author_counts.empty else 0.0,
            "tracked_weighted_mentions": float((group["is_active_tracked_account"] * group["tracked_discovery_score"]).sum()),
            "tracked_weighted_engagement": float((group["engagement_score"] * group["tracked_discovery_score"]).sum()),
            "total_engagement": float(group["engagement_score"].sum()),
            "avg_engagement": float(group["engagement_score"].mean()),
            "minutes_to_close_min": float(group["post_timestamp"].map(business_minutes_until_close).min()),
            "sentiment_open": float(first_row["sentiment_score"]),
            "sentiment_high": float(group["sentiment_score"].max()),
            "sentiment_low": float(group["sentiment_score"].min()),
            "sentiment_close": float(last_row["sentiment_score"]),
            "sentiment_avg": float(group["sentiment_score"].mean()),
            "sentiment_std": float(group["sentiment_score"].std(ddof=0)) if len(group) > 1 else 0.0,
            "sentiment_range": float(group["sentiment_score"].max() - group["sentiment_score"].min()),
            "semantic_market_relevance_avg": float(group["semantic_market_relevance"].mean()),
            "semantic_urgency_avg": float(group["semantic_urgency"].mean()),
            "semantic_topic_markets": float(topic_counts.get("markets", 0.0)),
            "semantic_topic_trade": float(topic_counts.get("trade", 0.0)),
            "semantic_topic_geopolitics": float(topic_counts.get("geopolitics", 0.0)),
            "semantic_topic_immigration": float(topic_counts.get("immigration", 0.0)),
            "semantic_topic_judiciary": float(topic_counts.get("judiciary", 0.0)),
            "semantic_topic_campaign": float(topic_counts.get("campaign", 0.0)),
            "semantic_topic_other": float(topic_counts.get("other", 0.0)),
            "policy_economy": float(policy_counts.get("economy", 0.0)),
            "policy_trade": float(policy_counts.get("trade", 0.0)),
            "policy_foreign_policy": float(policy_counts.get("foreign_policy", 0.0)),
            "policy_immigration": float(policy_counts.get("immigration", 0.0)),
            "policy_legal": float(policy_counts.get("legal", 0.0)),
            "policy_other": float(policy_counts.get("other", 0.0)),
            "session_return": market_row["session_return"],
            "prev_return_1d": market_row["prev_return_1d"],
            "prev_return_3d": market_row["prev_return_3d"],
            "prev_return_5d": market_row["prev_return_5d"],
            "rolling_vol_5d": market_row["rolling_vol_5d"],
            "close_vs_ma_5": market_row["close_vs_ma_5"],
            "volume_z_5": market_row["volume_z_5"],
            "target_next_session_return": market_row["target_next_session_return"],
            "target_available": market_row["target_available"],
        }
        if pd.notna(row["next_session_open_ts"]) and pd.notna(row["feature_source_max_ts"]):
            row["feature_cutoff_before_next_open"] = bool(row["feature_source_max_ts"] < row["next_session_open_ts"])
        return row


def latest_feature_preview(feature_rows: pd.DataFrame) -> dict[str, Any]:
    if feature_rows.empty:
        return {}
    row = feature_rows.sort_values("signal_session_date").iloc[-1]
    return {
        "signal_session_date": row["signal_session_date"],
        "post_count": int(row["post_count"]),
        "trump_post_count": int(row["trump_post_count"]),
        "tracked_account_post_count": int(row["tracked_account_post_count"]),
        "sentiment_avg": float(row["sentiment_avg"]),
        "market_context": f"prev 1d {row['prev_return_1d']:+.2%}, prev 5d {row['prev_return_5d']:+.2%}",
    }


def preview_post_texts(group: pd.DataFrame, max_items: int = 3) -> str:
    if group.empty:
        return ""
    bits = []
    for _, row in group.sort_values("post_timestamp").head(max_items).iterrows():
        handle = row["author_handle"] or row["author_display_name"] or row["source_platform"]
        bits.append(f"@{handle}: {truncate_text(row['cleaned_text'], max_chars=120)}")
    return " | ".join(bits)
