from __future__ import annotations

from collections import defaultdict
from typing import Any
from typing import Protocol
import re

import pandas as pd

from .storage import DuckDBStore
from .utils import stable_text_id

SEMANTIC_SCHEMA_VERSION = "narrative-v1"
SEMANTIC_COLUMNS = [
    "semantic_key",
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
]
SEMANTIC_DEFAULTS: dict[str, Any] = {
    "semantic_topic": "other",
    "semantic_policy_bucket": "other",
    "semantic_stance": "unknown",
    "semantic_market_relevance": 0.0,
    "semantic_urgency": 0.0,
    "semantic_primary_asset": "",
    "semantic_asset_targets": "",
    "semantic_confidence": 0.0,
    "semantic_summary": "",
    "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
    "semantic_provider": "heuristic-fallback",
}
MANDATORY_SEMANTIC_FIELDS = [
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
]

TOPIC_KEYWORDS = {
    "markets": ["stocks", "market", "dow", "nasdaq", "s&p", "spy"],
    "trade": ["tariff", "trade", "import", "export", "china", "deal", "supply chain", "reshoring"],
    "geopolitics": ["war", "ukraine", "russia", "iran", "china", "nato", "missile", "sanction"],
    "immigration": ["border", "migrant", "immigration", "asylum", "deport", "ice"],
    "judiciary": ["judge", "court", "indictment", "trial", "law", "justice"],
    "campaign": ["election", "vote", "campaign", "poll", "rally", "primary", "ballot"],
}

POLICY_BUCKET_KEYWORDS = {
    "economy": ["inflation", "jobs", "growth", "recession", "economy", "market", "fed", "rates", "tax", "nasdaq", "ai"],
    "trade": ["tariff", "trade", "deal", "import", "export", "supply chain"],
    "foreign_policy": ["war", "peace", "sanction", "nato", "iran", "russia", "china"],
    "immigration": ["border", "immigration", "migrant", "asylum"],
    "legal": ["court", "judge", "law", "trial", "indictment"],
}

MARKET_TERMS = {
    "stocks",
    "market",
    "inflation",
    "tariff",
    "recession",
    "growth",
    "economy",
    "rates",
    "fed",
    "jobs",
    "trade",
}

URGENT_TERMS = {"urgent", "breaking", "immediately", "now", "emergency", "crisis", "alert"}
POSITIVE_STANCE_TERMS = {"bullish", "surge", "strong", "winning", "booming", "optimistic", "rally"}
NEGATIVE_STANCE_TERMS = {"bearish", "risk", "weak", "pressure", "warning", "threat", "selloff"}
ASSET_ALIAS_TERMS = {
    "SPY": ["spy", "s&p 500", "s&p", "sp500", "broad market", "stock market"],
    "QQQ": ["qqq", "nasdaq", "nasdaq 100", "big tech"],
    "XLK": ["xlk", "technology sector", "tech sector", "software", "hardware"],
    "XLF": ["xlf", "financial sector", "financials", "banks", "banking"],
    "XLE": ["xle", "energy sector", "oil", "gas", "crude", "drilling"],
    "SMH": ["smh", "semiconductor", "semiconductors", "chip", "chips", "ai chip", "ai chips"],
    "NVDA": ["nvda", "nvidia", "geforce", "cuda"],
    "AAPL": ["aapl", "apple", "iphone", "ipad"],
    "MSFT": ["msft", "microsoft", "azure"],
    "AMZN": ["amzn", "amazon", "aws"],
    "META": ["meta", "facebook", "instagram"],
    "GOOGL": ["googl", "google", "alphabet"],
    "TSLA": ["tsla", "tesla", "elon"],
    "XOM": ["xom", "exxon", "exxonmobil"],
    "CVX": ["cvx", "chevron"],
    "JPM": ["jpm", "jpmorgan", "jamie dimon"],
    "BAC": ["bac", "bank of america", "bofa"],
}
TOPIC_ASSET_TARGETS = {
    "markets": ("SPY", "QQQ", "XLK", "SMH"),
    "trade": ("SPY", "XLE", "XLF"),
    "geopolitics": ("SPY", "XLE"),
    "immigration": ("SPY",),
    "judiciary": ("SPY",),
    "campaign": ("SPY", "QQQ"),
    "other": ("SPY",),
}
POLICY_ASSET_TARGETS = {
    "economy": ("SPY", "QQQ", "XLF"),
    "trade": ("SPY", "XLE", "XLF"),
    "foreign_policy": ("SPY", "XLE"),
    "immigration": ("SPY",),
    "legal": ("SPY",),
    "other": ("SPY",),
}


class NarrativeEnrichmentProvider(Protocol):
    provider_name: str

    def enrich_narrative(
        self,
        *,
        semantic_key: str,
        text: str,
        sentiment_label: str,
    ) -> dict[str, Any]:
        ...


def parse_semantic_asset_targets(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip().upper() for part in text.split(",") if part.strip()]


class LLMEnrichmentService:
    def __init__(self, store: DuckDBStore, provider: NarrativeEnrichmentProvider | None = None) -> None:
        self.store = store
        self.provider = provider

    def enrich_posts(self, posts: pd.DataFrame, enabled: bool) -> pd.DataFrame:
        return self._enrich_posts(posts, enabled=enabled, persist_cache=True)

    def enrich_posts_readonly(self, posts: pd.DataFrame, enabled: bool) -> pd.DataFrame:
        return self._enrich_posts(posts, enabled=enabled, persist_cache=False)

    def _enrich_posts(self, posts: pd.DataFrame, enabled: bool, persist_cache: bool) -> pd.DataFrame:
        out = posts.copy()
        if out.empty:
            for column in SEMANTIC_COLUMNS[1:]:
                default = SEMANTIC_DEFAULTS.get(column, "")
                dtype = float if isinstance(default, float) else object
                out[column] = pd.Series(dtype=dtype)
            out["semantic_cache_hit"] = pd.Series(dtype=bool)
            return out

        if not enabled:
            out["semantic_key"] = out["post_id"].fillna("").astype(str)
            out["semantic_topic"] = "disabled"
            out["semantic_policy_bucket"] = "disabled"
            out["semantic_stance"] = out["sentiment_label"].fillna("unknown")
            out["semantic_market_relevance"] = 0.0
            out["semantic_urgency"] = 0.0
            out["semantic_primary_asset"] = ""
            out["semantic_asset_targets"] = ""
            out["semantic_confidence"] = 0.0
            out["semantic_summary"] = "Narrative enrichment disabled."
            out["semantic_schema_version"] = SEMANTIC_SCHEMA_VERSION
            out["semantic_provider"] = "disabled"
            out["semantic_cache_hit"] = False
            return out

        cache = self._prepare_cache(self.store.read_frame("semantic_cache"))
        complete_cache_keys = set(cache.loc[self._cache_ready_mask(cache), "semantic_key"].astype(str))
        keyed = out.copy()
        keyed["semantic_key"] = keyed.apply(
            lambda row: row["post_id"] if str(row["post_id"]).strip() else stable_text_id(row["cleaned_text"]),
            axis=1,
        )
        uncached = keyed.loc[
            ~keyed["semantic_key"].astype(str).isin(complete_cache_keys),
            ["semantic_key", "cleaned_text", "sentiment_label"],
        ].drop_duplicates("semantic_key")

        if not uncached.empty:
            new_rows = uncached.apply(
                lambda row: self._build_enrichment_row(
                    semantic_key=row["semantic_key"],
                    text=str(row["cleaned_text"]),
                    sentiment_label=str(row["sentiment_label"]),
                ),
                axis=1,
                result_type="expand",
            )
            new_rows = self._prepare_cache(new_rows)
            if not cache.empty:
                cache = cache.loc[~cache["semantic_key"].astype(str).isin(new_rows["semantic_key"].astype(str))].copy()
            frames = [frame for frame in [cache, new_rows] if not frame.empty]
            cache = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SEMANTIC_COLUMNS)
            cache = self._prepare_cache(cache)
            if persist_cache:
                provider_name = self.provider.provider_name if self.provider is not None else "heuristic-fallback"
                self.store.save_frame(
                    "semantic_cache",
                    cache,
                    metadata={
                        "provider": provider_name,
                        "schema_version": SEMANTIC_SCHEMA_VERSION,
                    },
                )

        cached = keyed.merge(cache, on="semantic_key", how="left")
        cached["semantic_cache_hit"] = keyed["semantic_key"].astype(str).isin(complete_cache_keys).values
        cached = self._finalize_enriched_frame(cached)
        return cached

    def _build_enrichment_row(
        self,
        semantic_key: str,
        text: str,
        sentiment_label: str,
    ) -> dict[str, Any]:
        heuristic = self._heuristic_enrichment(
            semantic_key=semantic_key,
            text=text,
            sentiment_label=sentiment_label,
            provider_label="heuristic-fallback",
        )
        if self.provider is None:
            heuristic["semantic_provider"] = "heuristic-cache"
            return heuristic
        try:
            provider_payload = self.provider.enrich_narrative(
                semantic_key=semantic_key,
                text=text,
                sentiment_label=sentiment_label,
            )
        except Exception:
            return heuristic

        merged = heuristic.copy()
        merged.update(provider_payload or {})
        merged["semantic_provider"] = getattr(self.provider, "provider_name", "hosted-provider")
        return self._normalize_enrichment_row(merged)

    def _heuristic_enrichment(
        self,
        semantic_key: str,
        text: str,
        sentiment_label: str,
        provider_label: str,
    ) -> dict[str, Any]:
        lowered = text.lower()
        topic = self._pick_bucket(lowered, TOPIC_KEYWORDS, default="other")
        policy_bucket = self._pick_bucket(lowered, POLICY_BUCKET_KEYWORDS, default="other")
        stance = self._resolve_stance(lowered, sentiment_label)
        explicit_assets = self._extract_explicit_asset_targets(lowered, text)
        market_hits = sum(term in lowered for term in MARKET_TERMS) + len(explicit_assets)
        urgent_hits = sum(term in lowered for term in URGENT_TERMS)
        exclamations = lowered.count("!")
        caps_ratio = 0.0
        letters = [char for char in text if char.isalpha()]
        if letters:
            caps_ratio = sum(char.isupper() for char in text if char.isalpha()) / len(letters)
        market_relevance = min(1.0, market_hits / 3.0)
        urgency = min(1.0, urgent_hits * 0.25 + exclamations * 0.05 + caps_ratio * 0.5)
        asset_targets = self._rank_asset_targets(explicit_assets, topic, policy_bucket, market_relevance, urgency)
        primary_asset = explicit_assets[0] if explicit_assets else (asset_targets[0] if asset_targets else "")
        if primary_asset and primary_asset not in asset_targets:
            asset_targets = [primary_asset] + [symbol for symbol in asset_targets if symbol != primary_asset]
        confidence = min(
            1.0,
            0.3
            + (0.2 if topic != "other" else 0.0)
            + (0.15 if policy_bucket != "other" else 0.0)
            + (0.15 if explicit_assets else 0.0)
            + market_relevance * 0.15
            + urgency * 0.05,
        )
        return {
            "semantic_key": semantic_key,
            "semantic_topic": topic,
            "semantic_policy_bucket": policy_bucket,
            "semantic_stance": stance,
            "semantic_market_relevance": market_relevance,
            "semantic_urgency": urgency,
            "semantic_primary_asset": primary_asset,
            "semantic_asset_targets": ",".join(asset_targets),
            "semantic_confidence": confidence,
            "semantic_summary": self._build_summary(topic, policy_bucket, stance, primary_asset),
            "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
            "semantic_provider": provider_label,
        }

    @staticmethod
    def _pick_bucket(text: str, mapping: dict[str, list[str]], default: str) -> str:
        best_label = default
        best_hits = 0
        for label, keywords in mapping.items():
            hits = sum(keyword in text for keyword in keywords)
            if hits > best_hits:
                best_hits = hits
                best_label = label
        return best_label

    @staticmethod
    def _resolve_stance(text: str, sentiment_label: str) -> str:
        positive_hits = sum(term in text for term in POSITIVE_STANCE_TERMS)
        negative_hits = sum(term in text for term in NEGATIVE_STANCE_TERMS)
        if positive_hits > negative_hits:
            return "positive"
        if negative_hits > positive_hits:
            return "negative"
        normalized = str(sentiment_label or "unknown").strip().lower()
        return normalized or "unknown"

    @staticmethod
    def _extract_explicit_asset_targets(lowered_text: str, original_text: str) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        first_positions: dict[str, int] = {}
        for match in re.finditer(r"\$([A-Za-z]{1,5})\b", original_text):
            symbol = match.group(1).upper()
            scores[symbol] += 1.3
            first_positions[symbol] = min(first_positions.get(symbol, match.start()), match.start())
        for symbol, aliases in ASSET_ALIAS_TERMS.items():
            alias_positions = [lowered_text.find(alias) for alias in aliases if alias in lowered_text]
            hits = len(alias_positions)
            if hits:
                scores[symbol] += 0.75 + hits * 0.15
                first_positions[symbol] = min(first_positions.get(symbol, min(alias_positions)), min(alias_positions))
        ranked = sorted(scores.items(), key=lambda item: (-item[1], first_positions.get(item[0], 10**9), item[0]))
        return [symbol for symbol, _ in ranked[:4]]

    @staticmethod
    def _rank_asset_targets(
        explicit_assets: list[str],
        topic: str,
        policy_bucket: str,
        market_relevance: float,
        urgency: float,
    ) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        for idx, symbol in enumerate(explicit_assets):
            scores[symbol] += 1.0 - idx * 0.1
        for idx, symbol in enumerate(TOPIC_ASSET_TARGETS.get(topic, ())):
            scores[symbol] += 0.4 - idx * 0.05 + market_relevance * 0.1
        for idx, symbol in enumerate(POLICY_ASSET_TARGETS.get(policy_bucket, ())):
            scores[symbol] += 0.3 - idx * 0.04 + urgency * 0.05
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [symbol for symbol, _ in ranked[:4]]

    @staticmethod
    def _build_summary(topic: str, policy_bucket: str, stance: str, primary_asset: str) -> str:
        label_topic = topic.replace("_", " ")
        label_policy = policy_bucket.replace("_", " ")
        focus = primary_asset or "broad market"
        return f"{label_topic.title()} narrative in {label_policy} policy with {stance} tone; focus {focus}."

    @staticmethod
    def _prepare_cache(cache: pd.DataFrame) -> pd.DataFrame:
        normalized = cache.copy()
        if normalized.empty:
            return pd.DataFrame(columns=SEMANTIC_COLUMNS)
        for column in SEMANTIC_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = pd.NA
        normalized = normalized.loc[normalized["semantic_key"].notna()].copy()
        normalized["semantic_key"] = normalized["semantic_key"].astype(str)
        normalized = normalized.drop_duplicates(subset=["semantic_key"], keep="last").reset_index(drop=True)
        return normalized[SEMANTIC_COLUMNS]

    @staticmethod
    def _cache_ready_mask(cache: pd.DataFrame) -> pd.Series:
        if cache.empty:
            return pd.Series(dtype=bool)
        mask = pd.Series(True, index=cache.index)
        for column in MANDATORY_SEMANTIC_FIELDS:
            if column not in cache.columns:
                return pd.Series(False, index=cache.index)
            mask &= cache[column].notna()
        mask &= cache["semantic_schema_version"].astype(str) == SEMANTIC_SCHEMA_VERSION
        return mask

    @staticmethod
    def _normalize_enrichment_row(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = {"semantic_key": str(payload.get("semantic_key", "") or "")}
        for column, default in SEMANTIC_DEFAULTS.items():
            value = payload.get(column, default)
            if column in {"semantic_market_relevance", "semantic_urgency", "semantic_confidence"}:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = float(default)
                value = max(0.0, min(1.0, value))
            else:
                value = str(value or default)
            normalized[column] = value
        if normalized["semantic_schema_version"] != SEMANTIC_SCHEMA_VERSION:
            normalized["semantic_schema_version"] = SEMANTIC_SCHEMA_VERSION
        normalized["semantic_asset_targets"] = ",".join(parse_semantic_asset_targets(normalized["semantic_asset_targets"]))
        asset_targets = parse_semantic_asset_targets(normalized["semantic_asset_targets"])
        if not normalized["semantic_primary_asset"] and asset_targets:
            normalized["semantic_primary_asset"] = asset_targets[0]
        return normalized

    def _finalize_enriched_frame(self, enriched: pd.DataFrame) -> pd.DataFrame:
        out = enriched.copy()
        for column, default in SEMANTIC_DEFAULTS.items():
            if column not in out.columns:
                out[column] = default
            if isinstance(default, float):
                out[column] = pd.to_numeric(out[column], errors="coerce").fillna(float(default)).clip(lower=0.0, upper=1.0)
            else:
                out[column] = out[column].fillna(default).astype(str)
        out["semantic_asset_targets"] = out["semantic_asset_targets"].map(lambda value: ",".join(parse_semantic_asset_targets(value)))
        missing_primary = out["semantic_primary_asset"].eq("")
        out.loc[missing_primary, "semantic_primary_asset"] = out.loc[missing_primary, "semantic_asset_targets"].map(
            lambda value: parse_semantic_asset_targets(value)[0] if parse_semantic_asset_targets(value) else "",
        )
        out["semantic_schema_version"] = SEMANTIC_SCHEMA_VERSION
        return out
