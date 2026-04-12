from __future__ import annotations

from typing import Any

import pandas as pd

from .storage import DuckDBStore
from .utils import stable_text_id

SEMANTIC_COLUMNS = [
    "semantic_key",
    "semantic_topic",
    "semantic_policy_bucket",
    "semantic_stance",
    "semantic_market_relevance",
    "semantic_urgency",
    "semantic_provider",
]

TOPIC_KEYWORDS = {
    "markets": ["stocks", "market", "dow", "nasdaq", "s&p", "spy"],
    "trade": ["tariff", "trade", "import", "export", "china", "deal"],
    "geopolitics": ["war", "ukraine", "russia", "iran", "china", "nato", "missile"],
    "immigration": ["border", "migrant", "immigration", "asylum", "deport"],
    "judiciary": ["judge", "court", "indictment", "trial", "law", "justice"],
    "campaign": ["election", "vote", "campaign", "poll", "rally"],
}

POLICY_BUCKET_KEYWORDS = {
    "economy": ["inflation", "jobs", "growth", "recession", "economy", "market"],
    "trade": ["tariff", "trade", "deal", "import", "export"],
    "foreign_policy": ["war", "peace", "sanction", "nato", "iran", "russia"],
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


class LLMEnrichmentService:
    def __init__(self, store: DuckDBStore) -> None:
        self.store = store

    def enrich_posts(self, posts: pd.DataFrame, enabled: bool) -> pd.DataFrame:
        out = posts.copy()
        if out.empty:
            for column in SEMANTIC_COLUMNS[1:]:
                out[column] = pd.Series(dtype=object)
            out["semantic_cache_hit"] = pd.Series(dtype=bool)
            return out

        if not enabled:
            out["semantic_key"] = out["post_id"].fillna("").astype(str)
            out["semantic_topic"] = "disabled"
            out["semantic_policy_bucket"] = "disabled"
            out["semantic_stance"] = out["sentiment_label"].fillna("unknown")
            out["semantic_market_relevance"] = 0.0
            out["semantic_urgency"] = 0.0
            out["semantic_provider"] = "disabled"
            out["semantic_cache_hit"] = False
            return out

        cache = self.store.read_frame("semantic_cache")
        if cache.empty:
            cache = pd.DataFrame(columns=SEMANTIC_COLUMNS)
        cache = cache.drop_duplicates(subset=["semantic_key"], keep="last")

        keyed = out.copy()
        keyed["semantic_key"] = keyed.apply(
            lambda row: row["post_id"] if str(row["post_id"]).strip() else stable_text_id(row["cleaned_text"]),
            axis=1,
        )
        cached = keyed.merge(cache, on="semantic_key", how="left", suffixes=("", "_cached"))
        cache_hit = cached["semantic_topic"].notna()
        uncached = cached.loc[~cache_hit, ["semantic_key", "cleaned_text", "sentiment_label"]].drop_duplicates("semantic_key")

        if not uncached.empty:
            new_rows = uncached.apply(
                lambda row: self._heuristic_enrichment(
                    semantic_key=row["semantic_key"],
                    text=str(row["cleaned_text"]),
                    sentiment_label=str(row["sentiment_label"]),
                ),
                axis=1,
                result_type="expand",
            )
            cache = new_rows if cache.empty else pd.concat([cache, new_rows], ignore_index=True)
            cache = cache.drop_duplicates(subset=["semantic_key"], keep="last").reset_index(drop=True)
            self.store.save_frame("semantic_cache", cache, metadata={"provider": "heuristic-cache"})
            cached = keyed.merge(cache, on="semantic_key", how="left")
            cache_hit = keyed["semantic_key"].isin(uncached["semantic_key"]).map(lambda value: not value)

        cached["semantic_cache_hit"] = cache_hit.values
        return cached

    def _heuristic_enrichment(
        self,
        semantic_key: str,
        text: str,
        sentiment_label: str,
    ) -> dict[str, Any]:
        lowered = text.lower()
        topic = self._pick_bucket(lowered, TOPIC_KEYWORDS, default="other")
        policy_bucket = self._pick_bucket(lowered, POLICY_BUCKET_KEYWORDS, default="other")
        market_hits = sum(term in lowered for term in MARKET_TERMS)
        urgent_hits = sum(term in lowered for term in URGENT_TERMS)
        exclamations = lowered.count("!")
        caps_ratio = 0.0
        letters = [char for char in text if char.isalpha()]
        if letters:
            caps_ratio = sum(char.isupper() for char in text if char.isalpha()) / len(letters)
        return {
            "semantic_key": semantic_key,
            "semantic_topic": topic,
            "semantic_policy_bucket": policy_bucket,
            "semantic_stance": sentiment_label,
            "semantic_market_relevance": min(1.0, market_hits / 3.0),
            "semantic_urgency": min(1.0, urgent_hits * 0.25 + exclamations * 0.05 + caps_ratio * 0.5),
            "semantic_provider": "heuristic-cache",
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
