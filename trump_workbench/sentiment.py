from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .utils import clean_text

NEUTRAL_SENTIMENT_THRESHOLD = 0.05

FALLBACK_POSITIVE_LEXICON = {
    "agreement": 1.3,
    "approval": 1.2,
    "beautiful": 1.5,
    "best": 1.6,
    "boom": 1.6,
    "booming": 1.7,
    "bull": 1.4,
    "bullish": 1.6,
    "deal": 1.3,
    "deals": 1.2,
    "excellent": 1.6,
    "fair": 0.8,
    "fantastic": 1.8,
    "good": 1.0,
    "great": 1.4,
    "growth": 1.4,
    "historic": 1.2,
    "incredible": 1.7,
    "lower": 0.8,
    "peace": 1.3,
    "productive": 1.1,
    "record": 1.2,
    "records": 1.2,
    "secure": 1.1,
    "safe": 1.0,
    "strong": 1.3,
    "success": 1.5,
    "successful": 1.6,
    "tremendous": 1.8,
    "victory": 1.6,
    "win": 1.4,
    "winning": 1.5,
}

FALLBACK_NEGATIVE_LEXICON = {
    "attack": 1.4,
    "attacks": 1.4,
    "bad": 1.0,
    "conflict": 1.5,
    "corrupt": 1.5,
    "crisis": 1.7,
    "crooked": 1.3,
    "danger": 1.3,
    "dangerous": 1.5,
    "decline": 1.3,
    "disaster": 1.8,
    "enemy": 1.2,
    "enemies": 1.2,
    "failed": 1.5,
    "failure": 1.6,
    "falling": 1.3,
    "fraud": 1.4,
    "illegal": 1.3,
    "inflation": 1.6,
    "loss": 1.4,
    "losses": 1.4,
    "radical": 1.0,
    "recession": 1.8,
    "risk": 1.2,
    "tariff": 1.4,
    "tariffs": 1.5,
    "terrible": 1.7,
    "threat": 1.4,
    "threats": 1.4,
    "unfair": 1.0,
    "war": 1.8,
    "weak": 1.2,
    "worst": 1.7,
}


@lru_cache(maxsize=1)
def get_sentiment_backend() -> dict[str, Any]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

        return {
            "name": "VADER compound sentiment (-1 to +1)",
            "backend": "vaderSentiment",
            "analyzer": SentimentIntensityAnalyzer(),
            "fallback": False,
        }
    except Exception:
        return {
            "name": "Fallback keyword-weighted sentiment (-1 to +1)",
            "backend": "fallback_lexicon",
            "analyzer": None,
            "fallback": True,
        }


def tokenize_for_sentiment(text: str) -> list[str]:
    normalized = "".join(char.lower() if char.isalnum() or char in "'-" else " " for char in text)
    return [token.strip("'-") for token in normalized.split() if token.strip("'-")]


def fallback_sentiment_score(text: str) -> float:
    tokens = tokenize_for_sentiment(text)
    if not tokens:
        return 0.0

    negations = {
        "not",
        "no",
        "never",
        "none",
        "can't",
        "won't",
        "isn't",
        "aren't",
        "don't",
        "didn't",
        "shouldn't",
        "couldn't",
        "wouldn't",
        "n't",
    }
    boosters = {
        "very": 1.25,
        "extremely": 1.5,
        "highly": 1.35,
        "really": 1.2,
        "so": 1.1,
        "too": 1.1,
    }

    score = 0.0
    for idx, token in enumerate(tokens):
        base = 0.0
        if token in FALLBACK_POSITIVE_LEXICON:
            base = FALLBACK_POSITIVE_LEXICON[token]
        elif token in FALLBACK_NEGATIVE_LEXICON:
            base = -FALLBACK_NEGATIVE_LEXICON[token]

        if base != 0.0 and idx > 0:
            prev = tokens[idx - 1]
            if prev in negations:
                base *= -0.85
            elif prev in boosters:
                base *= boosters[prev]
        score += base

    normalized = score / max(np.sqrt(len(tokens)) * 2.4, 1.0)
    return float(np.clip(normalized, -1.0, 1.0))


def score_post_sentiment(text: str) -> float:
    backend = get_sentiment_backend()
    analyzer = backend.get("analyzer")
    if analyzer is not None:
        try:
            return float(analyzer.polarity_scores(text).get("compound", 0.0))
        except Exception:
            pass
    return fallback_sentiment_score(text)


def sentiment_label(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score >= NEUTRAL_SENTIMENT_THRESHOLD:
        return "positive"
    if score <= -NEUTRAL_SENTIMENT_THRESHOLD:
        return "negative"
    return "neutral"


def add_sentiment_scores(posts: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    backend = get_sentiment_backend()
    out = posts.copy()
    if out.empty:
        out["sentiment_score"] = pd.Series(dtype=float)
        out["sentiment_label"] = pd.Series(dtype=str)
        return out, backend

    cleaned = out["cleaned_text"].fillna("").astype(str).map(clean_text)
    scores = cleaned.map(score_post_sentiment)
    out["sentiment_score"] = pd.to_numeric(scores, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    out["sentiment_label"] = out["sentiment_score"].map(sentiment_label)
    return out, backend
