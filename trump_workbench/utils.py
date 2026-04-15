from __future__ import annotations

import hashlib
import html
import io
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import EASTERN

MOJIBAKE_REPLACEMENTS = {
    "‚Äì": "–",
    "‚Äî": "—",
    "‚Äú": "“",
    "‚Äù": "”",
    "‚Äô": "’",
    "‚Äôs": "’s",
    "‚Ä¶": "…",
    "Ã©": "é",
    "Ã¨": "è",
    "Ã¼": "ü",
    "Â": "",
}

TRUMP_MENTION_PATTERNS = [
    r"\btrump\b",
    r"\bdonald trump\b",
    r"@realdonaldtrump",
    r"\bpotus\b",
    r"\bpresident trump\b",
]


def clean_text(value: object, media_hint: object = None) -> str:
    text = "" if pd.isna(value) else str(value)
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = re.sub(r"\s+", " ", text).strip()

    media_text = "" if pd.isna(media_hint) else str(media_hint).strip()
    if not text and media_text:
        return "[media-only post]"
    if not text:
        return "[empty post]"
    return text


def truncate_text(value: object, max_chars: int = 200) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def parse_timestamp_to_eastern(
    value: object,
    assume_tz: str = EASTERN,
) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return pd.NaT
    try:
        if ts.tzinfo is None:
            return ts.tz_localize(assume_tz)
        return ts.tz_convert(assume_tz)
    except Exception:
        return pd.NaT


def normalize_boolean(series: pd.Series, default: bool = False) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series.fillna(default)

    truthy = {"1", "true", "yes", "y", "t"}
    falsy = {"0", "false", "no", "n", "f", "nan", "none", "", "<na>", "null"}

    def parse(value: object) -> bool:
        if pd.isna(value):
            return False
        lowered = str(value).strip().lower()
        if lowered in truthy:
            return True
        if lowered in falsy:
            return False
        return default

    return series.map(parse)


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to parse CSV bytes: {last_error}")


def normalize_column_lookup(columns: Iterable[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for original in columns:
        key = re.sub(r"[^a-z0-9]+", "", str(original).strip().lower())
        if key and key not in lookup:
            lookup[key] = str(original)
    return lookup


def first_matching_column(
    lookup: dict[str, str],
    candidates: Iterable[str],
) -> Optional[str]:
    normalized_candidates = [
        re.sub(r"[^a-z0-9]+", "", candidate.strip().lower())
        for candidate in candidates
    ]
    for candidate in normalized_candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def infer_mentions_trump(text: object) -> bool:
    cleaned = clean_text(text)
    lowered = cleaned.lower()
    return any(re.search(pattern, lowered) for pattern in TRUMP_MENTION_PATTERNS)


def infer_author_is_trump(handle: object, display_name: object) -> bool:
    handle_text = "" if pd.isna(handle) else str(handle).strip().lower().lstrip("@")
    display_text = "" if pd.isna(display_name) else str(display_name).strip().lower()
    return handle_text == "realdonaldtrump" or "donald trump" in display_text


def stable_text_id(*parts: object) -> str:
    joined = "||".join("" if pd.isna(part) else str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def ensure_tz_naive_date(value: pd.Timestamp | object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(EASTERN).tz_localize(None)
    return ts.normalize()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def fmt_pct(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:,.2f}%"


def fmt_score(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):+.4f}"


def business_minutes_until_close(ts: pd.Timestamp) -> float:
    local = ts.tz_convert(EASTERN)
    close = pd.Timestamp(f"{local.date()} 16:00", tz=EASTERN)
    minutes = (close - local).total_seconds() / 60.0
    return float(max(minutes, 0.0))
