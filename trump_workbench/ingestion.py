from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from .config import AppSettings
from .contracts import NORMALIZED_POST_COLUMNS, SourceAdapter
from .sentiment import add_sentiment_scores
from .utils import (
    clean_text,
    first_matching_column,
    infer_author_is_trump,
    infer_mentions_trump,
    normalize_boolean,
    normalize_column_lookup,
    parse_timestamp_to_eastern,
    read_csv_bytes,
    stable_text_id,
)

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TrumpTradingWorkbench/2.0; +https://example.invalid)",
}

TRUTH_ARCHIVE_URLS = [
    "https://ix.cnn.io/data/truth-social/truth_archive.csv",
    "https://stilesdata.com/trump-truth-social-archive/truth_archive.csv",
    "https://raw.githubusercontent.com/stiles/trump-truth-social-archive/main/data/truth_archive.csv",
]


def _request_text(url: str, timeout: int = 45) -> tuple[str, dict[str, str]]:
    response = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text, dict(response.headers)


def _ensure_normalized_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in NORMALIZED_POST_COLUMNS:
        if column not in out.columns:
            if column in {"author_is_trump", "is_reshare", "has_media", "mentions_trump"}:
                out[column] = False
            elif column in {
                "replies_count",
                "reblogs_count",
                "favourites_count",
            }:
                out[column] = 0
            elif column in {"engagement_score", "sentiment_score"}:
                out[column] = 0.0
            else:
                out[column] = ""

    out["post_timestamp"] = out["post_timestamp"].map(parse_timestamp_to_eastern)
    out["author_is_trump"] = normalize_boolean(out["author_is_trump"], default=False)
    out["is_reshare"] = normalize_boolean(out["is_reshare"], default=False)
    out["has_media"] = normalize_boolean(out["has_media"], default=False)
    out["mentions_trump"] = normalize_boolean(out["mentions_trump"], default=False)
    for column in ["replies_count", "reblogs_count", "favourites_count"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    out["engagement_score"] = (
        out["replies_count"] + out["reblogs_count"] + out["favourites_count"]
    ).astype(float)
    out["cleaned_text"] = out["cleaned_text"].fillna("").astype(str)
    out["raw_text"] = out["raw_text"].fillna("").astype(str)
    out["post_url"] = out["post_url"].fillna("").astype(str)
    out["post_id"] = out["post_id"].fillna("").astype(str)
    out["author_account_id"] = out["author_account_id"].fillna("").astype(str)
    out["author_handle"] = out["author_handle"].fillna("").astype(str)
    out["author_display_name"] = out["author_display_name"].fillna("").astype(str)
    out["source_platform"] = out["source_platform"].fillna("").astype(str)
    out["source_type"] = out["source_type"].fillna("").astype(str)
    out["source_provenance"] = out["source_provenance"].fillna("").astype(str)
    out = out.dropna(subset=["post_timestamp"]).copy()
    out = out.loc[out["post_timestamp"].dt.tz_convert("America/New_York") >= pd.Timestamp("2025-01-20", tz="America/New_York")]

    blank_ids = out["post_id"].str.strip() == ""
    out.loc[blank_ids, "post_id"] = out.loc[blank_ids].apply(
        lambda row: stable_text_id(
            row["source_platform"],
            row["author_handle"],
            row["post_timestamp"],
            row["cleaned_text"],
        ),
        axis=1,
    )
    blank_accounts = out["author_account_id"].str.strip() == ""
    out.loc[blank_accounts, "author_account_id"] = out.loc[blank_accounts].apply(
        lambda row: stable_text_id(row["author_handle"], row["author_display_name"], row["source_platform"]),
        axis=1,
    )

    dedupe_key = np.where(
        out["post_url"].str.strip() != "",
        out["post_url"].str.strip(),
        out["source_platform"].astype(str)
        + "|"
        + out["author_account_id"].astype(str)
        + "|"
        + out["post_timestamp"].astype(str)
        + "|"
        + out["cleaned_text"].astype(str),
    )
    out["dedupe_key"] = dedupe_key
    out = out.drop_duplicates(subset=["dedupe_key"], keep="last").drop(columns=["dedupe_key"])
    return out.sort_values(["post_timestamp", "source_platform", "author_handle"]).reset_index(drop=True)


def _series_or_default(raw_df: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in raw_df.columns:
        return raw_df[column]
    return pd.Series([default] * len(raw_df))


@dataclass
class TruthSocialArchiveAdapter(SourceAdapter):
    settings: AppSettings
    name: str = "Truth Social archive"

    def fetch_history(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        errors: list[str] = []
        text: Optional[str] = None
        headers: dict[str, str] = {}
        source_url = ""
        for url in TRUTH_ARCHIVE_URLS:
            try:
                text, headers = _request_text(url)
                source_url = url
                self.settings.truth_cache_file.write_text(text, encoding="utf-8")
                break
            except Exception as exc:
                errors.append(f"{url}: {exc}")

        if text is None and self.settings.truth_cache_file.exists():
            text = self.settings.truth_cache_file.read_text(encoding="utf-8")
            source_url = f"local cache ({self.settings.truth_cache_file.name})"

        if text is None:
            raise RuntimeError("Unable to load Truth Social archive. " + " | ".join(errors))

        raw_df = pd.read_csv(io.StringIO(text))
        raw_df = raw_df.rename(
            columns={
                "created_at": "raw_timestamp",
                "content": "raw_text",
                "url": "raw_url",
                "media": "raw_media",
            },
        )
        if "raw_timestamp" not in raw_df.columns:
            raise RuntimeError("Truth Social archive did not include a created_at column.")

        posts = pd.DataFrame(
            {
                "source_platform": "Truth Social",
                "source_type": "truth_archive",
                "author_account_id": "donald-trump",
                "author_handle": "realDonaldTrump",
                "author_display_name": "Donald Trump",
                "author_is_trump": True,
                "post_id": _series_or_default(raw_df, "id", "").fillna("").astype(str),
                "post_url": _series_or_default(raw_df, "raw_url", "").fillna("").astype(str),
                "post_timestamp": raw_df["raw_timestamp"].map(parse_timestamp_to_eastern),
                "raw_text": _series_or_default(raw_df, "raw_text", "").fillna("").astype(str),
                "cleaned_text": [
                    clean_text(text_value, media_value)
                    for text_value, media_value in zip(
                        _series_or_default(raw_df, "raw_text", ""),
                        _series_or_default(raw_df, "raw_media", ""),
                    )
                ],
                "is_reshare": False,
                "has_media": _series_or_default(raw_df, "raw_media", "").fillna("").astype(str).str.strip() != "",
                "replies_count": _series_or_default(raw_df, "replies_count", 0),
                "reblogs_count": _series_or_default(raw_df, "reblogs_count", 0),
                "favourites_count": _series_or_default(raw_df, "favourites_count", 0),
                "mentions_trump": False,
                "source_provenance": source_url,
            },
        )
        posts = _ensure_normalized_schema(posts)
        posts, sentiment_meta = add_sentiment_scores(posts)
        meta = {
            "source": self.name,
            "provenance": source_url,
            "last_modified": headers.get("Last-Modified"),
            "post_count": int(len(posts)),
            "coverage_start": posts["post_timestamp"].min() if not posts.empty else pd.NaT,
            "coverage_end": posts["post_timestamp"].max() if not posts.empty else pd.NaT,
            "sentiment_backend": sentiment_meta["backend"],
        }
        return posts, meta

    def fetch_since(
        self,
        last_cursor: Optional[pd.Timestamp],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        posts, meta = self.fetch_history()
        if last_cursor is not None:
            posts = posts.loc[posts["post_timestamp"] > last_cursor].copy()
        meta["incremental"] = True
        meta["post_count"] = int(len(posts))
        return posts.reset_index(drop=True), meta


@dataclass
class XCsvAdapter(SourceAdapter):
    settings: AppSettings
    name: str
    provenance: str
    raw_bytes: bytes
    source_type: str = "x_csv"

    def _parse_x_frame(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        lookup = normalize_column_lookup(raw_df.columns)
        timestamp_col = first_matching_column(
            lookup,
            ["timestamp", "created_at", "datetime", "date", "time", "tweet_timestamp"],
        )
        text_col = first_matching_column(
            lookup,
            ["text", "full_text", "tweet_text", "content", "body", "message"],
        )
        url_col = first_matching_column(lookup, ["url", "tweet_url", "link", "status_url", "permalink"])
        post_id_col = first_matching_column(lookup, ["post_id", "tweet_id", "status_id", "id"])
        is_retweet_col = first_matching_column(
            lookup,
            ["is_retweet", "retweet", "is_reshare", "is_repost", "retweeted"],
        )
        author_id_col = first_matching_column(
            lookup,
            ["author_id", "account_id", "user_id", "accountid", "authorid"],
        )
        handle_col = first_matching_column(
            lookup,
            ["author_handle", "handle", "username", "screen_name", "user_handle", "account_handle"],
        )
        display_col = first_matching_column(
            lookup,
            ["author_name", "display_name", "name", "user_name", "account_name"],
        )
        replies_col = first_matching_column(lookup, ["replies_count", "reply_count"])
        reblogs_col = first_matching_column(lookup, ["reblogs_count", "retweet_count", "reposts_count"])
        favs_col = first_matching_column(lookup, ["favourites_count", "favorite_count", "like_count", "likes"])
        mentions_col = first_matching_column(lookup, ["mentions_trump", "mentions_donald_trump"])

        if timestamp_col is None or text_col is None:
            raise RuntimeError(
                "X CSV must include at least a timestamp column and a text column.",
            )

        handles = raw_df[handle_col] if handle_col is not None else ""
        display_names = raw_df[display_col] if display_col is not None else ""
        cleaned_text = raw_df[text_col].map(clean_text)

        posts = pd.DataFrame(
            {
                "source_platform": "X",
                "source_type": self.source_type,
                "author_account_id": raw_df[author_id_col] if author_id_col is not None else "",
                "author_handle": handles,
                "author_display_name": display_names,
                "author_is_trump": [
                    infer_author_is_trump(handle, display)
                    for handle, display in zip(handles if isinstance(handles, pd.Series) else ["" for _ in range(len(raw_df))], display_names if isinstance(display_names, pd.Series) else ["" for _ in range(len(raw_df))])
                ],
                "post_id": raw_df[post_id_col] if post_id_col is not None else "",
                "post_url": raw_df[url_col] if url_col is not None else "",
                "post_timestamp": raw_df[timestamp_col].map(parse_timestamp_to_eastern),
                "raw_text": raw_df[text_col].fillna("").astype(str),
                "cleaned_text": cleaned_text,
                "is_reshare": normalize_boolean(raw_df[is_retweet_col], default=False)
                if is_retweet_col is not None
                else cleaned_text.str.startswith("RT @"),
                "has_media": False,
                "replies_count": raw_df[replies_col] if replies_col is not None else 0,
                "reblogs_count": raw_df[reblogs_col] if reblogs_col is not None else 0,
                "favourites_count": raw_df[favs_col] if favs_col is not None else 0,
                "mentions_trump": normalize_boolean(raw_df[mentions_col], default=False)
                if mentions_col is not None
                else cleaned_text.map(infer_mentions_trump),
                "source_provenance": self.provenance,
            },
        )
        posts = _ensure_normalized_schema(posts)
        trump_authored = posts["author_is_trump"]
        posts.loc[trump_authored, "mentions_trump"] = False
        posts, _ = add_sentiment_scores(posts)
        return posts

    def fetch_history(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        raw_df = read_csv_bytes(self.raw_bytes)
        posts = self._parse_x_frame(raw_df)
        meta = {
            "source": self.name,
            "provenance": self.provenance,
            "post_count": int(len(posts)),
            "coverage_start": posts["post_timestamp"].min() if not posts.empty else pd.NaT,
            "coverage_end": posts["post_timestamp"].max() if not posts.empty else pd.NaT,
        }
        return posts, meta

    def fetch_since(
        self,
        last_cursor: Optional[pd.Timestamp],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        posts, meta = self.fetch_history()
        if last_cursor is not None:
            posts = posts.loc[posts["post_timestamp"] > last_cursor].copy()
        meta["incremental"] = True
        meta["post_count"] = int(len(posts))
        return posts.reset_index(drop=True), meta

    @classmethod
    def from_local_file(
        cls,
        settings: AppSettings,
        path: str,
        name: str,
    ) -> "XCsvAdapter":
        raw_bytes = Path(path).read_bytes()
        return cls(settings=settings, name=name, provenance=f"file:{path}", raw_bytes=raw_bytes)

    @classmethod
    def from_remote_url(
        cls,
        settings: AppSettings,
        url: str,
        name: str,
    ) -> "XCsvAdapter":
        response = requests.get(url, headers=HTTP_HEADERS, timeout=45)
        response.raise_for_status()
        return cls(settings=settings, name=name, provenance=url, raw_bytes=response.content)


class IngestionService:
    def run_refresh(
        self,
        adapters: list[SourceAdapter],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        manifest_rows: list[dict[str, Any]] = []
        for adapter in adapters:
            posts, meta = adapter.fetch_history()
            frames.append(posts)
            meta = dict(meta)
            meta.setdefault("status", "ok")
            meta.setdefault("detail", "")
            manifest_rows.append(meta)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NORMALIZED_POST_COLUMNS)
        combined = _ensure_normalized_schema(combined) if not combined.empty else pd.DataFrame(columns=NORMALIZED_POST_COLUMNS)
        manifest = pd.DataFrame(manifest_rows)
        return combined, manifest

    def run_incremental_refresh(
        self,
        adapters: list[SourceAdapter],
        last_cursor: Optional[pd.Timestamp],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        manifest_rows: list[dict[str, Any]] = []
        for adapter in adapters:
            posts, meta = adapter.fetch_since(last_cursor)
            frames.append(posts)
            meta = dict(meta)
            meta.setdefault("status", "ok")
            meta.setdefault("detail", "")
            manifest_rows.append(meta)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NORMALIZED_POST_COLUMNS)
        combined = _ensure_normalized_schema(combined) if not combined.empty else pd.DataFrame(columns=NORMALIZED_POST_COLUMNS)
        manifest = pd.DataFrame(manifest_rows)
        return combined, manifest
