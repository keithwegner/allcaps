from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import AppSettings
from .contracts import MANUAL_OVERRIDE_COLUMNS
from .discovery import DiscoveryService
from .storage import DuckDBStore


class DiscoveryAdminError(ValueError):
    """Raised when a Discovery override mutation request is invalid."""


@dataclass(frozen=True)
class DiscoveryOverrideMutation:
    account_id: str
    handle: str = ""
    display_name: str = ""
    source_platform: str = "X"
    action: str = "pin"
    effective_from: str = ""
    effective_to: str | None = None
    note: str = ""


def _normalized_posts_for_discovery(store: DuckDBStore, settings: AppSettings) -> pd.DataFrame:
    posts = store.read_frame("normalized_posts")
    if posts.empty:
        raise DiscoveryAdminError("Refresh datasets first so Discovery overrides have stored posts to evaluate.")

    out = posts.copy()
    for column, default in {
        "source_platform": "",
        "mentions_trump": False,
        "author_is_trump": False,
        "author_account_id": "",
        "author_handle": "",
        "author_display_name": "",
        "post_id": "",
        "engagement_score": 0.0,
        "sentiment_score": 0.0,
    }.items():
        if column not in out.columns:
            out[column] = default
    if "post_timestamp" not in out.columns:
        raise DiscoveryAdminError("Stored posts are missing post timestamps; refresh datasets before managing Discovery overrides.")
    out["post_timestamp"] = pd.to_datetime(out["post_timestamp"], errors="coerce", utc=True).dt.tz_convert(settings.timezone)
    out = out.dropna(subset=["post_timestamp"]).copy()
    if out.empty:
        raise DiscoveryAdminError("Stored posts do not contain valid timestamps; refresh datasets before managing Discovery overrides.")
    out["source_platform"] = out["source_platform"].fillna("").astype(str)
    out["mentions_trump"] = out["mentions_trump"].fillna(False).astype(bool)
    out["author_is_trump"] = out["author_is_trump"].fillna(False).astype(bool)
    return out


def _x_candidate_posts(posts: pd.DataFrame) -> pd.DataFrame:
    return posts.loc[
        (posts["source_platform"] == "X")
        & posts["mentions_trump"]
        & (~posts["author_is_trump"])
    ].copy()


def _parse_effective_date(value: str | None, field_name: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise DiscoveryAdminError(f"{field_name} must be a valid date.")
    return pd.Timestamp(timestamp).normalize()


def _validate_override_request(request: DiscoveryOverrideMutation) -> tuple[str, str, pd.Timestamp, pd.Timestamp | None]:
    account_id = str(request.account_id or "").strip()
    if not account_id:
        raise DiscoveryAdminError("account_id is required.")
    action = str(request.action or "").strip().lower()
    if action not in {"pin", "suppress"}:
        raise DiscoveryAdminError("action must be either pin or suppress.")
    effective_from = _parse_effective_date(request.effective_from, "effective_from")
    effective_to = _parse_effective_date(request.effective_to, "effective_to") if request.effective_to else None
    if effective_to is not None and effective_to <= effective_from:
        raise DiscoveryAdminError("effective_to must be after effective_from.")
    return account_id, action, effective_from, effective_to


def _load_overrides(store: DuckDBStore, discovery_service: DiscoveryService) -> pd.DataFrame:
    overrides = discovery_service.normalize_manual_overrides(store.read_frame("manual_account_overrides"))
    if overrides.empty:
        return pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)
    return overrides


def recompute_discovery_frames(
    *,
    settings: AppSettings,
    store: DuckDBStore,
    discovery_service: DiscoveryService,
    posts: pd.DataFrame | None = None,
    overrides: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    discovery_posts = posts if posts is not None else _normalized_posts_for_discovery(store, settings)
    normalized_overrides = discovery_service.normalize_manual_overrides(overrides if overrides is not None else store.read_frame("manual_account_overrides"))
    as_of = pd.to_datetime(discovery_posts["post_timestamp"], errors="coerce").dropna().max()
    if pd.isna(as_of):
        raise DiscoveryAdminError("Stored posts do not contain valid timestamps; refresh datasets before managing Discovery overrides.")
    tracked_accounts, rankings = discovery_service.refresh_accounts(
        posts=discovery_posts,
        existing_accounts=store.read_frame("tracked_accounts"),
        as_of=pd.Timestamp(as_of),
        manual_overrides=normalized_overrides,
    )
    store.save_frame("manual_account_overrides", normalized_overrides, metadata={"row_count": int(len(normalized_overrides))})
    store.save_frame("tracked_accounts", tracked_accounts, metadata={"row_count": int(len(tracked_accounts))})
    store.save_frame("account_rankings", rankings, metadata={"row_count": int(len(rankings))})
    return tracked_accounts, rankings


def create_discovery_override(
    *,
    settings: AppSettings,
    store: DuckDBStore,
    discovery_service: DiscoveryService,
    request: DiscoveryOverrideMutation,
) -> None:
    posts = _normalized_posts_for_discovery(store, settings)
    if _x_candidate_posts(posts).empty:
        raise DiscoveryAdminError("Discovery overrides require non-Trump X mention data.")
    account_id, action, effective_from, effective_to = _validate_override_request(request)
    overrides = _load_overrides(store, discovery_service)
    updated = discovery_service.add_manual_override(
        overrides=overrides,
        account_id=account_id,
        handle=str(request.handle or "").strip(),
        display_name=str(request.display_name or "").strip(),
        action=action,
        effective_from=effective_from,
        effective_to=effective_to,
        note=str(request.note or "").strip(),
        source_platform=str(request.source_platform or "X").strip() or "X",
    )
    recompute_discovery_frames(
        settings=settings,
        store=store,
        discovery_service=discovery_service,
        posts=posts,
        overrides=updated,
    )


def delete_discovery_override(
    *,
    settings: AppSettings,
    store: DuckDBStore,
    discovery_service: DiscoveryService,
    override_id: str,
) -> None:
    normalized_override_id = str(override_id or "").strip()
    if not normalized_override_id:
        raise DiscoveryAdminError("override_id is required.")
    posts = _normalized_posts_for_discovery(store, settings)
    overrides = _load_overrides(store, discovery_service)
    if overrides.empty or normalized_override_id not in set(overrides["override_id"].astype(str)):
        raise DiscoveryAdminError("Override was not found.")
    updated = discovery_service.remove_manual_override(overrides, normalized_override_id)
    recompute_discovery_frames(
        settings=settings,
        store=store,
        discovery_service=discovery_service,
        posts=posts,
        overrides=updated,
    )
