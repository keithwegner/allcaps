from __future__ import annotations

from typing import Any, Mapping

from .config import AppSettings

ADMIN_SESSION_KEY = "allcaps_admin_authenticated"


def is_public_mode(settings: AppSettings) -> bool:
    return bool(settings.public_mode)


def is_admin_session(settings: AppSettings, session_state: Mapping[str, Any]) -> bool:
    if not is_public_mode(settings):
        return True
    return bool(session_state.get(ADMIN_SESSION_KEY, False))


def writes_enabled(settings: AppSettings, session_state: Mapping[str, Any]) -> bool:
    return is_admin_session(settings, session_state)


def verify_admin_password(settings: AppSettings, password: str) -> bool:
    if not is_public_mode(settings):
        return True
    expected = str(settings.admin_password or "")
    return bool(expected) and password == expected


def app_mode_label(settings: AppSettings, session_state: Mapping[str, Any]) -> str:
    if not is_public_mode(settings):
        return "private"
    if is_admin_session(settings, session_state):
        return "admin"
    return "public_read_only"
