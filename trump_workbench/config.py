from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import pandas as pd

APP_TITLE = "Trump Social Trading Research Workbench"
EASTERN = "America/New_York"
CURRENT_TERM_START = pd.Timestamp("2025-01-20")
DEFAULT_ETF_SYMBOLS = ("SPY", "QQQ", "XLK", "XLF", "XLE", "SMH")


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value.strip())
    except ValueError:
        return default


@dataclass
class AppSettings:
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
    )
    title: str = APP_TITLE
    timezone: str = EASTERN
    term_start: pd.Timestamp = CURRENT_TERM_START
    default_poll_seconds: int = 300
    state_root_override: str = field(default_factory=lambda: os.getenv("ALLCAPS_STATE_DIR", "").strip())
    public_mode: bool = field(default_factory=lambda: _env_flag("ALLCAPS_PUBLIC_MODE", False))
    admin_password: str = field(default_factory=lambda: os.getenv("ALLCAPS_ADMIN_PASSWORD", ""))
    auto_bootstrap_on_start: bool = field(default_factory=lambda: _env_flag("ALLCAPS_AUTO_BOOTSTRAP_ON_START", True))
    scheduler_enabled: bool = field(default_factory=lambda: _env_flag("ALLCAPS_SCHEDULER_ENABLED", False))
    scheduler_incremental_minutes: int = field(default_factory=lambda: max(5, _env_int("ALLCAPS_SCHEDULER_INCREMENTAL_MINUTES", 30)))
    scheduler_full_hour: int = field(default_factory=lambda: min(23, max(0, _env_int("ALLCAPS_SCHEDULER_FULL_HOUR", 3))))
    scheduler_full_minute: int = field(default_factory=lambda: min(59, max(0, _env_int("ALLCAPS_SCHEDULER_FULL_MINUTE", 0))))
    scheduler_loop_seconds: int = field(default_factory=lambda: max(15, _env_int("ALLCAPS_SCHEDULER_LOOP_SECONDS", 60)))
    api_cors_origins_raw: str = field(
        default_factory=lambda: os.getenv(
            "ALLCAPS_API_CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        ).strip(),
    )
    remote_x_csv_url: str = field(
        default_factory=lambda: os.getenv("ALLCAPS_REMOTE_X_CSV_URL", os.getenv("TRUMP_X_CSV_URL", "")).strip(),
    )

    def __post_init__(self) -> None:
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workbench_dir.mkdir(parents=True, exist_ok=True)
        self.lake_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    @property
    def code_root(self) -> Path:
        return self.base_dir

    @property
    def state_root(self) -> Path:
        if self.state_root_override:
            return Path(self.state_root_override).expanduser().resolve()
        return self.base_dir

    @property
    def cache_dir(self) -> Path:
        return self.state_root / ".cache"

    @property
    def workbench_dir(self) -> Path:
        return self.state_root / ".workbench"

    @property
    def lake_dir(self) -> Path:
        return self.workbench_dir / "lake"

    @property
    def artifact_dir(self) -> Path:
        return self.workbench_dir / "artifacts"

    @property
    def db_path(self) -> Path:
        return self.workbench_dir / "workbench.duckdb"

    @property
    def truth_cache_file(self) -> Path:
        return self.cache_dir / "truth_archive.csv"

    @property
    def local_x_path(self) -> Path:
        return self.code_root / "data" / "realDonaldTrump_x_current_term.csv"

    @property
    def local_mentions_path(self) -> Path:
        return self.code_root / "data" / "influential_x_mentions.csv"

    @property
    def x_template_path(self) -> Path:
        return self.code_root / "templates" / "x_posts_template.csv"

    @property
    def mention_template_path(self) -> Path:
        return self.code_root / "templates" / "x_mentions_template.csv"

    @property
    def scheduler_lock_path(self) -> Path:
        return self.workbench_dir / "scheduler_refresh.lock"

    @property
    def api_cors_origins(self) -> tuple[str, ...]:
        return tuple(
            origin.strip()
            for origin in self.api_cors_origins_raw.split(",")
            if origin.strip()
        )
