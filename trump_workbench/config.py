from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

APP_TITLE = "Trump Social Trading Research Workbench"
EASTERN = "America/New_York"
CURRENT_TERM_START = pd.Timestamp("2025-01-20")
DEFAULT_ETF_SYMBOLS = ("SPY", "QQQ", "XLK", "XLF", "XLE", "SMH")


@dataclass
class AppSettings:
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
    )
    title: str = APP_TITLE
    timezone: str = EASTERN
    term_start: pd.Timestamp = CURRENT_TERM_START
    default_poll_seconds: int = 300

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workbench_dir.mkdir(parents=True, exist_ok=True)
        self.lake_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self.base_dir / ".cache"

    @property
    def workbench_dir(self) -> Path:
        return self.base_dir / ".workbench"

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
        return self.base_dir / "data" / "realDonaldTrump_x_current_term.csv"

    @property
    def local_mentions_path(self) -> Path:
        return self.base_dir / "data" / "influential_x_mentions.csv"

    @property
    def x_template_path(self) -> Path:
        return self.base_dir / "templates" / "x_posts_template.csv"

    @property
    def mention_template_path(self) -> Path:
        return self.base_dir / "templates" / "x_mentions_template.csv"
