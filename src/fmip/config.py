"""Application configuration loaded from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

TRADING_DAYS_PER_YEAR = 252


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Values come from environment variables with safe defaults."""

    database_url: str
    default_start_date: str
    default_end_date: str
    risk_free_rate: float
    zscore_threshold: float
    rolling_window: int
    isolation_contamination: float
    log_level: str

    @property
    def database_configured(self) -> bool:
        return bool(self.database_url)


def load_settings() -> Settings:
    """Build a Settings object from the current environment."""
    return Settings(
        database_url=_env_str("DATABASE_URL", ""),
        default_start_date=_env_str("DEFAULT_START_DATE", "2018-01-01"),
        default_end_date=_env_str("DEFAULT_END_DATE", ""),
        risk_free_rate=_env_float("RISK_FREE_RATE", 0.04),
        zscore_threshold=_env_float("ZSCORE_THRESHOLD", 3.0),
        rolling_window=int(_env_float("ROLLING_WINDOW", 60)),
        isolation_contamination=_env_float("ISOLATION_CONTAMINATION", 0.01),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
    )
