"""Centralized configuration management.

All environment variables and defaults are accessed through this module,
ensuring type safety and single-point-of-change for configuration values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables."""

    openai_api_key: str | None
    openai_model: str
    tavily_api_key: str | None
    max_context_chars: int
    web_search_recency_days: int
    web_search_max_results: int

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_tavily_key(self) -> bool:
        return bool(self.tavily_api_key)


# Module-level singleton, lazily initialized
_config: Config | None = None


def load_config(env_path: Path | None = None) -> Config:
    """Load configuration from environment variables.

    Args:
        env_path: Optional path to .env file. Defaults to ".env" in cwd.

    Returns:
        Config instance with all settings loaded.
    """
    global _config

    if env_path is None:
        env_path = Path(".env")

    if env_path.exists():
        load_dotenv(env_path)

    _config = Config(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "60000")),
        web_search_recency_days=int(os.getenv("WEB_SEARCH_RECENCY_DAYS", "365")),
        web_search_max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5")),
    )
    return _config


def get_config() -> Config:
    """Get the current configuration, loading if not already loaded."""
    global _config
    if _config is None:
        return load_config()
    return _config


# Agent-specific limits (not from env, but centralized here for clarity)
SOLUTION_DESIGNER_MAX_SEARCH_CALLS = 2
SOLUTION_DESIGNER_MAX_ITERATIONS = 4

PROTOTYPE_BUILDER_MAX_SEARCH_CALLS = 3
PROTOTYPE_BUILDER_MAX_ITERATIONS = 6

LLM_TEMPERATURE = 0.2

