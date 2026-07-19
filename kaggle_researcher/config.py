from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBED_DIM = 1024
DEFAULT_MAX_EMBED_BATCH_SIZE = 8

from dotenv import load_dotenv

load_dotenv(".env")

class ConfigError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class Settings:
    deepseek_api_key: str
    deepseek_v4_pro: str = "deepseek-v4-pro"
    deepseek_v4_flash: str = "deepseek-v4-flash"
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dim: int = DEFAULT_EMBED_DIM
    max_embed_batch_size: int = DEFAULT_MAX_EMBED_BATCH_SIZE
    pg_dsn: str = "postgresql://researcher:researcher@localhost:5432/kaggle_research"
    top_k: int = 10
    max_notebooks: int = 20
    max_papers: int = 15
    max_repos: int = 10
    pdf_cache_dir: str = "./data/pdfs"
    kaggle_username: str | None = None
    kaggle_key: str | None = None
    github_token: str | None = None
    source_registry_enabled: bool = True
    source_refresh_mode: str = "auto"
    source_search_ttl_kaggle_hours: int = 24
    source_search_ttl_github_hours: int = 24
    source_search_ttl_arxiv_hours: int = 168
    source_search_ttl_papers_with_code_hours: int = 168
    source_cache_allow_stale_offline: bool = True
    source_content_dir: str = "./data/source_content"
    source_artifact_dir: str = "./data/source_artifacts"

    def source_search_ttls(self) -> dict[str, timedelta]:
        return {
            "kaggle": timedelta(hours=self.source_search_ttl_kaggle_hours),
            "github": timedelta(hours=self.source_search_ttl_github_hours),
            "arxiv": timedelta(hours=self.source_search_ttl_arxiv_hours),
            "papers_with_code": timedelta(hours=self.source_search_ttl_papers_with_code_hours),
        }


def load_config() -> Settings:
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise ConfigError("DEEPSEEK_API_KEY is required")

    return Settings(
        deepseek_api_key=deepseek_api_key,
        deepseek_v4_pro="deepseek-v4-pro",
        deepseek_v4_flash="deepseek-v4-flash",
        embed_model=os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL),
        embed_dim=_get_positive_int_env("EMBED_DIM", DEFAULT_EMBED_DIM),
        max_embed_batch_size=_get_positive_int_env(
            "MAX_EMBED_BATCH_SIZE",
            DEFAULT_MAX_EMBED_BATCH_SIZE,
        ),
        pg_dsn=os.getenv("PG_DSN", "postgresql://researcher:researcher@localhost:5432/kaggle_research"),
        top_k=10,
        max_notebooks=20,
        max_papers=15,
        max_repos=10,
        pdf_cache_dir="./data/pdfs",
        kaggle_username=os.getenv("KAGGLE_USERNAME"),
        kaggle_key=os.getenv("KAGGLE_KEY"),
        github_token=os.getenv("GITHUB_TOKEN"),
        source_registry_enabled=_get_bool_env("SOURCE_REGISTRY_ENABLED", True),
        source_refresh_mode=_get_choice_env("SOURCE_REFRESH_MODE", "auto", {"auto", "always", "never"}),
        source_search_ttl_kaggle_hours=_get_positive_int_env("SOURCE_SEARCH_TTL_KAGGLE_HOURS", 24),
        source_search_ttl_github_hours=_get_positive_int_env("SOURCE_SEARCH_TTL_GITHUB_HOURS", 24),
        source_search_ttl_arxiv_hours=_get_positive_int_env("SOURCE_SEARCH_TTL_ARXIV_HOURS", 168),
        source_search_ttl_papers_with_code_hours=_get_positive_int_env("SOURCE_SEARCH_TTL_PAPERS_WITH_CODE_HOURS", 168),
        source_cache_allow_stale_offline=_get_bool_env("SOURCE_CACHE_ALLOW_STALE_OFFLINE", True),
        source_content_dir=os.getenv("SOURCE_CONTENT_DIR", "./data/source_content"),
        source_artifact_dir=os.getenv("SOURCE_ARTIFACT_DIR", "./data/source_artifacts"),
    )


def _get_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc

    if value <= 0:
        raise ConfigError(f"{name} must be a positive integer")

    return value


def _get_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean")


def _get_choice_env(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        raise ConfigError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return value
