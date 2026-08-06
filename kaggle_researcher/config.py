from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBED_DIM = 1024
DEFAULT_MAX_EMBED_BATCH_SIZE = 8
DEFAULT_WRITEUPS_PER_COMPETITION = 10
DEFAULT_NOTEBOOK_CONCURRENCY = 2

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
    notebook_concurrency: int = DEFAULT_NOTEBOOK_CONCURRENCY
    max_papers: int = 15
    max_repos: int = 10
    pdf_cache_dir: str = "./data/pdfs"
    max_discussions: int = 200
    writeups_per_competition: int = DEFAULT_WRITEUPS_PER_COMPETITION
    max_context_tokens: int = 120_000
    max_sample_sub_bytes: int = 5_000_000
    meta_kaggle_dir: str | None = None
    run_budget_tokens: int | None = None
    kaggle_api_token: str | None = None
    kaggle_username: str | None = None
    kaggle_key: str | None = None
    github_token: str | None = None


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
        pg_dsn=os.getenv(
            "PG_DSN", "postgresql://researcher:researcher@localhost:5432/kaggle_research"
        ),
        top_k=_get_positive_int_env("TOP_K", 10),
        max_notebooks=_get_positive_int_env("MAX_NOTEBOOKS", 20),
        notebook_concurrency=get_notebook_concurrency(),
        max_papers=_get_positive_int_env("MAX_PAPERS", 15),
        max_repos=_get_positive_int_env("MAX_REPOS", 10),
        pdf_cache_dir=os.getenv("PDF_CACHE_DIR", "./data/pdfs"),
        max_discussions=_get_positive_int_env("MAX_DISCUSSIONS", 200),
        writeups_per_competition=get_writeups_per_competition(),
        max_context_tokens=_get_positive_int_env("MAX_CONTEXT_TOKENS", 120_000),
        max_sample_sub_bytes=_get_positive_int_env(
            "MAX_SAMPLE_SUB_BYTES",
            5_000_000,
        ),
        meta_kaggle_dir=os.getenv("META_KAGGLE_DIR"),
        run_budget_tokens=_get_optional_positive_int_env("RUN_BUDGET_TOKENS"),
        kaggle_api_token=os.getenv("KAGGLE_API_TOKEN"),
        kaggle_username=os.getenv("KAGGLE_USERNAME"),
        kaggle_key=os.getenv("KAGGLE_KEY"),
        github_token=os.getenv("GITHUB_TOKEN"),
    )


def get_writeups_per_competition() -> int:
    """Return the facts-only writeup cap without requiring model credentials."""
    return _get_positive_int_env(
        "WRITEUPS_PER_COMPETITION",
        DEFAULT_WRITEUPS_PER_COMPETITION,
    )


def get_notebook_concurrency() -> int:
    """Return the facts notebook download concurrency without model credentials."""
    return _get_positive_int_env(
        "NOTEBOOK_CONCURRENCY",
        DEFAULT_NOTEBOOK_CONCURRENCY,
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


def _get_optional_positive_int_env(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc

    if value <= 0:
        raise ConfigError(f"{name} must be a positive integer")

    return value
