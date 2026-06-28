from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBED_DIM = 1024
DEFAULT_MAX_EMBED_BATCH_SIZE = 8


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
