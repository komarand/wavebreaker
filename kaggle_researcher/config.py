from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class Settings:
    deepseek_api_key: str
    deepseek_v4_pro: str = "deepseek-v4-pro"
    deepseek_v4_flash: str = "deepseek-v4-flash"
    vllm_base_url: str = "http://localhost:8000/v1"
    embed_model: str = "Qwen/Qwen3-Embedding-4B"
    embed_dim: int = 2560
    max_embed_batch_size: int = 64
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
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        embed_model="Qwen/Qwen3-Embedding-4B",
        embed_dim=2560,
        max_embed_batch_size=64,
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
