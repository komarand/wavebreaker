from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class Settings:
    deepseek_api_key: str | None = None
    deepseek_v4_pro: str = "deepseek-v4-pro"
    deepseek_v4_flash: str = "deepseek-v4-flash"
    vllm_base_url: str = "http://localhost:8000/v1"
    embed_model: str = "Qwen/Qwen3-Embedding-4B"
    pg_dsn: str = "postgresql://researcher:researcher@localhost:5432/kaggle_research"
    kaggle_username: str | None = None
    kaggle_key: str | None = None
    github_token: str | None = None


def load_config() -> Settings:
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_v4_pro=os.getenv("DEEPSEEK_V4_PRO", "deepseek-v4-pro"),
        deepseek_v4_flash=os.getenv("DEEPSEEK_V4_FLASH", "deepseek-v4-flash"),
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        embed_model=os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B"),
        pg_dsn=os.getenv("PG_DSN", "postgresql://researcher:researcher@localhost:5432/kaggle_research"),
        kaggle_username=os.getenv("KAGGLE_USERNAME"),
        kaggle_key=os.getenv("KAGGLE_KEY"),
        github_token=os.getenv("GITHUB_TOKEN"),
    )
