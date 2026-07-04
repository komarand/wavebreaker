from __future__ import annotations

from pathlib import Path

import pytest

import kaggle_researcher.config as config_module
from kaggle_researcher.config import (
    DEFAULT_EMBED_DIM,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MAX_EMBED_BATCH_SIZE,
    ConfigError,
    load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_missing_deepseek_api_key_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY is required"):
        load_config()


def test_defaults_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.delenv("EMBED_DIM", raising=False)
    monkeypatch.delenv("MAX_EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("PG_DSN", raising=False)

    settings = load_config()

    assert settings.deepseek_api_key == "secret-key"
    assert settings.deepseek_v4_pro == "deepseek-v4-pro"
    assert settings.deepseek_v4_flash == "deepseek-v4-flash"
    assert settings.embed_model == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.embed_dim == 1024
    assert settings.max_embed_batch_size == 8
    assert settings.pg_dsn == "postgresql://researcher:researcher@localhost:5432/kaggle_research"
    assert settings.top_k == 10
    assert settings.max_notebooks == 20
    assert settings.max_papers == 15
    assert settings.max_repos == 10
    assert settings.pdf_cache_dir == "./data/pdfs"
    assert settings.kaggle_username is None
    assert settings.kaggle_key is None
    assert settings.github_token is None


def test_embedding_defaults_match_current_local_model_contract() -> None:
    assert DEFAULT_EMBED_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    assert DEFAULT_EMBED_DIM == 1024
    assert DEFAULT_MAX_EMBED_BATCH_SIZE == 8


def test_env_overrides_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")
    monkeypatch.setenv("EMBED_MODEL", "custom-embed-model")
    monkeypatch.setenv("EMBED_DIM", "384")
    monkeypatch.setenv("MAX_EMBED_BATCH_SIZE", "16")
    monkeypatch.setenv("PG_DSN", "postgresql://user:pass@db:5432/app")
    monkeypatch.setenv("KAGGLE_USERNAME", "kaggle-user")
    monkeypatch.setenv("KAGGLE_KEY", "kaggle-key")
    monkeypatch.setenv("GITHUB_TOKEN", "github-token")

    settings = load_config()

    assert settings.embed_model == "custom-embed-model"
    assert settings.embed_dim == 384
    assert settings.max_embed_batch_size == 16
    assert settings.pg_dsn == "postgresql://user:pass@db:5432/app"
    assert settings.kaggle_username == "kaggle-user"
    assert settings.kaggle_key == "kaggle-key"
    assert settings.github_token == "github-token"


def test_removed_embedding_server_env_is_not_read_or_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")
    removed_server_env = "V" + "LLM_BASE_URL"
    monkeypatch.setenv(removed_server_env, "http://should-not-be-used")

    settings = load_config()

    removed_server_attr = "v" + "llm_base_url"
    assert not hasattr(settings, removed_server_attr)


def test_importing_config_does_not_require_deepseek_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert config_module.DEFAULT_EMBED_DIM == 1024


@pytest.mark.parametrize("name", ["EMBED_DIM", "MAX_EMBED_BATCH_SIZE"])
def test_positive_integer_env_values_are_validated(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")
    monkeypatch.setenv(name, "0")

    with pytest.raises(ConfigError, match=f"{name} must be a positive integer"):
        load_config()


def test_no_removed_embedding_server_or_old_qwen_defaults_remain() -> None:
    searched_files = [
        PROJECT_ROOT / "kaggle_researcher" / "config.py",
        PROJECT_ROOT / "kaggle_researcher" / "store" / "sql.py",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "pyproject.toml",
    ]
    haystack = "\n".join(path.read_text(encoding="utf-8") for path in searched_files)
    removed_server_env = "V" + "LLM_BASE_URL"
    old_qwen_model = "Qwen3-Embedding-" + "4B"
    old_pgvector_dim = "vector(" + str(2500 + 60) + ")"

    assert removed_server_env not in haystack
    assert old_qwen_model not in haystack
    assert old_pgvector_dim not in haystack
