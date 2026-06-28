from __future__ import annotations

import pytest

from kaggle_researcher.config import ConfigError, load_config


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


@pytest.mark.parametrize("name", ["EMBED_DIM", "MAX_EMBED_BATCH_SIZE"])
def test_positive_integer_env_values_are_validated(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")
    monkeypatch.setenv(name, "0")

    with pytest.raises(ConfigError, match=f"{name} must be a positive integer"):
        load_config()
