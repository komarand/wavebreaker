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
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("PG_DSN", raising=False)

    settings = load_config()

    assert settings.deepseek_api_key == "secret-key"
    assert settings.deepseek_v4_pro == "deepseek-v4-pro"
    assert settings.deepseek_v4_flash == "deepseek-v4-flash"
    assert settings.vllm_base_url == "http://localhost:8000/v1"
    assert settings.embed_model == "Qwen/Qwen3-Embedding-4B"
    assert settings.embed_dim == 2560
    assert settings.max_embed_batch_size == 64
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
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.setenv("PG_DSN", "postgresql://user:pass@db:5432/app")
    monkeypatch.setenv("KAGGLE_USERNAME", "kaggle-user")
    monkeypatch.setenv("KAGGLE_KEY", "kaggle-key")
    monkeypatch.setenv("GITHUB_TOKEN", "github-token")

    settings = load_config()

    assert settings.vllm_base_url == "http://127.0.0.1:9000/v1"
    assert settings.pg_dsn == "postgresql://user:pass@db:5432/app"
    assert settings.kaggle_username == "kaggle-user"
    assert settings.kaggle_key == "kaggle-key"
    assert settings.github_token == "github-token"
