from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from kaggle_researcher.clients.deepseek_client import DeepSeekClient


def run(coro):
    return asyncio.run(coro)


@pytest.mark.network
def test_github_api_rate_limit_endpoint_is_reachable() -> None:
    response = httpx.get("https://api.github.com/rate_limit", timeout=15)

    assert response.status_code == 200
    assert "rate" in response.json()


@pytest.mark.network
def test_deepseek_chat_text_live_sanity() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY is required for this live test")

    client = DeepSeekClient(api_key=api_key)
    result = run(
        client.chat_text(
            model="deepseek-v4-flash",
            system_prompt="Return one short word.",
            user_prompt="Return the word ok.",
            timeout=30,
            max_tokens=8,
        )
    )

    assert result.strip()


@pytest.mark.network
@pytest.mark.slow
def test_real_embedding_model_returns_configured_dimension() -> None:
    from kaggle_researcher import embedder

    embedder._model = None
    embedder._embedding_dim = None

    vector = embedder.embed_one("offline architecture sanity check")

    assert len(vector) == 1024
    assert all(isinstance(value, float) for value in vector)
