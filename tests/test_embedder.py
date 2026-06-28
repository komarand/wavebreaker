from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from kaggle_researcher.embedder import EmbedderError, embed_one, embed_texts


def run(coro):
    return asyncio.run(coro)


def patch_async_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def async_client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("kaggle_researcher.embedder.httpx.AsyncClient", async_client_factory)


def test_shuffled_response_indexes_return_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 2, "embedding": [2.0]},
                    {"index": 0, "embedding": [0.0]},
                    {"index": 1, "embedding": [1.0]},
                ]
            },
        )

    patch_async_client(monkeypatch, handler)

    result = run(embed_texts(["a", "b", "c"], "http://vllm.test/v1", "embed-model"))

    payload = json.loads(requests[0].content)
    assert result == [[0.0], [1.0], [2.0]]
    assert str(requests[0].url) == "http://vllm.test/v1/embeddings"
    assert payload == {"model": "embed-model", "input": ["a", "b", "c"]}


def test_hundred_texts_with_batch_size_64_makes_two_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        batch = payload["input"]
        batch_sizes.append(len(batch))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [float(index)]}
                    for index, _text in enumerate(batch)
                ]
            },
        )

    patch_async_client(monkeypatch, handler)

    result = run(
        embed_texts(
            [f"text-{index}" for index in range(100)],
            "http://vllm.test/v1",
            "embed-model",
            batch_size=64,
        )
    )

    assert batch_sizes == [64, 36]
    assert len(result) == 100


def test_failed_final_batch_raises_embedder_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_sleep(delay: float) -> None:
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, json={"error": "server error"})

    patch_async_client(monkeypatch, handler)
    monkeypatch.setattr("kaggle_researcher.embedder.asyncio.sleep", fake_sleep)

    with pytest.raises(EmbedderError, match="retryable status 500"):
        run(embed_texts(["a"], "http://vllm.test/v1", "embed-model"))

    assert calls == 3


def test_embed_one_returns_single_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    patch_async_client(monkeypatch, handler)

    result = run(embed_one("query", "http://vllm.test/v1", "embed-model"))

    assert result == [1.0, 2.0]
