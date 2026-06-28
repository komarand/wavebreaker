from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx


class EmbedderError(RuntimeError):
    """Raised when vLLM embedding generation fails."""


async def embed_texts(
    texts: list[str],
    base_url: str,
    model: str,
    batch_size: int = 64,
) -> list[list[float]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not texts:
        return []

    embeddings: list[list[float]] = []
    endpoint = f"{base_url.rstrip('/')}/embeddings"

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_embeddings = await _embed_batch(endpoint=endpoint, model=model, batch=batch)
        embeddings.extend(batch_embeddings)

    return embeddings


async def embed_one(text: str, base_url: str, model: str) -> list[float]:
    embeddings = await embed_texts([text], base_url=base_url, model=model, batch_size=1)
    return embeddings[0]


async def _embed_batch(endpoint: str, model: str, batch: list[str]) -> list[list[float]]:
    payload = {"model": model, "input": batch}
    data = await _post_embeddings(endpoint=endpoint, payload=payload)

    try:
        raw_items = data["data"]
    except (KeyError, TypeError) as exc:
        raise EmbedderError("Embedding response did not contain data") from exc

    if not isinstance(raw_items, list):
        raise EmbedderError("Embedding response data was not a list")

    try:
        sorted_items = sorted(raw_items, key=lambda item: item["index"])
        embeddings = [item["embedding"] for item in sorted_items]
    except (KeyError, TypeError) as exc:
        raise EmbedderError("Embedding response items were malformed") from exc

    if len(embeddings) != len(batch):
        raise EmbedderError("Embedding response size did not match input batch size")

    return embeddings


async def _post_embeddings(
    endpoint: str,
    payload: dict[str, Any],
    timeout: float = 90,
) -> dict[str, Any]:
    max_attempts = 3
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload)

            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt < max_attempts - 1:
                    await _sleep_before_retry(attempt)
                    continue
                raise EmbedderError(
                    f"Embedding request failed with retryable status {response.status_code}"
                )

            if response.status_code >= 400:
                raise EmbedderError(f"Embedding request failed with status {response.status_code}")

            return response.json()
        except httpx.RequestError as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                await _sleep_before_retry(attempt)
                continue
            raise EmbedderError("Embedding network request failed") from exc
        except json.JSONDecodeError as exc:
            raise EmbedderError("Embedding response body was not valid JSON") from exc

    raise EmbedderError("Embedding request failed") from last_error


async def _sleep_before_retry(attempt: int) -> None:
    await asyncio.sleep(0.1 * (2**attempt))
