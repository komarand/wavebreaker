from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx


class DeepSeekClientError(RuntimeError):
    """Raised when a DeepSeek API call cannot be completed safely."""


class DeepSeekClient:
    """Small OpenAI-compatible async wrapper for DeepSeek chat calls."""

    _MAX_ATTEMPTS = 3
    _BACKOFF_BASE_SECONDS = 0.1

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport

    async def chat_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: float = 90,
    ) -> dict[str, Any]:
        payload = self._build_payload(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "json_object"},
        )
        data = await self._post_chat_completions(payload, timeout=timeout)
        content = self._extract_content(data)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return await self._repair_json(model=model, invalid_json=content, timeout=timeout)

        if not isinstance(parsed, dict):
            raise DeepSeekClientError("DeepSeek JSON response was not an object")
        return parsed

    async def chat_text(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: float = 90,
        max_tokens: int | None = None,
    ) -> str:
        payload = self._build_payload(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
        data = await self._post_chat_completions(payload, timeout=timeout)
        return self._extract_content(data)

    async def _repair_json(self, model: str, invalid_json: str, timeout: float) -> dict[str, Any]:
        payload = self._build_payload(
            model=model,
            system_prompt=(
                "You repair malformed JSON. Return only one valid JSON object. "
                "Do not add prose, markdown, or keys that are not implied by the input."
            ),
            user_prompt=f"Repair this malformed JSON object:\n{invalid_json}",
            response_format={"type": "json_object"},
        )
        data = await self._post_chat_completions(payload, timeout=timeout)
        content = self._extract_content(data)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise DeepSeekClientError("DeepSeek JSON repair returned invalid JSON") from exc

        if not isinstance(parsed, dict):
            raise DeepSeekClientError("DeepSeek JSON repair response was not an object")
        return parsed

    async def _post_chat_completions(
        self,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self._MAX_ATTEMPTS):
            try:
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    transport=self._transport,
                    timeout=timeout,
                ) as client:
                    response = await client.post("/chat/completions", json=payload)

                if response.status_code in {429} or 500 <= response.status_code < 600:
                    if attempt < self._MAX_ATTEMPTS - 1:
                        await self._sleep_before_retry(attempt)
                        continue
                    raise DeepSeekClientError(
                        f"DeepSeek request failed with retryable status {response.status_code}"
                    )

                if response.status_code >= 400:
                    raise DeepSeekClientError(
                        f"DeepSeek request failed with status {response.status_code}"
                    )

                return response.json()
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self._MAX_ATTEMPTS - 1:
                    await self._sleep_before_retry(attempt)
                    continue
                raise DeepSeekClientError("DeepSeek network request failed") from exc
            except json.JSONDecodeError as exc:
                raise DeepSeekClientError("DeepSeek response body was not valid JSON") from exc

        raise DeepSeekClientError("DeepSeek request failed") from last_error

    async def _sleep_before_retry(self, attempt: int) -> None:
        await asyncio.sleep(self._BACKOFF_BASE_SECONDS * (2**attempt))

    @staticmethod
    def _build_payload(
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepSeekClientError("DeepSeek response did not contain message content") from exc

        if not isinstance(content, str):
            raise DeepSeekClientError("DeepSeek message content was not text")
        return content
