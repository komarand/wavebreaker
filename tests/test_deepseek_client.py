from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from kaggle_researcher.clients.deepseek_client import DeepSeekClient, DeepSeekClientError


def completion_response(content: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"choices": [{"message": {"content": content}}]},
    )


def run(coro):
    return asyncio.run(coro)


def test_chat_json_returns_dict_from_mocked_response() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return completion_response('{"answer": 42}')

    client = DeepSeekClient(
        api_key="secret-key",
        base_url="https://deepseek.test",
        transport=httpx.MockTransport(handler),
    )

    result = run(client.chat_json("deepseek-v4-pro", "system", "user"))

    payload = json.loads(requests[0].content)
    assert result == {"answer": 42}
    assert requests[0].url.path == "/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer secret-key"
    assert payload["response_format"] == {"type": "json_object"}


def test_chat_text_returns_message_content_and_honors_max_tokens() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return completion_response("plain text")

    client = DeepSeekClient(
        api_key="secret-key",
        base_url="https://deepseek.test",
        transport=httpx.MockTransport(handler),
    )

    result = run(client.chat_text("deepseek-v4-flash", "system", "user", max_tokens=25))

    payload = json.loads(requests[0].content)
    assert result == "plain text"
    assert payload["max_tokens"] == 25
    assert "response_format" not in payload


def test_429_triggers_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    calls = 0

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return completion_response('{"ok": true}')

    monkeypatch.setattr("kaggle_researcher.clients.deepseek_client.asyncio.sleep", fake_sleep)
    client = DeepSeekClient(
        api_key="secret-key",
        base_url="https://deepseek.test",
        transport=httpx.MockTransport(handler),
    )

    result = run(client.chat_json("deepseek-v4-pro", "system", "user"))

    assert result == {"ok": True}
    assert calls == 2
    assert sleeps == [0.1]


def test_invalid_json_triggers_one_repair_attempt() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return completion_response("{broken")
        return completion_response('{"fixed": true}')

    client = DeepSeekClient(
        api_key="secret-key",
        base_url="https://deepseek.test",
        transport=httpx.MockTransport(handler),
    )

    result = run(client.chat_json("deepseek-v4-pro", "system", "user"))

    repair_payload = json.loads(requests[1].content)
    assert result == {"fixed": True}
    assert len(requests) == 2
    assert repair_payload["response_format"] == {"type": "json_object"}
    assert "{broken" in repair_payload["messages"][1]["content"]


def test_api_key_is_not_present_in_exceptions_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    secret = "secret-key-that-must-not-leak"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    client = DeepSeekClient(
        api_key=secret,
        base_url="https://deepseek.test",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(DeepSeekClientError) as exc_info:
        run(client.chat_json("deepseek-v4-pro", "system", "user"))

    assert secret not in str(exc_info.value)
    assert secret not in caplog.text
