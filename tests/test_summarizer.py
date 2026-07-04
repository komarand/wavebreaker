from __future__ import annotations

import asyncio

import pytest

from kaggle_researcher.schemas import SourceDocument
from kaggle_researcher.summarizer import summarize_all, summarize_one


def run(coro):
    return asyncio.run(coro)


def make_doc(doc_id: str, content: str) -> SourceDocument:
    return SourceDocument(
        id=doc_id,
        competition_id="comp-1",
        source="kaggle",
        title=f"Document {doc_id}",
        url="https://example.com/doc",
        content=content,
    )


class FakeDeepSeekClient:
    def __init__(self, responses: list[str] | None = None, fail: bool = False) -> None:
        self.responses = list(responses or [])
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    async def chat_text(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: int = 90,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "timeout": timeout,
                "max_tokens": max_tokens,
            }
        )
        if self.fail:
            raise RuntimeError("api unavailable")
        return self.responses.pop(0) if self.responses else "summary"


def test_short_content_does_not_call_api() -> None:
    client = FakeDeepSeekClient()
    doc = make_doc("short", "short content")

    result = run(summarize_one(client, doc, "deepseek-v4-flash"))

    assert result.summary == "short content"
    assert client.calls == []
    assert doc.summary is None


def test_api_failure_uses_first_800_characters() -> None:
    client = FakeDeepSeekClient(fail=True)
    content = "x" * 1000
    doc = make_doc("long", content)

    result = run(summarize_one(client, doc, "deepseek-v4-flash"))

    assert result.summary == "x" * 800
    assert len(client.calls) == 1


def test_summarize_one_uses_deepseek_flash_client() -> None:
    client = FakeDeepSeekClient(responses=["model summary"])
    doc = make_doc("long", "Long content. " * 30)

    result = run(summarize_one(client, doc, "deepseek-v4-flash"))

    assert result.summary == "model summary"
    assert client.calls[0]["model"] == "deepseek-v4-flash"
    assert "250-300 words" in str(client.calls[0]["system_prompt"])
    assert "Long content." in str(client.calls[0]["user_prompt"])


def test_summarize_all_preserves_order() -> None:
    client = FakeDeepSeekClient(responses=["summary-1", "summary-2", "summary-3"])
    docs = [
        make_doc("doc-1", "content one " * 20),
        make_doc("doc-2", "content two " * 20),
        make_doc("doc-3", "content three " * 20),
    ]

    results = run(summarize_all(client, docs, "deepseek-v4-flash", concurrency=2))

    assert [doc.id for doc in results] == ["doc-1", "doc-2", "doc-3"]
    assert [doc.summary for doc in results] == ["summary-1", "summary-2", "summary-3"]


def test_summarize_all_rejects_invalid_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency must be positive"):
        run(summarize_all(FakeDeepSeekClient(), [], "deepseek-v4-flash", concurrency=0))
