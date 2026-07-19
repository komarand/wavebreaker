from __future__ import annotations

import asyncio

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.schemas import SourceDocument


SYSTEM_PROMPT = """You summarize Kaggle research sources for a retrieval pipeline.

Write a concise, factual summary of the source in 250-300 words when enough
content is available. Focus on competition-relevant methods, validation hints,
metric details, and reusable ideas. Do not claim that train/test data was
analyzed, notebooks were executed, or leakage was confirmed."""


async def summarize_one(
    client: DeepSeekClient,
    doc: SourceDocument,
    model: str,
) -> SourceDocument:
    content = doc.content or ""
    if len(content) < 150:
        return doc.model_copy(update={
            "summary": content,
            "metadata": {
                **doc.metadata,
                "summary_fallback_used": True,
                "summary_fallback_reason": "short_content",
                "summary_actual_input_kind": "raw_or_parsed_text",
            },
        })

    try:
        summary = await client.chat_text(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(doc),
            timeout=90,
        )
    except Exception as exc:
        summary = content[:800]
        return doc.model_copy(update={
            "summary": summary,
            "metadata": {
                **doc.metadata,
                "summary_fallback_used": True,
                "summary_fallback_reason": type(exc).__name__,
                "summary_actual_input_kind": "raw_or_parsed_text",
            },
        })

    return doc.model_copy(update={"summary": summary})


async def summarize_all(
    client: DeepSeekClient,
    docs: list[SourceDocument],
    model: str,
    concurrency: int = 8,
) -> list[SourceDocument]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    semaphore = asyncio.Semaphore(concurrency)

    async def summarize_with_limit(doc: SourceDocument) -> SourceDocument:
        async with semaphore:
            return await summarize_one(client=client, doc=doc, model=model)

    return list(await asyncio.gather(*(summarize_with_limit(doc) for doc in docs)))


def _build_user_prompt(doc: SourceDocument) -> str:
    return (
        f"Source: {doc.source}\n"
        f"Title: {doc.title}\n"
        f"URL: {doc.url or ''}\n\n"
        "Content:\n"
        f"{doc.content}"
    )
