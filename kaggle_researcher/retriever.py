from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from kaggle_researcher.embedder import embed_one
from kaggle_researcher.schemas import RetrievedDocument

if TYPE_CHECKING:
    from kaggle_researcher.store.pg_store import PgStore


def reciprocal_rank_fusion(
    vector_results: list[RetrievedDocument],
    fts_results: list[RetrievedDocument],
    k: int = 60,
) -> list[RetrievedDocument]:
    scores_by_id: dict[str, float] = {}
    documents_by_id: dict[str, RetrievedDocument] = {}

    for ranked_results in (vector_results, fts_results):
        for rank, document in enumerate(ranked_results):
            scores_by_id[document.id] = scores_by_id.get(document.id, 0.0) + (
                1.0 / (k + rank + 1)
            )

            existing_document = documents_by_id.get(document.id)
            if existing_document is None:
                documents_by_id[document.id] = document
            else:
                documents_by_id[document.id] = _best_available_document(
                    existing_document,
                    document,
                )

    fused_documents = [
        document.model_copy(update={"rrf_score": scores_by_id[document_id]})
        for document_id, document in documents_by_id.items()
    ]
    return sorted(fused_documents, key=lambda document: document.rrf_score, reverse=True)


async def hybrid_search(
    store: "PgStore",
    query: str,
    top_k: int = 10,
) -> list[RetrievedDocument]:
    query_embedding = embed_one(query)
    search_top_k = top_k * 2
    vector_results, fts_results = await asyncio.gather(
        store.vector_search(query_embedding, top_k=search_top_k),
        store.fts_search(query, top_k=search_top_k),
    )
    return reciprocal_rank_fusion(vector_results, fts_results)[:top_k]


def _best_available_document(
    first: RetrievedDocument,
    second: RetrievedDocument,
) -> RetrievedDocument:
    if second.score > first.score:
        return second
    if not first.metadata and second.metadata:
        return second
    return first
