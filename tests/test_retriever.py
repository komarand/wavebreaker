from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from kaggle_researcher.retriever import hybrid_search, reciprocal_rank_fusion
from kaggle_researcher.schemas import RetrievedDocument


def run(coro):
    return asyncio.run(coro)


def make_doc(
    doc_id: str,
    score: float,
    metadata: dict | None = None,
    title: str | None = None,
) -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        competition_id="comp-1",
        source="kaggle",
        title=title or doc_id,
        url="https://example.com/doc",
        content=f"content for {doc_id}",
        score=score,
        rrf_score=0.0,
        metadata=metadata or {},
    )


def test_rrf_sums_duplicate_ids_and_sorts_descending() -> None:
    vector_results = [make_doc("doc-1", 0.9), make_doc("doc-2", 0.8)]
    fts_results = [make_doc("doc-2", 0.7), make_doc("doc-3", 0.6)]

    results = reciprocal_rank_fusion(vector_results, fts_results, k=60)

    expected_doc_2_score = (1 / (60 + 1 + 1)) + (1 / (60 + 0 + 1))
    assert [document.id for document in results] == ["doc-2", "doc-1", "doc-3"]
    assert results[0].rrf_score == expected_doc_2_score
    assert results[0].score == 0.8


def test_rrf_preserves_best_available_metadata() -> None:
    vector_results = [make_doc("doc-1", 0.4, metadata={})]
    fts_results = [make_doc("doc-1", 0.3, metadata={"source_rank": 1})]

    results = reciprocal_rank_fusion(vector_results, fts_results)

    assert results[0].metadata == {"source_rank": 1}


def test_hybrid_search_calls_embedder_and_both_searches(monkeypatch) -> None:
    vector_results = [make_doc("vector", 0.9)]
    fts_results = [make_doc("fts", 0.8)]
    store = type(
        "FakeStore",
        (),
        {
            "vector_search": AsyncMock(return_value=vector_results),
            "fts_search": AsyncMock(return_value=fts_results),
        },
    )()

    monkeypatch.setattr("kaggle_researcher.retriever.embed_one", lambda query: [0.1, 0.2])

    results = run(hybrid_search(store, "baseline model", top_k=5))

    store.vector_search.assert_awaited_once_with([0.1, 0.2], top_k=10)
    store.fts_search.assert_awaited_once_with("baseline model", top_k=10)
    assert [document.id for document in results] == ["vector", "fts"]


def test_hybrid_search_returns_top_k_after_fusion(monkeypatch) -> None:
    store = type(
        "FakeStore",
        (),
        {
            "vector_search": AsyncMock(
                return_value=[make_doc(f"vector-{index}", 1.0 - index / 10) for index in range(4)]
            ),
            "fts_search": AsyncMock(
                return_value=[make_doc(f"fts-{index}", 1.0 - index / 10) for index in range(4)]
            ),
        },
    )()

    monkeypatch.setattr("kaggle_researcher.retriever.embed_one", lambda query: [0.1, 0.2])

    results = run(hybrid_search(store, "query", top_k=3))

    assert len(results) == 3
    assert all(results[index].rrf_score >= results[index + 1].rrf_score for index in range(2))
