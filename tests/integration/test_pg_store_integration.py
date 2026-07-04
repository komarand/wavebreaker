from __future__ import annotations

import uuid
from typing import Any

import asyncpg
import pytest

from kaggle_researcher.config import DEFAULT_EMBED_DIM
from kaggle_researcher.retriever import hybrid_search
from kaggle_researcher.schemas import SourceDocument
from kaggle_researcher.store.pg_store import PgStore


def vector_with_one(index: int, dim: int = DEFAULT_EMBED_DIM) -> list[float]:
    vector = [0.0] * dim
    vector[index] = 1.0
    return vector


def source_doc(doc_id: str, competition_id: str, title: str, content: str) -> SourceDocument:
    return SourceDocument(
        id=doc_id,
        competition_id=competition_id,
        source="kaggle",
        title=title,
        url="https://example.com/doc",
        content=content,
        summary=content,
        metadata={"integration": True},
    )


@pytest.fixture
async def initialized_store(pg_dsn: str) -> Any:
    competition_id = f"pytest-{uuid.uuid4().hex}"
    store = PgStore(competition_id=competition_id, dsn=pg_dsn, embed_dim=DEFAULT_EMBED_DIM)
    await store.init()
    try:
        yield store
    finally:
        pool = store.pool
        if pool is not None:
            async with pool.acquire() as connection:
                await connection.execute("DELETE FROM documents WHERE competition_id = $1", competition_id)
        await store.close()


@pytest.mark.integration
async def test_pg_store_init_upsert_search_hybrid_and_close(
    initialized_store: PgStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = initialized_store
    docs = [
        source_doc("doc-a", store.competition_id, "Baseline notebook", "baseline feature engineering"),
        source_doc("doc-b", store.competition_id, "Neural notebook", "neural network ensembling"),
    ]

    await store.upsert(docs, [vector_with_one(0), vector_with_one(1)])
    await store.upsert(
        [docs[0].model_copy(update={"summary": "updated baseline feature engineering"})],
        [vector_with_one(0)],
    )

    async with store.pool.acquire() as connection:
        count = await connection.fetchval(
            "SELECT count(*) FROM documents WHERE competition_id = $1",
            store.competition_id,
        )
        vector_type = await connection.fetchval(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = 'documents'
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
            LIMIT 1
            """
        )

    assert count == 2
    assert vector_type == f"vector({DEFAULT_EMBED_DIM})"

    vector_results = await store.vector_search(vector_with_one(0), top_k=1)
    assert [document.id for document in vector_results] == ["doc-a"]
    assert vector_results[0].content == "updated baseline feature engineering"

    fts_results = await store.fts_search("baseline", top_k=5)
    assert [document.id for document in fts_results] == ["doc-a"]

    monkeypatch.setattr("kaggle_researcher.retriever.embed_one", lambda query: vector_with_one(0))
    hybrid_results = await hybrid_search(store, "baseline", top_k=2)
    assert hybrid_results[0].id == "doc-a"

    await store.close()
    assert store.pool is None


@pytest.mark.integration
async def test_pg_store_dimension_mismatch_raises_clear_error(pg_dsn: str) -> None:
    store = PgStore(
        competition_id=f"pytest-{uuid.uuid4().hex}",
        dsn=pg_dsn,
        embed_dim=DEFAULT_EMBED_DIM,
    )
    await store.init()
    await store.close()

    mismatched_store = PgStore(
        competition_id=f"pytest-{uuid.uuid4().hex}",
        dsn=pg_dsn,
        embed_dim=2,
    )
    with pytest.raises(RuntimeError, match="configured embedding dimension"):
        await mismatched_store.init()
    await mismatched_store.close()


@pytest.mark.integration
async def test_pg_store_vector_dimension_mismatch_on_upsert(initialized_store: PgStore) -> None:
    doc = source_doc("doc-mismatch", initialized_store.competition_id, "Bad", "bad vector")

    with pytest.raises(ValueError, match="does not match configured dimension"):
        await initialized_store.upsert([doc], [[0.1, 0.2]])
