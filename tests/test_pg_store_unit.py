from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kaggle_researcher.schemas import SourceDocument
from kaggle_researcher.store.pg_store import PgStore
from kaggle_researcher.store.sql import (
    CREATE_COMPETITION_PATTERNS_EMBEDDING_HNSW_INDEX_SQL,
    CREATE_COMPETITION_PATTERNS_TABLE_SQL,
    CREATE_DOCUMENTS_COMPETITION_ID_INDEX_SQL,
    CREATE_DOCUMENTS_EMBEDDING_HNSW_INDEX_SQL,
    CREATE_DOCUMENTS_TABLE_SQL,
    CREATE_DOCUMENTS_TS_CONTENT_GIN_INDEX_SQL,
    CREATE_VECTOR_EXTENSION_SQL,
)


class FakeTransaction:
    async def __aenter__(self) -> FakeTransaction:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class FakeAcquire:
    def __init__(self, connection: object) -> None:
        self.connection = connection

    async def __aenter__(self) -> object:
        return self.connection

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def make_store_with_connection(connection: object) -> PgStore:
    store = PgStore(competition_id="comp-123", dsn="postgresql://test")
    store.pool = SimpleNamespace(acquire=lambda: FakeAcquire(connection), close=AsyncMock())
    return store


def test_upsert_len_mismatch_raises_error() -> None:
    store = PgStore(competition_id="comp-123", dsn="postgresql://test")
    doc = SourceDocument(
        id="doc-1",
        competition_id="comp-123",
        source="kaggle",
        title="Notebook",
        url="https://example.com/doc-1",
        content="content",
    )

    with pytest.raises(ValueError, match="same length"):
        asyncio.run(store.upsert([doc], []))


def test_init_runs_expected_ddl(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = SimpleNamespace(execute=AsyncMock())
    pool = SimpleNamespace(acquire=lambda: FakeAcquire(connection))
    create_pool = AsyncMock(return_value=pool)

    from kaggle_researcher.store import pg_store as pg_store_module

    monkeypatch.setattr(pg_store_module, "asyncpg", SimpleNamespace(create_pool=create_pool))

    store = PgStore(competition_id="comp-123", dsn="postgresql://test")
    asyncio.run(store.init())

    create_pool.assert_awaited_once_with(dsn="postgresql://test")
    executed_sql = [call.args[0] for call in connection.execute.await_args_list]
    assert executed_sql == [
        CREATE_VECTOR_EXTENSION_SQL,
        CREATE_DOCUMENTS_TABLE_SQL,
        CREATE_COMPETITION_PATTERNS_TABLE_SQL,
        CREATE_DOCUMENTS_EMBEDDING_HNSW_INDEX_SQL,
        CREATE_DOCUMENTS_TS_CONTENT_GIN_INDEX_SQL,
        CREATE_DOCUMENTS_COMPETITION_ID_INDEX_SQL,
        CREATE_COMPETITION_PATTERNS_EMBEDDING_HNSW_INDEX_SQL,
    ]


def test_upsert_uses_single_transaction_and_executemany() -> None:
    connection = SimpleNamespace(
        executemany=AsyncMock(),
        transaction=lambda: FakeTransaction(),
    )
    store = make_store_with_connection(connection)
    doc = SourceDocument(
        id="doc-1",
        competition_id="comp-123",
        source="kaggle",
        title="Notebook",
        url="https://example.com/doc-1",
        content="full content",
        summary="short summary",
        metadata={"votes": 10},
    )

    asyncio.run(store.upsert([doc], [[0.1, 0.2]]))

    connection.executemany.assert_awaited_once()
    query, rows = connection.executemany.await_args.args
    assert "INSERT INTO documents" in query
    assert len(rows) == 1
    assert rows[0][0] == "doc-1"
    assert rows[0][1] == "comp-123"
    assert rows[0][6] == "short summary"


def test_vector_search_filters_by_competition_id_and_prefers_summary() -> None:
    connection = SimpleNamespace(
        fetch=AsyncMock(
            return_value=[
                {
                    "id": "doc-1",
                    "competition_id": "comp-123",
                    "source": "arxiv",
                    "title": "Paper",
                    "url": "https://example.com/paper",
                    "content": "summary text",
                    "score": 0.91,
                    "metadata": {"pdf_url": "https://example.com/paper.pdf"},
                }
            ]
        )
    )
    store = make_store_with_connection(connection)

    results = asyncio.run(store.vector_search([0.1, 0.2], top_k=5))

    sql, embedding, competition_id, top_k = connection.fetch.await_args.args
    assert "WHERE competition_id = $2" in sql
    assert embedding == [0.1, 0.2]
    assert competition_id == "comp-123"
    assert top_k == 5
    assert len(results) == 1
    assert results[0].content == "summary text"
    assert results[0].rrf_score == 0.0


def test_fts_search_filters_by_competition_id() -> None:
    connection = SimpleNamespace(
        fetch=AsyncMock(
            return_value=[
                {
                    "id": "doc-2",
                    "competition_id": "comp-123",
                    "source": "github",
                    "title": "Repo",
                    "url": "https://example.com/repo",
                    "content": "readme snippet",
                    "score": 0.77,
                    "metadata": {"stars": 100},
                }
            ]
        )
    )
    store = make_store_with_connection(connection)

    results = asyncio.run(store.fts_search("baseline model", top_k=3))

    sql, query, competition_id, top_k = connection.fetch.await_args.args
    assert "WHERE competition_id = $2" in sql
    assert "plainto_tsquery('english', $1)" in sql
    assert query == "baseline model"
    assert competition_id == "comp-123"
    assert top_k == 3
    assert len(results) == 1
    assert results[0].metadata["stars"] == 100
