from __future__ import annotations

from typing import Any

from kaggle_researcher.schemas import RetrievedDocument, SourceDocument
from kaggle_researcher.store.sql import (
    CREATE_COMPETITION_PATTERNS_EMBEDDING_HNSW_INDEX_SQL,
    CREATE_COMPETITION_PATTERNS_TABLE_SQL,
    CREATE_DOCUMENTS_COMPETITION_ID_INDEX_SQL,
    CREATE_DOCUMENTS_EMBEDDING_HNSW_INDEX_SQL,
    CREATE_DOCUMENTS_TABLE_SQL,
    CREATE_DOCUMENTS_TS_CONTENT_GIN_INDEX_SQL,
    CREATE_VECTOR_EXTENSION_SQL,
)

try:
    import asyncpg
except ImportError:  # pragma: no cover - covered indirectly via tests without dependency
    asyncpg = None


class PgStore:
    def __init__(self, competition_id: str, dsn: str) -> None:
        self.competition_id = competition_id
        self.dsn = dsn
        self.pool: Any | None = None

    async def init(self) -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required to initialize PgStore")

        self.pool = await asyncpg.create_pool(dsn=self.dsn)
        async with self.pool.acquire() as connection:
            await connection.execute(CREATE_VECTOR_EXTENSION_SQL)
            await connection.execute(CREATE_DOCUMENTS_TABLE_SQL)
            await connection.execute(CREATE_COMPETITION_PATTERNS_TABLE_SQL)
            await connection.execute(CREATE_DOCUMENTS_EMBEDDING_HNSW_INDEX_SQL)
            await connection.execute(CREATE_DOCUMENTS_TS_CONTENT_GIN_INDEX_SQL)
            await connection.execute(CREATE_DOCUMENTS_COMPETITION_ID_INDEX_SQL)
            await connection.execute(CREATE_COMPETITION_PATTERNS_EMBEDDING_HNSW_INDEX_SQL)

    async def upsert(self, docs: list[SourceDocument], embeddings: list[list[float]]) -> None:
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings must have the same length")

        if not docs:
            return

        pool = self._require_pool()
        rows = [
            (
                doc.id,
                doc.competition_id,
                doc.source,
                doc.title,
                str(doc.url) if doc.url is not None else None,
                doc.content,
                doc.summary,
                doc.metadata,
                embedding,
            )
            for doc, embedding in zip(docs, embeddings, strict=True)
        ]

        query = """
        INSERT INTO documents (
            id, competition_id, source, title, url, content, summary, metadata, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (id) DO UPDATE SET
            competition_id = EXCLUDED.competition_id,
            source = EXCLUDED.source,
            title = EXCLUDED.title,
            url = EXCLUDED.url,
            content = EXCLUDED.content,
            summary = EXCLUDED.summary,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding,
            updated_at = now()
        """

        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.executemany(query, rows)

    async def vector_search(self, embedding: list[float], top_k: int = 10) -> list[RetrievedDocument]:
        pool = self._require_pool()
        query = """
        SELECT
            id,
            competition_id,
            source,
            title,
            url,
            COALESCE(summary, content) AS content,
            1 - (embedding <=> $1::vector) AS score,
            metadata
        FROM documents
        WHERE competition_id = $2
        ORDER BY embedding <=> $1::vector
        LIMIT $3
        """

        async with pool.acquire() as connection:
            rows = await connection.fetch(query, embedding, self.competition_id, top_k)

        return [self._row_to_retrieved_document(row) for row in rows]

    async def fts_search(self, query: str, top_k: int = 10) -> list[RetrievedDocument]:
        pool = self._require_pool()
        sql = """
        SELECT
            id,
            competition_id,
            source,
            title,
            url,
            COALESCE(summary, content) AS content,
            ts_rank(ts_content, plainto_tsquery('english', $1)) AS score,
            metadata
        FROM documents
        WHERE competition_id = $2
          AND ts_content @@ plainto_tsquery('english', $1)
        ORDER BY score DESC
        LIMIT $3
        """

        async with pool.acquire() as connection:
            rows = await connection.fetch(sql, query, self.competition_id, top_k)

        return [self._row_to_retrieved_document(row) for row in rows]

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    def _require_pool(self) -> Any:
        if self.pool is None:
            raise RuntimeError("PgStore is not initialized")
        return self.pool

    @staticmethod
    def _row_to_retrieved_document(row: Any) -> RetrievedDocument:
        return RetrievedDocument(
            id=row["id"],
            competition_id=row["competition_id"],
            source=row["source"],
            title=row["title"],
            url=row["url"],
            content=row["content"],
            score=float(row["score"]),
            rrf_score=0.0,
            metadata=dict(row["metadata"] or {}),
        )
