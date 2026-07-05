from __future__ import annotations

import re
from typing import Any

from kaggle_researcher.config import DEFAULT_EMBED_DIM
from kaggle_researcher.schemas import RetrievedDocument, SourceDocument
from kaggle_researcher.store.sql import (
    CREATE_COMPETITION_PATTERNS_EMBEDDING_HNSW_INDEX_SQL,
    CREATE_DOCUMENTS_COMPETITION_ID_INDEX_SQL,
    CREATE_DOCUMENTS_EMBEDDING_HNSW_INDEX_SQL,
    CREATE_DOCUMENTS_TS_CONTENT_GIN_INDEX_SQL,
    CREATE_VECTOR_EXTENSION_SQL,
    create_competition_patterns_table_sql,
    create_documents_table_sql,
)

try:
    import asyncpg
except ImportError:  # pragma: no cover - covered indirectly via tests without dependency
    asyncpg = None


class PgStore:
    def __init__(self, competition_id: str, dsn: str, embed_dim: int = DEFAULT_EMBED_DIM) -> None:
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        self.competition_id = competition_id
        self.dsn = dsn
        self.embed_dim = embed_dim
        self.pool: Any | None = None

    async def init(self) -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required to initialize PgStore")

        self.pool = await asyncpg.create_pool(dsn=self.dsn, ssl=False)
        async with self.pool.acquire() as connection:
            await connection.execute(CREATE_VECTOR_EXTENSION_SQL)
            await connection.execute(create_documents_table_sql(self.embed_dim))
            await connection.execute(create_competition_patterns_table_sql(self.embed_dim))
            await self._validate_vector_columns(connection)
            await connection.execute(CREATE_DOCUMENTS_EMBEDDING_HNSW_INDEX_SQL)
            await connection.execute(CREATE_DOCUMENTS_TS_CONTENT_GIN_INDEX_SQL)
            await connection.execute(CREATE_DOCUMENTS_COMPETITION_ID_INDEX_SQL)
            await connection.execute(CREATE_COMPETITION_PATTERNS_EMBEDDING_HNSW_INDEX_SQL)

    async def upsert(self, docs: list[SourceDocument], embeddings: list[list[float]]) -> None:
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings must have the same length")

        if not docs:
            return

        self._validate_embedding_dimensions(embeddings)
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
        self._validate_embedding_dimensions([embedding])
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

    def _validate_embedding_dimensions(self, embeddings: list[list[float]]) -> None:
        for embedding in embeddings:
            if len(embedding) != self.embed_dim:
                raise ValueError(
                    f"Embedding dimension {len(embedding)} does not match configured "
                    f"dimension {self.embed_dim}"
                )

    async def _validate_vector_columns(self, connection: Any) -> None:
        for table_name in ("documents", "competition_patterns"):
            actual_dim = await self._get_vector_column_dim(connection, table_name, "embedding")
            if actual_dim != self.embed_dim:
                raise RuntimeError(
                    f"Existing PostgreSQL column {table_name}.embedding uses vector({actual_dim}), "
                    f"but the configured embedding dimension is {self.embed_dim}. Recreate the "
                    "database volume or table because pgvector dimensions cannot be changed safely "
                    "in place."
                )

    @staticmethod
    async def _get_vector_column_dim(connection: Any, table_name: str, column_name: str) -> int:
        data_type = await connection.fetchval(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = $1
              AND a.attname = $2
              AND NOT a.attisdropped
              AND n.nspname = ANY (current_schemas(false))
            ORDER BY array_position(current_schemas(false), n.nspname)
            LIMIT 1
            """,
            table_name,
            column_name,
        )

        match = re.fullmatch(r"vector\((\d+)\)", str(data_type or ""))
        if match is None:
            raise RuntimeError(
                f"PostgreSQL column {table_name}.{column_name} is {data_type!r}, expected "
                "a pgvector column with a fixed dimension."
            )

        return int(match.group(1))

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
