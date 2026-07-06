from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kaggle_researcher.config import DEFAULT_EMBED_DIM
from kaggle_researcher.embedder import embed_one
from kaggle_researcher.store.sql import (
    CREATE_COMPETITION_PATTERNS_EMBEDDING_HNSW_INDEX_SQL,
    CREATE_VECTOR_EXTENSION_SQL,
    create_competition_patterns_table_sql,
)

try:
    import asyncpg
except ImportError:  # pragma: no cover - covered by tests via monkeypatch
    asyncpg = None

try:
    from pgvector.asyncpg import register_vector
except ImportError:  # pragma: no cover - dependency is present in runtime requirements
    register_vector = None


class DomainMemory:
    def __init__(self, dsn: str, embed_dim: int = DEFAULT_EMBED_DIM) -> None:
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        self.dsn = dsn
        self.embed_dim = embed_dim
        self.pool: Any | None = None

    async def init(self) -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required to initialize DomainMemory")

        self.pool = await asyncpg.create_pool(
            dsn=self.dsn,
            ssl=False,
            init=_register_vector_codec,
        )
        async with self.pool.acquire() as connection:
            await connection.execute(CREATE_VECTOR_EXTENSION_SQL)
            await connection.execute(create_competition_patterns_table_sql(self.embed_dim))
            await connection.execute(CREATE_COMPETITION_PATTERNS_EMBEDDING_HNSW_INDEX_SQL)

    async def find_similar(self, task_type: str, domain: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        embedding = embed_one(f"{task_type} {domain}".strip())
        self._validate_embedding(embedding)
        pool = self._require_pool()
        query = """
        SELECT
            id,
            competition_family,
            task_type,
            domain,
            pattern_text,
            1 - (embedding <=> $1::vector) AS score,
            typical_models,
            typical_features,
            typical_validation,
            common_traps,
            source_competition_id
        FROM competition_patterns
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """

        async with pool.acquire() as connection:
            rows = await connection.fetch(query, embedding, top_k)

        return [_row_to_pattern(row) for row in rows]

    async def save_pattern(self, pattern: dict[str, Any]) -> None:
        pattern_text = build_pattern_text(pattern)
        embedding = embed_one(pattern_text)
        self._validate_embedding(embedding)
        pattern_id = stable_pattern_id(pattern)
        pool = self._require_pool()
        query = """
        INSERT INTO competition_patterns (
            id,
            competition_family,
            task_type,
            domain,
            pattern_text,
            embedding,
            typical_models,
            typical_features,
            typical_validation,
            common_traps,
            source_competition_id
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (id) DO UPDATE SET
            competition_family = EXCLUDED.competition_family,
            task_type = EXCLUDED.task_type,
            domain = EXCLUDED.domain,
            pattern_text = EXCLUDED.pattern_text,
            embedding = EXCLUDED.embedding,
            typical_models = EXCLUDED.typical_models,
            typical_features = EXCLUDED.typical_features,
            typical_validation = EXCLUDED.typical_validation,
            common_traps = EXCLUDED.common_traps,
            source_competition_id = EXCLUDED.source_competition_id,
            updated_at = now()
        """
        row = (
            pattern_id,
            pattern["competition_family"],
            pattern.get("task_type"),
            pattern.get("domain"),
            pattern_text,
            embedding,
            _json_dumps(pattern.get("typical_models", [])),
            _json_dumps(pattern.get("typical_features", [])),
            pattern.get("typical_validation"),
            _json_dumps(pattern.get("common_traps", [])),
            pattern.get("source_competition_id"),
        )

        async with pool.acquire() as connection:
            await connection.execute(query, *row)

    async def seed_from_file(self, path: str | Path) -> int:
        raw_patterns = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw_patterns, list):
            raise ValueError("seed pattern file must contain a list")

        for pattern in raw_patterns:
            if not isinstance(pattern, dict):
                raise ValueError("each seed pattern must be an object")
            await self.save_pattern(pattern)
        return len(raw_patterns)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    def _require_pool(self) -> Any:
        if self.pool is None:
            raise RuntimeError("DomainMemory is not initialized")
        return self.pool

    def _validate_embedding(self, embedding: list[float]) -> None:
        if len(embedding) != self.embed_dim:
            raise ValueError(
                f"Embedding dimension {len(embedding)} does not match configured "
                f"dimension {self.embed_dim}"
            )


def build_pattern_text(pattern: dict[str, Any]) -> str:
    typical_models = pattern.get("typical_models", [])
    if isinstance(typical_models, str):
        models_text = typical_models
    else:
        models_text = ", ".join(str(model) for model in typical_models)
    return "\n".join(
        [
            f"competition_family: {pattern.get('competition_family', '')}",
            f"task_type: {pattern.get('task_type', '')}",
            f"domain: {pattern.get('domain', '')}",
            f"typical_models: {models_text}",
        ]
    )


def stable_pattern_id(pattern: dict[str, Any]) -> str:
    key = f"{pattern.get('competition_family', '')}:{pattern.get('source_competition_id', '')}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"pattern-{digest}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value or [], ensure_ascii=False)


def _json_loads(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_to_pattern(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "competition_family": row["competition_family"],
        "task_type": row["task_type"],
        "domain": row["domain"],
        "pattern_text": row["pattern_text"],
        "score": float(row["score"]),
        "typical_models": _json_loads(row["typical_models"]),
        "typical_features": _json_loads(row["typical_features"]),
        "typical_validation": row["typical_validation"],
        "common_traps": _json_loads(row["common_traps"]),
        "source_competition_id": row["source_competition_id"],
    }


async def _register_vector_codec(connection: Any) -> None:
    if register_vector is not None:
        await register_vector(connection)
