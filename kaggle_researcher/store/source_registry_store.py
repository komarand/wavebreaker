from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from kaggle_researcher.config import DEFAULT_EMBED_DIM
from kaggle_researcher.schemas import RetrievedDocument
from kaggle_researcher.source_registry.errors import EmbeddingCompatibilityError, SourceMigrationError
from kaggle_researcher.source_registry.schemas import (
    ArtifactRecord,
    CanonicalSourceIdentity,
    ContentHashes,
    EmbeddingRecord,
    SearchCacheEntry,
    SourceRecord,
    SourceVersion,
)
from kaggle_researcher.store.migrations import create_source_registry_migration_sql
from kaggle_researcher.store.pg_store import _metadata_to_dict, _register_vector_codec
from kaggle_researcher.store.sql import CREATE_VECTOR_EXTENSION_SQL

try:
    import asyncpg
except ImportError:  # pragma: no cover
    asyncpg = None


class SourceRegistryStore:
    def __init__(self, dsn: str, embed_dim: int = DEFAULT_EMBED_DIM, *, pool: Any | None = None,
                 competition_id: str | None = None) -> None:
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        self.dsn = dsn
        self.embed_dim = embed_dim
        self.pool = pool
        self.competition_id = competition_id
        self._owns_pool = pool is None

    async def init(self) -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required to initialize SourceRegistryStore")
        if self.pool is None:
            bootstrap = await asyncpg.connect(dsn=self.dsn, ssl=False)
            try:
                await bootstrap.execute(CREATE_VECTOR_EXTENSION_SQL)
            finally:
                await bootstrap.close()
            self.pool = await asyncpg.create_pool(dsn=self.dsn, ssl=False, init=_register_vector_codec)
        try:
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    await connection.execute(create_source_registry_migration_sql(self.embed_dim))
                await self._validate_embedding_dimension(connection)
        except Exception as exc:
            raise SourceMigrationError(f"Source registry migration failed: {type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        if self.pool is not None and self._owns_pool:
            await self.pool.close()
            self.pool = None

    async def get_source(self, source_id: str) -> SourceRecord | None:
        async with self._require_pool().acquire() as connection:
            return _source_record(await connection.fetchrow("SELECT * FROM sources WHERE source_id=$1", source_id))

    async def upsert_source(
        self,
        identity: CanonicalSourceIdentity,
        title: str | None,
        canonical_url: str | None,
        metadata: dict[str, Any],
        checked_at: datetime | None = None,
    ) -> SourceRecord:
        sql = """
        INSERT INTO sources(source_id,source_type,external_id,canonical_url,title,metadata,last_checked_at,identity_version)
        VALUES($1,$2,$3,$4,$5,$6::jsonb,$7,$8)
        ON CONFLICT(source_type,external_id) DO UPDATE SET
            canonical_url=COALESCE(EXCLUDED.canonical_url,sources.canonical_url),
            title=COALESCE(EXCLUDED.title,sources.title),
            metadata=sources.metadata || EXCLUDED.metadata,
            last_seen_at=now(),
            last_checked_at=COALESCE(EXCLUDED.last_checked_at,sources.last_checked_at),
            source_status='active'
        RETURNING *
        """
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                sql, identity.source_id, identity.source_type, identity.external_id,
                canonical_url or identity.canonical_url, title, _json(metadata), checked_at,
                identity.identity_version,
            )
        return _require_record(_source_record(row), "source")

    async def mark_source_status(self, source_id: str, status: str, checked_at: datetime) -> None:
        if status not in {"active", "unavailable", "deleted", "blocked", "unknown"}:
            raise ValueError(f"Invalid source status: {status}")
        async with self._require_pool().acquire() as connection:
            await connection.execute(
                "UPDATE sources SET source_status=$2,last_checked_at=$3 WHERE source_id=$1",
                source_id, status, checked_at,
            )

    async def get_current_version(self, source_id: str) -> SourceVersion | None:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                "SELECT v.* FROM sources s JOIN source_versions v ON v.version_id=s.current_version_id WHERE s.source_id=$1",
                source_id,
            )
        return _version_record(row)

    async def get_version_by_hash(self, source_id: str, normalized_content_hash: str) -> SourceVersion | None:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM source_versions WHERE source_id=$1 AND normalized_content_hash=$2",
                source_id, normalized_content_hash,
            )
        return _version_record(row)

    async def create_or_reuse_version(
        self,
        source_id: str,
        source_revision: str | None,
        raw_content: str | None,
        content_location: str | None,
        content_mime_type: str | None,
        hashes: ContentHashes,
        metadata: dict[str, Any],
    ) -> tuple[SourceVersion, bool]:
        version_id = uuid4()
        insert = """
        INSERT INTO source_versions(version_id,source_id,source_revision,raw_content_hash,normalized_content_hash,
            raw_content,content_location,content_mime_type,content_size_bytes,metadata,is_current)
        VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,TRUE)
        ON CONFLICT(source_id,normalized_content_hash) DO NOTHING RETURNING *
        """
        async with self._require_pool().acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    insert, version_id, source_id, source_revision, hashes.raw_hash, hashes.normalized_hash,
                    raw_content, content_location, content_mime_type,
                    len(raw_content.encode("utf-8")) if raw_content is not None else None,
                    _json({**metadata, "normalization_policy_version": hashes.normalization_policy_version}),
                )
                created = row is not None
                if row is None:
                    row = await connection.fetchrow(
                        "SELECT * FROM source_versions WHERE source_id=$1 AND normalized_content_hash=$2",
                        source_id, hashes.normalized_hash,
                    )
                    await connection.execute(
                        """UPDATE source_versions SET
                            source_revision=COALESCE($2,source_revision),
                            raw_content=COALESCE(raw_content,$3),
                            content_location=COALESCE(content_location,$4),
                            content_size_bytes=COALESCE(content_size_bytes,$5),
                            metadata=metadata || $6::jsonb
                           WHERE version_id=$1""",
                        row["version_id"], source_revision, raw_content, content_location,
                        len(raw_content.encode("utf-8")) if raw_content is not None else None,
                        _json(metadata),
                    )
                selected_id = row["version_id"]
                await connection.execute(
                    "UPDATE source_versions SET is_current=(version_id=$2) WHERE source_id=$1",
                    source_id, selected_id,
                )
                await connection.execute(
                    "UPDATE sources SET current_version_id=$2 WHERE source_id=$1",
                    source_id, selected_id,
                )
                row = await connection.fetchrow("SELECT * FROM source_versions WHERE version_id=$1", selected_id)
        return _require_record(_version_record(row), "source version"), created

    async def set_current_version(self, source_id: str, version_id: UUID) -> None:
        async with self._require_pool().acquire() as connection:
            async with connection.transaction():
                belongs = await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM source_versions WHERE source_id=$1 AND version_id=$2)", source_id, version_id
                )
                if not belongs:
                    raise ValueError("Version does not belong to source")
                await connection.execute("UPDATE source_versions SET is_current=(version_id=$2) WHERE source_id=$1", source_id, version_id)
                await connection.execute("UPDATE sources SET current_version_id=$2 WHERE source_id=$1", source_id, version_id)

    async def find_artifact(self, version_id: UUID, artifact_type: str, processor_fingerprint: str, input_hash: str) -> ArtifactRecord | None:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM source_artifacts WHERE version_id=$1 AND artifact_type=$2 AND processor_fingerprint=$3 AND input_hash=$4",
                version_id, artifact_type, processor_fingerprint, input_hash,
            )
        return _artifact_record(row)

    async def find_latest_artifact(self, version_id: UUID, artifact_type: str) -> ArtifactRecord | None:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM source_artifacts WHERE version_id=$1 AND artifact_type=$2 ORDER BY created_at DESC LIMIT 1",
                version_id, artifact_type,
            )
        return _artifact_record(row)

    async def save_artifact(self, version_id: UUID, artifact_type: str, processor_fingerprint: str,
                            input_hash: str, output_hash: str, payload: Any = None,
                            content_location: str | None = None, metadata: dict[str, Any] | None = None) -> ArtifactRecord:
        sql = """
        INSERT INTO source_artifacts(artifact_id,version_id,artifact_type,processor_fingerprint,input_hash,output_hash,payload,content_location,metadata)
        VALUES($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9::jsonb)
        ON CONFLICT(version_id,artifact_type,processor_fingerprint,input_hash) DO UPDATE SET artifact_type=EXCLUDED.artifact_type
        RETURNING *
        """
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                sql, uuid4(), version_id, artifact_type, processor_fingerprint, input_hash,
                output_hash, _json(payload), content_location, _json(metadata or {}),
            )
        return _require_record(_artifact_record(row), "artifact")

    async def find_embedding(self, version_id: UUID, input_kind: str, embedding_fingerprint: str,
                             input_hash: str) -> EmbeddingRecord | None:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM source_embeddings WHERE version_id=$1 AND input_kind=$2 AND embedding_fingerprint=$3 AND input_hash=$4",
                version_id, input_kind, embedding_fingerprint, input_hash,
            )
        record = _embedding_record(row)
        return record.model_copy(update={"embedding": list(row["embedding"])}) if record is not None else None

    async def find_latest_embedding(self, version_id: UUID, input_kind: str) -> EmbeddingRecord | None:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM source_embeddings WHERE version_id=$1 AND input_kind=$2 ORDER BY created_at DESC LIMIT 1",
                version_id, input_kind,
            )
        record = _embedding_record(row)
        return record.model_copy(update={"embedding": list(row["embedding"])}) if record is not None else None

    async def save_embedding(self, version_id: UUID, input_kind: str, embedding_fingerprint: str,
                             input_hash: str, embedding: list[float], metadata: dict[str, Any] | None = None) -> EmbeddingRecord:
        if len(embedding) != self.embed_dim:
            raise EmbeddingCompatibilityError(
                f"Embedding dimension {len(embedding)} does not match configured dimension {self.embed_dim}"
            )
        sql = """
        INSERT INTO source_embeddings(embedding_id,version_id,input_kind,embedding_fingerprint,input_hash,
            embedding_dimension,embedding,metadata)
        VALUES($1,$2,$3,$4,$5,$6,$7,$8::jsonb)
        ON CONFLICT(version_id,input_kind,embedding_fingerprint,input_hash) DO UPDATE SET input_kind=EXCLUDED.input_kind
        RETURNING *
        """
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                sql, uuid4(), version_id, input_kind, embedding_fingerprint, input_hash,
                len(embedding), embedding, _json(metadata or {}),
            )
        record = _require_record(_embedding_record(row), "embedding")
        return record.model_copy(update={"embedding": list(row["embedding"])})

    async def associate_source_with_competition(self, competition_id: str, source_id: str, provider: str,
                                                query: str, discovery_rank: int | None,
                                                relevance_metadata: dict[str, Any]) -> None:
        sql = """
        INSERT INTO competition_sources(competition_id,source_id,discovery_queries,discovery_providers,best_discovery_rank,relevance_metadata)
        VALUES($1,$2,$3::jsonb,$4::jsonb,$5,$6::jsonb)
        ON CONFLICT(competition_id,source_id) DO UPDATE SET
            last_seen_at=now(),
            discovery_queries=(SELECT jsonb_agg(DISTINCT value) FROM jsonb_array_elements(
                competition_sources.discovery_queries || EXCLUDED.discovery_queries)),
            discovery_providers=(SELECT jsonb_agg(DISTINCT value) FROM jsonb_array_elements(
                competition_sources.discovery_providers || EXCLUDED.discovery_providers)),
            best_discovery_rank=LEAST(competition_sources.best_discovery_rank,EXCLUDED.best_discovery_rank),
            relevance_metadata=competition_sources.relevance_metadata || EXCLUDED.relevance_metadata
        """
        async with self._require_pool().acquire() as connection:
            await connection.execute(sql, competition_id, source_id, _json([query] if query else []),
                                     _json([provider] if provider else []), discovery_rank, _json(relevance_metadata))

    async def record_run_source(self, run_id: str, competition_id: str, source_id: str, **values: Any) -> None:
        sql = """
        INSERT INTO research_run_sources(run_id,competition_id,source_id,version_id,selected_for_retrieval,
            selected_for_reasoning,retrieval_score,rrf_score,artifact_ids,embedding_id,cache_decisions)
        VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11::jsonb)
        ON CONFLICT(run_id,source_id) DO UPDATE SET
            version_id=COALESCE(EXCLUDED.version_id,research_run_sources.version_id),
            selected_for_retrieval=research_run_sources.selected_for_retrieval OR EXCLUDED.selected_for_retrieval,
            selected_for_reasoning=research_run_sources.selected_for_reasoning OR EXCLUDED.selected_for_reasoning,
            retrieval_score=COALESCE(EXCLUDED.retrieval_score,research_run_sources.retrieval_score),
            rrf_score=COALESCE(EXCLUDED.rrf_score,research_run_sources.rrf_score),
            artifact_ids=CASE WHEN EXCLUDED.artifact_ids='[]'::jsonb THEN research_run_sources.artifact_ids ELSE EXCLUDED.artifact_ids END,
            embedding_id=COALESCE(EXCLUDED.embedding_id,research_run_sources.embedding_id),
            cache_decisions=research_run_sources.cache_decisions || EXCLUDED.cache_decisions
        """
        version_id = values.get("version_id")
        if isinstance(version_id, str):
            version_id = UUID(version_id)
        embedding_id = values.get("embedding_id")
        if isinstance(embedding_id, str):
            embedding_id = UUID(embedding_id)
        async with self._require_pool().acquire() as connection:
            await connection.execute(
                sql, run_id, competition_id, source_id, version_id,
                values.get("selected_for_retrieval", False), values.get("selected_for_reasoning", False),
                values.get("retrieval_score"), values.get("rrf_score"), _json(values.get("artifact_ids", [])),
                embedding_id, _json(values.get("cache_decisions", {})),
            )

    async def get_search_cache(self, provider: str, query_hash: str, request_fingerprint: str) -> SearchCacheEntry | None:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM source_search_cache WHERE provider=$1 AND query_hash=$2 AND request_fingerprint=$3",
                provider, query_hash, request_fingerprint,
            )
        return _search_record(row)

    async def save_search_cache(self, entry: SearchCacheEntry) -> SearchCacheEntry:
        sql = """
        INSERT INTO source_search_cache(provider,query_hash,normalized_query,request_fingerprint,result_source_ids,
            raw_result_metadata,fetched_at,expires_at)
        VALUES($1,$2,$3,$4,$5::jsonb,$6::jsonb,$7,$8)
        ON CONFLICT(provider,query_hash,request_fingerprint) DO UPDATE SET
            normalized_query=EXCLUDED.normalized_query,result_source_ids=EXCLUDED.result_source_ids,
            raw_result_metadata=EXCLUDED.raw_result_metadata,fetched_at=EXCLUDED.fetched_at,expires_at=EXCLUDED.expires_at
        RETURNING *
        """
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                sql, entry.provider, entry.query_hash, entry.normalized_query, entry.request_fingerprint,
                _json(entry.result_source_ids), _json(entry.raw_result_metadata), entry.fetched_at, entry.expires_at,
            )
        return _require_record(_search_record(row), "search cache")

    async def prune_expired_search_cache(self, before: datetime) -> int:
        async with self._require_pool().acquire() as connection:
            status = await connection.execute("DELETE FROM source_search_cache WHERE expires_at < $1", before)
        match = re.search(r"(\d+)$", status)
        return int(match.group(1)) if match else 0

    async def list_competition_source_ids(self, competition_id: str) -> list[str]:
        async with self._require_pool().acquire() as connection:
            rows = await connection.fetch(
                "SELECT source_id FROM competition_sources WHERE competition_id=$1 ORDER BY first_seen_at", competition_id
            )
        return [str(row["source_id"]) for row in rows]

    async def get_embedding_vector(self, embedding_id: UUID) -> list[float] | None:
        async with self._require_pool().acquire() as connection:
            value = await connection.fetchval(
                "SELECT embedding FROM source_embeddings WHERE embedding_id=$1", embedding_id
            )
        return list(value) if value is not None else None

    async def vector_search(self, embedding: list[float], top_k: int = 10) -> list[RetrievedDocument]:
        if len(embedding) != self.embed_dim:
            raise EmbeddingCompatibilityError("Query embedding dimension is incompatible with source registry")
        sql = """
        SELECT s.source_id,s.source_type,s.title,s.canonical_url,v.version_id,
            COALESCE(a.payload->>'text',a.payload->>'summary',v.raw_content,'') content,
            1-(e.embedding <=> $1::vector) score,s.metadata
        FROM competition_sources cs JOIN sources s ON s.source_id=cs.source_id
        JOIN source_versions v ON v.version_id=s.current_version_id
        JOIN LATERAL (SELECT * FROM source_embeddings x WHERE x.version_id=v.version_id ORDER BY created_at DESC LIMIT 1) e ON true
        LEFT JOIN LATERAL (SELECT * FROM source_artifacts x WHERE x.version_id=v.version_id AND x.artifact_type IN ('summary','parsed_text')
            ORDER BY CASE WHEN x.artifact_type='summary' THEN 0 ELSE 1 END,created_at DESC LIMIT 1) a ON true
        WHERE cs.competition_id=$2 ORDER BY e.embedding <=> $1::vector LIMIT $3
        """
        competition_id = self._require_competition_id()
        async with self._require_pool().acquire() as connection:
            rows = await connection.fetch(sql, embedding, competition_id, top_k)
        return [self._retrieved_document(row, competition_id) for row in rows]

    async def fts_search(self, query: str, top_k: int = 10) -> list[RetrievedDocument]:
        sql = """
        SELECT s.source_id,s.source_type,s.title,s.canonical_url,v.version_id,
            COALESCE(a.payload->>'text',a.payload->>'summary',v.raw_content,'') content,
            ts_rank(a.ts_content,plainto_tsquery('english',$1)) score,s.metadata
        FROM competition_sources cs JOIN sources s ON s.source_id=cs.source_id
        JOIN source_versions v ON v.version_id=s.current_version_id
        JOIN source_artifacts a ON a.version_id=v.version_id AND a.artifact_type IN ('summary','parsed_text')
        WHERE cs.competition_id=$2 AND a.ts_content @@ plainto_tsquery('english',$1)
        ORDER BY score DESC LIMIT $3
        """
        competition_id = self._require_competition_id()
        async with self._require_pool().acquire() as connection:
            rows = await connection.fetch(sql, query, competition_id, top_k)
        return [self._retrieved_document(row, competition_id) for row in rows]

    def _require_pool(self) -> Any:
        if self.pool is None:
            raise RuntimeError("SourceRegistryStore is not initialized")
        return self.pool

    def _require_competition_id(self) -> str:
        if not self.competition_id:
            raise RuntimeError("competition_id is required for registry retrieval")
        return self.competition_id

    @staticmethod
    def _retrieved_document(row: Any, competition_id: str) -> RetrievedDocument:
        metadata = _metadata_to_dict(row["metadata"])
        metadata["version_id"] = str(row["version_id"])
        return RetrievedDocument(
            id=row["source_id"], competition_id=competition_id, source=row["source_type"],
            title=row["title"] or row["source_id"], url=row["canonical_url"],
            content=row["content"], score=float(row["score"]), rrf_score=0.0, metadata=metadata,
        )

    async def _validate_embedding_dimension(self, connection: Any) -> None:
        data_type = await connection.fetchval(
            """SELECT format_type(a.atttypid,a.atttypmod) FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
            WHERE c.relname='source_embeddings' AND a.attname='embedding' AND NOT a.attisdropped LIMIT 1"""
        )
        match = re.fullmatch(r"vector\((\d+)\)", str(data_type or ""))
        if match is None or int(match.group(1)) != self.embed_dim:
            raise EmbeddingCompatibilityError(
                f"source_embeddings.embedding is {data_type!r}; configured dimension is {self.embed_dim}. "
                "Embeddings are never truncated automatically."
            )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _require_record(value: Any, label: str) -> Any:
    if value is None:
        raise RuntimeError(f"Database did not return the expected {label} record")
    return value


def _source_record(row: Any) -> SourceRecord | None:
    if row is None: return None
    return SourceRecord(**{**dict(row), "metadata": _metadata_to_dict(row["metadata"])})


def _version_record(row: Any) -> SourceVersion | None:
    if row is None: return None
    return SourceVersion(**{**dict(row), "metadata": _metadata_to_dict(row["metadata"])})


def _artifact_record(row: Any) -> ArtifactRecord | None:
    if row is None: return None
    data = dict(row); data.pop("ts_content", None)
    data["metadata"] = _metadata_to_dict(row["metadata"])
    data["payload"] = _json_value(row["payload"])
    return ArtifactRecord(**data)


def _embedding_record(row: Any) -> EmbeddingRecord | None:
    if row is None: return None
    data = dict(row); data.pop("embedding", None); data["metadata"] = _metadata_to_dict(row["metadata"])
    return EmbeddingRecord(**data)


def _search_record(row: Any) -> SearchCacheEntry | None:
    if row is None: return None
    data = dict(row); data["result_source_ids"] = list(_json_value(row["result_source_ids"]) or [])
    data["raw_result_metadata"] = _metadata_to_dict(row["raw_result_metadata"])
    return SearchCacheEntry(**data)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try: return json.loads(value)
        except json.JSONDecodeError: return value
    return value
