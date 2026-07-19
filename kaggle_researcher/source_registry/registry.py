from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from kaggle_researcher.source_registry.errors import EmbeddingCompatibilityError, SourceRegistryError
from kaggle_researcher.source_registry.schemas import (
    ArtifactRecord,
    CanonicalSourceIdentity,
    ContentHashes,
    EmbeddingRecord,
    SearchCacheEntry,
    SourceRecord,
    SourceVersion,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class InMemorySourceRegistryStore:
    """Transaction-like offline store used by deterministic unit and pipeline tests."""

    def __init__(self, *, embed_dim: int = 1024) -> None:
        self.embed_dim = embed_dim
        self.sources: dict[str, SourceRecord] = {}
        self.versions: dict[UUID, SourceVersion] = {}
        self.version_keys: dict[tuple[str, str], UUID] = {}
        self.artifacts: dict[UUID, ArtifactRecord] = {}
        self.artifact_keys: dict[tuple[UUID, str, str, str], UUID] = {}
        self.embeddings: dict[UUID, EmbeddingRecord] = {}
        self.embedding_keys: dict[tuple[UUID, str, str, str], UUID] = {}
        self.competition_sources: dict[tuple[str, str], dict[str, Any]] = {}
        self.run_sources: dict[tuple[str, str], dict[str, Any]] = {}
        self.search_entries: dict[tuple[str, str, str], SearchCacheEntry] = {}
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def get_source(self, source_id: str) -> SourceRecord | None:
        value = self.sources.get(source_id)
        return deepcopy(value) if value else None

    async def upsert_source(
        self,
        identity: CanonicalSourceIdentity,
        title: str | None,
        canonical_url: str | None,
        metadata: dict[str, Any],
        checked_at: datetime | None = None,
    ) -> SourceRecord:
        now = _now()
        async with self._lock:
            existing = self.sources.get(identity.source_id)
            if existing is None:
                existing = SourceRecord(
                    source_id=identity.source_id,
                    source_type=identity.source_type,
                    external_id=identity.external_id,
                    canonical_url=canonical_url or identity.canonical_url,
                    title=title,
                    metadata=deepcopy(metadata),
                    first_seen_at=now,
                    last_seen_at=now,
                    last_checked_at=checked_at,
                    source_status="active",
                    identity_version=identity.identity_version,
                )
            else:
                existing = existing.model_copy(update={
                    "canonical_url": canonical_url or identity.canonical_url or existing.canonical_url,
                    "title": title or existing.title,
                    "metadata": {**existing.metadata, **deepcopy(metadata)},
                    "last_seen_at": now,
                    "last_checked_at": checked_at or existing.last_checked_at,
                    "source_status": "active",
                })
            self.sources[identity.source_id] = existing
            return deepcopy(existing)

    async def mark_source_status(self, source_id: str, status: str, checked_at: datetime) -> None:
        if status not in {"active", "unavailable", "deleted", "blocked", "unknown"}:
            raise ValueError(f"Invalid source status: {status}")
        async with self._lock:
            source = self.sources.get(source_id)
            if source is None:
                raise SourceRegistryError(f"Unknown source: {source_id}")
            self.sources[source_id] = source.model_copy(update={"source_status": status, "last_checked_at": checked_at})

    async def get_current_version(self, source_id: str) -> SourceVersion | None:
        source = self.sources.get(source_id)
        if source is None or source.current_version_id is None:
            return None
        value = self.versions.get(source.current_version_id)
        return deepcopy(value) if value else None

    async def get_version_by_hash(self, source_id: str, normalized_content_hash: str) -> SourceVersion | None:
        version_id = self.version_keys.get((source_id, normalized_content_hash))
        value = self.versions.get(version_id) if version_id else None
        return deepcopy(value) if value else None

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
        async with self._lock:
            if source_id not in self.sources:
                raise SourceRegistryError(f"Unknown source: {source_id}")
            key = (source_id, hashes.normalized_hash)
            existing_id = self.version_keys.get(key)
            created = existing_id is None
            if existing_id is None:
                version = SourceVersion(
                    version_id=uuid4(),
                    source_id=source_id,
                    source_revision=source_revision,
                    raw_content_hash=hashes.raw_hash,
                    normalized_content_hash=hashes.normalized_hash,
                    raw_content=raw_content,
                    content_location=content_location,
                    content_mime_type=content_mime_type,
                    content_size_bytes=len(raw_content.encode("utf-8")) if raw_content is not None else None,
                    metadata={**deepcopy(metadata), "normalization_policy_version": hashes.normalization_policy_version},
                    is_current=True,
                )
                self.versions[version.version_id] = version
                self.version_keys[key] = version.version_id
            else:
                version = self.versions[existing_id]
                updates: dict[str, Any] = {
                    "metadata": {**version.metadata, **deepcopy(metadata)},
                }
                if source_revision is not None:
                    updates["source_revision"] = source_revision
                if version.raw_content is None and raw_content is not None:
                    updates["raw_content"] = raw_content
                    updates["content_size_bytes"] = len(raw_content.encode("utf-8"))
                version = version.model_copy(update=updates)
                self.versions[existing_id] = version
            for version_id, old in list(self.versions.items()):
                if old.source_id == source_id and old.is_current != (version_id == version.version_id):
                    self.versions[version_id] = old.model_copy(update={"is_current": version_id == version.version_id})
            self.sources[source_id] = self.sources[source_id].model_copy(update={"current_version_id": version.version_id})
            return deepcopy(self.versions[version.version_id]), created

    async def set_current_version(self, source_id: str, version_id: UUID) -> None:
        async with self._lock:
            version = self.versions.get(version_id)
            if version is None or version.source_id != source_id:
                raise SourceRegistryError("Version does not belong to source")
            for current_id, old in list(self.versions.items()):
                if old.source_id == source_id:
                    self.versions[current_id] = old.model_copy(update={"is_current": current_id == version_id})
            self.sources[source_id] = self.sources[source_id].model_copy(update={"current_version_id": version_id})

    async def find_artifact(
        self, version_id: UUID, artifact_type: str, processor_fingerprint: str, input_hash: str
    ) -> ArtifactRecord | None:
        artifact_id = self.artifact_keys.get((version_id, artifact_type, processor_fingerprint, input_hash))
        value = self.artifacts.get(artifact_id) if artifact_id else None
        return deepcopy(value) if value else None

    async def find_latest_artifact(self, version_id: UUID, artifact_type: str) -> ArtifactRecord | None:
        values = [item for item in self.artifacts.values() if item.version_id == version_id and item.artifact_type == artifact_type]
        value = max(values, key=lambda item: item.created_at) if values else None
        return deepcopy(value) if value else None

    async def save_artifact(
        self,
        version_id: UUID,
        artifact_type: str,
        processor_fingerprint: str,
        input_hash: str,
        output_hash: str,
        payload: Any = None,
        content_location: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        key = (version_id, artifact_type, processor_fingerprint, input_hash)
        async with self._lock:
            artifact_id = self.artifact_keys.get(key)
            if artifact_id is None:
                record = ArtifactRecord(
                    artifact_id=uuid4(), version_id=version_id, artifact_type=artifact_type,
                    processor_fingerprint=processor_fingerprint, input_hash=input_hash,
                    output_hash=output_hash, payload=deepcopy(payload), content_location=content_location,
                    metadata=deepcopy(metadata or {}),
                )
                self.artifacts[record.artifact_id] = record
                self.artifact_keys[key] = record.artifact_id
            else:
                record = self.artifacts[artifact_id]
            return deepcopy(record)

    async def find_embedding(
        self, version_id: UUID, input_kind: str, embedding_fingerprint: str, input_hash: str
    ) -> EmbeddingRecord | None:
        embedding_id = self.embedding_keys.get((version_id, input_kind, embedding_fingerprint, input_hash))
        value = self.embeddings.get(embedding_id) if embedding_id else None
        return deepcopy(value) if value else None

    async def find_latest_embedding(self, version_id: UUID, input_kind: str) -> EmbeddingRecord | None:
        values = [item for item in self.embeddings.values() if item.version_id == version_id and item.input_kind == input_kind]
        value = max(values, key=lambda item: item.created_at) if values else None
        return deepcopy(value) if value else None

    async def save_embedding(
        self,
        version_id: UUID,
        input_kind: str,
        embedding_fingerprint: str,
        input_hash: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingRecord:
        if len(embedding) != self.embed_dim:
            raise EmbeddingCompatibilityError(
                f"Embedding dimension {len(embedding)} does not match configured dimension {self.embed_dim}"
            )
        key = (version_id, input_kind, embedding_fingerprint, input_hash)
        async with self._lock:
            embedding_id = self.embedding_keys.get(key)
            if embedding_id is None:
                record = EmbeddingRecord(
                    embedding_id=uuid4(), version_id=version_id, input_kind=input_kind,
                    embedding_fingerprint=embedding_fingerprint, input_hash=input_hash,
                    embedding_dimension=len(embedding), embedding=list(embedding), metadata=deepcopy(metadata or {}),
                )
                self.embeddings[record.embedding_id] = record
                self.embedding_keys[key] = record.embedding_id
            else:
                record = self.embeddings[embedding_id]
            return deepcopy(record)

    async def associate_source_with_competition(
        self,
        competition_id: str,
        source_id: str,
        provider: str,
        query: str,
        discovery_rank: int | None,
        relevance_metadata: dict[str, Any],
    ) -> None:
        now = _now()
        key = (competition_id, source_id)
        async with self._lock:
            existing = self.competition_sources.get(key, {})
            queries = list(existing.get("discovery_queries", []))
            providers = list(existing.get("discovery_providers", []))
            if query and query not in queries:
                queries.append(query)
            if provider and provider not in providers:
                providers.append(provider)
            ranks = [rank for rank in (existing.get("best_discovery_rank"), discovery_rank) if rank is not None]
            self.competition_sources[key] = {
                "competition_id": competition_id, "source_id": source_id,
                "first_seen_at": existing.get("first_seen_at", now), "last_seen_at": now,
                "discovery_queries": queries, "discovery_providers": providers,
                "best_discovery_rank": min(ranks) if ranks else None,
                "relevance_metadata": {**existing.get("relevance_metadata", {}), **deepcopy(relevance_metadata)},
            }

    async def record_run_source(self, run_id: str, competition_id: str, source_id: str, **values: Any) -> None:
        key = (run_id, source_id)
        async with self._lock:
            existing = self.run_sources.get(key, {})
            self.run_sources[key] = {
                **existing, "run_id": run_id, "competition_id": competition_id,
                "source_id": source_id, **deepcopy(values), "recorded_at": existing.get("recorded_at", _now()),
            }

    async def get_search_cache(self, provider: str, query_hash: str, request_fingerprint: str) -> SearchCacheEntry | None:
        value = self.search_entries.get((provider, query_hash, request_fingerprint))
        return deepcopy(value) if value else None

    async def save_search_cache(self, entry: SearchCacheEntry) -> SearchCacheEntry:
        async with self._lock:
            self.search_entries[(entry.provider, entry.query_hash, entry.request_fingerprint)] = deepcopy(entry)
        return deepcopy(entry)

    async def prune_expired_search_cache(self, before: datetime) -> int:
        async with self._lock:
            keys = [key for key, value in self.search_entries.items() if value.expires_at < before]
            for key in keys:
                del self.search_entries[key]
            return len(keys)

    async def list_competition_source_ids(self, competition_id: str) -> list[str]:
        return [source_id for (current_competition, source_id) in self.competition_sources if current_competition == competition_id]

    async def get_embedding_vector(self, embedding_id: UUID) -> list[float] | None:
        record = self.embeddings.get(embedding_id)
        return list(record.embedding) if record and record.embedding is not None else None
