from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from kaggle_researcher.source_registry.fingerprints import build_processor_fingerprint
from kaggle_researcher.source_registry.hashing import compute_content_hashes, sha256_text
from kaggle_researcher.source_registry.identity import canonicalize_source_identity
from kaggle_researcher.source_registry.repository import SQLiteSourceRegistryRepository
from kaggle_researcher.source_registry.schemas import (
    ArtifactRecord,
    CacheDecision,
    CachePolicy,
    CompetitionSourceLink,
    EmbeddingRecord,
    ProcessingManifest,
    SourceDescriptor,
    SourceProcessingResult,
    SourceRecord,
    SourceVersion,
    utc_now,
)


class SourceRegistryService:
    def __init__(self, repository: SQLiteSourceRegistryRepository) -> None:
        self.repository = repository
        self.repository.initialize()

    def register_source(
        self,
        descriptor: SourceDescriptor,
        *,
        competition_id: str,
        policy: CachePolicy | None = None,
    ) -> SourceProcessingResult:
        policy = policy or CachePolicy()
        identity = canonicalize_source_identity(
            descriptor.source_type,
            descriptor.external_id,
            descriptor.canonical_url,
            descriptor.provider_metadata,
        )
        now = utc_now()
        existing_source = self.repository.get_source(identity.source_id)
        source = SourceRecord(
            source_id=identity.source_id,
            source_type=identity.source_type,
            external_id=identity.external_id,
            canonical_url=identity.canonical_url,
            title=descriptor.title or (existing_source.title if existing_source else None),
            metadata={
                **(existing_source.metadata if existing_source else {}),
                **descriptor.provider_metadata,
            },
            current_version_id=existing_source.current_version_id if existing_source else None,
            first_seen_at=existing_source.first_seen_at if existing_source else now,
            last_seen_at=now,
            last_checked_at=now,
            source_status="active",
            identity_version=identity.identity_version,
        )
        decisions: list[CacheDecision] = []

        if descriptor.raw_content is None:
            if source.current_version_id is None:
                raise ValueError(
                    f"Source {source.source_id} has no supplied content and no cached version"
                )
            version = self.repository.get_version(source.current_version_id)
            if version is None:
                raise RuntimeError(
                    f"Source {source.source_id} points to missing version {source.current_version_id}"
                )
            decisions.append(CacheDecision(
                stage="source_version", decision="cache_hit", source_id=source.source_id,
                version_id=version.version_id, reason="Reused current version without a provider fetch.",
            ))
        else:
            content_type, normalization_policy = _content_hash_policy(descriptor.content_mime_type)
            hashes = compute_content_hashes(
                descriptor.raw_content,
                content_type=content_type,
                policy_version=normalization_policy,
            )
            cached_version = self.repository.find_version_by_hash(
                source.source_id, hashes.normalized_hash
            )
            if cached_version is not None:
                version = cached_version
                decisions.append(CacheDecision(
                    stage="source_version", decision="cache_hit", source_id=source.source_id,
                    version_id=version.version_id, reason="Normalized content hash already exists.",
                ))
            else:
                raw_bytes = (
                    descriptor.raw_content.encode("utf-8")
                    if isinstance(descriptor.raw_content, str)
                    else bytes(descriptor.raw_content)
                )
                version = SourceVersion(
                    version_id=uuid4(),
                    source_id=source.source_id,
                    source_revision=descriptor.source_revision,
                    raw_content_hash=hashes.raw_hash,
                    normalized_content_hash=hashes.normalized_hash,
                    raw_content=descriptor.raw_content if isinstance(descriptor.raw_content, str) else None,
                    content_location=None,
                    content_mime_type=descriptor.content_mime_type,
                    content_size_bytes=len(raw_bytes),
                    fetched_at=now,
                    metadata={
                        "normalization_policy_version": hashes.normalization_policy_version,
                        "revision_is_reliable": descriptor.revision_is_reliable,
                    },
                    is_current=True,
                )
                self.repository.save_source(source)
                self.repository.save_version(version, raw_bytes=raw_bytes)
                decisions.append(CacheDecision(
                    stage="source_version",
                    decision="forced_rebuild" if cached_version else "cache_miss",
                    source_id=source.source_id,
                    version_id=version.version_id,
                    reason="Stored a new immutable source version.",
                ))

        source = source.model_copy(update={"current_version_id": version.version_id})
        self.repository.save_source(source)
        self._link_competition(source.source_id, competition_id, descriptor)
        return SourceProcessingResult(
            source=source,
            version=version,
            cache_decisions=decisions,
            warnings=list(identity.warnings),
        )

    def get_or_create_artifact(
        self,
        result: SourceProcessingResult,
        *,
        artifact_type: str,
        processor_name: str,
        processor_version: str,
        configuration: dict[str, Any] | None,
        input_hash: str,
        producer: Callable[[], Any],
        policy: CachePolicy | None = None,
    ) -> ArtifactRecord:
        policy = policy or CachePolicy()
        fingerprint = build_processor_fingerprint(
            processor_name, processor_version, configuration
        )
        cached = self.repository.find_artifact(
            result.version.version_id, artifact_type, fingerprint.fingerprint, input_hash
        )
        if cached is not None and not policy.rebuilds(artifact_type):
            result.cache_decisions.append(CacheDecision(
                stage=artifact_type, decision="cache_hit", source_id=result.source.source_id,
                version_id=result.version.version_id, artifact_id=cached.artifact_id,
                reason="Artifact input and processor fingerprint match.",
                current_fingerprint=fingerprint.fingerprint,
            ))
            return cached

        payload = producer()
        output_hash = sha256_text(_canonical_payload(payload))
        artifact = ArtifactRecord(
            artifact_id=cached.artifact_id if cached is not None else uuid4(),
            version_id=result.version.version_id,
            artifact_type=artifact_type,
            processor_fingerprint=fingerprint.fingerprint,
            input_hash=input_hash,
            output_hash=output_hash,
            payload=payload,
            metadata={
                "processor_name": processor_name,
                "processor_version": processor_version,
                "processor_configuration": configuration or {},
            },
        )
        self.repository.save_artifact(artifact)
        result.cache_decisions.append(CacheDecision(
            stage=artifact_type,
            decision="forced_rebuild" if cached else "cache_miss",
            source_id=result.source.source_id, version_id=result.version.version_id,
            artifact_id=artifact.artifact_id, reason="Computed and persisted artifact.",
            current_fingerprint=fingerprint.fingerprint,
        ))
        return artifact

    def get_or_create_embedding(
        self,
        result: SourceProcessingResult,
        *,
        input_kind: str,
        input_text: str,
        model_name: str,
        model_revision: str,
        dimension: int,
        producer: Callable[[], list[float]],
        configuration: dict[str, Any] | None = None,
        policy: CachePolicy | None = None,
    ) -> EmbeddingRecord:
        policy = policy or CachePolicy()
        input_hash = sha256_text(input_text)
        fingerprint = build_processor_fingerprint(
            f"embedding:{model_name}", model_revision, configuration
        )
        cached = self.repository.find_embedding(
            result.version.version_id, input_kind, fingerprint.fingerprint, input_hash
        )
        if cached is not None and not policy.rebuilds("embeddings"):
            result.cache_decisions.append(CacheDecision(
                stage="embeddings", decision="cache_hit", source_id=result.source.source_id,
                version_id=result.version.version_id, embedding_id=cached.embedding_id,
                reason="Embedding input and model fingerprint match.",
                current_fingerprint=fingerprint.fingerprint,
            ))
            return cached

        vector = producer()
        if len(vector) != dimension:
            raise ValueError(
                f"Embedding dimension {len(vector)} does not match declared dimension {dimension}"
            )
        embedding = EmbeddingRecord(
            embedding_id=cached.embedding_id if cached is not None else uuid4(),
            version_id=result.version.version_id,
            input_kind=input_kind,
            embedding_fingerprint=fingerprint.fingerprint,
            input_hash=input_hash,
            embedding_dimension=dimension,
            embedding=[float(value) for value in vector],
            metadata={"model_name": model_name, "model_revision": model_revision},
        )
        self.repository.save_embedding(embedding)
        result.cache_decisions.append(CacheDecision(
            stage="embeddings",
            decision="forced_rebuild" if cached is not None else "cache_miss",
            source_id=result.source.source_id,
            version_id=result.version.version_id, embedding_id=embedding.embedding_id,
            reason="Computed and persisted embedding.",
            current_fingerprint=fingerprint.fingerprint,
        ))
        return embedding

    def write_manifest(
        self,
        result: SourceProcessingResult,
        *,
        run_id: str,
        competition_id: str,
        started_at: datetime,
        status: str = "success",
    ) -> ProcessingManifest:
        manifest = ProcessingManifest(
            manifest_id=uuid4(), run_id=run_id, competition_id=competition_id,
            source_id=result.source.source_id, version_id=result.version.version_id,
            input_hash=result.version.normalized_content_hash,
            cache_decisions=result.cache_decisions,
            started_at=started_at, completed_at=utc_now(), status=status,
            warnings=result.warnings,
        )
        self.repository.save_manifest(manifest)
        return manifest

    def _link_competition(
        self, source_id: str, competition_id: str, descriptor: SourceDescriptor
    ) -> None:
        if not competition_id.strip():
            raise ValueError("competition_id must not be empty")
        current = self.repository.list_competition_links(competition_id)
        existing = next((item for item in current if item.source_id == source_id), None)
        queries = list(existing.discovery_queries if existing else [])
        if descriptor.discovery_query and descriptor.discovery_query not in queries:
            queries.append(descriptor.discovery_query)
        now = utc_now()
        self.repository.save_competition_link(CompetitionSourceLink(
            competition_id=competition_id, source_id=source_id,
            discovery_queries=queries,
            first_seen_at=existing.first_seen_at if existing else now,
            last_seen_at=now,
            metadata={"discovery_rank": descriptor.discovery_rank},
        ))


def _content_hash_policy(mime_type: str | None) -> tuple[str, str]:
    normalized = (mime_type or "text/plain").lower()
    if "ipynb" in normalized or "notebook" in normalized:
        return "notebook", "notebook-v1"
    if normalized == "application/pdf" or not normalized.startswith("text/"):
        return "binary", "binary-v1"
    return "text", "text-v1"


def _canonical_payload(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
