from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class CanonicalSourceIdentity(StrictModel):
    source_id: str
    source_type: str
    external_id: str
    canonical_url: str | None = None
    identity_version: str = "1.0"
    identity_basis: str
    warnings: list[str] = Field(default_factory=list)


class ContentHashes(StrictModel):
    raw_hash: str
    normalized_hash: str
    normalization_policy_version: str
    normalized_size: int


SourceStatus = Literal["active", "unavailable", "deleted", "blocked", "unknown"]


class SourceRecord(StrictModel):
    source_id: str
    source_type: str
    external_id: str
    canonical_url: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    current_version_id: UUID | None = None
    first_seen_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)
    last_checked_at: datetime | None = None
    source_status: SourceStatus = "active"
    identity_version: str = "1.0"


class SourceVersion(StrictModel):
    version_id: UUID
    source_id: str
    source_revision: str | None = None
    raw_content_hash: str
    normalized_content_hash: str
    raw_content: str | None = Field(default=None, repr=False)
    content_location: str | None = None
    content_mime_type: str | None = None
    content_size_bytes: int | None = None
    fetched_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_current: bool = True


class ProcessorFingerprint(StrictModel):
    processor_name: str
    processor_version: str
    configuration: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str

    def __str__(self) -> str:
        return self.fingerprint


class ArtifactRecord(StrictModel):
    artifact_id: UUID
    version_id: UUID
    artifact_type: str
    processor_fingerprint: str
    input_hash: str
    output_hash: str
    payload: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    content_location: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingRecord(StrictModel):
    embedding_id: UUID
    version_id: UUID
    input_kind: str
    embedding_fingerprint: str
    input_hash: str
    embedding_dimension: int
    embedding: list[float] | None = Field(default=None, exclude=True, repr=False)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


CacheDecisionName = Literal[
    "cache_hit",
    "cache_miss",
    "forced_rebuild",
    "source_changed",
    "processor_changed",
    "input_changed",
    "not_cacheable",
    "error",
]


class CacheDecision(StrictModel):
    stage: str
    decision: CacheDecisionName
    source_id: str | None = None
    version_id: UUID | None = None
    artifact_id: UUID | None = None
    embedding_id: UUID | None = None
    reason: str
    previous_fingerprint: str | None = None
    current_fingerprint: str | None = None


class SourceDescriptor(StrictModel):
    source_type: str
    external_id: str | None = None
    canonical_url: str | None = None
    title: str | None = None
    source_revision: str | None = None
    revision_is_reliable: bool = False
    content_url: str | None = None
    content_mime_type: str | None = "text/plain"
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    discovery_query: str = ""
    discovery_rank: int | None = None
    raw_content: str | bytes | None = Field(default=None, repr=False)


class SourceProcessingResult(StrictModel):
    source: SourceRecord
    version: SourceVersion
    parsed_artifact: ArtifactRecord | None = None
    summary_artifact: ArtifactRecord | None = None
    static_analysis_artifacts: list[ArtifactRecord] = Field(default_factory=list)
    embedding: EmbeddingRecord | None = None
    cache_decisions: list[CacheDecision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SearchCacheEntry(StrictModel):
    provider: str
    query_hash: str
    normalized_query: str
    request_fingerprint: str
    result_source_ids: list[str]
    raw_result_metadata: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        return self.expires_at <= utc_now()


class SourceRefreshMode(str, Enum):
    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


class ArtifactRebuildMode(str, Enum):
    NONE = "none"
    PARSED = "parsed"
    SUMMARIES = "summaries"
    EMBEDDINGS = "embeddings"
    STATIC_ANALYSIS = "static_analysis"
    ALL = "all"


class CachePolicy(StrictModel):
    source_refresh_mode: SourceRefreshMode = SourceRefreshMode.AUTO
    rebuild_artifacts: set[str] = Field(default_factory=set)
    search_ttl_by_provider: dict[str, timedelta] = Field(
        default_factory=lambda: {
            "kaggle": timedelta(hours=24),
            "github": timedelta(hours=24),
            "arxiv": timedelta(days=7),
            "papers_with_code": timedelta(days=7),
        }
    )
    allow_stale_search_cache_when_offline: bool = True
    verify_source_revision: bool = True
    verify_content_hash: bool = True
    write_cache_telemetry: bool = False
    cache_enabled: bool = True

    @field_validator("rebuild_artifacts", mode="before")
    @classmethod
    def normalize_rebuild_artifacts(cls, value: Any) -> set[str]:
        if value is None:
            return set()
        values = {str(item).strip().lower().replace("-", "_") for item in value}
        aliases = {"summary": "summaries", "embedding": "embeddings", "static": "static_analysis"}
        values = {aliases.get(item, item) for item in values if item and item != "none"}
        allowed = {"parsed", "summaries", "embeddings", "static_analysis", "all"}
        unknown = values - allowed
        if unknown:
            raise ValueError(f"Unknown artifact rebuild stages: {', '.join(sorted(unknown))}")
        return values

    def rebuilds(self, stage: str) -> bool:
        normalized = stage.lower().replace("-", "_")
        aliases = {
            "parsed_text": "parsed",
            "summary": "summaries",
            "embedding": "embeddings",
            "notebook_static_analysis": "static_analysis",
            "repository_static_analysis": "static_analysis",
        }
        normalized = aliases.get(normalized, normalized)
        return not self.cache_enabled or "all" in self.rebuild_artifacts or normalized in self.rebuild_artifacts


class CacheRunTelemetry(StrictModel):
    run_id: str
    competition_id: str = ""
    sources_discovered: int = 0
    sources_reused: int = 0
    sources_new: int = 0
    sources_changed: int = 0
    sources_unavailable: int = 0
    downloads_skipped: int = 0
    downloads_performed: int = 0
    parses_reused: int = 0
    parses_computed: int = 0
    summaries_reused: int = 0
    summaries_computed: int = 0
    embeddings_reused: int = 0
    embeddings_computed: int = 0
    static_analyses_reused: int = 0
    static_analyses_computed: int = 0
    search_cache_hits: int = 0
    search_cache_misses: int = 0
    search_stale_hits: int = 0
    provider_calls: int = 0
    estimated_avoided_operations: int = 0
    per_source: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CompetitionSourceLink(StrictModel):
    competition_id: str
    source_id: str
    discovery_queries: list[str] = Field(default_factory=list)
    relevance_score: float | None = None
    relevance_summary: str | None = None
    first_seen_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


ProcessingStatus = Literal["success", "degraded", "failed"]


class ProcessingManifest(StrictModel):
    manifest_id: UUID
    run_id: str
    competition_id: str
    source_id: str
    version_id: UUID
    input_hash: str
    cache_decisions: list[CacheDecision] = Field(default_factory=list)
    processor_fingerprints: list[str] = Field(default_factory=list)
    started_at: datetime
    completed_at: datetime
    status: ProcessingStatus
    warnings: list[str] = Field(default_factory=list)
