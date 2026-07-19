from __future__ import annotations


SOURCE_REGISTRY_MIGRATION_VERSION = "source_registry_001"


def create_source_registry_migration_sql(embed_dim: int) -> str:
    if embed_dim <= 0:
        raise ValueError("embed_dim must be positive")
    return f"""
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    external_id TEXT NOT NULL,
    canonical_url TEXT,
    title TEXT,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    current_version_id UUID,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_checked_at TIMESTAMPTZ,
    source_status TEXT NOT NULL DEFAULT 'active'
        CHECK (source_status IN ('active','unavailable','deleted','blocked','unknown')),
    identity_version TEXT NOT NULL,
    UNIQUE (source_type, external_id)
);

CREATE TABLE IF NOT EXISTS source_versions (
    version_id UUID PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    source_revision TEXT,
    raw_content_hash TEXT NOT NULL,
    normalized_content_hash TEXT NOT NULL,
    raw_content TEXT,
    content_location TEXT,
    content_mime_type TEXT,
    content_size_bytes BIGINT,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE (source_id, normalized_content_hash)
);

CREATE TABLE IF NOT EXISTS source_artifacts (
    artifact_id UUID PRIMARY KEY,
    version_id UUID NOT NULL REFERENCES source_versions(version_id),
    artifact_type TEXT NOT NULL,
    processor_fingerprint TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    output_hash TEXT NOT NULL,
    payload JSONB,
    content_location TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    ts_content tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(payload->>'text', payload->>'summary', ''))
    ) STORED,
    UNIQUE (version_id, artifact_type, processor_fingerprint, input_hash)
);

CREATE TABLE IF NOT EXISTS source_embeddings (
    embedding_id UUID PRIMARY KEY,
    version_id UUID NOT NULL REFERENCES source_versions(version_id),
    input_kind TEXT NOT NULL,
    embedding_fingerprint TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    embedding vector({embed_dim}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    UNIQUE (version_id, input_kind, embedding_fingerprint, input_hash)
);

CREATE TABLE IF NOT EXISTS competition_sources (
    competition_id TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    discovery_queries JSONB NOT NULL DEFAULT '[]'::jsonb,
    discovery_providers JSONB NOT NULL DEFAULT '[]'::jsonb,
    best_discovery_rank INTEGER,
    relevance_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    PRIMARY KEY (competition_id, source_id)
);

CREATE TABLE IF NOT EXISTS research_run_sources (
    run_id TEXT NOT NULL,
    competition_id TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    version_id UUID REFERENCES source_versions(version_id),
    selected_for_retrieval BOOLEAN NOT NULL DEFAULT FALSE,
    selected_for_reasoning BOOLEAN NOT NULL DEFAULT FALSE,
    retrieval_score DOUBLE PRECISION,
    rrf_score DOUBLE PRECISION,
    artifact_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding_id UUID REFERENCES source_embeddings(embedding_id),
    cache_decisions JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, source_id)
);

CREATE TABLE IF NOT EXISTS source_search_cache (
    provider TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    normalized_query TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    result_source_ids JSONB NOT NULL,
    raw_result_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    fetched_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (provider, query_hash, request_fingerprint)
);

CREATE INDEX IF NOT EXISTS source_versions_source_id_idx ON source_versions(source_id);
CREATE INDEX IF NOT EXISTS source_versions_normalized_hash_idx ON source_versions(normalized_content_hash);
CREATE INDEX IF NOT EXISTS source_artifacts_version_type_idx ON source_artifacts(version_id, artifact_type);
CREATE INDEX IF NOT EXISTS source_artifacts_ts_content_gin_idx ON source_artifacts USING gin(ts_content);
CREATE INDEX IF NOT EXISTS source_embeddings_version_kind_idx ON source_embeddings(version_id, input_kind);
CREATE INDEX IF NOT EXISTS source_embeddings_hnsw_idx ON source_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS competition_sources_competition_idx ON competition_sources(competition_id);
CREATE INDEX IF NOT EXISTS research_run_sources_run_idx ON research_run_sources(run_id);
CREATE INDEX IF NOT EXISTS source_search_cache_expires_idx ON source_search_cache(expires_at);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'sources_current_version_fk') THEN
        ALTER TABLE sources ADD CONSTRAINT sources_current_version_fk
            FOREIGN KEY (current_version_id) REFERENCES source_versions(version_id);
    END IF;
END $$;

INSERT INTO schema_migrations(version) VALUES ('{SOURCE_REGISTRY_MIGRATION_VERSION}')
ON CONFLICT (version) DO NOTHING;
""".strip()
