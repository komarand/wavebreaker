from __future__ import annotations

from kaggle_researcher.config import DEFAULT_EMBED_DIM


CREATE_VECTOR_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"


def create_documents_table_sql(embed_dim: int) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    competition_id  TEXT NOT NULL,
    source          TEXT NOT NULL,
    title           TEXT NOT NULL,
    url             TEXT,
    content         TEXT NOT NULL,
    summary         TEXT,
    metadata        JSONB DEFAULT '{{}}'::jsonb,
    embedding       vector({embed_dim}),
    ts_content      tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(summary, content))
    ) STORED,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);
""".strip()


def create_competition_patterns_table_sql(embed_dim: int) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS competition_patterns (
    id                    TEXT PRIMARY KEY,
    competition_family    TEXT NOT NULL,
    task_type             TEXT,
    domain                TEXT,
    pattern_text          TEXT NOT NULL,
    embedding             vector({embed_dim}),
    typical_models        JSONB,
    typical_features      JSONB,
    typical_validation    TEXT,
    common_traps          JSONB,
    source_competition_id TEXT,
    created_at            TIMESTAMPTZ DEFAULT now(),
    updated_at            TIMESTAMPTZ DEFAULT now()
);
""".strip()


CREATE_DOCUMENTS_TABLE_SQL = create_documents_table_sql(DEFAULT_EMBED_DIM)
CREATE_COMPETITION_PATTERNS_TABLE_SQL = create_competition_patterns_table_sql(DEFAULT_EMBED_DIM)


CREATE_DOCUMENTS_EMBEDDING_HNSW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
""".strip()


CREATE_DOCUMENTS_TS_CONTENT_GIN_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS documents_ts_content_gin_idx
ON documents USING gin (ts_content);
""".strip()


CREATE_DOCUMENTS_COMPETITION_ID_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS documents_competition_id_idx
ON documents (competition_id);
""".strip()


CREATE_COMPETITION_PATTERNS_EMBEDDING_HNSW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS competition_patterns_embedding_hnsw_idx
ON competition_patterns USING hnsw (embedding vector_cosine_ops);
""".strip()
