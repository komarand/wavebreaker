from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

from kaggle_researcher.source_registry.schemas import (
    ArtifactRecord,
    CompetitionSourceLink,
    EmbeddingRecord,
    ProcessingManifest,
    SearchCacheEntry,
    SourceRecord,
    SourceVersion,
)


SCHEMA_VERSION = 1


class SQLiteSourceRegistryRepository:
    """Persistent, deterministic source registry backed by stdlib SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA_SQL)
            current = connection.execute("PRAGMA user_version").fetchone()[0]
            if current not in {0, SCHEMA_VERSION}:
                raise RuntimeError(
                    f"Unsupported source registry schema version {current}; expected {SCHEMA_VERSION}"
                )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def get_source(self, source_id: str) -> SourceRecord | None:
        row = self._fetchone("SELECT payload FROM sources WHERE source_id = ?", (source_id,))
        return SourceRecord.model_validate_json(row["payload"]) if row else None

    def save_source(self, source: SourceRecord) -> None:
        self._execute(
            """
            INSERT INTO sources(source_id, payload) VALUES (?, ?)
            ON CONFLICT(source_id) DO UPDATE SET payload = excluded.payload
            """,
            (source.source_id, source.model_dump_json()),
        )

    def get_version(self, version_id: UUID) -> SourceVersion | None:
        row = self._fetchone(
            "SELECT payload FROM source_versions WHERE version_id = ?", (str(version_id),)
        )
        return SourceVersion.model_validate_json(row["payload"]) if row else None

    def find_version_by_hash(self, source_id: str, normalized_hash: str) -> SourceVersion | None:
        row = self._fetchone(
            """
            SELECT payload FROM source_versions
            WHERE source_id = ? AND normalized_content_hash = ?
            ORDER BY fetched_at DESC LIMIT 1
            """,
            (source_id, normalized_hash),
        )
        return SourceVersion.model_validate_json(row["payload"]) if row else None

    def save_version(self, version: SourceVersion, raw_bytes: bytes | None = None) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if version.is_current:
                rows = connection.execute(
                    "SELECT version_id, payload FROM source_versions WHERE source_id = ? AND is_current = 1",
                    (version.source_id,),
                ).fetchall()
                for row in rows:
                    previous = SourceVersion.model_validate_json(row["payload"]).model_copy(
                        update={"is_current": False}
                    )
                    connection.execute(
                        "UPDATE source_versions SET is_current = 0, payload = ? WHERE version_id = ?",
                        (previous.model_dump_json(), row["version_id"]),
                    )
            connection.execute(
                """
                INSERT INTO source_versions(
                    version_id, source_id, normalized_content_hash, fetched_at, is_current,
                    raw_bytes, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(version_id) DO UPDATE SET
                    is_current = excluded.is_current,
                    raw_bytes = COALESCE(excluded.raw_bytes, source_versions.raw_bytes),
                    payload = excluded.payload
                """,
                (
                    str(version.version_id),
                    version.source_id,
                    version.normalized_content_hash,
                    version.fetched_at.isoformat(),
                    int(version.is_current),
                    raw_bytes,
                    version.model_dump_json(),
                ),
            )

    def load_raw_bytes(self, version_id: UUID) -> bytes | None:
        row = self._fetchone(
            "SELECT raw_bytes FROM source_versions WHERE version_id = ?", (str(version_id),)
        )
        return bytes(row["raw_bytes"]) if row and row["raw_bytes"] is not None else None

    def find_artifact(
        self,
        version_id: UUID,
        artifact_type: str,
        processor_fingerprint: str,
        input_hash: str,
    ) -> ArtifactRecord | None:
        row = self._fetchone(
            """
            SELECT payload FROM artifacts
            WHERE version_id = ? AND artifact_type = ?
              AND processor_fingerprint = ? AND input_hash = ?
            """,
            (str(version_id), artifact_type, processor_fingerprint, input_hash),
        )
        return ArtifactRecord.model_validate_json(row["payload"]) if row else None

    def save_artifact(self, artifact: ArtifactRecord) -> None:
        self._execute(
            """
            INSERT INTO artifacts(
                artifact_id, version_id, artifact_type, processor_fingerprint, input_hash, payload
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(version_id, artifact_type, processor_fingerprint, input_hash)
            DO UPDATE SET payload = excluded.payload
            """,
            (
                str(artifact.artifact_id), str(artifact.version_id), artifact.artifact_type,
                artifact.processor_fingerprint, artifact.input_hash, artifact.model_dump_json(),
            ),
        )

    def find_embedding(
        self,
        version_id: UUID,
        input_kind: str,
        embedding_fingerprint: str,
        input_hash: str,
    ) -> EmbeddingRecord | None:
        row = self._fetchone(
            """
            SELECT payload, embedding FROM embeddings
            WHERE version_id = ? AND input_kind = ?
              AND embedding_fingerprint = ? AND input_hash = ?
            """,
            (str(version_id), input_kind, embedding_fingerprint, input_hash),
        )
        if not row:
            return None
        record = EmbeddingRecord.model_validate_json(row["payload"])
        vector = json.loads(row["embedding"]) if row["embedding"] is not None else None
        return record.model_copy(update={"embedding": vector})

    def save_embedding(self, embedding: EmbeddingRecord) -> None:
        self._execute(
            """
            INSERT INTO embeddings(
                embedding_id, version_id, input_kind, embedding_fingerprint,
                input_hash, embedding, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(version_id, input_kind, embedding_fingerprint, input_hash)
            DO UPDATE SET embedding = excluded.embedding, payload = excluded.payload
            """,
            (
                str(embedding.embedding_id), str(embedding.version_id), embedding.input_kind,
                embedding.embedding_fingerprint, embedding.input_hash,
                json.dumps(embedding.embedding) if embedding.embedding is not None else None,
                embedding.model_dump_json(),
            ),
        )

    def save_competition_link(self, link: CompetitionSourceLink) -> None:
        self._execute(
            """
            INSERT INTO competition_sources(competition_id, source_id, payload) VALUES (?, ?, ?)
            ON CONFLICT(competition_id, source_id) DO UPDATE SET payload = excluded.payload
            """,
            (link.competition_id, link.source_id, link.model_dump_json()),
        )

    def list_competition_links(self, competition_id: str) -> list[CompetitionSourceLink]:
        rows = self._fetchall(
            "SELECT payload FROM competition_sources WHERE competition_id = ? ORDER BY source_id",
            (competition_id,),
        )
        return [CompetitionSourceLink.model_validate_json(row["payload"]) for row in rows]

    def save_manifest(self, manifest: ProcessingManifest) -> None:
        self._execute(
            """
            INSERT INTO processing_manifests(manifest_id, run_id, source_id, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(manifest_id) DO UPDATE SET payload = excluded.payload
            """,
            (str(manifest.manifest_id), manifest.run_id, manifest.source_id, manifest.model_dump_json()),
        )

    def save_search_cache(self, entry: SearchCacheEntry) -> None:
        self._execute(
            """
            INSERT INTO search_cache(provider, query_hash, request_fingerprint, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(provider, query_hash, request_fingerprint)
            DO UPDATE SET payload = excluded.payload
            """,
            (entry.provider, entry.query_hash, entry.request_fingerprint, entry.model_dump_json()),
        )

    def get_search_cache(
        self, provider: str, query_hash: str, request_fingerprint: str
    ) -> SearchCacheEntry | None:
        row = self._fetchone(
            """
            SELECT payload FROM search_cache
            WHERE provider = ? AND query_hash = ? AND request_fingerprint = ?
            """,
            (provider, query_hash, request_fingerprint),
        )
        return SearchCacheEntry.model_validate_json(row["payload"]) if row else None

    def count(self, table: str) -> int:
        allowed = {
            "sources", "source_versions", "artifacts", "embeddings",
            "competition_sources", "processing_manifests", "search_cache",
        }
        if table not in allowed:
            raise ValueError(f"Unsupported registry table: {table}")
        row = self._fetchone(f"SELECT COUNT(*) AS count FROM {table}", ())
        return int(row["count"])

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _execute(self, sql: str, params: Iterable[Any]) -> None:
        with self._connect() as connection:
            connection.execute(sql, tuple(params))

    def _fetchone(self, sql: str, params: Iterable[Any]) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(sql, tuple(params)).fetchone()

    def _fetchall(self, sql: str, params: Iterable[Any]) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(connection.execute(sql, tuple(params)).fetchall())


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_versions (
    version_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    normalized_content_hash TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    is_current INTEGER NOT NULL,
    raw_bytes BLOB,
    payload TEXT NOT NULL,
    UNIQUE(source_id, normalized_content_hash)
);
CREATE UNIQUE INDEX IF NOT EXISTS source_versions_one_current
ON source_versions(source_id) WHERE is_current = 1;
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL REFERENCES source_versions(version_id),
    artifact_type TEXT NOT NULL,
    processor_fingerprint TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(version_id, artifact_type, processor_fingerprint, input_hash)
);
CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL REFERENCES source_versions(version_id),
    input_kind TEXT NOT NULL,
    embedding_fingerprint TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    embedding TEXT,
    payload TEXT NOT NULL,
    UNIQUE(version_id, input_kind, embedding_fingerprint, input_hash)
);
CREATE TABLE IF NOT EXISTS competition_sources (
    competition_id TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    payload TEXT NOT NULL,
    PRIMARY KEY(competition_id, source_id)
);
CREATE TABLE IF NOT EXISTS processing_manifests (
    manifest_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS search_cache (
    provider TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(provider, query_hash, request_fingerprint)
);
"""
