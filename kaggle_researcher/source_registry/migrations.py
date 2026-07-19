from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from kaggle_researcher.config import DEFAULT_EMBED_DIM
from kaggle_researcher.source_registry.hashing import compute_content_hashes, sha256_text
from kaggle_researcher.source_registry.identity import canonicalize_source_identity
from kaggle_researcher.store.source_registry_store import SourceRegistryStore


LEGACY_UNKNOWN_SUMMARY_FINGERPRINT = "legacy:unknown:summary"
LEGACY_UNKNOWN_EMBEDDING_FINGERPRINT = "legacy:unknown:embedding"


class MigrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    legacy_rows_seen: int = 0
    sources_created: int = 0
    sources_reused: int = 0
    versions_created: int = 0
    artifacts_migrated: int = 0
    embeddings_migrated: int = 0
    embeddings_skipped: int = 0
    ambiguous_identities: int = 0
    failed_rows: int = 0
    warnings: list[str] = Field(default_factory=list)


async def migrate_legacy_documents(
    registry: Any,
    legacy_rows: Iterable[dict[str, Any]],
    *,
    dry_run: bool = False,
    known_summary_fingerprint: str | None = None,
    known_embedding_fingerprint: str | None = None,
) -> MigrationReport:
    report = MigrationReport()
    seen_sources: set[str] = set()
    seen_versions: set[tuple[str, str]] = set()
    for raw_row in legacy_rows:
        report.legacy_rows_seen += 1
        row = dict(raw_row)
        try:
            metadata = _dict_value(row.get("metadata"))
            source_type = str(row.get("source") or metadata.get("source") or "")
            external_id = (
                metadata.get("arxiv_id") or metadata.get("entry_id") or metadata.get("kernel_ref")
                or metadata.get("full_name") or metadata.get("id")
            )
            identity = canonicalize_source_identity(
                source_type, str(external_id) if external_id else None, row.get("url"), metadata
            )
            if identity.identity_basis == "canonical_url_hash":
                report.ambiguous_identities += 1
                report.warnings.extend(identity.warnings)
            content = str(row.get("content") or "")
            if not content:
                raise ValueError("legacy document content is empty")
            hashes = compute_content_hashes(content)
            source_was_seen = identity.source_id in seen_sources
            if not source_was_seen and not dry_run:
                source_was_seen = await registry.get_source(identity.source_id) is not None
            if source_was_seen:
                report.sources_reused += 1
            else:
                report.sources_created += 1
            seen_sources.add(identity.source_id)
            version_key = (identity.source_id, hashes.normalized_hash)
            version_created = version_key not in seen_versions
            if version_created and not dry_run:
                version_created = await registry.get_version_by_hash(*version_key) is None
            if version_created:
                report.versions_created += 1
            seen_versions.add(version_key)
            summary = row.get("summary")
            embedding = row.get("embedding")
            if dry_run:
                if summary:
                    report.artifacts_migrated += 1
                if embedding is not None:
                    if known_embedding_fingerprint and len(embedding) == registry.embed_dim:
                        report.embeddings_migrated += 1
                    else:
                        report.embeddings_skipped += 1
                continue
            source = await registry.upsert_source(
                identity, row.get("title"), row.get("url"), metadata, row.get("updated_at")
            )
            version, _ = await registry.create_or_reuse_version(
                source.source_id, metadata.get("source_revision"), content, None, "text/plain",
                hashes, {"migrated_from": "documents", "legacy_document_id": row.get("id")},
            )
            await registry.associate_source_with_competition(
                str(row.get("competition_id") or "unknown"), source.source_id,
                source.source_type, "legacy_migration", None, {},
            )
            if summary:
                fingerprint = known_summary_fingerprint or LEGACY_UNKNOWN_SUMMARY_FINGERPRINT
                await registry.save_artifact(
                    version.version_id, "summary", fingerprint, hashes.normalized_hash,
                    sha256_text(json.dumps({"text": str(summary)}, sort_keys=True, separators=(",", ":"))),
                    payload={"text": str(summary)}, metadata={"legacy_fingerprint_known": bool(known_summary_fingerprint)},
                )
                report.artifacts_migrated += 1
            if embedding is not None:
                if known_embedding_fingerprint and len(embedding) == registry.embed_dim:
                    await registry.save_embedding(
                        version.version_id, "summary" if summary else "raw_content",
                        known_embedding_fingerprint,
                        sha256_text(str(summary or content)), list(embedding),
                        {"migrated_from": "documents"},
                    )
                    report.embeddings_migrated += 1
                else:
                    report.embeddings_skipped += 1
                    report.warnings.append(
                        f"Skipped legacy embedding for {source.source_id}: compatible model fingerprint is unknown."
                    )
        except Exception as exc:
            report.failed_rows += 1
            report.warnings.append(
                f"Skipped malformed legacy document {row.get('id', '<unknown>')!r}: {type(exc).__name__}: {exc}"
            )
    return report


async def migrate_legacy_documents_from_postgres(
    registry: SourceRegistryStore, *, dry_run: bool = False
) -> MigrationReport:
    pool = registry._require_pool()
    async with pool.acquire() as connection:
        exists = await connection.fetchval("SELECT to_regclass('documents') IS NOT NULL")
        rows = await connection.fetch("SELECT * FROM documents ORDER BY created_at,id") if exists else []
    return await migrate_legacy_documents(registry, (dict(row) for row in rows), dry_run=dry_run)


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


async def _run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kaggle_researcher.source_registry.migrations")
    parser.add_argument("command", choices=("migrate-legacy-documents",))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-path", default="source_registry_migration_report.json")
    args = parser.parse_args(argv)
    registry = SourceRegistryStore(
        dsn=os.getenv("PG_DSN", "postgresql://researcher:researcher@localhost:5432/kaggle_research"),
        embed_dim=int(os.getenv("EMBED_DIM", str(DEFAULT_EMBED_DIM))),
    )
    await registry.init()
    try:
        report = await migrate_legacy_documents_from_postgres(registry, dry_run=args.dry_run)
    finally:
        await registry.close()
    Path(args.report_path).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(_run_cli()))
