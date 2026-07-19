from __future__ import annotations

import os
from uuid import uuid4

import pytest

from kaggle_researcher.source_registry.hashing import compute_content_hashes, sha256_text
from kaggle_researcher.source_registry.identity import canonicalize_source_identity
from kaggle_researcher.store.source_registry_store import SourceRegistryStore


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_registry_round_trip_and_dimension_guard() -> None:
    dsn = os.getenv("SOURCE_REGISTRY_TEST_DSN")
    if not dsn:
        pytest.skip("SOURCE_REGISTRY_TEST_DSN is not configured")
    embed_dim = int(os.getenv("SOURCE_REGISTRY_TEST_EMBED_DIM", "3"))
    store = SourceRegistryStore(dsn=dsn, embed_dim=embed_dim, competition_id="registry-integration")
    await store.init()
    external_id = f"owner/repo-{uuid4().hex}"
    identity = canonicalize_source_identity("github", external_id, None)
    try:
        source = await store.upsert_source(identity, "Repository", identity.canonical_url, {})
        version, created = await store.create_or_reuse_version(
            source.source_id, "abc123", "readme", None, "text/plain",
            compute_content_hashes("readme"), {"content_scope": "readme_only"},
        )
        assert created
        artifact = await store.save_artifact(
            version.version_id, "summary", "summary-fp", version.normalized_content_hash,
            sha256_text("summary"), payload={"text": "summary"},
        )
        embedding = await store.save_embedding(
            version.version_id, "summary", "embedding-fp", artifact.output_hash,
            [0.1] * embed_dim,
        )
        await store.associate_source_with_competition(
            "registry-integration", source.source_id, "github", "query", 1, {}
        )
        assert (await store.get_current_version(source.source_id)).version_id == version.version_id
        assert (await store.find_embedding(
            version.version_id, "summary", "embedding-fp", artifact.output_hash
        )).embedding_id == embedding.embedding_id
        with pytest.raises(Exception, match="dimension"):
            await store.save_embedding(
                version.version_id, "summary", "wrong-fp", "wrong", [0.1] * (embed_dim + 1)
            )
    finally:
        await store.close()
