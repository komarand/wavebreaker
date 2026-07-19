import asyncio

from kaggle_researcher.source_registry.hashing import compute_content_hashes
from kaggle_researcher.source_registry.identity import canonicalize_source_identity
from kaggle_researcher.source_registry.registry import InMemorySourceRegistryStore


def run(value):
    return asyncio.run(value)


def test_versions_are_reused_and_old_versions_remain() -> None:
    async def scenario() -> None:
        store = InMemorySourceRegistryStore(embed_dim=2)
        identity = canonicalize_source_identity("arxiv", "2401.12345", None)
        await store.upsert_source(identity, "Old", identity.canonical_url, {"votes": 1})
        first, created = await store.create_or_reuse_version(identity.source_id, "v1", "one", None, "text/plain", compute_content_hashes("one"), {})
        assert created
        same, created = await store.create_or_reuse_version(identity.source_id, "v1", "one", None, "text/plain", compute_content_hashes("one"), {})
        assert not created and same.version_id == first.version_id
        second, created = await store.create_or_reuse_version(identity.source_id, "v2", "two", None, "text/plain", compute_content_hashes("two"), {})
        assert created and second.version_id != first.version_id
        assert not store.versions[first.version_id].is_current
        await store.upsert_source(identity, "New", identity.canonical_url, {"votes": 2})
        assert (await store.get_current_version(identity.source_id)).version_id == second.version_id

    run(scenario())
