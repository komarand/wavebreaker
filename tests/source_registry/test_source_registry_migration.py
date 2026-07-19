import asyncio

from kaggle_researcher.source_registry.migrations import migrate_legacy_documents
from kaggle_researcher.source_registry.registry import InMemorySourceRegistryStore


def run(value): return asyncio.run(value)


def test_legacy_migration_deduplicates_and_is_idempotent() -> None:
    async def scenario() -> None:
        store = InMemorySourceRegistryStore(embed_dim=2)
        rows = [
            {"id": "a", "competition_id": "one", "source": "arxiv", "url": "https://arxiv.org/abs/2401.12345v1", "title": "P", "content": "text", "summary": "sum", "embedding": [0.1, 0.2]},
            {"id": "b", "competition_id": "two", "source": "arxiv", "url": "https://arxiv.org/pdf/2401.12345v2.pdf", "title": "P", "content": "text", "summary": "sum", "embedding": [0.1, 0.2]},
        ]
        report = await migrate_legacy_documents(store, rows)
        assert len(store.sources) == 1 and len(store.versions) == 1 and len(store.competition_sources) == 2
        assert report.embeddings_skipped == 2 and len(store.embeddings) == 0
        await migrate_legacy_documents(store, rows)
        assert len(store.sources) == 1 and len(store.versions) == 1
        dry_store = InMemorySourceRegistryStore(embed_dim=2)
        await migrate_legacy_documents(dry_store, rows, dry_run=True)
        assert not dry_store.sources

    run(scenario())
