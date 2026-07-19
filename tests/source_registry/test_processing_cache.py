import asyncio

from kaggle_researcher.source_registry.fingerprints import build_embedding_fingerprint, build_summary_fingerprint
from kaggle_researcher.source_registry.processing_cache import process_source
from kaggle_researcher.source_registry.registry import InMemorySourceRegistryStore
from kaggle_researcher.source_registry.schemas import CachePolicy, SourceDescriptor


def run(value):
    return asyncio.run(value)


def test_repeated_and_cross_competition_processing_calls_expensive_stages_once() -> None:
    async def scenario() -> None:
        store = InMemorySourceRegistryStore(embed_dim=2)
        calls = {"summary": 0, "embedding": 0}

        async def summarize(text, descriptor):
            calls["summary"] += 1
            return f"summary:{text}"

        def embed(text):
            calls["embedding"] += 1
            return [0.1, 0.2]

        descriptor = SourceDescriptor(source_type="arxiv", external_id="2401.12345v2", source_revision="v2", revision_is_reliable=True, raw_content="paper")
        kwargs = dict(
            registry=store, cache_policy=CachePolicy(), summarizer=summarize, embedder=embed,
            summary_fingerprint=build_summary_fingerprint(model="m", prompt="p"),
            embedding_fingerprint=build_embedding_fingerprint(model="e", dimension=2),
        )
        first = await process_source(descriptor, "one", "run-1", **kwargs)
        second = await process_source(descriptor, "one", "run-2", **kwargs)
        third = await process_source(descriptor, "two", "run-3", **kwargs)
        assert calls == {"summary": 1, "embedding": 1}
        assert first.version.version_id == second.version.version_id == third.version.version_id
        assert len(store.competition_sources) == 2
        assert len(store.embeddings) == 1

    run(scenario())


def test_summary_and_embedding_invalidation_are_dependency_scoped() -> None:
    async def scenario() -> None:
        store = InMemorySourceRegistryStore(embed_dim=2)
        calls = {"summary": 0, "embedding": 0}
        async def summary(text, descriptor): calls["summary"] += 1; return text.upper()
        def embedding(text): calls["embedding"] += 1; return [0.1, 0.2]
        descriptor = SourceDescriptor(source_type="github", external_id="a/b", raw_content="readme")
        base = dict(registry=store, cache_policy=CachePolicy(), summarizer=summary, embedder=embedding)
        await process_source(descriptor, "c", "r1", **base,
            summary_fingerprint=build_summary_fingerprint(model="s", prompt="one"),
            embedding_fingerprint=build_embedding_fingerprint(model="e1", dimension=2))
        await process_source(descriptor, "c", "r2", **base,
            summary_fingerprint=build_summary_fingerprint(model="s", prompt="two"),
            embedding_fingerprint=build_embedding_fingerprint(model="e1", dimension=2))
        assert calls == {"summary": 2, "embedding": 2}
        await process_source(descriptor, "c", "r3", **base,
            summary_fingerprint=build_summary_fingerprint(model="s", prompt="two"),
            embedding_fingerprint=build_embedding_fingerprint(model="e2", dimension=2))
        assert calls == {"summary": 2, "embedding": 3}

    run(scenario())


def test_single_flight_prevents_duplicate_concurrent_calls() -> None:
    async def scenario() -> None:
        store = InMemorySourceRegistryStore(embed_dim=2)
        calls = 0
        async def summary(text, descriptor):
            nonlocal calls; calls += 1; await asyncio.sleep(0); return text
        descriptor = SourceDescriptor(source_type="kaggle", external_id="a/b", raw_content="code")
        await asyncio.gather(*(
            process_source(descriptor, "c", f"r{i}", store, CachePolicy(), summarizer=summary)
            for i in range(5)
        ))
        assert calls == 1

    run(scenario())
