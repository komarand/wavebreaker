import asyncio

from kaggle_researcher.schemas import SourceDocument
from kaggle_researcher.source_registry.processing_cache import process_source_documents
from kaggle_researcher.source_registry.registry import InMemorySourceRegistryStore
from kaggle_researcher.source_registry.schemas import CachePolicy


def test_mocked_document_pipeline_reuses_and_selectively_invalidates() -> None:
    async def scenario() -> None:
        store = InMemorySourceRegistryStore(embed_dim=2)
        calls = {"summary": 0, "embedding": 0}

        async def summarize(document: SourceDocument) -> SourceDocument:
            calls["summary"] += 1
            return document.model_copy(update={"summary": f"summary:{document.content}"})

        def embed(text: str) -> list[float]:
            calls["embedding"] += 1
            return [0.1, 0.2]

        def document(content: str) -> SourceDocument:
            return SourceDocument(
                id="legacy-id",
                competition_id="one",
                source="arxiv",
                title="Paper",
                url="https://arxiv.org/abs/2401.12345v1",
                content=content,
                metadata={"entry_id": "2401.12345v1", "source_revision": "v1", "revision_is_reliable": True},
            )

        kwargs = dict(
            registry=store,
            cache_policy=CachePolicy(),
            summarize_one=summarize,
            embed_one=embed,
            summary_model="summary-model",
            embed_model="embedding-model",
        )
        first_docs, _, _ = await process_source_documents(
            [document("unchanged")], competition_id="one", run_id="run-1", **kwargs
        )
        second_docs, _, _ = await process_source_documents(
            [document("unchanged")], competition_id="one", run_id="run-2", **kwargs
        )
        await process_source_documents(
            [document("unchanged")], competition_id="two", run_id="run-3", **kwargs
        )
        assert calls == {"summary": 1, "embedding": 1}
        assert first_docs[0].id == second_docs[0].id == "arxiv:2401.12345"
        assert len(store.competition_sources) == 2

        await process_source_documents(
            [document("changed")], competition_id="one", run_id="run-4", **kwargs
        )
        assert calls == {"summary": 2, "embedding": 2}
        assert len(store.versions) == 2

    asyncio.run(scenario())


def test_embedding_misses_preserve_batching() -> None:
    async def scenario() -> None:
        store = InMemorySourceRegistryStore(embed_dim=2)
        batch_sizes: list[int] = []

        async def summarize(document: SourceDocument) -> SourceDocument:
            return document.model_copy(update={"summary": document.content})

        def embed_one(text: str) -> list[float]:
            raise AssertionError("single-item embedder should not be used when batching is available")

        def embed_many(texts: list[str]) -> list[list[float]]:
            batch_sizes.append(len(texts))
            return [[0.1, 0.2] for _ in texts]

        docs = [
            SourceDocument(
                id=f"legacy-{index}", competition_id="one", source="github",
                title=f"Repo {index}", url=f"https://github.com/owner/repo-{index}",
                content=f"readme {index}", metadata={"full_name": f"owner/repo-{index}"},
            )
            for index in range(3)
        ]
        await process_source_documents(
            docs, competition_id="one", run_id="batch-run", registry=store,
            cache_policy=CachePolicy(), summarize_one=summarize, embed_one=embed_one,
            embed_many=embed_many, summary_model="s", embed_model="e",
        )
        assert batch_sizes == [3]

    asyncio.run(scenario())
