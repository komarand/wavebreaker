from __future__ import annotations

from datetime import datetime, timezone

import pytest

from kaggle_researcher.source_registry.fingerprints import build_processor_fingerprint
from kaggle_researcher.source_registry.repository import SQLiteSourceRegistryRepository
from kaggle_researcher.source_registry.schemas import CachePolicy, SourceDescriptor
from kaggle_researcher.source_registry.service import SourceRegistryService


def _service(tmp_path) -> SourceRegistryService:
    return SourceRegistryService(SQLiteSourceRegistryRepository(tmp_path / "registry.sqlite3"))


def _paper(content: bytes = b"%PDF-stage-one") -> SourceDescriptor:
    return SourceDescriptor(
        source_type="arxiv",
        external_id="https://arxiv.org/abs/2401.12345v2",
        canonical_url="https://arxiv.org/pdf/2401.12345v2.pdf?utm_source=test",
        title="Stable source identity",
        source_revision="v2",
        revision_is_reliable=True,
        content_mime_type="application/pdf",
        discovery_query="tabular validation",
        raw_content=content,
    )


def test_source_and_version_are_reused_across_competitions(tmp_path) -> None:
    service = _service(tmp_path)

    first = service.register_source(_paper(), competition_id="competition-a")
    second = service.register_source(_paper(), competition_id="competition-b")

    assert first.source.source_id == "arxiv:2401.12345"
    assert first.version.version_id == second.version.version_id
    assert second.cache_decisions[-1].decision == "cache_hit"
    assert service.repository.count("sources") == 1
    assert service.repository.count("source_versions") == 1
    assert service.repository.count("competition_sources") == 2


def test_changed_content_creates_new_immutable_current_version(tmp_path) -> None:
    service = _service(tmp_path)
    first = service.register_source(_paper(b"%PDF-v1"), competition_id="competition-a")
    second = service.register_source(_paper(b"%PDF-v2"), competition_id="competition-a")

    assert first.version.version_id != second.version.version_id
    assert service.repository.count("source_versions") == 2
    previous = service.repository.get_version(first.version.version_id)
    assert previous is not None and previous.is_current is False
    assert service.repository.get_source(first.source.source_id).current_version_id == second.version.version_id


def test_artifact_cache_is_invalidated_by_processor_fingerprint(tmp_path) -> None:
    service = _service(tmp_path)
    result = service.register_source(_paper(), competition_id="competition-a")
    calls = 0

    def produce():
        nonlocal calls
        calls += 1
        return {"text": "parsed"}

    first = service.get_or_create_artifact(
        result, artifact_type="parsed", processor_name="pdf-parser",
        processor_version="1", configuration={"all_pages": True},
        input_hash=result.version.normalized_content_hash, producer=produce,
    )
    cached = service.get_or_create_artifact(
        result, artifact_type="parsed", processor_name="pdf-parser",
        processor_version="1", configuration={"all_pages": True},
        input_hash=result.version.normalized_content_hash, producer=produce,
    )
    changed = service.get_or_create_artifact(
        result, artifact_type="parsed", processor_name="pdf-parser",
        processor_version="2", configuration={"all_pages": True},
        input_hash=result.version.normalized_content_hash, producer=produce,
    )

    assert first.artifact_id == cached.artifact_id
    assert changed.artifact_id != first.artifact_id
    assert calls == 2


def test_embedding_cache_is_model_revision_aware(tmp_path) -> None:
    service = _service(tmp_path)
    result = service.register_source(_paper(), competition_id="competition-a")
    calls = 0

    def embed():
        nonlocal calls
        calls += 1
        return [0.25, 0.75]

    first = service.get_or_create_embedding(
        result, input_kind="summary", input_text="same input", model_name="model",
        model_revision="rev-a", dimension=2, producer=embed,
    )
    cached = service.get_or_create_embedding(
        result, input_kind="summary", input_text="same input", model_name="model",
        model_revision="rev-a", dimension=2, producer=embed,
    )
    changed = service.get_or_create_embedding(
        result, input_kind="summary", input_text="same input", model_name="model",
        model_revision="rev-b", dimension=2, producer=embed,
    )

    assert first.embedding_id == cached.embedding_id
    assert cached.embedding == [0.25, 0.75]
    assert changed.embedding_id != first.embedding_id
    assert calls == 2


def test_force_rebuild_reuses_unique_artifact_identity(tmp_path) -> None:
    service = _service(tmp_path)
    result = service.register_source(_paper(), competition_id="competition-a")
    kwargs = dict(
        artifact_type="parsed", processor_name="parser", processor_version="1",
        configuration={}, input_hash=result.version.normalized_content_hash,
    )
    first = service.get_or_create_artifact(result, producer=lambda: "one", **kwargs)
    rebuilt = service.get_or_create_artifact(
        result, producer=lambda: "two",
        policy=CachePolicy(rebuild_artifacts={"parsed"}), **kwargs,
    )
    assert rebuilt.artifact_id == first.artifact_id
    assert rebuilt.payload == "two"
    assert service.repository.count("artifacts") == 1


def test_manifest_and_processor_fingerprint_are_deterministic(tmp_path) -> None:
    service = _service(tmp_path)
    result = service.register_source(_paper(), competition_id="competition-a")
    manifest = service.write_manifest(
        result, run_id="run-1", competition_id="competition-a",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    left = build_processor_fingerprint("parser", "1", {"b": 2, "a": 1})
    right = build_processor_fingerprint("parser", "1", {"a": 1, "b": 2})

    assert manifest.status == "success"
    assert service.repository.count("processing_manifests") == 1
    assert left.fingerprint == right.fingerprint


def test_embedding_dimension_mismatch_is_rejected(tmp_path) -> None:
    service = _service(tmp_path)
    result = service.register_source(_paper(), competition_id="competition-a")
    with pytest.raises(ValueError, match="Embedding dimension"):
        service.get_or_create_embedding(
            result, input_kind="summary", input_text="text", model_name="model",
            model_revision="rev", dimension=3, producer=lambda: [1.0, 2.0],
        )
