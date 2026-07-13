from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaggle_researcher.schemas import (
    RetrievedDocument,
    ReasoningBaseResult,
    SourceDocument,
    ValidationResult,
)


def test_valid_source_document_accepts_optional_metadata() -> None:
    document = SourceDocument(
        id="doc-1",
        competition_id="comp-1",
        source="arxiv",
        title="Paper",
        url="https://example.com/paper",
        content="parsed paper text",
        metadata={"pdf_url": "https://example.com/paper.pdf"},
    )

    assert str(document.url) == "https://example.com/paper"
    assert document.metadata["pdf_url"] == "https://example.com/paper.pdf"


def test_valid_retrieved_document_accepts_scores_and_metadata() -> None:
    document = RetrievedDocument(
        id="doc-1",
        competition_id="comp-1",
        source="github",
        title="Repo",
        url="https://github.com/example/repo",
        content="readme text",
        score=0.8,
        rrf_score=0.25,
        metadata={"stars": 50},
    )

    assert document.score == 0.8
    assert document.rrf_score == 0.25
    assert document.metadata["stars"] == 50


def test_missing_required_source_fields_fail_clearly() -> None:
    with pytest.raises(ValidationError) as exc_info:
        SourceDocument(id="doc-1", competition_id="comp-1", source="kaggle", title="Missing content")

    assert "content" in str(exc_info.value)


def test_invalid_source_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        SourceDocument(
            id="doc-1",
            competition_id="comp-1",
            source="blog",
            title="Title",
            url="https://example.com/doc-1",
            content="content",
        )


def test_invalid_confidence_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        ReasoningBaseResult(confidence="certain", evidence_ids=["doc-1"])


def test_optional_defaults_are_independent_per_instance() -> None:
    first = SourceDocument(
        id="doc-1",
        competition_id="comp-1",
        source="kaggle",
        title="Title 1",
        url="https://example.com/doc-1",
        content="content",
    )
    second = SourceDocument(
        id="doc-2",
        competition_id="comp-2",
        source="github",
        title="Title 2",
        url="https://example.com/doc-2",
        content="content",
    )

    first.metadata["votes"] = 10

    assert second.metadata == {}


def test_reasoning_list_defaults_are_independent() -> None:
    first = ValidationResult(
        confidence="low",
        evidence_ids=[],
        recommended_cv="GroupKFold",
        validation_risk="medium",
        likely_split="group",
        failure_modes=[],
        reasoning="Use group-aware validation.",
        primary_validation={"method": "group_kfold"},
    )
    second = ValidationResult(
        confidence="medium",
        evidence_ids=[],
        recommended_cv="TimeSeriesSplit",
        validation_risk="high",
        likely_split="time",
        failure_modes=[],
        reasoning="Respect temporal ordering.",
        primary_validation={"method": "temporal_cv"},
    )

    first.failure_modes.append("leakage")
    first.evidence_ids.append("doc-1")

    assert second.failure_modes == []
    assert second.evidence_ids == []
