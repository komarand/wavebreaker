from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaggle_researcher.schemas import (
    ReasoningBaseResult,
    SourceDocument,
    ValidationResult,
)


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
    )
    second = ValidationResult(
        confidence="medium",
        evidence_ids=[],
        recommended_cv="TimeSeriesSplit",
        validation_risk="high",
        likely_split="time",
        failure_modes=[],
        reasoning="Respect temporal ordering.",
    )

    first.failure_modes.append("leakage")
    first.evidence_ids.append("doc-1")

    assert second.failure_modes == []
    assert second.evidence_ids == []
