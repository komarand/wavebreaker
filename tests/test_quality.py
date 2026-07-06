from __future__ import annotations

from kaggle_researcher.quality import (
    validate_reasoning_outputs,
    validate_report_text,
    validate_retrieved_documents,
)
from kaggle_researcher.reasoning.report_composer import SECTION_HEADINGS
from kaggle_researcher.schemas import MetricResult, RetrievedDocument


def _full_report(extra_text: str = "Confidence: medium.") -> str:
    return "\n\n".join(f"## {heading}\n{extra_text}" for heading in SECTION_HEADINGS)


def test_validate_report_text_warns_for_missing_sections() -> None:
    warnings = validate_report_text("## Executive summary\nConfidence: medium.")

    assert len(warnings) == 1
    assert "15 required section headings" in warnings[0]


def test_validate_report_text_warns_for_forbidden_phrases() -> None:
    warnings = validate_report_text(_full_report("EDA showed a strong split signal."))

    assert any("forbidden data-execution claim" in warning for warning in warnings)
    assert any("eda showed" in warning for warning in warnings)


def test_validate_reasoning_outputs_warns_for_missing_evidence_ids() -> None:
    warnings = validate_reasoning_outputs(
        {
            "metric": {"confidence": "medium", "metric_explanation": "auc"},
            "experiments": [{"priority": "P0", "experiment": "baseline"}],
        }
    )

    assert "Reasoning output 'metric' is missing 'evidence_ids'." in warnings
    assert "Reasoning output 'experiments'[0] is missing 'evidence_ids'." in warnings


def test_validate_reasoning_outputs_warns_for_missing_confidence() -> None:
    warnings = validate_reasoning_outputs({"validation": {"evidence_ids": ["doc-1"]}})

    assert "Reasoning output 'validation' is missing 'confidence'." in warnings


def test_validate_reasoning_outputs_accepts_pydantic_models() -> None:
    warnings = validate_reasoning_outputs(
        {
            "metric": MetricResult(
                confidence="medium",
                evidence_ids=["doc-1"],
                metric_explanation="auc",
                needs_calibration=False,
                rank_averaging_useful=True,
                threshold_search_needed=False,
                surrogate_loss_suggestion="logloss",
            )
        }
    )

    assert warnings == []


def test_validate_reasoning_outputs_warns_for_confirmed_leakage_claim() -> None:
    warnings = validate_reasoning_outputs(
        {
            "leakage": {
                "confidence": "high",
                "evidence_ids": ["doc-1"],
                "possible_issues": ["Leakage is confirmed by train/test analysis."],
            }
        }
    )

    assert "Leakage Risk Analyst output must not say leakage is confirmed." in warnings


def test_validate_retrieved_documents_warns_when_empty() -> None:
    assert validate_retrieved_documents([]) == ["retrieved_documents must not be empty."]


def test_validate_retrieved_documents_accepts_non_empty_list() -> None:
    docs = [
        RetrievedDocument(
            id="doc-1",
            competition_id="comp-1",
            source="kaggle",
            title="Notebook",
            url="https://example.com/notebook",
            content="retrieved evidence",
            score=0.9,
            rrf_score=0.2,
        )
    ]

    assert validate_retrieved_documents(docs) == []
