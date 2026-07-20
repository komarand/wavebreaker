from __future__ import annotations

from kaggle_researcher.eda.quality import (
    validate_evidence_pack,
    validate_evidence_refs,
    validate_hypothesis_results,
    validate_no_unsupported_summary_claims,
)
from kaggle_researcher.eda.schemas import (
    EdaEvidencePack,
    HypothesisResult,
    RecommendedNextAction,
    ResearchHypotheses,
    ResearchHypothesis,
)


def test_missing_hypothesis_result_creates_warning() -> None:
    warnings = validate_hypothesis_results(
        _pack(hypothesis_results=[]),
        _hypotheses(["schema_001", "val_001"]),
    )

    assert any("Missing hypothesis result for schema_001" in warning for warning in warnings)
    assert any("Missing hypothesis result for val_001" in warning for warning in warnings)


def test_duplicate_hypothesis_result_creates_warning() -> None:
    result = _result("val_001", ["validation_evidence.primary_validation"])
    warnings = validate_hypothesis_results(
        _pack(hypothesis_results=[result, result]),
        _hypotheses(["val_001"]),
    )

    assert any("val_001 has 2 results" in warning for warning in warnings)


def test_broken_evidence_ref_creates_warning() -> None:
    warnings = validate_evidence_refs(
        _pack(
            hypothesis_results=[
                _result("val_001", ["validation_evidence.missing_policy"])
            ]
        )
    )

    assert any("broken evidence_ref: validation_evidence.missing_policy" in warning for warning in warnings)


def test_empty_recommended_action_evidence_refs_creates_warning() -> None:
    warnings = validate_evidence_pack(
        _pack(
            recommended_next_actions=[
                RecommendedNextAction(
                    priority="P0",
                    action="Use selected validation.",
                    why="Validation evidence should support this action.",
                    evidence_refs=[],
                )
            ]
        )
    )

    assert any("recommended_next_actions[0] has no evidence_refs" in warning for warning in warnings)


def test_malformed_recommended_action_text_creates_warning() -> None:
    warnings = validate_evidence_pack(
        _pack(
            recommended_next_actions=[
                RecommendedNextAction(
                    priority="P1",
                    action=".",
                    why=".",
                    evidence_refs=["metric_evidence.metric_name"],
                )
            ]
        )
    )

    assert any("recommended_next_actions[0] has empty or malformed action text" in warning for warning in warnings)
    assert any("recommended_next_actions[0] has empty or malformed why" in warning for warning in warnings)


def test_temporal_overclaim_creates_warning() -> None:
    warnings = validate_no_unsupported_summary_claims(
        "Temporal validation is required for this competition.",
        _pack(primary_validation="stratified_kfold"),
    )

    assert any("temporal validation is required" in warning.lower() for warning in warnings)


def test_leakage_found_without_failed_check_creates_warning() -> None:
    warnings = validate_no_unsupported_summary_claims(
        "Leakage found in the data.",
        _pack(leakage_evidence=[{"check_id": "leak_001", "status": "passed", "evidence": {}}]),
    )

    assert any("leakage found" in warning.lower() for warning in warnings)


def test_forbidden_baseline_phrase_creates_warning() -> None:
    warnings = validate_no_unsupported_summary_claims(
        "The baseline proves final solution quality.",
        _pack(),
    )

    assert any("baseline" in warning.lower() and "final solution" in warning.lower() for warning in warnings)


def test_quality_functions_return_warnings_not_exceptions() -> None:
    pack = _pack(
        hypothesis_results=[
            HypothesisResult(
                hypothesis_id="metric_001",
                category="metric",
                status="confirmed",
                confidence_after_eda="medium",
                finding="Metric resolved.",
                evidence_refs=[],
                impact_on_strategy="Use metric evidence.",
            )
        ],
        recommended_next_actions=[
            RecommendedNextAction(
                priority="P0",
                action="Use metric evidence.",
                why="Metric evidence exists.",
                evidence_refs=["metric_evidence.not_present"],
            )
        ],
    )

    warnings = [
        *validate_evidence_pack(pack),
        *validate_hypothesis_results(pack, _hypotheses(["schema_001"])),
        *validate_no_unsupported_summary_claims("Probably confirmed.", pack),
    ]

    assert warnings
    assert all(isinstance(warning, str) for warning in warnings)


def _pack(
    *,
    primary_validation: str = "stratified_kfold",
    hypothesis_results: list[HypothesisResult] | None = None,
    recommended_next_actions: list[RecommendedNextAction] | None = None,
    leakage_evidence: list[dict] | None = None,
) -> EdaEvidencePack:
    return EdaEvidencePack(
        competition_id="quality_tiny",
        created_at="2026-07-08T12:00:00+03:00",
        run_id="quality_tiny_20260708_120000",
        metric_evidence={"metric_name": "roc_auc"},
        validation_evidence={"primary_validation": {"method": primary_validation}},
        leakage_evidence=leakage_evidence if leakage_evidence is not None else [],
        hypothesis_results=hypothesis_results if hypothesis_results is not None else [
            _result("val_001", ["validation_evidence.primary_validation"])
        ],
        recommended_next_actions=recommended_next_actions
        if recommended_next_actions is not None
        else [],
        warnings=["source coverage limited"],
        limitations=["tiny fixture"],
    )


def _result(hypothesis_id: str, evidence_refs: list[str]) -> HypothesisResult:
    return HypothesisResult(
        hypothesis_id=hypothesis_id,
        category="validation",
        status="confirmed",
        confidence_after_eda="high",
        finding="Validation policy selected.",
        evidence_refs=evidence_refs,
        impact_on_strategy="Use selected validation.",
    )


def _hypotheses(hypothesis_ids: list[str]) -> ResearchHypotheses:
    return ResearchHypotheses(
        competition_id="quality_tiny",
        hypotheses=[
            ResearchHypothesis(
                hypothesis_id=hypothesis_id,
                category="validation",
                claim="Validation policy should be selected.",
                rationale="Validation controls model comparison.",
                expected_eda_checks=["validation_analyzer.primary_validation"],
                priority="P0",
                confidence_before_eda="medium",
            )
            for hypothesis_id in hypothesis_ids
        ],
    )
