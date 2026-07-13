from kaggle_researcher.eda.modules.testable_hypotheses import (
    HypothesisGenerationConfig,
    build_safety_constraints,
    build_testable_hypotheses,
    build_validation_requirements,
    validate_testable_hypotheses_output,
)


def test_safety_and_validation_rules_are_not_hypotheses() -> None:
    evidence = {
        "inferred_schema": {"target_column": "target", "primary_id_column": "id", "sample_submission_table": "submission.csv"},
        "validation_evidence": {"primary_validation": {"method": "stratified_kfold"}},
        "metric_evidence": {"requires_threshold": True, "threshold_search_needed": True},
        "leakage_evidence": [{"status": "warning"}],
    }
    constraints = build_safety_constraints(evidence)
    requirements = build_validation_requirements(evidence)
    hypotheses = build_testable_hypotheses(evidence_pack=evidence)
    assert any("target and primary ID" in item["rule"] for item in constraints)
    assert any("target distribution" in item["rule"] for item in requirements)
    assert any("threshold" in item["rule"] for item in requirements)
    assert all("exclude" not in item["statement"].lower() for item in hypotheses)
    assert all("build" not in item["statement"].lower() or "baseline" not in item["statement"].lower() for item in hypotheses)


def test_unstable_high_cardinality_ablation_creates_one_controlled_hypothesis() -> None:
    evidence = {"baseline_ablation_evidence": {"feature_block_findings": [
        {"feature_block": "high_cardinality_categorical", "status": "neutral", "confidence": "medium", "evidence_refs": ["ablations.a"]},
        {"feature_block": "high_cardinality_categorical", "status": "unstable", "confidence": "medium", "evidence_refs": ["ablations.b"]},
        {"feature_block": "low_cardinality_categorical", "status": "helped", "materiality": "material", "stability": "stable", "confidence": "high"},
    ]}}
    result = build_testable_hypotheses(evidence_pack=evidence)
    high_card = [item for item in result if "high-cardinality" in item["statement"]]
    assert len(high_card) == 1
    assert high_card[0]["baseline_ref"]
    assert "same folds" in high_card[0]["required_controls"]
    assert not any("low-cardinality" in item["statement"] for item in result)


def test_reliable_interaction_is_bounded_and_unreliable_one_is_excluded() -> None:
    evidence = {"interaction_diagnostics": {"interaction_hypotheses": [
        {"columns": ["b", "a"], "reliability": "reliable", "materiality": "material", "confidence": "high", "hypothesis": "Supported pair", "evidence_refs": ["pairs.a_b"]},
        {"columns": ["tiny", "x"], "reliability": "caution_small_sample", "materiality": "material", "evidence_refs": ["pairs.tiny"]},
    ]}}
    result = build_testable_hypotheses(evidence_pack=evidence)
    assert len(result) == 1
    assert result[0]["scope"] == "interaction"
    assert "a, b" in result[0]["statement"]


def test_source_claims_and_completed_process_steps_are_never_sources() -> None:
    evidence = {
        "baseline_evidence": {"status": "completed"},
        "source_claim_validation": {"recommended_experiments": [{"action": "Use model X"}]},
    }
    assert build_testable_hypotheses(evidence_pack=evidence) == []


def test_cap_projection_validation_and_determinism() -> None:
    interactions = [
        {"columns": [f"a{i}", f"b{i}"], "reliability": "reliable", "materiality": "material", "confidence": "high", "evidence_refs": [f"pair.{i}"]}
        for i in range(20)
    ]
    evidence = {"interaction_diagnostics": {"interaction_hypotheses": interactions}}
    config = HypothesisGenerationConfig(max_testable_hypotheses=2, max_per_scope=2)
    first = build_testable_hypotheses(evidence_pack=evidence, config=config)
    second = build_testable_hypotheses(evidence_pack={"interaction_diagnostics": {"interaction_hypotheses": list(reversed(interactions))}}, config=config)
    assert first == second
    assert len(first) == 2
    assert validate_testable_hypotheses_output({"testable_hypotheses": first, "experiment_candidates": first}, max_count=2) == []
