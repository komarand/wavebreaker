from __future__ import annotations

from kaggle_researcher.eda.modules.source_claim_validation import (
    collect_source_claims,
    validate_source_claim_validation,
    validate_source_claims,
)


def test_no_source_claims_is_a_safe_skip() -> None:
    result = validate_source_claims([], _evidence())

    assert result["status"] == "skipped"
    assert result["reason"] == "no_source_claims"


def test_schema_and_metric_claims_are_confirmed_or_contradicted() -> None:
    result = validate_source_claims([
        {"source_id": "direct", "source_type": "discussion", "source_scope": "direct_competition", "claim_type": "schema", "claim_text": "The target is target_label."},
        {"source_id": "direct", "source_type": "discussion", "source_scope": "direct_competition", "claim_type": "metric", "claim_text": "The competition uses AUC."},
    ], _evidence())

    schema = next(item for item in result["validated_claims"] if item["claim_type"] == "schema")
    metric = next(item for item in result["validated_claims"] if item["claim_type"] == "metric")
    assert schema["validation_status"] == "confirmed"
    assert schema["safe_strategy_use"] == "adopt"
    assert "inferred_schema.target_column" in schema["supporting_eda_refs"]
    assert metric["validation_status"] == "contradicted"
    assert metric["safe_strategy_use"] == "reject"


def test_unsafe_encoding_and_analogous_model_claims_are_not_adopted() -> None:
    result = validate_source_claims([
        {"source_id": "notebook", "source_type": "notebook", "source_scope": "direct_competition", "claim_type": "leakage", "claim_text": "Fit target encoding on all training rows before CV."},
        {"source_id": "analogy", "source_type": "analogous_competition", "source_scope": "analogous_task", "claim_type": "model", "claim_text": "Gradient boosting is best."},
    ], _evidence())

    unsafe = next(item for item in result["validated_claims"] if item["validation_status"] == "unsafe")
    analogous = next(item for item in result["validated_claims"] if item["validation_status"] == "analogous_only")
    assert unsafe["validation_status"] == "unsafe"
    assert unsafe["safe_strategy_use"] == "reject"
    assert analogous["validation_status"] == "analogous_only"
    assert analogous["safe_strategy_use"] == "test_as_hypothesis"
    assert analogous["evidence_origin"] != "eda_confirmed"


def test_missingness_and_validation_conflicts_remain_conservative() -> None:
    result = validate_source_claims([
        {"source_id": "a", "source_scope": "direct_competition", "claim_type": "missingness", "claim_text": "Missingness indicators materially improve CV."},
        {"source_id": "b", "source_scope": "direct_competition", "claim_type": "validation", "claim_text": "Use temporal validation."},
        {"source_id": "c", "source_scope": "direct_competition", "claim_type": "validation", "claim_text": "Use StratifiedKFold."},
    ], _evidence())

    missingness = next(item for item in result["validated_claims"] if item["claim_type"] == "missingness")
    assert missingness["validation_status"] == "contradicted"
    assert result["claim_conflicts"]
    assert any(item["safe_strategy_use"] == "adopt" for item in result["validated_claims"])
    assert validate_source_claim_validation(result) == []


def test_scout_adapter_and_deduplication_are_deterministic() -> None:
    raw = {"hypotheses": [{"category": "metric", "claim": "The metric is accuracy.", "source_refs": ["doc"], "confidence_before_eda": "medium"}], "structured_findings": []}
    claims = collect_source_claims(raw)
    result = validate_source_claims([*claims, *claims], _evidence())

    assert len(result["claim_inventory"]) == 1
    assert result["validated_claims"][0]["claim_id"] == "claim_001"


def _evidence() -> dict:
    return {
        "inferred_schema": {"target_column": "target_label", "primary_id_column": "customer_id", "tables": [{"columns": [{"name": "target_label"}, {"name": "customer_id"}]}]},
        "metric_evidence": {"metric_name": "accuracy", "requires_threshold": False},
        "validation_evidence": {"primary_validation": {"method": "stratified_kfold"}},
        "target_diagnostics": {"target_by_missingness": [{"column": "income", "target_rate_difference": 0.2}]},
        "baseline_ablation_evidence": {"feature_block_findings": [{"feature_block": "missingness_indicators", "status": "neutral", "materiality": "negligible"}]},
        "leakage_evidence": [],
        "feature_diagnostics": {},
        "interaction_diagnostics": {},
    }
