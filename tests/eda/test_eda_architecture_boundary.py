from kaggle_researcher.eda.boundary import (
    MODULE_CLASSIFICATION,
    build_eda_implications,
    stage_status,
)


def test_module_classification_keeps_post_eda_and_rendering_out_of_core() -> None:
    assert MODULE_CLASSIFICATION["source_claim_validation"] == "post_eda_reasoning"
    assert MODULE_CLASSIFICATION["visual_diagnostics"] == "evidence_rendering"
    assert MODULE_CLASSIFICATION["baseline_runner"] == "model_assisted_diagnostic"


def test_compatibility_inputs_project_to_stable_boundary_outputs() -> None:
    hints = {"feature_engineering": [{
        "priority": "P1", "action": "Add fold-fitted categorical encoders.",
        "why": "A stable categorical signal was measured.",
        "evidence_refs": ["feature_diagnostics.categorical"], "confidence": "high",
    }]}
    implications = build_eda_implications(hints)
    assert implications[0]["implication_id"] == "eda_implication_001"
    assert implications[0]["evidence_refs"]
    assert "primary model" not in implications[0]["implication"].lower()

def test_core_status_is_independent_of_post_eda_and_visual_modules() -> None:
    statuses = {
        "file_inventory": "completed", "schema_inferer": "completed",
        "table_profiler": "completed", "metric_analyzer": "completed",
        "validation_analyzer": "completed", "target_diagnostics": "completed",
        "leakage_checker": "completed", "feature_diagnostics": "completed",
        "relationship_inferer": "skipped", "drift_analyzer": "skipped",
        "source_claim_validation": "skipped", "visual_diagnostics": "skipped",
    }
    result = stage_status(statuses)
    assert result["core_eda"] == "completed"
    assert result["post_eda_reasoning"] == "skipped"
    assert result["visual_artifacts"] == "skipped"
