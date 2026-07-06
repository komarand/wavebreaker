from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kaggle_researcher import main as main_module
from kaggle_researcher.main import run_research
from kaggle_researcher.research_scout import (
    ALLOWED_EDA_MODULES,
    attach_supporting_source_ids,
    build_research_hypotheses,
    build_research_scout_summary,
    cleanup_generic_eda_tasks,
    correct_hypothesis_categories,
    enforce_scout_validation_policy,
    enforce_stratified_groupkfold_caveat,
    ensure_task_ids,
    expand_verification_steps,
    is_generic_verification_text,
    normalize_research_hypotheses,
    relink_eda_tasks,
    split_recommended_sequences,
    split_eda_task_plan,
    validate_research_hypotheses,
)
from kaggle_researcher.research_scout_schemas import (
    EdaTask,
    ResearchHypothesesPayload,
    ResearchHypothesis,
    ScoutFinding,
    VerificationStep,
)
from kaggle_researcher.schemas import PlanData, RetrievedDocument, SourceDocument


def run(coro):
    return asyncio.run(coro)


def complete_payload() -> dict[str, Any]:
    return normalize_research_hypotheses(
        {
            "competition_id": "home-credit-credit-risk-model-stability",
            "competition_url": "https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability",
            "competition_desc": "Binary classification. Metric: Gini Stability. Tabular credit data.",
            "task_type": "binary_classification",
            "metric": {"name": "gini_stability"},
            "domain": "tabular_credit_risk",
        }
    )


def minimal_hypothesis(
    hypothesis_id: str,
    category: str,
    claim: str,
    *,
    priority: str = "P1",
    how_to_verify: list[str] | None = None,
    verification_steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": hypothesis_id,
        "category": category,
        "priority": priority,
        "claim": claim,
        "why_it_matters": "This affects validation, feature work, or model selection.",
        "how_to_verify": how_to_verify or ["Check the relevant evidence on the real train/test files."],
        "verification_steps": verification_steps or [],
        "expected_evidence_keys": ["evidence.result"],
        "provenance": ["heuristic", "not_verified_on_data"],
        "confidence": "medium",
        "status": "needs_eda",
    }


def test_research_hypothesis_schema_valid() -> None:
    hypothesis = ResearchHypothesis(
        id="val_001",
        category="validation",
        priority="P0",
        claim="Use out-of-time validation.",
        why_it_matters="Random CV may overestimate leaderboard performance.",
        how_to_verify=["Find candidate time columns."],
        provenance=["heuristic", "not_verified_on_data"],
        confidence="high",
    )

    assert hypothesis.id == "val_001"


def test_research_hypothesis_rejects_invalid_priority() -> None:
    with pytest.raises(ValidationError):
        ResearchHypothesis(
            id="val_001",
            category="validation",
            priority="P9",
            claim="Bad priority.",
            why_it_matters="It should fail.",
            how_to_verify=["Validate schema."],
            provenance=["heuristic"],
        )


def test_eda_task_schema_valid() -> None:
    task = EdaTask(
        id="eda_val_001",
        priority="P0",
        module="validation_analyzer",
        question="Which split is trustworthy?",
        rationale="Validation controls strategy.",
        expected_outputs=["validation_evidence.recommended_validation"],
    )

    assert task.module == "validation_analyzer"


def test_eda_task_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        EdaTask(
            id="",
            priority="P0",
            module="schema_inferer",
            question="Which schema roles exist?",
            rationale="Schema roles are required before modeling.",
            expected_outputs=["inferred_schema.global_roles"],
        )


def test_research_payload_schema_valid() -> None:
    payload = ResearchHypothesesPayload.model_validate(complete_payload())

    assert payload.schema_version == "1.0"


def test_normalize_adds_missing_ids() -> None:
    payload = normalize_research_hypotheses(
        {
            "competition_id": "comp",
            "competition_desc": "Credit risk.",
            "task_type": "binary_classification",
            "metric": "gini",
            "hypotheses": [
                {
                    "category": "validation",
                    "priority": "P0",
                    "claim": "Check validation.",
                    "why_it_matters": "It matters.",
                    "how_to_verify": ["Inspect folds."],
                    "provenance": ["heuristic"],
                }
            ],
        }
    )

    assert payload["hypotheses"][0]["id"] == "val_001"


def test_normalize_adds_not_verified_on_data() -> None:
    payload = complete_payload()

    assert all("not_verified_on_data" in item["provenance"] for item in payload["hypotheses"])


def test_normalize_adds_default_validation_for_stability_metric() -> None:
    payload = complete_payload()

    validation_claims = " ".join(
        item["claim"].lower()
        for item in payload["hypotheses"]
        if item["category"] == "validation"
    )
    assert "out-of-time" in validation_claims or "temporal" in validation_claims


def test_normalize_adds_default_leakage_checks() -> None:
    payload = complete_payload()

    leakage_claims = " ".join(
        item["claim"].lower()
        for item in payload["hypotheses"]
        if item["category"] == "leakage"
    )
    assert "id overlap" in leakage_claims
    assert "target-like" in leakage_claims
    assert "future" in leakage_claims


def test_normalize_ensures_p0_tasks_exist() -> None:
    payload = complete_payload()
    p0_ids = {
        item["id"]
        for item in payload["hypotheses"]
        if item["priority"] == "P0"
    }
    related_ids = {
        hypothesis_id
        for task in payload["eda_tasks"]
        for hypothesis_id in task["related_hypothesis_ids"]
    }

    assert p0_ids <= related_ids


def test_validate_research_hypotheses_accepts_complete_payload() -> None:
    validate_research_hypotheses(complete_payload())


def test_validate_research_hypotheses_rejects_no_validation_hypothesis() -> None:
    payload = complete_payload()
    payload["hypotheses"] = [
        item for item in payload["hypotheses"] if item["category"] != "validation"
    ]

    with pytest.raises(ValueError, match="P0 validation"):
        validate_research_hypotheses(payload)


def test_validate_research_hypotheses_rejects_no_leakage_hypothesis() -> None:
    payload = complete_payload()
    payload["hypotheses"] = [
        item for item in payload["hypotheses"] if item["category"] != "leakage"
    ]

    with pytest.raises(ValueError, match="P0 leakage"):
        validate_research_hypotheses(payload)


def test_validate_research_hypotheses_requires_temporal_for_stability_metric() -> None:
    payload = complete_payload()
    for item in payload["hypotheses"]:
        if item["category"] == "validation":
            item["claim"] = "Use stratified validation."

    with pytest.raises(ValueError, match="temporal"):
        validate_research_hypotheses(payload)


def test_build_research_scout_summary_contains_p0() -> None:
    summary = build_research_scout_summary(complete_payload())

    assert "## P0 EDA checks" in summary
    assert "[P0]" in summary


def test_build_research_scout_summary_contains_limitations() -> None:
    summary = build_research_scout_summary(complete_payload())

    assert "No real EDA was performed" in summary


def test_split_eda_task_plan_is_focused_subset() -> None:
    plan = split_eda_task_plan(complete_payload())

    assert "eda_tasks" in plan
    assert "hypothesis_index" in plan
    assert "hypotheses" not in plan


def test_ensure_task_ids_fills_missing_ids() -> None:
    tasks = ensure_task_ids(
        [
            {
                "id": "",
                "priority": "P1",
                "module": "schema_inferer",
                "question": "Which schema roles exist in the files?",
                "rationale": "Schema roles are required before downstream checks.",
                "expected_outputs": ["inferred_schema.global_roles"],
                "related_hypothesis_ids": ["schema_002"],
            }
        ]
    )

    assert tasks[0]["id"] == "eda_schema_schema_002"


def test_ensure_task_ids_uniqueness() -> None:
    tasks = ensure_task_ids(
        [
            {"id": "", "module": "schema_inferer", "related_hypothesis_ids": ["schema_002"]},
            {"id": "", "module": "schema_inferer", "related_hypothesis_ids": ["schema_002"]},
        ]
    )

    assert [task["id"] for task in tasks] == ["eda_schema_schema_002", "eda_schema_schema_002_2"]


def test_cleanup_removes_generic_table_profiler_tasks() -> None:
    tasks = cleanup_generic_eda_tasks(
        [
            {
                "id": "eda_table_profiler_001",
                "priority": "P1",
                "module": "table_profiler",
                "question": "What does the dataset show for this hypothesis?",
                "rationale": "Generic.",
                "expected_outputs": ["table_profiles"],
                "related_hypothesis_ids": [],
            }
        ],
        {"schema_001", "schema_004"},
    )

    assert all(task["id"] != "eda_table_profiler_001" for task in tasks)


def test_cleanup_adds_single_global_profile_task() -> None:
    tasks = cleanup_generic_eda_tasks(
        [
            {
                "id": "eda_table_profiler_001",
                "priority": "P1",
                "module": "table_profiler",
                "question": "What does the dataset show for this hypothesis?",
                "rationale": "Generic.",
                "expected_outputs": ["table_profiles"],
                "related_hypothesis_ids": [],
            },
            {
                "id": "eda_table_profiler_002",
                "priority": "P1",
                "module": "table_profiler",
                "question": "What does the dataset show for this hypothesis?",
                "rationale": "Generic.",
                "expected_outputs": ["table_profiles"],
                "related_hypothesis_ids": [],
            },
        ],
        {"schema_001", "schema_004"},
    )

    global_tasks = [task for task in tasks if task["id"] == "eda_profile_global"]
    assert len(global_tasks) == 1
    assert global_tasks[0]["related_hypothesis_ids"] == ["schema_001", "schema_004"]


def test_cleanup_keeps_specific_profile_tasks() -> None:
    specific = {
        "id": "eda_profile_schema_001",
        "priority": "P1",
        "module": "table_profiler",
        "question": "Which columns have high missingness and cardinality in the base table?",
        "rationale": "Specific profiling supports schema and leakage checks.",
        "expected_outputs": ["table_profiles.base_table"],
        "related_hypothesis_ids": ["schema_001"],
    }

    assert cleanup_generic_eda_tasks([specific]) == [specific]


def test_scout_validation_policy_rewrites_stratified_group_kfold_primary() -> None:
    payload = normalize_research_hypotheses(
        {
            "competition_id": "comp",
            "competition_desc": "Metric uses Gini stability over WEEK_NUM.",
            "task_type": "binary_classification",
            "metric": {"name": "gini_stability"},
            "hypotheses": [
                {
                    "id": "val_001",
                    "category": "validation",
                    "priority": "P0",
                    "claim": "Using StratifiedGroupKFold with groups=WEEK_NUM ensures safe validation.",
                    "why_it_matters": "Validation matters.",
                    "how_to_verify": ["Build grouped folds."],
                    "expected_evidence_keys": ["validation_evidence.recommended_validation"],
                    "provenance": ["heuristic", "not_verified_on_data"],
                }
            ],
        }
    )

    val_001 = next(item for item in payload["hypotheses"] if item["id"] == "val_001")
    assert "out-of-time holdout" in val_001["claim"]
    assert "secondary diagnostic" in val_001["claim"]


def test_scout_validation_policy_requires_oot_for_stability_metric() -> None:
    payload = complete_payload()
    validation_claims = " ".join(
        item["claim"].lower() for item in payload["hypotheses"] if item["category"] == "validation"
    )

    assert "out-of-time" in validation_claims
    assert "rolling" in validation_claims or "expanding" in validation_claims


def test_scout_validation_policy_updates_validation_task_question() -> None:
    payload = complete_payload()
    task = next(task for task in payload["eda_tasks"] if task["id"] == "eda_val_001")

    assert "strict temporal or out-of-time validation split" in task["question"]
    assert "validation_evidence.temporal_folds" in task["expected_outputs"]
    assert "validation_evidence.oot_holdout" in task["expected_outputs"]


def test_attach_supporting_source_ids_metric_prefers_kaggle_metric_source() -> None:
    payload = {"hypotheses": [{"id": "metric_001", "category": "metric", "provenance": ["heuristic"]}]}
    result = attach_supporting_source_ids(
        payload,
        [
            {"id": "arxiv-1", "source": "arxiv", "title": "AUC paper", "content": "auc metric evaluation"},
            {"id": "kaggle-1", "source": "kaggle", "title": "Metric notebook", "content": "gini stability metric WEEK_NUM evaluation"},
        ],
    )

    assert result["hypotheses"][0]["supporting_source_ids"][0] == "kaggle-1"


def test_attach_supporting_source_ids_validation_prefers_week_num_source() -> None:
    payload = {"hypotheses": [{"id": "val_001", "category": "validation", "provenance": ["heuristic"]}]}
    result = attach_supporting_source_ids(
        payload,
        [
            {"id": "doc-1", "source": "github", "title": "CV helper", "content": "fold validation"},
            {"id": "doc-2", "source": "kaggle", "title": "WEEK_NUM validation", "content": "WEEK_NUM temporal out-of-time CV"},
        ],
    )

    assert result["hypotheses"][0]["supporting_source_ids"][0] == "doc-2"


def test_attach_supporting_source_ids_notebook_prefers_kaggle() -> None:
    payload = {"hypotheses": [{"id": "nb_001", "category": "notebook_reverse_engineering", "provenance": ["heuristic"]}]}
    result = attach_supporting_source_ids(
        payload,
        [
            {"id": "github-1", "source": "github", "title": "Notebook mirror", "content": "notebook model code"},
            {"id": "kaggle-1", "source": "kaggle", "title": "Kaggle notebook", "content": "notebook cv feature model"},
        ],
    )

    assert result["hypotheses"][0]["supporting_source_ids"][0] == "kaggle-1"


def test_attach_supporting_source_ids_adds_provenance() -> None:
    payload = {"hypotheses": [{"id": "base_001", "category": "baseline", "provenance": ["heuristic"]}]}
    result = attach_supporting_source_ids(
        payload,
        [{"id": "github-1", "source": "github", "title": "LightGBM baseline", "content": "baseline lightgbm model"}],
    )

    assert "github" in result["hypotheses"][0]["provenance"]


def test_split_recommended_sequences() -> None:
    result = split_recommended_sequences(
        {
            "recommended_eda_sequence": [
                "Implement and test gini_stability metric",
                "file_inventory",
                "schema_inferer",
            ]
        }
    )

    assert result["recommended_module_sequence"][:2] == ["file_inventory", "schema_inferer"]
    assert result["recommended_eda_sequence"] == result["recommended_module_sequence"]
    assert result["recommended_human_checklist"] == ["Implement and test gini_stability metric."]


def test_recommended_module_sequence_only_allowed_modules() -> None:
    payload = complete_payload()

    assert set(payload["recommended_module_sequence"]) <= ALLOWED_EDA_MODULES


def test_human_checklist_is_natural_language() -> None:
    payload = complete_payload()

    assert payload["recommended_human_checklist"]
    assert not any(item in ALLOWED_EDA_MODULES for item in payload["recommended_human_checklist"])


def test_validator_rejects_empty_task_id() -> None:
    payload = complete_payload()
    payload["eda_tasks"][0]["id"] = ""

    with pytest.raises(ValueError, match="EDA task IDs must not be empty"):
        validate_research_hypotheses(payload)


def test_validator_rejects_duplicate_task_id() -> None:
    payload = complete_payload()
    payload["eda_tasks"][1]["id"] = payload["eda_tasks"][0]["id"]

    with pytest.raises(ValueError, match="duplicate task IDs"):
        validate_research_hypotheses(payload)


def test_validator_rejects_generic_task_question() -> None:
    payload = complete_payload()
    payload["eda_tasks"].append(
        {
            "id": "eda_profile_generic",
            "priority": "P1",
            "module": "table_profiler",
            "question": "What does the dataset show for this hypothesis?",
            "rationale": "Generic profiling is not actionable.",
            "expected_outputs": ["table_profiles"],
            "related_hypothesis_ids": [],
        }
    )

    with pytest.raises(ValueError, match="generic task question"):
        validate_research_hypotheses(payload)


def test_validator_rejects_missing_related_hypothesis_ids_for_non_global_task() -> None:
    payload = complete_payload()
    payload["eda_tasks"].append(
        {
            "id": "eda_feat_unlinked",
            "priority": "P1",
            "module": "feature_probe",
            "question": "Which feature family should be tested first?",
            "rationale": "Feature probes must map to a hypothesis.",
            "expected_outputs": ["feature_evidence.family_lift"],
            "related_hypothesis_ids": [],
        }
    )

    with pytest.raises(ValueError, match="missing related_hypothesis_ids"):
        validate_research_hypotheses(payload)


def test_validator_rejects_unsafe_stability_validation() -> None:
    payload = complete_payload()
    for item in payload["hypotheses"]:
        if item["category"] == "validation" and item["priority"] == "P0":
            item["claim"] = "StratifiedGroupKFold is the primary validation for WEEK_NUM."

    with pytest.raises(ValueError, match="out-of-time"):
        validate_research_hypotheses(payload)


def test_validator_accepts_clean_scout_payload() -> None:
    validate_research_hypotheses(complete_payload())


def test_research_scout_integration_lite_normalizes_unsafe_response() -> None:
    raw_payload = {
        "competition_id": "home-credit-credit-risk-model-stability",
        "competition_desc": "Gini stability over WEEK_NUM with temporal drift.",
        "task_type": "binary_classification",
        "metric": {"name": "gini_stability"},
        "source_summary": {"kaggle": 2, "github": 1},
        "hypotheses": [
            {
                "id": "val_001",
                "category": "validation",
                "priority": "P0",
                "claim": "Using StratifiedGroupKFold with groups=WEEK_NUM ensures safe validation.",
                "why_it_matters": "Validation controls model selection.",
                "how_to_verify": ["Build grouped folds."],
                "expected_evidence_keys": ["validation_evidence.recommended_validation"],
                "provenance": ["heuristic", "not_verified_on_data"],
            }
        ],
        "eda_tasks": [
            {
                "id": "",
                "priority": "P1",
                "module": "schema_inferer",
                "question": "Which schema roles exist in the files?",
                "rationale": "Schema roles are needed before validation.",
                "expected_outputs": ["inferred_schema.global_roles"],
                "related_hypothesis_ids": ["schema_001"],
            },
            {
                "id": "eda_table_profiler_001",
                "priority": "P1",
                "module": "table_profiler",
                "question": "What does the dataset show for this hypothesis?",
                "rationale": "Generic profiling is not actionable.",
                "expected_outputs": ["table_profiles"],
                "related_hypothesis_ids": [],
            },
        ],
        "recommended_eda_sequence": [
            "Implement and test gini_stability metric",
            "file_inventory",
            "schema_inferer",
        ],
    }

    normalized = normalize_research_hypotheses(raw_payload)
    normalized = attach_supporting_source_ids(
        normalized,
        [
            {"id": "kaggle-week", "source": "kaggle", "title": "WEEK_NUM validation", "content": "WEEK_NUM temporal validation gini stability"},
            {"id": "github-base", "source": "github", "title": "LightGBM baseline", "content": "baseline lightgbm model features"},
        ],
    )

    validate_research_hypotheses(normalized)
    val_001 = next(item for item in normalized["hypotheses"] if item["id"] == "val_001")
    assert "out-of-time holdout" in val_001["claim"]
    assert all(task["id"] for task in normalized["eda_tasks"])
    assert not any("What does the dataset show" in task["question"] for task in normalized["eda_tasks"])
    assert len([task for task in normalized["eda_tasks"] if task["id"] == "eda_profile_global"]) == 1
    assert any(item["supporting_source_ids"] for item in normalized["hypotheses"])
    assert normalized["recommended_eda_sequence"] == normalized["recommended_module_sequence"]


def test_verification_step_schema_valid() -> None:
    step = VerificationStep(
        id="verify_val_001_time_columns",
        module="schema_inferer",
        operation="identify_time_columns",
        question="Which columns can define chronological validation splits?",
        outputs=["inferred_schema.global_roles.candidate_time_columns"],
        success_criteria=["At least one reliable time column exists."],
    )

    assert step.module == "schema_inferer"


def test_verification_step_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        VerificationStep(
            id="",
            module="schema_inferer",
            operation="identify_time_columns",
            question="Which columns can define chronological validation splits?",
            outputs=["inferred_schema.global_roles.candidate_time_columns"],
        )


def test_research_hypothesis_accepts_verification_steps() -> None:
    hypothesis = ResearchHypothesis(
        id="val_001",
        category="validation",
        priority="P0",
        claim="Use strict temporal validation.",
        why_it_matters="Temporal validation controls model selection.",
        how_to_verify=["Identify time columns."],
        verification_steps=[
            VerificationStep(
                id="verify_val_001_time_columns",
                module="schema_inferer",
                operation="identify_time_columns",
                question="Which columns can define chronological validation splits?",
                outputs=["inferred_schema.global_roles.candidate_time_columns"],
                success_criteria=["A time column exists."],
            )
        ],
        expected_evidence_keys=["inferred_schema.global_roles.candidate_time_columns"],
        provenance=["heuristic", "not_verified_on_data"],
    )

    assert hypothesis.verification_steps[0].operation == "identify_time_columns"


def test_expand_verification_steps_adds_validation_steps() -> None:
    payload = {
        "hypotheses": [
            minimal_hypothesis(
                "val_001",
                "validation",
                "Primary validation should use out-of-time and rolling temporal CV with WEEK_NUM.",
                priority="P0",
            )
        ],
        "eda_tasks": [{"id": "eda_val_001", "module": "validation_analyzer"}],
    }

    expanded = expand_verification_steps(payload)
    outputs = {
        output
        for step in expanded["hypotheses"][0]["verification_steps"]
        for output in step["outputs"]
    }

    assert "validation_evidence.oot_holdout" in outputs
    assert "validation_evidence.temporal_folds" in outputs


def test_expand_verification_steps_adds_drift_steps_for_default_rate_over_time() -> None:
    expanded = expand_verification_steps(
        {"hypotheses": [minimal_hypothesis("drift_001", "drift", "Default rate changes over time by WEEK_NUM.")]}
    )
    outputs = {output for step in expanded["hypotheses"][0]["verification_steps"] for output in step["outputs"]}

    assert "validation_evidence.target_by_period" in outputs
    assert "drift_evidence.adversarial_auc" in outputs


def test_expand_verification_steps_adds_metric_steps_for_residual_std() -> None:
    expanded = expand_verification_steps(
        {"hypotheses": [minimal_hypothesis("metric_001", "metric", "Residual std penalty affects Gini stability.")]}
    )
    outputs = {output for step in expanded["hypotheses"][0]["verification_steps"] for output in step["outputs"]}

    assert "metric_evidence.weekly_components" in outputs
    assert "baseline_evidence.metric_components" in outputs


def test_expand_verification_steps_derives_how_to_verify() -> None:
    expanded = expand_verification_steps(
        {"hypotheses": [minimal_hypothesis("leak_001", "leakage", "Check future-information leakage.")]}
    )

    assert "check the relevant evidence" not in " ".join(expanded["hypotheses"][0]["how_to_verify"]).lower()
    assert expanded["hypotheses"][0]["verification_steps"]


def test_is_generic_verification_text_detects_generic_phrase() -> None:
    assert is_generic_verification_text("Check the dataset.")
    assert not is_generic_verification_text("Check table_profiles.missingness for columns above 95%.")


def test_validator_rejects_generic_how_to_verify_without_steps() -> None:
    payload = complete_payload()
    target = payload["hypotheses"][0]
    target["how_to_verify"] = ["Check the dataset."]
    target["verification_steps"] = []

    with pytest.raises(ValueError, match="generic how_to_verify"):
        validate_research_hypotheses(payload)


def test_validator_accepts_generic_how_to_verify_when_structured_steps_exist_and_rewrites_legacy_text() -> None:
    payload = normalize_research_hypotheses(
        {
            "competition_id": "comp",
            "competition_desc": "Metric gini stability with WEEK_NUM.",
            "task_type": "binary_classification",
            "metric": {"name": "gini_stability"},
            "hypotheses": [
                minimal_hypothesis("val_001", "validation", "Primary validation should use out-of-time plus rolling temporal CV.", priority="P0")
            ],
        }
    )

    validate_research_hypotheses(payload)
    val = next(item for item in payload["hypotheses"] if item["id"] == "val_001")
    assert not any(is_generic_verification_text(text) for text in val["how_to_verify"])


def test_category_correction_default_rate_to_drift() -> None:
    result = correct_hypothesis_categories(
        {"hypotheses": [minimal_hypothesis("feat_001", "feature_engineering", "Default rate changes over time by WEEK_NUM.")], "eda_tasks": []}
    )

    assert result["hypotheses"][0]["category"] == "drift"
    assert result["hypotheses"][0]["id"] == "drift_001"


def test_category_correction_covariate_shift_to_drift() -> None:
    result = correct_hypothesis_categories(
        {"hypotheses": [minimal_hypothesis("feat_002", "feature_engineering", "Covariate shift may separate train/test rows.")], "eda_tasks": []}
    )

    assert result["hypotheses"][0]["category"] == "drift"


def test_category_correction_residual_std_to_metric() -> None:
    result = correct_hypothesis_categories(
        {"hypotheses": [minimal_hypothesis("feat_005", "feature_engineering", "Residual std penalty drives the metric.")], "eda_tasks": []}
    )

    assert result["hypotheses"][0]["category"] == "metric"


def test_category_correction_prediction_ties_to_metric() -> None:
    result = correct_hypothesis_categories(
        {"hypotheses": [minimal_hypothesis("feat_006", "feature_engineering", "Prediction ties can change rank-based scoring.")], "eda_tasks": []}
    )

    assert result["hypotheses"][0]["category"] == "metric"


def test_category_correction_high_missing_to_dataset_schema() -> None:
    result = correct_hypothesis_categories(
        {"hypotheses": [minimal_hypothesis("feat_007", "feature_engineering", "High missing rates above >95% missing require schema profiling.")], "eda_tasks": []}
    )

    assert result["hypotheses"][0]["category"] == "dataset_schema"


def test_category_correction_updates_task_links() -> None:
    result = correct_hypothesis_categories(
        {
            "hypotheses": [minimal_hypothesis("feat_001", "feature_engineering", "Default rate changes over time by WEEK_NUM.")],
            "eda_tasks": [{"id": "eda_feat_001", "module": "feature_probe", "related_hypothesis_ids": ["feat_001"]}],
        }
    )

    assert result["eda_tasks"][0]["related_hypothesis_ids"] == ["drift_001"]


def test_category_correction_records_migration_map() -> None:
    result = correct_hypothesis_categories(
        {"hypotheses": [minimal_hypothesis("feat_001", "feature_engineering", "Default rate changes over time by WEEK_NUM.")], "eda_tasks": []}
    )

    assert result["category_corrections"][0]["old_id"] == "feat_001"
    assert result["category_corrections"][0]["new_id"] == "drift_001"


def test_structured_findings_schema_valid() -> None:
    finding = ScoutFinding(
        id="finding_001",
        finding_type="observed_in_sources",
        claim="Top notebooks use grouped validation by WEEK_NUM.",
        caveat="Use only as diagnostic.",
        provenance=["kaggle"],
    )

    assert finding.finding_type == "observed_in_sources"


def test_legacy_scout_findings_generated_from_structured() -> None:
    payload = normalize_research_hypotheses(
        {
            "competition_id": "comp",
            "competition_desc": "Metric gini stability with WEEK_NUM.",
            "task_type": "binary_classification",
            "metric": {"name": "gini_stability"},
            "structured_findings": [
                {
                    "id": "finding_001",
                    "finding_type": "warning",
                    "claim": "Public leaderboard may be noisy over time.",
                    "provenance": ["heuristic"],
                }
            ],
        }
    )

    assert payload["scout_findings"][0].startswith("warning: Public leaderboard")


def test_findings_grouped_in_summary() -> None:
    summary = build_research_scout_summary(complete_payload())

    assert "## Structured findings" in summary
    assert "### Observed in sources" in summary
    assert "### Caveats" in summary


def test_enforce_stratified_groupkfold_caveat_adds_caveat_finding() -> None:
    payload = enforce_stratified_groupkfold_caveat(
        {
            "competition_desc": "Gini stability over WEEK_NUM.",
            "metric": {"name": "gini_stability"},
            "hypotheses": [minimal_hypothesis("val_001", "validation", "StratifiedGroupKFold is the primary validation.", priority="P0")],
            "structured_findings": [],
            "eda_tasks": [],
        }
    )

    assert any(item["id"] == "caveat_stratified_groupkfold_temporal" for item in payload["structured_findings"])


def test_enforce_stratified_groupkfold_caveat_rewrites_recommendation_to_observed() -> None:
    payload = enforce_stratified_groupkfold_caveat(
        {
            "competition_desc": "Gini stability over WEEK_NUM.",
            "metric": {"name": "gini_stability"},
            "hypotheses": [minimal_hypothesis("val_001", "validation", "Use out-of-time plus rolling temporal CV.", priority="P0")],
            "structured_findings": [
                {
                    "id": "finding_001",
                    "finding_type": "recommendation",
                    "claim": "Use StratifiedGroupKFold with WEEK_NUM.",
                    "provenance": ["kaggle"],
                }
            ],
            "eda_tasks": [],
        }
    )

    finding = next(item for item in payload["structured_findings"] if item["id"] == "finding_001")
    assert finding["finding_type"] == "observed_in_sources"
    assert "not sufficient as primary temporal validation" in finding["caveat"]


def test_enforce_stratified_groupkfold_caveat_keeps_temporal_validation_primary() -> None:
    payload = enforce_stratified_groupkfold_caveat(
        {
            "competition_desc": "Gini stability over WEEK_NUM.",
            "metric": {"name": "gini_stability"},
            "hypotheses": [minimal_hypothesis("val_001", "validation", "StratifiedGroupKFold is the primary validation.", priority="P0")],
            "eda_tasks": [],
        }
    )

    claim = payload["hypotheses"][0]["claim"]
    assert "out-of-time holdout" in claim
    assert "secondary diagnostic" in claim


def test_relink_eda_tasks_links_drift_hypotheses_to_drift_task() -> None:
    payload = relink_eda_tasks({"hypotheses": [minimal_hypothesis("drift_001", "drift", "Default rate drift.")], "eda_tasks": []})

    task = next(task for task in payload["eda_tasks"] if task["module"] == "drift_analyzer")
    assert "drift_001" in task["related_hypothesis_ids"]


def test_relink_eda_tasks_links_metric_hypotheses_to_metric_task() -> None:
    payload = relink_eda_tasks({"hypotheses": [minimal_hypothesis("metric_001", "metric", "Residual std penalty.")], "eda_tasks": []})

    task = next(task for task in payload["eda_tasks"] if task["module"] == "metric_analyzer")
    assert "metric_001" in task["related_hypothesis_ids"]


def test_relink_eda_tasks_links_schema_hypotheses_to_schema_or_profile_tasks() -> None:
    payload = relink_eda_tasks({"hypotheses": [minimal_hypothesis("schema_001", "dataset_schema", "High missing rates.")], "eda_tasks": []})

    schema_task = next(task for task in payload["eda_tasks"] if task["module"] == "schema_inferer")
    profile_task = next(task for task in payload["eda_tasks"] if task["module"] == "table_profiler")
    assert "schema_001" in schema_task["related_hypothesis_ids"]
    assert "schema_001" in profile_task["related_hypothesis_ids"]


def test_relink_eda_tasks_keeps_feature_task_feature_only() -> None:
    payload = relink_eda_tasks(
        {
            "hypotheses": [
                minimal_hypothesis("feat_001", "feature_engineering", "Feature family lift."),
                minimal_hypothesis("drift_001", "drift", "Default rate drift."),
            ],
            "eda_tasks": [],
        }
    )

    feature_task = next(task for task in payload["eda_tasks"] if task["module"] == "feature_probe")
    assert feature_task["related_hypothesis_ids"] == ["feat_001"]


def test_research_scout_integration_lite_structures_and_relinks_llm_drift() -> None:
    raw_payload = {
        "competition_id": "home-credit-credit-risk-model-stability",
        "competition_desc": "Gini stability over WEEK_NUM.",
        "task_type": "binary_classification",
        "metric": {"name": "gini_stability"},
        "source_summary": {"kaggle": 2},
        "hypotheses": [
            minimal_hypothesis("feat_001", "feature_engineering", "Default rate changes over time by WEEK_NUM."),
            minimal_hypothesis("feat_002", "feature_engineering", "Covariate shift separates train/test rows."),
            minimal_hypothesis("feat_005", "feature_engineering", "Residual std penalty affects weekly Gini."),
            minimal_hypothesis("feat_006", "feature_engineering", "Prediction ties affect rank-based scoring."),
        ],
        "structured_findings": [
            {
                "id": "finding_sgkf",
                "finding_type": "recommendation",
                "claim": "Notebooks recommend StratifiedGroupKFold with WEEK_NUM.",
                "provenance": ["kaggle"],
            }
        ],
        "eda_tasks": [
            {
                "id": "eda_feat_001",
                "priority": "P1",
                "module": "feature_probe",
                "question": "Which feature families deserve experiments?",
                "rationale": "Feature probes prioritize model work.",
                "expected_outputs": ["feature_evidence.family_lift"],
                "related_hypothesis_ids": ["feat_001", "feat_002", "feat_005", "feat_006"],
            }
        ],
    }

    normalized = normalize_research_hypotheses(raw_payload)
    validate_research_hypotheses(normalized)

    category_by_old = {
        correction["old_id"]: correction["new_category"]
        for correction in normalized["category_corrections"]
    }
    assert category_by_old["feat_001"] == "drift"
    assert category_by_old["feat_002"] == "drift"
    assert category_by_old["feat_005"] == "metric"
    assert category_by_old["feat_006"] == "metric"
    assert all(item["verification_steps"] for item in normalized["hypotheses"] if item["priority"] in {"P0", "P1"})
    assert not any(
        is_generic_verification_text(text)
        for item in normalized["hypotheses"]
        for text in item["how_to_verify"]
    )
    assert any(item["id"] == "caveat_stratified_groupkfold_temporal" for item in normalized["structured_findings"])
    feature_task = next(task for task in normalized["eda_tasks"] if task["module"] == "feature_probe")
    assert all(hypothesis_id.startswith("feat_") for hypothesis_id in feature_task["related_hypothesis_ids"])


def test_build_research_hypotheses_uses_mocked_llm_response() -> None:
    class FakeClient:
        async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["model"] == "deepseek-v4-pro"
            return {
                "hypotheses": [
                    {
                        "category": "validation",
                        "priority": "P0",
                        "claim": "Identify WEEK_NUM for temporal validation.",
                        "why_it_matters": "Stability scoring depends on time.",
                        "how_to_verify": ["Inspect columns."],
                        "provenance": ["kaggle"],
                        "confidence": "high",
                    }
                ],
                "eda_tasks": [],
            }

    payload = run(
        build_research_hypotheses(
            competition_id="home-credit-credit-risk-model-stability",
            competition_url="https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability",
            competition_desc="Metric: Gini Stability. Tabular credit data.",
            plan_data={"task_type": "binary_classification", "metric": "gini_stability", "domain": "tabular_credit_risk"},
            retrieved_documents=[],
            client=FakeClient(),
            model="deepseek-v4-pro",
        )
    )

    assert payload["models_used"]["research_scout"] == "deepseek-v4-pro"
    validate_research_hypotheses(payload)


def test_scout_mode_writes_artifacts_and_skips_docx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeSettings:
        deepseek_api_key = "secret"
        deepseek_v4_pro = "deepseek-v4-pro"
        deepseek_v4_flash = "deepseek-v4-flash"
        embed_model = "fake-embedder"
        embed_dim = 2
        max_embed_batch_size = 2
        pg_dsn = "postgresql://example"
        top_k = 2
        max_notebooks = 1
        max_papers = 1
        max_repos = 1
        pdf_cache_dir = "./data/pdfs"
        github_token = None

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
            return {"hypotheses": [], "eda_tasks": []}

    class FakeStore:
        def __init__(self, competition_id: str, dsn: str, embed_dim: int) -> None:
            self.competition_id = competition_id

        async def init(self) -> None:
            return None

        async def upsert(self, docs: list[SourceDocument], embeddings: list[list[float]]) -> None:
            return None

        async def close(self) -> None:
            return None

    class FakeDomainMemory:
        def __init__(self, dsn: str, embed_dim: int) -> None:
            return None

        async def init(self) -> None:
            return None

        async def find_similar(self, task_type: str, domain: str, top_k: int = 5) -> list[dict[str, Any]]:
            return []

        async def close(self) -> None:
            return None

    async def fake_plan(description: str, client: Any, model: str) -> PlanData:
        return PlanData(
            task_type="binary_classification",
            metric="gini_stability",
            domain="tabular_credit_risk",
            kaggle_queries=["credit"],
            arxiv_queries=[],
            github_queries=[],
        )

    async def fake_collect_sources(**kwargs: Any) -> list[SourceDocument]:
        return [
            SourceDocument(
                id="doc-1",
                competition_id="comp-1",
                source="kaggle",
                title="Notebook",
                url="https://example.com/notebook",
                content="Uses WEEK_NUM and Gini stability.",
                summary="Uses WEEK_NUM and Gini stability.",
            )
        ]

    async def fake_summarize_documents(**kwargs: Any) -> list[SourceDocument]:
        return kwargs["docs"]

    async def fake_retrieve_documents(**kwargs: Any) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                id="doc-1",
                competition_id="comp-1",
                source="kaggle",
                title="Notebook",
                url="https://example.com/notebook",
                content="Uses WEEK_NUM and Gini stability.",
                score=0.9,
                rrf_score=0.1,
            )
        ]

    def fail_generate_report(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("DOCX report should not be generated in scout mode")

    monkeypatch.setattr(main_module, "load_config", lambda: FakeSettings())
    monkeypatch.setattr(main_module, "DeepSeekClient", FakeClient)
    monkeypatch.setattr(main_module, "PgStore", FakeStore)
    monkeypatch.setattr(main_module, "DomainMemory", FakeDomainMemory)
    monkeypatch.setattr(main_module, "plan", fake_plan)
    monkeypatch.setattr(main_module, "_collect_sources", fake_collect_sources)
    monkeypatch.setattr(main_module, "summarize_documents", fake_summarize_documents)
    monkeypatch.setattr(main_module, "_embed_documents", lambda texts, batch_size, show_progress: [[0.1, 0.2]])
    monkeypatch.setattr(main_module, "_retrieve_documents", fake_retrieve_documents)
    monkeypatch.setattr(main_module, "generate_report", fail_generate_report)
    monkeypatch.setattr(main_module, "_create_run_dir", lambda competition_id: tmp_path / "runs" / competition_id)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/comp-1",
            "Metric: Gini Stability. Tabular credit data.",
            mode="scout",
            show_progress=False,
        )
    )

    run_path = Path(result.run_artifacts_path)
    assert result.mode == "scout"
    assert Path(result.research_hypotheses_path).exists()
    assert Path(result.eda_task_plan_path).exists()
    assert Path(result.summary_path).exists()
    assert not any(path.suffix == ".docx" for path in tmp_path.rglob("*"))
    payload = json.loads((run_path / "research_hypotheses.json").read_text(encoding="utf-8"))
    assert any(item["category"] == "validation" and item["priority"] == "P0" for item in payload["hypotheses"])
    assert any(item["category"] == "leakage" and item["priority"] == "P0" for item in payload["hypotheses"])
    assert any(task["module"] == "schema_inferer" for task in payload["eda_tasks"])
    assert any(task["module"] == "validation_analyzer" for task in payload["eda_tasks"])
    assert any(task["module"] == "leakage_checker" for task in payload["eda_tasks"])
    assert any(task["module"] == "metric_analyzer" for task in payload["eda_tasks"])
    assert any(task["module"] == "baseline_runner" for task in payload["eda_tasks"])
    assert payload["models_used"]["research_scout"] == "deepseek-v4-pro"
