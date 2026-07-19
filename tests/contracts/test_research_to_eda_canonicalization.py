from __future__ import annotations

import pytest

from kaggle_researcher.contracts.research_to_eda import (
    ResearchEdaModulePlanError,
    canonicalize_research_to_eda_contract,
    extract_check_module,
    normalize_check_reference,
    normalize_hypothesis_category,
    normalize_module_name,
    require_valid_research_to_eda_contract,
    validate_research_to_eda_contract,
)
from kaggle_researcher.research_scout import (
    build_deterministic_research_scout_fallback,
    run_research_scout,
)
from kaggle_researcher.schemas import PlanData
from tests.contracts.factories import valid_research_payload, valid_task_plan_payload


pytestmark = pytest.mark.contract


def test_alias_and_check_module_normalization_is_structural_only() -> None:
    assert normalize_hypothesis_category("feature_engineering") == "feature"
    assert normalize_hypothesis_category("relationships") == "relationship"
    assert normalize_hypothesis_category(
        "legacy_relationship_group", ["relationship_inferer.relationships"]
    ) == "relationship"
    assert normalize_module_name(" relationship_analyzer ") == "relationship_inferer"
    assert normalize_check_reference(
        " relationship_analyzer.join_key_coverage "
    ) == "relationship_inferer.join_key_coverage"
    assert extract_check_module("validation_analyzer.period_distribution") == (
        "validation_analyzer"
    )
    assert extract_check_module(" validation_analyzer ") == "validation_analyzer"
    assert extract_check_module("") == ""


def test_valid_category_check_pair_is_unchanged() -> None:
    research = valid_research_payload()
    research["hypotheses"][2]["expected_eda_checks"] = [
        "validation_analyzer.period_distribution"
    ]

    result = canonicalize_research_to_eda_contract(research, valid_task_plan_payload())

    validation = result.research_hypotheses.hypotheses[2]
    assert validation.category == "validation"
    assert validation.expected_eda_checks == [
        "validation_analyzer.period_distribution"
    ]


def test_legacy_relationship_category_and_module_are_normalized_safely() -> None:
    research = valid_research_payload()
    research["hypotheses"].append({
        "hypothesis_id": "relationship_001",
        "category": "relationships",
        "claim": "Check whether table joins are structurally reliable.",
        "rationale": "Join coverage must be measured before relational features.",
        "expected_eda_checks": ["relationship_analyzer.join_key_coverage"],
        "priority": "P1",
        "confidence_before_eda": "medium",
        "source_refs": [],
        "status": "needs_eda",
        "limitations": [],
    })

    result = canonicalize_research_to_eda_contract(research, valid_task_plan_payload())
    hypothesis = result.research_hypotheses.hypotheses[-1]

    assert hypothesis.category == "relationship"
    assert hypothesis.expected_eda_checks == [
        "relationship_inferer.join_key_coverage"
    ]


@pytest.mark.parametrize("checks", [
    ["relationship_inferer.join_key_coverage"],
    [
        "feature_probe.feature_family_probe",
        "relationship_inferer.join_key_coverage",
    ],
])
def test_real_or_mixed_category_mismatch_remains_strict(checks: list[str]) -> None:
    research = valid_research_payload()
    research["hypotheses"].append({
        "hypothesis_id": "feat_001",
        "category": "feature",
        "claim": "Test candidate feature behavior.",
        "rationale": "Feature claims require category-specific evidence.",
        "expected_eda_checks": checks,
        "priority": "P1",
        "confidence_before_eda": "medium",
        "source_refs": [],
        "status": "needs_eda",
        "limitations": [],
    })
    canonical = canonicalize_research_to_eda_contract(
        research, valid_task_plan_payload()
    )

    result = validate_research_to_eda_contract(
        canonical.research_hypotheses, canonical.eda_task_plan
    )

    mismatch = next(
        issue for issue in result.errors
        if issue.code == "hypothesis_check_category_mismatch"
    )
    assert mismatch.hypothesis_id == "feat_001"
    assert mismatch.category == "feature"
    assert mismatch.check_module == "relationship_inferer"
    assert mismatch.allowed_modules == ["feature_probe"]
    assert "relationship_inferer" in mismatch.check_ref


def test_blocking_status_and_plan_list_are_rebuilt_from_registry() -> None:
    plan = valid_task_plan_payload()
    plan["eda_tasks"] = [
        {
            "task_id": "validation_001",
            "module": "validation_analyzer",
            "priority": "P0",
            "blocking": False,
            "related_hypothesis_ids": ["val_core"],
            "dependencies": [],
            "expected_outputs": [],
            "params": {},
        },
        {
            "task_id": "drift_001",
            "module": "drift_analyzer",
            "priority": "P1",
            "blocking": True,
            "related_hypothesis_ids": [],
            "dependencies": [],
            "expected_outputs": [],
            "params": {},
        },
    ]
    plan["blocking_tasks"] = [
        "validation_analyzer", "drift_analyzer", "unknown_module"
    ]

    result = canonicalize_research_to_eda_contract(valid_research_payload(), plan)
    tasks = {task.module: task for task in result.eda_task_plan.eda_tasks}

    assert tasks["validation_analyzer"].blocking is True
    assert tasks["drift_analyzer"].blocking is False
    assert result.eda_task_plan.blocking_tasks == ["validation_analyzer"]


def test_duplicate_module_aliases_merge_and_rebuild_reference_indexes() -> None:
    research = valid_research_payload()
    research["hypotheses"].append({
        "hypothesis_id": "relationship_001",
        "category": "relationship",
        "claim": "Check table relationships.",
        "rationale": "Relational coverage is dataset-dependent.",
        "expected_eda_checks": ["relationship_inferer.relationships"],
        "priority": "P1",
        "confidence_before_eda": "medium",
        "source_refs": [],
        "status": "needs_eda",
        "limitations": [],
    })
    plan = valid_task_plan_payload()
    plan["eda_tasks"].extend([
        {
            "task_id": "relationship_legacy",
            "module": "relationship_analyzer",
            "priority": "P1",
            "blocking": True,
            "related_hypothesis_ids": ["relationship_001", "missing_001"],
            "dependencies": [],
            "expected_outputs": [],
            "params": {},
        },
        {
            "task_id": "relationship_current",
            "module": "relationship_inferer",
            "priority": "P1",
            "blocking": False,
            "related_hypothesis_ids": ["relationship_001"],
            "dependencies": [],
            "expected_outputs": [],
            "params": {},
        },
    ])

    result = canonicalize_research_to_eda_contract(research, plan)
    relationship_tasks = [
        task for task in result.eda_task_plan.eda_tasks
        if task.module == "relationship_inferer"
    ]

    assert len(relationship_tasks) == 1
    assert relationship_tasks[0].related_hypothesis_ids == ["relationship_001"]
    assert result.eda_task_plan.hypothesis_index["relationship_001"] == [
        "relationship_legacy"
    ]
    assert "missing_001" not in result.eda_task_plan.hypothesis_index
    assert result.eda_task_plan.recommended_module_sequence.count(
        "relationship_inferer"
    ) <= 1


def test_deterministic_fallback_passes_strict_contract_validation() -> None:
    fallback = build_deterministic_research_scout_fallback(
        competition_id="fallback-demo",
        competition_url="https://example.invalid/fallback-demo",
        competition_desc="Generic binary classification.",
        plan_data=PlanData(
            task_type="binary_classification",
            metric="roc_auc",
            domain="generic_tabular",
            kaggle_queries=[],
            arxiv_queries=[],
            github_queries=[],
        ),
        retrieved_documents=[],
    )
    canonical = canonicalize_research_to_eda_contract(
        fallback.to_research_hypotheses_payload(),
        fallback.to_eda_task_plan_payload(),
    )

    validation = require_valid_research_to_eda_contract(
        canonical.research_hypotheses, canonical.eda_task_plan
    )

    assert validation.valid
    assert {
        "file_inventory", "schema_inferer", "table_profiler", "metric_analyzer",
        "validation_analyzer", "leakage_checker",
    } == set(canonical.eda_task_plan.blocking_tasks)


@pytest.mark.asyncio
async def test_semantic_mismatch_uses_bounded_repair_then_fallback() -> None:
    invalid = {
        "hypotheses": [{
            "hypothesis_id": "feat_001",
            "category": "feature",
            "claim": "Test whether relational joins create useful features.",
            "rationale": "The claim spans incompatible check categories.",
            "expected_eda_checks": ["relationship_inferer.join_key_coverage"],
            "priority": "P1",
            "confidence_before_eda": "medium",
        }],
        "eda_task_plan": {},
    }

    class InvalidTwiceClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def chat_json(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(kwargs)
            return invalid

    client = InvalidTwiceClient()
    output = await run_research_scout(
        competition_id="fallback-demo",
        competition_url="https://example.invalid/fallback-demo",
        competition_desc="Generic binary classification.",
        plan_data=PlanData(
            task_type="binary_classification",
            metric="roc_auc",
            domain="generic_tabular",
            kaggle_queries=[],
            arxiv_queries=[],
            github_queries=[],
        ),
        retrieved_documents=[],
        client=client,  # type: ignore[arg-type]
    )

    assert len(client.calls) == 2
    assert output.models_used["fallback"] is True
    canonical = canonicalize_research_to_eda_contract(
        output.to_research_hypotheses_payload(), output.to_eda_task_plan_payload()
    )
    assert require_valid_research_to_eda_contract(
        canonical.research_hypotheses, canonical.eda_task_plan
    ).valid


def test_direct_validator_remains_strict_and_error_message_has_task_context() -> None:
    canonical = canonicalize_research_to_eda_contract(
        valid_research_payload(), valid_task_plan_payload()
    )
    invalid_plan = canonical.eda_task_plan.model_copy(deep=True)
    invalid_plan.eda_tasks[0].blocking = False

    with pytest.raises(ResearchEdaModulePlanError) as raised:
        require_valid_research_to_eda_contract(
            canonical.research_hypotheses, invalid_plan
        )

    assert "blocking_task_conflict[inventory]" in str(raised.value)
    issue = next(
        item for item in raised.value.result.errors
        if item.code == "blocking_task_conflict"
    )
    assert issue.task_id == "inventory"
    assert issue.module == "file_inventory"
    assert issue.task_blocking is False
    assert issue.listed_in_blocking_tasks is True
