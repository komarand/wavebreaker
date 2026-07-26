from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import kaggle_researcher.reasoning.final_synthesizer as final_synthesizer_module
from kaggle_researcher.eda.schemas import EdaEvidencePack, ResearchHypotheses, ResearchHypothesis
from kaggle_researcher.contracts.action_support import UnsupportedFinalStrategyActionError
from kaggle_researcher.contracts.experiments import (
    CrossNamespaceReferenceError,
    ReferenceIssue,
)
from kaggle_researcher.contracts.final_strategy_compilation import (
    FinalStrategyCompilationError,
    FinalStrategySchemaValidationError,
)
from kaggle_researcher.contracts.final_synthesis_diagnostics import (
    FinalSynthesisDiagnostics,
)
from tests.contracts.factories import build_final_strategy_reference_catalog
from kaggle_researcher.reasoning.final_synthesizer import (
    REQUIRED_SECTION_IDS,
    FinalStrategyResult,
    build_deterministic_provenance_links,
    build_fallback_final_strategy,
    postprocess_final_strategy_result,
    repair_final_strategy_payload,
    render_final_strategy,
    render_final_strategy_summary,
    validate_rendered_strategy_quality,
)
from kaggle_researcher.schemas import PlanData, RetrievedDocument
from tests.fixtures.final_synthesis import synthesize_for_test as synthesize_final_strategy


@pytest.fixture(autouse=True)
def _exercise_legacy_monolithic_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module is the retained v1 protocol compatibility suite."""

    monkeypatch.setenv("FINAL_SYNTHESIS_PROTOCOL", "monolithic_legacy")


class FakeFinalSynthesizerClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.response


class SequentialFinalSynthesizerClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.responses[len(self.calls) - 1]


class InvalidJsonFinalSynthesizerClient:
    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        raise ValueError(
            "JSON parse failed; authorization=Bearer mocked-secret and sk-mockedsecret123"
        )


@pytest.mark.asyncio
async def test_mock_llm_response_validates_into_final_strategy_result() -> None:
    client = FakeFinalSynthesizerClient(_strategy_payload())

    result = await synthesize_final_strategy(
        competition_desc="Generic iid binary classification with ROC AUC.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        domain_patterns=[{"competition_family": "generic_tabular"}],
        research_hypotheses=_research_hypotheses(),
        eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
        reasoning_outputs={"metric": {"confidence": "medium"}},
        client=client,
        model="deepseek-v4-pro",
    )

    assert isinstance(result, FinalStrategyResult)
    assert result.competition_id == "generic-binary"
    assert result.recommended_validation == "stratified_kfold"
    assert {section.section_id for section in result.sections} == set(REQUIRED_SECTION_IDS)
    assert result.actions[0].evidence_refs == ["validation_evidence.primary_validation"]
    assert result.actions[0].related_hypothesis_ids == ["val_001"]
    assert result.actions[0].source_refs == ["retrieved-1"]
    assert result.models_used["final_synthesizer"] == "deepseek-v4-pro"
    assert result.synthesis_status == "llm_success"
    assert result.llm_output_valid is True
    assert result.repair_attempted is False
    assert result.repair_succeeded is False
    assert result.fallback_used is False


@pytest.mark.asyncio
async def test_valid_initial_synthesis_writes_success_diagnostics(
    tmp_path: Path,
) -> None:
    result = await synthesize_final_strategy(
        competition_desc="Generic iid binary classification with ROC AUC.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        domain_patterns=[],
        research_hypotheses=_research_hypotheses(),
        eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
        reasoning_outputs={},
        client=FakeFinalSynthesizerClient(_strategy_payload()),
        model="deepseek-v4-pro",
        diagnostics_dir=tmp_path,
    )

    diagnostics = _load_synthesis_diagnostics(tmp_path)
    assert diagnostics.initial_output_valid is True
    assert diagnostics.repair_attempted is False
    assert diagnostics.fallback_required is False
    assert result.synthesis_status == "llm_success"
    assert result.llm_output_valid is True
    assert result.repair_attempted is False
    assert result.repair_succeeded is False
    assert result.fallback_used is False
    assert [attempt.attempt for attempt in diagnostics.attempts] == ["initial_llm"]


@pytest.mark.asyncio
async def test_invalid_json_writes_sanitized_parse_diagnostic(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="JSON parse failed"):
        await synthesize_final_strategy(
            competition_desc="Generic iid binary classification with ROC AUC.",
            plan_data=_plan(),
            retrieved_documents=[_doc()],
            domain_patterns=[],
            research_hypotheses=_research_hypotheses(),
            eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
            reasoning_outputs={},
            client=InvalidJsonFinalSynthesizerClient(),
            model="deepseek-v4-pro",
            diagnostics_dir=tmp_path,
        )

    diagnostics_path = tmp_path / "final_synthesis_diagnostics.json"
    diagnostics = _load_synthesis_diagnostics(tmp_path)
    initial = diagnostics.attempts[0]
    assert initial.output_received is True
    assert initial.json_parse_succeeded is False
    assert [issue.stage for issue in initial.issues] == ["llm_parse"]
    serialized = diagnostics_path.read_text(encoding="utf-8")
    assert "mocked-secret" not in serialized
    assert "sk-mockedsecret123" not in serialized


@pytest.mark.asyncio
async def test_missing_p0_refs_are_grounded_and_diagnostics_are_written(
    tmp_path: Path,
) -> None:
    payload = _strategy_payload()
    for action in [
        *payload["actions"],
        *[
            action
            for section in payload["sections"]
            for action in section.get("actions", [])
        ],
    ]:
        action["action_id"] = "action_val_001"
        action["evidence_refs"] = []
    client = FakeFinalSynthesizerClient(payload)

    result = await synthesize_final_strategy(
        competition_desc="Generic iid binary classification with ROC AUC.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        domain_patterns=[],
        research_hypotheses=_research_hypotheses(),
        eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
        reasoning_outputs={},
        client=client,
        model="deepseek-v4-pro",
        diagnostics_dir=tmp_path,
    )

    action = next(item for item in result.actions if item.action_id == "action_val_001")
    assert action.priority == "P0"
    assert action.evidence_refs == ["validation_evidence.primary_validation"]
    for name in (
        "final_strategy_raw_payload.json",
        "final_strategy_normalized_payload.json",
        "final_strategy_action_support_report.json",
        "final_strategy_validation_errors.json",
        "final_synthesis_diagnostics.json",
    ):
        assert (tmp_path / name).is_file()
    report = json.loads(
        (tmp_path / "final_strategy_action_support_report.json").read_text(
            encoding="utf-8"
        )
    )
    resolution = next(
        item
        for item in report["evidence_resolution"]["actions"]
        if item["action_id"] == "action_val_001"
    )
    assert resolution["original_refs"] == []
    assert resolution["added_refs"] == ["validation_evidence.primary_validation"]
    assert resolution["unresolved_refs"] == []
    assert report["strict_gate"]["failed_actions"] == []
    diagnostics = _load_synthesis_diagnostics(tmp_path)
    assert diagnostics.initial_output_valid is False
    assert diagnostics.repair_attempted is True
    assert diagnostics.repair_succeeded is True
    assert diagnostics.fallback_required is False
    assert result.synthesis_status == "repaired_success"
    assert result.llm_output_valid is False
    assert result.repair_attempted is True
    assert result.repair_succeeded is True
    assert result.fallback_used is False
    assert any(
        issue.issue_type == "missing_evidence_reference"
        and "action_val_001" in (issue.field_path or "")
        for issue in diagnostics.attempts[0].issues
    )


@pytest.mark.asyncio
async def test_raw_hypothesis_evidence_ref_is_migrated_before_action_validation() -> None:
    payload = _strategy_payload()
    for action in [
        *payload["actions"],
        *[
            action
            for section in payload["sections"]
            for action in section.get("actions", [])
        ],
    ]:
        action["evidence_refs"] = ["val_001"]
        action["related_hypothesis_ids"] = []
        action["hypothesis_ids"] = []
    client = FakeFinalSynthesizerClient(payload)

    result = await synthesize_final_strategy(
        competition_desc="Generic iid binary classification with ROC AUC.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        domain_patterns=[],
        research_hypotheses=_research_hypotheses(),
        eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
        reasoning_outputs={},
        client=client,
        model="deepseek-v4-pro",
    )

    assert result.actions[0].evidence_refs == ["validation_evidence.primary_validation"]
    assert result.actions[0].related_hypothesis_ids == ["val_001"]
    assert any(
        item["field_path"] == "hypothesis_reference_migration.moved_hypothesis_refs"
        for item in result.reference_repairs
    )


@pytest.mark.asyncio
async def test_traceback_composite_refs_resolve_before_namespace_validation() -> None:
    pack_payload = _eda_pack(primary_method="stratified_kfold").model_dump(mode="json")
    pack_payload.update({
        "eda_risks": [{
            "risk_id": "risk_leakage_001", "risk_type": "leakage",
            "severity": "high", "status": "confirmed", "confidence": "high",
            "title": "Leakage", "finding": "Groups can leak.",
            "impact": "Scores become optimistic.",
            "evidence_refs": ["validation_evidence.primary_validation"],
        }],
        "validation_requirements": [{
            "validation_requirement_id": "validation_requirement_003",
            "rule": "Use grouped validation.", "reason": "Groups repeat.",
            "status": "recommended", "mandatory": False,
            "evidence_refs": ["validation_evidence.primary_validation"],
        }],
        "safety_constraints": [{
            "safety_constraint_id": "safety_003", "scope": "validation",
            "rule": "Do not split groups.", "reason": "Prevent leakage.",
            "severity": "advisory", "blocking": False,
            "evidence_refs": ["validation_evidence.primary_validation"],
        }],
    })
    pack = EdaEvidencePack.model_validate(pack_payload)
    payload = _strategy_payload()
    composite_refs = [
        "risk_leakage_001",
        "validation_requirements.validation_requirement_003",
        "safety_constraints.safety_003",
    ]
    for action in [
        *payload["actions"],
        *[action for section in payload["sections"] for action in section.get("actions", [])],
    ]:
        action["evidence_refs"] = list(composite_refs)
        action["risk_ids"] = ["risk_leakage_001"]
        action["validation_requirement_ids"] = ["validation_requirement_003"]
        action["safety_constraint_ids"] = ["safety_003"]
    client = FakeFinalSynthesizerClient(payload)

    result = await synthesize_final_strategy(
        competition_desc="Generic grouped binary classification.",
        plan_data=_plan(), retrieved_documents=[_doc()], domain_patterns=[],
        research_hypotheses=_research_hypotheses(), eda_evidence_pack=pack,
        reasoning_outputs={}, client=client, model="deepseek-v4-pro",
    )

    assert len(client.calls) == 1
    assert result.actions[0].evidence_refs == ["validation_evidence.primary_validation"]
    assert not set(composite_refs) & set(result.actions[0].evidence_refs)
    assert all(action.evidence_refs for action in _result_actions(result))
    assert not {
        "unknown", "unresolved", "missing_evidence", "generated_by_llm",
    } & {ref for action in _result_actions(result) for ref in action.evidence_refs}
    assert any(
        item["field_path"] == "composite_reference_resolution.resolved_composite_refs"
        for item in result.reference_repairs
    )


@pytest.mark.asyncio
async def test_prompt_includes_final_synthesis_guardrails() -> None:
    client = FakeFinalSynthesizerClient(_strategy_payload())

    await synthesize_final_strategy(
        competition_desc="Generic iid binary classification with ROC AUC.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        domain_patterns=[],
        research_hypotheses=_research_hypotheses(),
        eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
        reasoning_outputs={},
        client=client,
        model="deepseek-v4-pro",
    )

    call = client.calls[0]
    prompt = call["user_prompt"]
    system_prompt = call["system_prompt"]
    combined = f"{system_prompt}\n{prompt}"
    assert call["model"] == "deepseek-v4-pro"
    assert "source -> hypothesis -> EDA -> strategy" in prompt
    assert "Respect validation_evidence.primary_validation" in prompt
    assert "Do not include raw chain-of-thought" in combined
    assert "Do not claim that notebooks were executed" in combined
    assert "Do not claim that baseline is final solution" in combined
    assert "typed support_refs" in combined
    assert '"allowed_hypothesis_ids"' in prompt
    assert "Do not invent IDs" in prompt
    payload = json.loads(prompt)
    assert "allowed_experiment_ids" in payload
    assert "approved_experiment_ids" in payload
    assert "rejected_experiment_ids" in payload
    assert "approved_experiments" in payload
    assert "allowed_evidence_refs" in payload
    assert "allowed_eda_result_refs" in payload
    assert "allowed_support_refs" in payload
    assert "approved_experiments is a context section" in payload["evidence_reference_instruction"]
    action_draft_schema = payload["expected_schema"]["$defs"]["FinalStrategyActionDraft"]
    assert "support_refs" in action_draft_schema["properties"]
    assert "evidence_refs" not in action_draft_schema["properties"]
    assert any(
        "Do not place hypothesis IDs into experiment_ids" in rule
        for rule in payload["guardrails"]
    )


@pytest.mark.asyncio
async def test_final_synthesizer_repairs_invalid_context_evidence_once() -> None:
    pack = _rich_eda_pack().model_copy(update={
        "baseline_ablation_evidence": {
            "feature_block_findings": [
                {"feature_block": "low_cardinality_categorical", "finding": "useful"},
                {"feature_block": "high_cardinality_categorical", "finding": "mixed"},
                {"feature_block": "text", "finding": "not_testable"},
            ]
        }
    })
    semantic_refs = [
        "baseline_ablation_evidence.feature_block_findings.low_cardinality_categorical",
        "baseline_ablation_evidence.feature_block_findings.high_cardinality_categorical",
        "baseline_ablation_evidence.feature_block_findings.text",
    ]
    invalid = _strategy_payload()
    ablation_action = deepcopy(invalid["actions"][0])
    ablation_action.update({
        "action_id": "action_ablation",
        "action": "Prioritize the feature blocks supported by baseline ablations.",
        "evidence_refs": [*semantic_refs, "approved_experiments"],
        "eda_result_refs": [*semantic_refs, "approved_experiments"],
        "experiment_ids": [],
    })
    invalid["actions"].append(ablation_action)
    corrected = deepcopy(invalid)
    corrected_action = corrected["actions"][-1]
    corrected_action["evidence_refs"] = semantic_refs
    corrected_action["eda_result_refs"] = semantic_refs
    corrected_action["experiment_ids"] = ["exp_001"]
    client = SequentialFinalSynthesizerClient([invalid, corrected])
    reasoning_outputs = {
        "experiments": [{
            "experiment_id": "exp_001",
            "source_hypothesis_ids": ["val_001"],
            "priority": "P1",
            "experiment": "Test feature blocks independently.",
            "why": "Measure incremental value from each feature family.",
            "cost": "low",
            "expected_gain": "diagnostic",
            "risk": "fold variance",
            "evidence_ids": semantic_refs,
        }],
        "review": {
            "confidence": "medium",
            "reviewed_experiment_ids": ["exp_001"],
            "approved_experiment_ids": ["exp_001"],
            "rejected_experiment_ids": [],
        },
    }

    result = await synthesize_final_strategy(
        competition_desc="Generic iid binary classification with ROC AUC.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        domain_patterns=[],
        research_hypotheses=_research_hypotheses(),
        eda_evidence_pack=pack,
        reasoning_outputs=reasoning_outputs,
        client=client,
        model="deepseek-v4-pro",
    )

    assert len(client.calls) == 2
    repaired = next(action for action in result.actions if action.action_id == "action_ablation")
    assert set(semantic_refs) <= set(repaired.evidence_refs)
    assert set(semantic_refs) <= set(repaired.eda_result_refs)
    assert repaired.experiment_ids == ["exp_001"]
    assert all(
        "approved_experiments" not in [*action.evidence_refs, *action.eda_result_refs]
        for action in _result_actions(result)
    )
    initial_prompt = json.loads(client.calls[0]["user_prompt"])
    assert set(semantic_refs) <= set(initial_prompt["allowed_eda_result_refs"])
    repair_prompt = json.loads(client.calls[1]["user_prompt"])
    assert any(
        issue["value"] == "approved_experiments"
        and issue["reason"] == "context_label_not_reference"
        for issue in repair_prompt["invalid_references"]
    )


@pytest.mark.asyncio
async def test_stratified_kfold_from_eda_cannot_be_overridden_with_temporal_cv() -> None:
    payload = _strategy_payload(recommended_validation="temporal_cv")
    payload["actions"][0]["validation_strategy"] = "temporal_cv"
    client = FakeFinalSynthesizerClient(payload)

    with pytest.raises(ValueError, match="stratified_kfold"):
        await synthesize_final_strategy(
            competition_desc="Generic iid binary classification with ROC AUC.",
            plan_data=_plan(),
            retrieved_documents=[_doc()],
            domain_patterns=[],
            research_hypotheses=_research_hypotheses(),
            eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
            reasoning_outputs={},
            client=client,
            model="deepseek-v4-pro",
        )


def test_repair_action_missing_related_hypothesis_ids() -> None:
    repaired = repair_final_strategy_payload(
        _repair_payload(
            action="Use StratifiedKFold for model validation.",
            evidence_refs=["validation_evidence.primary_validation"],
        ),
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=_repair_eda_payload(),
    )

    assert repaired["actions"][0]["related_hypothesis_ids"] == ["val_001"]


def test_repair_metric_action_missing_ids() -> None:
    repaired = repair_final_strategy_payload(
        _repair_payload(
            action="Tune the metric threshold only on validation folds.",
            evidence_refs=["metric_evidence.requires_threshold"],
        ),
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=_repair_eda_payload(),
    )

    assert repaired["actions"][0]["related_hypothesis_ids"] == [
        "metric_001",
        "val_001",
    ]


def test_repair_feature_action_missing_ids() -> None:
    repaired = repair_final_strategy_payload(
        _repair_payload(
            action="Use target encoding only when it is fold-fitted.",
            evidence_refs=["feature_probe_evidence"],
        ),
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=_repair_eda_payload(),
    )

    assert "leak_001" in repaired["actions"][0]["related_hypothesis_ids"]


def test_repair_empty_section() -> None:
    payload = _repair_payload(
        action="Use StratifiedKFold for model validation.",
        evidence_refs=["validation_evidence.primary_validation"],
    )
    payload["sections"] = [
        {
            "section_id": "baseline_findings",
            "title": "Baseline Findings",
            "summary": "The model response omitted links.",
        }
    ]

    repaired = repair_final_strategy_payload(
        payload,
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=_repair_eda_payload(),
    )
    result = FinalStrategyResult.model_validate(repaired)

    assert result.sections[0].action_ids or result.sections[0].evidence_refs
    assert REPAIR_NOTE in result.limitations


@pytest.mark.asyncio
async def test_result_from_payload_repairs_llm_payload() -> None:
    payload = _repair_payload(
        action="Use StratifiedKFold for model validation.",
        evidence_refs=["validation_evidence.primary_validation"],
    )
    payload["sections"] = [
        {
            "section_id": "baseline_findings",
            "title": "Baseline Findings",
            "summary": "Incomplete baseline response.",
        },
        {
            "section_id": "leakage_and_data_quality",
            "title": "Leakage And Data Quality",
            "summary": "Encoding safety.",
            "actions": [
                {
                    "priority": "P0",
                    "action": "Avoid naive target encoding.",
                    "reason": "The feature probe marked it unsafe.",
                    "evidence_refs": ["feature_probe_evidence"],
                    "related_hypothesis_ids": [],
                }
            ],
        },
    ]
    client = FakeFinalSynthesizerClient(payload)

    result = await synthesize_final_strategy(
        competition_desc="Generic binary classification.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        domain_patterns=[],
        research_hypotheses=_rich_research_hypotheses(),
        eda_evidence_pack=_rich_eda_pack(),
        reasoning_outputs={},
        client=client,
        model="deepseek-v4-pro",
    )

    action_text = " ".join(action.action.lower() for action in _result_actions(result))
    assert "stratifiedkfold" in action_text.replace("_", "")
    assert "target encoding" in action_text
    assert all(action.related_hypothesis_ids for action in _result_actions(result))


def test_fallback_strategy_validates() -> None:
    eda = _repair_eda_payload()
    eda["validation_requirements"] = [{
        "validation_requirement_id": "validation_requirement_001",
        "mandatory": True,
    }]
    eda["safety_constraints"] = [{
        "safety_constraint_id": "safety_001",
        "blocking": True,
    }]
    fallback = build_fallback_final_strategy(
        competition_id="generic-binary",
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=eda,
    )

    result = FinalStrategyResult.model_validate(fallback)

    assert len(result.sections) >= 8
    assert all(action.related_hypothesis_ids for action in _result_actions(result))
    assert all(action.evidence_refs for action in _result_actions(result))
    assert result.selected_validation_requirement_ids == ["validation_requirement_001"]
    assert result.enforced_safety_constraint_ids == ["safety_001"]


def test_degraded_fallback_contains_all_sections_once_in_canonical_order() -> None:
    payload = build_fallback_final_strategy(
        competition_id="generic-binary",
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=_repair_eda_payload(),
        task_type="binary_classification",
        metric_name="roc_auc",
    )

    result = FinalStrategyResult.model_validate(payload)
    section_ids = [section.section_id for section in result.sections]

    assert section_ids == REQUIRED_SECTION_IDS
    assert len(section_ids) == len(set(section_ids)) == 11


def test_fallback_missing_baseline_is_limitation_not_reproduction() -> None:
    eda = _repair_eda_payload()
    eda["baseline_evidence"] = {"status": "skipped", "reason": "disabled"}

    result = FinalStrategyResult.model_validate(build_fallback_final_strategy(
        competition_id="generic-binary",
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=eda,
        task_type="binary_classification",
        metric_name="roc_auc",
    ))
    section = next(item for item in result.sections if item.section_id == "baseline_findings")
    section_actions = [
        action for action in result.actions if action.action_id in section.action_ids
    ]

    assert section.availability == "not_available"
    assert "status=skipped" in section.summary
    assert section.limitations
    assert not any("reproduce" in action.action.lower() for action in section_actions)


def test_fallback_missing_drift_is_explicitly_not_available() -> None:
    eda = _repair_eda_payload()
    eda["drift_evidence"] = {}

    result = FinalStrategyResult.model_validate(build_fallback_final_strategy(
        competition_id="generic-binary",
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=eda,
    ))
    section = next(
        item for item in result.sections
        if item.section_id == "drift_and_leaderboard_risk"
    )

    assert section.availability == "not_available"
    assert "not_available" in section.summary
    assert "high drift" not in section.summary.lower()


def test_fallback_first_48_hours_references_existing_actions_and_experiments() -> None:
    result = FinalStrategyResult.model_validate(build_fallback_final_strategy(
        competition_id="generic-binary",
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=_repair_eda_payload(),
        task_type="binary_classification",
        metric_name="roc_auc",
    ))
    section = next(item for item in result.sections if item.section_id == "first_48_hours")
    action_ids = {action.action_id for action in result.actions}
    experiment_ids = {
        action.experiment_id for action in result.actions if action.experiment_id
    }

    assert [block.time_window for block in section.time_blocks] == [
        "0-4_hours", "4-12_hours", "12-24_hours", "24-48_hours",
    ]
    assert all(block.action_ids or block.experiment_ids for block in section.time_blocks)
    assert all(set(block.action_ids) <= action_ids for block in section.time_blocks)
    assert all(set(block.experiment_ids) <= experiment_ids for block in section.time_blocks)


def test_fallback_experiments_queue_contains_structured_experiments() -> None:
    result = FinalStrategyResult.model_validate(build_fallback_final_strategy(
        competition_id="generic-binary",
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=_repair_eda_payload(),
        task_type="binary_classification",
        metric_name="roc_auc",
    ))
    section = next(item for item in result.sections if item.section_id == "experiments_queue")
    experiments = [
        action for action in result.actions
        if action.action_id in section.action_ids and action.experiment_id
    ]

    assert experiments
    assert all(action.hypothesis for action in experiments)
    assert all(action.exact_change for action in experiments)
    assert all(action.validation_policy == "stratified_kfold" for action in experiments)
    assert all(action.success_criterion for action in experiments)
    assert all(action.risk for action in experiments)
    assert not any(
        action.action == "Run the evidence-backed P0 actions first."
        for action in experiments
    )


def test_fallback_executive_summary_and_markdown_expose_complete_degradation() -> None:
    result = FinalStrategyResult.model_validate(build_fallback_final_strategy(
        competition_id="generic-binary",
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=_repair_eda_payload(),
        task_type="binary_classification",
        metric_name="roc_auc",
    ))
    executive = result.sections[0]
    markdown = render_final_strategy(result)

    assert "degraded fallback" in executive.summary.lower()
    assert "binary_classification" in executive.summary
    assert "roc_auc" in executive.summary
    assert "stratified_kfold" in executive.summary
    for section in result.sections:
        assert f"## {section.title}" in markdown


def test_sparse_fallback_does_not_invent_model_or_experiment_actions() -> None:
    result = FinalStrategyResult.model_validate(build_fallback_final_strategy(
        competition_id="sparse",
        research_hypotheses=[{"hypothesis_id": "schema_001", "category": "schema"}],
        eda_evidence_pack={},
        task_type="unknown",
        metric_name="unknown",
    ))
    modeling = next(item for item in result.sections if item.section_id == "modeling_plan")
    experiments = next(item for item in result.sections if item.section_id == "experiments_queue")

    assert modeling.availability == "not_available"
    assert experiments.availability == "not_available"
    assert not experiments.action_ids
    assert not any(action.experiment_id for action in result.actions)
    assert all(
        action.action.startswith("Pause evidence-dependent modeling")
        for action in result.actions
    )


def test_degraded_fallback_schema_rejects_missing_or_reordered_sections() -> None:
    payload = build_fallback_final_strategy(
        competition_id="generic-binary",
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=_repair_eda_payload(),
    )
    payload["sections"][0], payload["sections"][1] = (
        payload["sections"][1], payload["sections"][0]
    )

    with pytest.raises(ValueError, match="canonical order"):
        FinalStrategyResult.model_validate(payload)


@pytest.mark.asyncio
async def test_invalid_repair_uses_fallback_strategy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _strategy_payload()
    payload["actions"][0]["priority"] = "not-a-priority"
    payload["actions"][0]["confidence"] = "not-a-confidence"
    client = FakeFinalSynthesizerClient(payload)
    monkeypatch.setattr(
        final_synthesizer_module,
        "repair_final_strategy_payload",
        lambda payload, **kwargs: payload,
    )

    result = await synthesize_final_strategy(
        competition_desc="Generic binary classification.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        domain_patterns=[],
        research_hypotheses=_rich_research_hypotheses(),
        eda_evidence_pack=_rich_eda_pack(),
        reasoning_outputs={},
        client=client,
        model="deepseek-v4-pro",
        diagnostics_dir=tmp_path,
    )

    assert any("built from available Scout hypotheses" in item for item in result.limitations)
    assert all(action.related_hypothesis_ids for action in _result_actions(result))
    diagnostics = _load_synthesis_diagnostics(tmp_path)
    assert [attempt.attempt for attempt in diagnostics.attempts] == [
        "initial_llm",
        "deterministic_repair",
    ]
    assert diagnostics.repair_attempted is True
    assert diagnostics.repair_succeeded is False
    assert diagnostics.fallback_required is True
    assert diagnostics.fallback_reason
    assert len(diagnostics.attempts[0].issues) > 1
    assert len(diagnostics.attempts[1].issues) > 1
    assert result.synthesis_status == "degraded_fallback"
    assert result.llm_output_valid is False
    assert result.repair_attempted is True
    assert result.repair_succeeded is False
    assert result.fallback_used is True


@pytest.mark.asyncio
async def test_final_synthesis_uses_allowed_hypothesis_ids_only(
    tmp_path: Path,
) -> None:
    payload = _repair_payload(
        action="Use StratifiedKFold for model validation.",
        evidence_refs=["validation_evidence.primary_validation"],
    )
    payload["actions"][0]["related_hypothesis_ids"] = ["invented_999"]
    client = FakeFinalSynthesizerClient(payload)

    result = await synthesize_final_strategy(
        competition_desc="Generic binary classification.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        domain_patterns=[],
        research_hypotheses=_rich_research_hypotheses(),
        eda_evidence_pack=_rich_eda_pack(),
        reasoning_outputs={},
        client=client,
        model="deepseek-v4-pro",
        diagnostics_dir=tmp_path,
    )

    allowed = {item["hypothesis_id"] for item in _repair_hypotheses()}
    assert {
        hypothesis_id
        for action in _result_actions(result)
        for hypothesis_id in action.related_hypothesis_ids
    } <= allowed
    diagnostics = _load_synthesis_diagnostics(tmp_path)
    initial_issues = diagnostics.attempts[0].issues
    assert any(
        issue.issue_type == "unknown_hypothesis_id"
        and issue.invalid_reference == "invented_999"
        and issue.field_path == "actions[0].related_hypothesis_ids"
        for issue in initial_issues
    )
    assert diagnostics.repair_succeeded is True
    assert diagnostics.fallback_required is False


def test_two_hypothesis_source_refs_create_two_validated_links() -> None:
    second_document = _doc().model_copy(update={
        "id": "retrieved-2",
        "title": "Validation paper",
        "source": "arxiv",
    })
    hypotheses = _research_hypotheses().model_copy(deep=True)
    hypotheses.hypotheses[0].source_refs = [
        "retrieved-1", "retrieved-2", "retrieved-1",
    ]
    pack = _eda_pack(primary_method="stratified_kfold")
    pack.hypothesis_results[0].evidence_refs = [
        "validation_evidence.primary_validation",
        "validation_evidence.primary_validation",
    ]
    catalog = build_final_strategy_reference_catalog(
        pack,
        research_hypotheses=hypotheses,
        retrieved_documents=[_doc(), second_document],
    )

    source_links, eda_links, repairs = build_deterministic_provenance_links(
        research_hypotheses=hypotheses,
        eda_evidence_pack=pack,
        reference_catalog=catalog,
    )

    assert [str(link.source_ref) for link in source_links] == [
        "retrieved-1", "retrieved-2",
    ]
    assert all(link.hypothesis_id == "val_001" for link in source_links)
    assert len(eda_links) == 1
    assert repairs == []
    second_entry = catalog.resolve("retrieved-2", "source_claim").entry
    assert second_entry is not None
    assert second_entry.title == "Validation paper"
    assert second_entry.source_type == "arxiv"


@pytest.mark.parametrize("status", ["confirmed", "rejected"])
def test_hypothesis_result_status_is_preserved_in_eda_link(status: str) -> None:
    pack = _eda_pack(primary_method="stratified_kfold").model_copy(deep=True)
    pack.hypothesis_results[0].status = status
    hypotheses = _research_hypotheses()
    catalog = build_final_strategy_reference_catalog(
        pack,
        research_hypotheses=hypotheses,
        retrieved_documents=[_doc()],
    )

    _, links, _ = build_deterministic_provenance_links(
        research_hypotheses=hypotheses,
        eda_evidence_pack=pack,
        reference_catalog=catalog,
    )

    assert len(links) == 1
    assert links[0].eda_result_ref == "validation_evidence.primary_validation"
    assert links[0].result_status == status
    assert links[0].finding_summary == pack.hypothesis_results[0].finding


def test_fallback_strategy_preserves_structured_provenance_links() -> None:
    hypotheses = _research_hypotheses()
    pack = _eda_pack(primary_method="stratified_kfold")

    payload = build_fallback_final_strategy(
        competition_id="generic-binary",
        research_hypotheses=[
            item.model_dump(mode="json") for item in hypotheses.hypotheses
        ],
        eda_evidence_pack=pack.model_dump(mode="json"),
    )
    result = FinalStrategyResult.model_validate(payload)

    assert result.source_to_hypothesis_links
    assert result.hypothesis_to_eda_links
    assert result.action_provenance
    assert result.hypothesis_to_eda_links[0].result_status == "confirmed"


def test_unknown_source_ref_is_removed_with_recorded_repair() -> None:
    hypotheses = _research_hypotheses().model_copy(deep=True)
    hypotheses.hypotheses[0].source_refs = ["retrieved-1", "invented-source"]
    pack = _eda_pack(primary_method="stratified_kfold")
    catalog = build_final_strategy_reference_catalog(
        pack,
        research_hypotheses=hypotheses,
        retrieved_documents=[_doc()],
    )

    links, _, repairs = build_deterministic_provenance_links(
        research_hypotheses=hypotheses,
        eda_evidence_pack=pack,
        reference_catalog=catalog,
    )

    assert [str(link.source_ref) for link in links] == ["retrieved-1"]
    assert {
        (repair["field_path"], repair["original_id"], repair["replacement_id"])
        for repair in repairs
    } == {(
        "research_hypotheses.val_001.source_refs",
        "invented-source",
        "",
    )}


def test_eda_only_safety_action_does_not_require_fake_source() -> None:
    payload = _strategy_payload()
    action = payload["actions"][0]
    action["source_refs"] = []
    action["source_claim"] = None
    action["evidence_origin"] = "Safety-warning"
    action["safety_constraint_ids"] = ["safety_001"]
    payload["source_to_hypothesis_links"] = []

    result = FinalStrategyResult.model_validate(payload)

    assert result.actions[0].source_refs == []
    assert result.actions[0].safety_constraint_ids == ["safety_001"]
    assert result.actions[0].eda_result_refs == [
        "validation_evidence.primary_validation"
    ]


def test_section_provenance_contains_only_its_action_links() -> None:
    result = FinalStrategyResult.model_validate(_strategy_payload())
    action = result.actions[0]
    section = next(
        section for section in result.sections
        if action.action_id in section.action_ids
    )

    assert section.related_hypothesis_ids == action.hypothesis_ids
    assert section.source_refs == action.source_refs
    assert section.eda_result_refs == action.eda_result_refs
    assert result.action_provenance[0].action_id == action.action_id


def test_duplicate_structured_links_are_rejected() -> None:
    payload = _strategy_payload()
    payload["source_to_hypothesis_links"] *= 2

    with pytest.raises(ValueError, match="duplicate links"):
        FinalStrategyResult.model_validate(payload)


def test_postprocessor_deduplicates_validation_actions() -> None:
    result = _quality_result(
        {
            "metric_and_validation": [
                _quality_action(
                    "Use stratified k-fold cross-validation for model comparison.",
                    ["validation_evidence.primary_validation"],
                    priority="P0",
                )
            ],
            "modeling_plan": [
                _quality_action(
                    "Use stratified k-fold cross-validation for model comparison.",
                    ["validation_evidence.diagnostic_validations"],
                    priority="P0",
                )
            ],
        }
    )

    cleaned = postprocess_final_strategy_result(
        result,
        eda_evidence_pack=_quality_eda(),
    )
    validation_actions = [
        action
        for action in cleaned.actions
        if "model comparison" in action.action.lower()
    ]

    assert len(validation_actions) == 1
    assert validation_actions[0].priority == "P0"
    assert set(validation_actions[0].evidence_refs) == {
        "validation_evidence.primary_validation",
        "validation_evidence.diagnostic_validations",
    }


def test_primary_id_split_is_rewritten_generically() -> None:
    result = _quality_result(
        {
            "metric_and_validation": [
                _quality_action(
                    "Split by row_id for validation.",
                    ["inferred_schema.primary_id_column"],
                )
            ]
        }
    )

    cleaned = postprocess_final_strategy_result(
        result,
        eda_evidence_pack=_quality_eda(),
    )
    text = render_final_strategy(cleaned).lower()

    assert "split by row_id" not in text
    assert "excluded from features" in text
    assert "alignment or tracking" in text or "alignment or row tracking" in text


def test_group_validation_allows_group_column_but_excludes_primary_id() -> None:
    eda = _quality_eda()
    eda["validation_evidence"]["primary_validation"] = {
        "method": "group_kfold",
        "group_column": "entity_group",
    }
    eda["validation_evidence"]["group_columns"] = [{"column": "entity_group"}]
    result = _quality_result(
        {
            "metric_and_validation": [
                _quality_action(
                    "Use entity_group for group-aware validation.",
                    ["validation_evidence.primary_validation"],
                )
            ]
        },
        validation="group_kfold",
    )

    cleaned = postprocess_final_strategy_result(result, eda_evidence_pack=eda)
    text = render_final_strategy(cleaned).lower()

    assert "entity_group for group-aware validation" in text
    assert "primary id excluded from model features" in text


def test_drift_from_primary_id_is_rendered_as_diagnostic_artifact() -> None:
    eda = _quality_eda()
    eda["drift_evidence"] = {
        "severity": "high",
        "feature_drift_severity": "low",
        "excluded_columns": ["row_id"],
        "numeric_psi": {"columns": [{"column": "row_id", "severity": "high"}]},
    }
    result = _quality_result(
        {
            "drift_and_leaderboard_risk": [
                _quality_action(
                    "High PSI proves feature drift and should be modeled.",
                    ["drift_evidence"],
                    priority="P1",
                )
            ]
        }
    )

    cleaned = postprocess_final_strategy_result(result, eda_evidence_pack=eda)
    text = render_final_strategy(cleaned).lower()

    assert "diagnostic artifact" in text
    assert "assess feature drift separately" in text
    assert "should be modeled" not in text


def test_source_suggestion_is_labeled_and_changed_to_a_test() -> None:
    result = _quality_result(
        {
            "feature_priorities": [
                _quality_action(
                    "Use derived features from categorical columns.",
                    ["source-doc-1"],
                    source_refs=["source-doc-1"],
                )
            ]
        }
    )

    cleaned = postprocess_final_strategy_result(
        result,
        eda_evidence_pack=_quality_eda(),
        source_evidence=[{"id": "source-doc-1", "source": "paper"}],
    )
    source_action = next(
        action
        for action in cleaned.actions
        if "source-suggested" in action.action.lower()
    )

    assert source_action.evidence_origin == "Source-supported"
    assert source_action.action.startswith("Test ")
    assert "only if validation improves" in source_action.action


def test_skipped_baseline_gets_conservative_p0_and_uncertainty() -> None:
    eda = _quality_eda()
    eda["baseline_evidence"] = {}
    eda["notebook_static_analysis"] = {}
    result = _quality_result(
        {
            "modeling_plan": [
                _quality_action(
                    "Use an advanced boosted ensemble.",
                    ["source-doc-1"],
                    priority="P0",
                    source_refs=["source-doc-1"],
                )
            ]
        }
    )

    cleaned = postprocess_final_strategy_result(
        result,
        eda_evidence_pack=eda,
        source_evidence=[{"id": "source-doc-1"}],
    )

    assert any(
        action.priority == "P0" and "simple linear/logistic or tree baseline" in action.action
        for action in cleaned.actions
    )
    advanced = next(action for action in cleaned.actions if "boosted ensemble" in action.action)
    assert advanced.priority == "P1"
    assert any("Baseline evidence unavailable or skipped" in item for item in cleaned.limitations)
    assert any("Notebook static analysis unavailable or skipped" in item for item in cleaned.limitations)


def test_summary_is_short_and_clean_strategy_has_no_quality_warnings() -> None:
    result = FinalStrategyResult.model_validate(_strategy_payload())
    result.task_type = "binary_classification"
    cleaned = postprocess_final_strategy_result(
        result,
        eda_evidence_pack=_quality_eda(),
        source_evidence=[{"id": "retrieved-1"}],
    )
    full_text = render_final_strategy(cleaned)
    summary_text = render_final_strategy_summary(cleaned)

    assert len(summary_text) < 0.5 * len(full_text)
    assert "Top P0 Actions" in summary_text
    assert "Top Risks" in summary_text
    assert "First Experiments" in summary_text
    assert "Do Not Do" in summary_text
    assert validate_rendered_strategy_quality(
        cleaned,
        full_text,
        summary_text,
        eda_evidence_pack=_quality_eda(),
    ) == []


def test_markdown_synthesis_status_banner_is_status_driven() -> None:
    llm_result = FinalStrategyResult.model_validate(_strategy_payload())
    llm_markdown = render_final_strategy(llm_result)
    assert "## Synthesis Status" not in llm_markdown
    assert "required deterministic contract repair" not in llm_markdown

    repaired_payload = llm_result.model_dump(mode="json")
    repaired_payload.update({
        "synthesis_status": "repaired_success",
        "llm_output_valid": False,
        "repair_attempted": True,
        "repair_succeeded": True,
        "fallback_used": False,
    })
    repaired_markdown = render_final_strategy(
        FinalStrategyResult.model_validate(repaired_payload)
    )
    assert (
        "The model-generated strategy required deterministic contract repair."
        in repaired_markdown
    )
    assert "## Synthesis Status" not in repaired_markdown

    fallback_payload = llm_result.model_dump(mode="json")
    fallback_payload.update({
        "synthesis_status": "degraded_fallback",
        "llm_output_valid": False,
        "repair_attempted": True,
        "repair_succeeded": False,
        "fallback_used": True,
        "limitations": [
            "The LLM output was invalid, so a deterministic fallback was used."
        ],
    })
    fallback_payload["sections"][0]["summary"] = (
        "Degraded fallback strategy assembled from validated evidence."
    )
    fallback_payload["sections"][-1]["time_blocks"] = [
        {
            "time_window": window,
            "summary": "Continue the evidence-backed validation action.",
            "action_ids": ["action_validation"],
        }
        for window in (
            "0-4_hours", "4-12_hours", "12-24_hours", "24-48_hours",
        )
    ]
    fallback_markdown = render_final_strategy(
        FinalStrategyResult.model_validate(fallback_payload)
    )
    assert "## Synthesis Status" in fallback_markdown
    assert "- Status: degraded fallback" in fallback_markdown
    assert (
        "The LLM-generated strategy did not satisfy the final strategy contract."
        in fallback_markdown
    )
    assert "assembled deterministically from validated EDA and Scout evidence" in fallback_markdown


def _quality_action(
    action: str,
    evidence_refs: list[str],
    *,
    priority: str = "P0",
    source_refs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "priority": priority,
        "action": action,
        "reason": "Generic evidence-linked recommendation.",
        "evidence_refs": evidence_refs,
        "related_hypothesis_ids": ["val_001"],
        "source_refs": source_refs or [],
    }


def _quality_result(
    actions_by_section: dict[str, list[dict[str, Any]]],
    *,
    validation: str = "stratified_kfold",
) -> FinalStrategyResult:
    sections = []
    all_actions = []
    for section_id, actions in actions_by_section.items():
        sections.append(
            {
                "section_id": section_id,
                "title": section_id.replace("_", " ").title(),
                "summary": "Generic strategy guidance.",
                "actions": actions,
            }
        )
        all_actions.extend(deepcopy(actions))
    return FinalStrategyResult.model_validate(
        {
            "competition_id": "generic-tabular",
            "synthesis_status": "llm_success",
            "llm_output_valid": True,
            "repair_attempted": False,
            "repair_succeeded": False,
            "fallback_used": False,
            "synthesis_diagnostics_path": None,
            "task_type": "binary_classification",
            "metric": {"name": "roc_auc"},
            "recommended_validation": validation,
            "sections": sections,
            "actions": all_actions,
        }
    )


def _quality_eda() -> dict[str, Any]:
    eda = _repair_eda_payload()
    eda["validation_evidence"].update(
        {
            "target_available": True,
            "time_columns": [],
            "group_columns": [],
            "diagnostic_validations": [],
        }
    )
    eda["notebook_static_analysis"] = {"status": "completed"}
    return eda


REPAIR_NOTE = (
    "Final strategy payload was repaired deterministically because the LLM "
    "omitted required linkage fields."
)


def _repair_payload(*, action: str, evidence_refs: list[str]) -> dict[str, Any]:
    return {
        "competition_id": "generic-binary",
        "synthesis_status": "llm_success",
        "llm_output_valid": True,
        "repair_attempted": False,
        "repair_succeeded": False,
        "fallback_used": False,
        "synthesis_diagnostics_path": None,
        "actions": [
            {
                "priority": "P0",
                "action": action,
                "reason": "Grounded in EDA.",
                "evidence_refs": evidence_refs,
                "related_hypothesis_ids": [],
            }
        ],
    }


def _repair_hypotheses() -> list[dict[str, Any]]:
    return [
        {"hypothesis_id": "schema_001", "category": "schema"},
        {"hypothesis_id": "metric_001", "category": "metric"},
        {"hypothesis_id": "val_001", "category": "validation"},
        {"hypothesis_id": "leak_001", "category": "leakage"},
        {"hypothesis_id": "drift_001", "category": "drift"},
    ]


def _repair_eda_payload() -> dict[str, Any]:
    return {
        "inferred_schema": {"primary_id_column": "row_id"},
        "metric_evidence": {"requires_threshold": True},
        "validation_evidence": {
            "primary_validation": {"method": "stratified_kfold"}
        },
        "leakage_evidence": [
            {"status": "warning", "finding": "Target encoding can leak labels."}
        ],
        "feature_probe_evidence": [
            {
                "feature_family": "target_encoding_or_woe",
                "status": "unsafe",
                "leakage_risk": "high",
            }
        ],
        "drift_evidence": {"severity": "high"},
        "baseline_evidence": {"status": "completed"},
        "hypothesis_results": [
            {
                **item,
                "status": "confirmed",
                "confidence_after_eda": "high",
                "finding": f"EDA evaluated {item['category']} evidence.",
                "evidence_refs": [f"{item['category']}_evidence"],
                "impact_on_strategy": f"Use the supported {item['category']} policy.",
            }
            for item in _repair_hypotheses()
        ],
    }


def _rich_research_hypotheses() -> ResearchHypotheses:
    hypotheses = []
    for item in _repair_hypotheses():
        hypotheses.append(
            ResearchHypothesis(
                hypothesis_id=item["hypothesis_id"],
                category=item["category"],
                claim=f"Check {item['category']} evidence.",
                priority="P0",
                confidence_before_eda="medium",
            )
        )
    return ResearchHypotheses(
        competition_id="generic-binary",
        hypotheses=hypotheses,
    )


def _rich_eda_pack() -> EdaEvidencePack:
    payload = _repair_eda_payload()
    return EdaEvidencePack(
        competition_id="generic-binary",
        created_at="2026-07-08T12:00:00+03:00",
        run_id="generic-binary_20260708_120000",
        **payload,
    )


def _result_actions(result: FinalStrategyResult) -> list[Any]:
    return list(result.actions)


@pytest.mark.asyncio
async def test_post_resolution_schema_error_is_typed_sanitized_and_chained() -> None:
    response = _strategy_payload()
    response["actions"][0]["priority"] = "not-a-priority"
    response["actions"][0]["private_payload"] = "TOP_SECRET_RAW_RESPONSE"
    client = FakeFinalSynthesizerClient(response)
    pack = _eda_pack(primary_method="stratified_kfold")
    hypotheses = _research_hypotheses()
    catalog = build_final_strategy_reference_catalog(
        pack,
        research_hypotheses=hypotheses,
        source_claim_ids=["retrieved-1"],
    )
    issue = ReferenceIssue(
        field_path="actions[0].evidence_refs",
        expected_namespace="evidence",
        invalid_value="approved_experiments",
        actual_namespace="context_label",
        reason="context_label_not_reference",
    )

    with pytest.raises(FinalStrategySchemaValidationError) as caught:
        await final_synthesizer_module._repair_final_references_once(
            client=client,
            model="deepseek-v4-pro",
            result=FinalStrategyResult.model_validate(_strategy_payload()),
            issues=[issue],
            allowed_evidence_refs=["validation_evidence.primary_validation"],
            allowed_eda_result_refs=["validation_evidence.primary_validation"],
            approved_experiment_ids=[],
            allowed_risk_ids=[],
            allowed_validation_requirement_ids=[],
            allowed_safety_constraint_ids=[],
            reference_catalog=catalog,
        )

    error = caught.value
    assert isinstance(error, FinalStrategyCompilationError)
    assert error.phase == "post_resolution_schema_validation"
    assert error.diagnostics.initial_reference_issues == 1
    assert error.diagnostics.resolved_references == 1
    assert error.diagnostics.unresolved_references == 0
    assert error.diagnostics.kept_actions >= 1
    assert error.action_ids == ("action_validation",)
    assert isinstance(error.__cause__, final_synthesizer_module.ValidationError)
    assert "approved_experiments" not in str(error)
    assert "TOP_SECRET_RAW_RESPONSE" not in str(error)
    assert all(
        "TOP_SECRET_RAW_RESPONSE" not in item.message
        for item in error.diagnostics.schema_validation_errors
    )


@pytest.mark.asyncio
async def test_full_final_synthesis_pipeline_surfaces_post_resolution_schema_error() -> None:
    pack = _rich_eda_pack().model_copy(update={
        "baseline_ablation_evidence": {
            "feature_block_findings": [
                {"feature_block": "low_cardinality_categorical", "finding": "useful"},
            ]
        }
    })
    evidence_ref = (
        "baseline_ablation_evidence.feature_block_findings.low_cardinality_categorical"
    )
    initial = _strategy_payload()
    extra_action = deepcopy(initial["actions"][0])
    extra_action.update({
        "action_id": "action_schema_failure",
        "evidence_refs": [evidence_ref, "approved_experiments"],
        "eda_result_refs": [evidence_ref, "approved_experiments"],
        "experiment_ids": [],
    })
    initial["actions"].append(extra_action)
    repaired = deepcopy(initial)
    for action in repaired["actions"]:
        if action["action_id"] == "action_schema_failure":
            action["evidence_refs"] = [evidence_ref]
            action["eda_result_refs"] = [evidence_ref]
            action["priority"] = "invalid-priority"
    client = SequentialFinalSynthesizerClient([initial, repaired])

    with pytest.raises(FinalStrategySchemaValidationError) as caught:
        await synthesize_final_strategy(
            competition_desc="Generic iid binary classification with ROC AUC.",
            plan_data=_plan(),
            retrieved_documents=[_doc()],
            domain_patterns=[],
            research_hypotheses=_research_hypotheses(),
            eda_evidence_pack=pack,
            reasoning_outputs={},
            client=client,
            model="deepseek-v4-pro",
        )

    assert len(client.calls) == 2
    assert caught.value.phase == "post_resolution_schema_validation"
    assert isinstance(caught.value.__cause__, final_synthesizer_module.ValidationError)
    assert "approved_experiments" not in str(caught.value)


@pytest.mark.asyncio
async def test_cross_namespace_error_contains_only_post_resolution_issues() -> None:
    pack = _rich_eda_pack().model_copy(update={
        "baseline_ablation_evidence": {
            "feature_block_findings": [
                {"feature_block": "low_cardinality_categorical", "finding": "useful"},
            ]
        }
    })
    evidence_ref = (
        "baseline_ablation_evidence.feature_block_findings.low_cardinality_categorical"
    )
    initial = _strategy_payload()
    extra_action = deepcopy(initial["actions"][0])
    extra_action.update({
        "action_id": "action_unresolved",
        "evidence_refs": [evidence_ref, "approved_experiments"],
        "eda_result_refs": [evidence_ref, "approved_experiments"],
        "experiment_ids": [],
    })
    initial["actions"].append(extra_action)
    client = SequentialFinalSynthesizerClient([initial, deepcopy(initial)])

    with pytest.raises(FinalStrategySchemaValidationError) as caught:
        await synthesize_final_strategy(
            competition_desc="Generic iid binary classification with ROC AUC.",
            plan_data=_plan(),
            retrieved_documents=[_doc()],
            domain_patterns=[],
            research_hypotheses=_research_hypotheses(),
            eda_evidence_pack=pack,
            reasoning_outputs={},
            client=client,
            model="deepseek-v4-pro",
        )

    assert caught.value.phase == "post_resolution_schema_validation"
    assert caught.value.diagnostics.schema_validation_errors


@pytest.mark.asyncio
async def test_post_resolution_support_gate_error_does_not_become_stale_namespace_error() -> None:
    response = _strategy_payload()
    response["actions"][0]["evidence_refs"] = ["risk_leakage_001"]
    response["actions"][0]["priority"] = "P0"
    client = FakeFinalSynthesizerClient(response)
    pack = _eda_pack(primary_method="stratified_kfold")
    catalog = build_final_strategy_reference_catalog(
        pack,
        research_hypotheses=_research_hypotheses(),
    )
    issue = ReferenceIssue(
        field_path="actions[0].evidence_refs",
        expected_namespace="evidence",
        invalid_value="approved_experiments",
        actual_namespace="context_label",
        reason="context_label_not_reference",
    )

    with pytest.raises(UnsupportedFinalStrategyActionError) as caught:
        await final_synthesizer_module._repair_final_references_once(
            client=client,
            model="deepseek-v4-pro",
            result=FinalStrategyResult.model_validate(_strategy_payload()),
            issues=[issue],
            allowed_evidence_refs=["validation_evidence.primary_validation"],
            allowed_eda_result_refs=["validation_evidence.primary_validation"],
            approved_experiment_ids=[],
            allowed_risk_ids=[],
            allowed_validation_requirement_ids=[],
            allowed_safety_constraint_ids=[],
            reference_catalog=catalog,
        )

    assert caught.value.phase == "action_support_gate"
    assert caught.value.action_id == "action_validation"
    assert "approved_experiments" not in str(caught.value)


def _strategy_payload(
    *,
    recommended_validation: str = "stratified_kfold",
) -> dict[str, Any]:
    action = {
        "action_id": "action_validation",
        "priority": "P0",
        "action": "Use StratifiedKFold for model comparison.",
        "reason": "The retrieved source claim, Scout hypothesis, and EDA validation evidence agree.",
        "evidence_refs": ["validation_evidence.primary_validation"],
        "related_hypothesis_ids": ["val_001"],
        "source_claim": "Sources describe iid binary classification rather than a time split.",
        "source_refs": ["retrieved-1"],
        "eda_result_refs": ["validation_evidence.primary_validation"],
        "validation_strategy": recommended_validation,
        "confidence": "high",
    }
    sections = []
    for section_id in REQUIRED_SECTION_IDS:
        section = {
            "section_id": section_id,
            "title": section_id.replace("_", " ").title(),
            "summary": "Strategy guidance tied to EDA evidence.",
            "evidence_refs": ["validation_evidence.primary_validation"],
            "related_hypothesis_ids": ["val_001"],
        }
        if section_id == "metric_and_validation":
            section["actions"] = [action]
        sections.append(section)
    return {
        "competition_id": "generic-binary",
        "synthesis_status": "llm_success",
        "llm_output_valid": True,
        "repair_attempted": False,
        "repair_succeeded": False,
        "fallback_used": False,
        "synthesis_diagnostics_path": None,
        "task_type": "binary_classification",
        "metric": {"name": "roc_auc"},
        "recommended_validation": recommended_validation,
        "sections": sections,
        "actions": [action],
        "source_to_hypothesis_links": [
            {
                "source_ref": "retrieved-1",
                "source_claim": "Sources describe iid binary classification.",
                "hypothesis_id": "val_001",
            }
        ],
        "hypothesis_to_eda_links": [
            {
                "hypothesis_id": "val_001",
                "eda_result_ref": "validation_evidence.primary_validation",
            }
        ],
    }


def _plan() -> PlanData:
    return PlanData(
        task_type="binary_classification",
        metric="roc_auc",
        domain="generic_tabular",
    )


def _doc() -> RetrievedDocument:
    return RetrievedDocument(
        id="retrieved-1",
        competition_id="generic-binary",
        source="kaggle",
        title="Notebook",
        url="https://example.com/notebook",
        content="Uses iid binary classification with stratified folds.",
        score=0.9,
        rrf_score=0.2,
    )


def _load_synthesis_diagnostics(path: Path) -> FinalSynthesisDiagnostics:
    return FinalSynthesisDiagnostics.model_validate_json(
        (path / "final_synthesis_diagnostics.json").read_text(encoding="utf-8")
    )


def _research_hypotheses() -> ResearchHypotheses:
    return ResearchHypotheses(
        competition_id="generic-binary",
        hypotheses=[
            ResearchHypothesis(
                hypothesis_id="val_001",
                category="validation",
                claim="StratifiedKFold should be checked for iid binary classification.",
                rationale="Class balance matters for ROC AUC validation.",
                expected_eda_checks=["validation_analyzer.primary_validation"],
                priority="P0",
                confidence_before_eda="medium",
                source_refs=["retrieved-1"],
            )
        ],
    )


def _eda_pack(*, primary_method: str) -> EdaEvidencePack:
    return EdaEvidencePack(
        competition_id="generic-binary",
        created_at="2026-07-08T12:00:00+03:00",
        run_id="generic-binary_20260708_120000",
        validation_evidence={
            "primary_validation": {"method": primary_method},
            "diagnostic_validations": [
                {
                    "method": "temporal_holdout",
                    "reason": "Date column exists, but time ordering was diagnostic only.",
                }
            ],
        },
        hypothesis_results=[
            {
                "hypothesis_id": "val_001",
                "category": "validation",
                "status": "confirmed",
                "confidence_after_eda": "high",
                "finding": "EDA selected StratifiedKFold as primary validation.",
                "evidence_refs": ["validation_evidence.primary_validation"],
                "impact_on_strategy": "Use StratifiedKFold for model comparison.",
            }
        ],
    )
