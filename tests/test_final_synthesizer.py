from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

import kaggle_researcher.reasoning.final_synthesizer as final_synthesizer_module
from kaggle_researcher.eda.schemas import EdaEvidencePack, ResearchHypotheses, ResearchHypothesis
from kaggle_researcher.reasoning.final_synthesizer import (
    REQUIRED_SECTION_IDS,
    FinalStrategyResult,
    build_fallback_final_strategy,
    postprocess_final_strategy_result,
    repair_final_strategy_payload,
    render_final_strategy,
    render_final_strategy_summary,
    synthesize_final_strategy,
    validate_rendered_strategy_quality,
)
from kaggle_researcher.schemas import PlanData, RetrievedDocument


class FakeFinalSynthesizerClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.response


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
    assert "Link every important recommendation to EDA evidence_refs" in combined
    assert '"allowed_hypothesis_ids"' in prompt
    assert "Do not invent IDs" in prompt


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

    assert result.sections[0].actions or result.sections[0].evidence_refs
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
    assert "stratifiedkfold" in action_text
    assert "target encoding" in action_text
    assert all(action.related_hypothesis_ids for action in _result_actions(result))


def test_fallback_strategy_validates() -> None:
    fallback = build_fallback_final_strategy(
        competition_id="generic-binary",
        research_hypotheses=_repair_hypotheses(),
        eda_evidence_pack=_repair_eda_payload(),
    )

    result = FinalStrategyResult.model_validate(fallback)

    assert len(result.sections) >= 8
    assert all(action.related_hypothesis_ids for action in _result_actions(result))
    assert all(action.evidence_refs for action in _result_actions(result))


@pytest.mark.asyncio
async def test_invalid_repair_uses_fallback_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeFinalSynthesizerClient(
        {
            "competition_id": "generic-binary",
            "sections": [{"section_id": "broken", "title": "Broken", "summary": "Empty"}],
        }
    )
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
    )

    assert any("built from available Scout hypotheses" in item for item in result.limitations)
    assert all(action.related_hypothesis_ids for action in _result_actions(result))


@pytest.mark.asyncio
async def test_final_synthesis_uses_allowed_hypothesis_ids_only() -> None:
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
    )

    allowed = {item["hypothesis_id"] for item in _repair_hypotheses()}
    assert {
        hypothesis_id
        for action in _result_actions(result)
        for hypothesis_id in action.related_hypothesis_ids
    } <= allowed


def test_postprocessor_deduplicates_validation_actions() -> None:
    result = _quality_result(
        {
            "metric_and_validation": [
                _quality_action(
                    "Use stratified k-fold cross-validation for model comparison.",
                    ["validation_evidence.primary_validation"],
                    priority="P1",
                )
            ],
            "modeling_plan": [
                _quality_action(
                    "Use StratifiedKFold CV for model comparison.",
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
        for section in cleaned.sections
        for action in section.actions
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

    assert len(summary_text) < 0.4 * len(full_text)
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
    actions = list(result.actions)
    for section in result.sections:
        actions.extend(section.actions)
    return actions


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
