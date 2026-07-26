from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher.contracts.pipeline import validate_full_run_artifacts
from kaggle_researcher.contracts.research_hypotheses import load_research_hypotheses
from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import EdaEvidencePack, EdaRunConfig
from kaggle_researcher.reasoning.experiment_planner import plan_experiments
from kaggle_researcher.reasoning.final_synthesizer import render_final_strategy
from tests.fixtures.final_synthesis import synthesize_for_test as synthesize_final_strategy
from kaggle_researcher.reasoning.leaderboard_auditor import audit_leaderboard_risk
from kaggle_researcher.reasoning.leakage_risk_analyst import analyze_leakage_risk
from kaggle_researcher.reasoning.metric_specialist import analyze_metric
from kaggle_researcher.reasoning.skeptical_reviewer import review
from kaggle_researcher.reasoning.validation_architect import design_validation
from kaggle_researcher.schemas import PlanData, RetrievedDocument


pytestmark = [pytest.mark.contract, pytest.mark.pipeline_smoke]


class JsonClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    async def chat_json(self, **_: Any) -> dict[str, Any]:
        return self.response


class FinalProtocolClient:
    def __init__(self, *, hypothesis_id: str, approved_experiment_id: str) -> None:
        self.hypothesis_id = hypothesis_id
        self.approved_experiment_id = approved_experiment_id

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        prompt = json.loads(kwargs["user_prompt"])
        if "selection_context" in prompt or "invalid_draft" in prompt:
            context = prompt.get("selection_context") or {}
            sections = (
                context.get("required_section_ids")
                or prompt.get("allowed_catalogs", {}).get("required_section_ids")
                or []
            )
            return {
                "contract_family": "strategy_selection_draft",
                "schema_version": "2.0",
                "selected_actions": [{
                    "client_action_key": "run_approved_baseline",
                    "action_kind": "baseline_reproduction",
                    "action": "Run the reviewer-approved baseline on fixed folds.",
                    "priority": "P0",
                    "confidence": "high",
                    "reason": "The approved experiment supplies the validation anchor.",
                    "primary_evidence_refs": ["validation_evidence.primary_validation"],
                    "supporting_evidence_refs": [],
                    "limitation_evidence_refs": [],
                    "source_refs": [],
                    "motivating_hypothesis_ids": [self.hypothesis_id],
                    "safety_hypothesis_ids": [],
                    "validation_context_ids": [],
                    "rejected_hypothesis_ids": [],
                    "safety_constraint_ids": prompt.get(
                        "allowed_safety_constraint_ids", []
                    ),
                    "validation_requirement_ids": prompt.get(
                        "allowed_validation_requirement_ids", []
                    ),
                    "approved_experiment_ids": [self.approved_experiment_id],
                    "feature_metadata": None,
                    "dependencies": [],
                    "limitations": [],
                }],
                "feature_experiment_families": [],
                "candidate_experiments": [],
                "proposed_core_experiment_ids": [],
                "proposed_backlog_experiment_ids": [],
                "section_plan": [
                    {
                        "section_id": section,
                        "selected_action_keys": ["run_approved_baseline"],
                        "selected_family_keys": [],
                        "selected_experiment_keys": [],
                        "summary_intent": "Execute the frozen approved plan.",
                    }
                    for section in sections
                ],
                "limitations": [],
            }
        immutable = prompt["immutable_strategy_payload"]
        return {
            "contract_family": "strategy_rendering_draft",
            "schema_version": "2.0",
            "skeleton_id": prompt.get("skeleton_id") or prompt["required_identity"]["skeleton_id"],
            "skeleton_hash": prompt.get("skeleton_hash") or prompt["required_identity"]["skeleton_hash"],
            "executive_summary": "Execute the frozen reviewer-approved validation plan.",
            "action_wording": [
                {
                    "action_id": item["action_id"],
                    "display_action": item["action"],
                    "display_reason": item["reason"],
                }
                for item in immutable["actions"]
            ],
            "experiment_wording": [],
            "family_wording": [],
            "section_summaries": [
                {"section_id": item["section_id"], "summary": item["summary"]}
                for item in immutable["section_structure"]
            ],
            "limitation_wording": [],
            "uncertainty_summary": "Claims remain bounded to the published evidence.",
        }


@pytest.mark.asyncio
async def test_real_internal_pipeline_boundaries_without_network_or_llm(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/eda/iid_binary_tiny")
    eda_result = await run_eda(EdaRunConfig(
        competition_id="iid_binary_tiny",
        hypotheses_path=fixture / "research_hypotheses.json",
        task_plan_path=fixture / "eda_task_plan.json",
        local_dataset_path=fixture,
        output_dir=tmp_path / "eda-runs",
        download_dataset=False,
        profile_sample_rows=1000,
    ))
    evidence_pack = EdaEvidencePack.model_validate_json(
        eda_result.evidence_pack_path.read_text(encoding="utf-8")
    )
    eda_hypothesis_id = evidence_pack.testable_hypotheses[0].hypothesis_id
    hypotheses, _ = load_research_hypotheses(fixture / "research_hypotheses.json")
    plan = PlanData(
        task_type="binary_classification",
        metric="roc_auc",
        domain="generic tabular classification",
    )
    document = RetrievedDocument(
        id="source-001",
        competition_id="iid_binary_tiny",
        source="kaggle",
        title="Validation note",
        content="Use stratified folds for imbalanced binary classification.",
        score=1.0,
        rrf_score=1.0,
    )
    docs = [document]

    validation = await design_validation(
        competition_desc="IID binary classification.",
        plan_data=plan,
        retrieved_documents=docs,
        client=JsonClient({
            "confidence": "medium",
            "evidence_ids": ["source-001"],
            "recommended_cv": "StratifiedKFold",
            "validation_risk": "medium",
            "likely_split": "iid",
            "failure_modes": None,
            "reasoning": "Class balance should be preserved across fixed folds.",
            "primary_validation": {"method": "stratified_kfold"},
            "secondary_validation": None,
            "do_not_use": [],
            "policy_notes": [],
        }),
    )
    metric = await analyze_metric(
        plan_data=plan,
        retrieved_documents=docs,
        client=JsonClient({
            "confidence": "medium",
            "evidence_ids": ["source-001"],
            "metric_explanation": "ROC AUC evaluates ranking quality.",
            "needs_calibration": False,
            "rank_averaging_useful": True,
            "threshold_search_needed": False,
            "surrogate_loss_suggestion": "Use probabilistic binary objectives.",
        }),
    )
    leakage = await analyze_leakage_risk(
        competition_desc="IID binary classification.",
        plan_data=plan,
        retrieved_documents=docs,
        client=JsonClient({
            "confidence": "low",
            "evidence_ids": [],
            "risk_level": "medium",
            "possible_issues": ["Possible identifier memorization risk."],
            "recommended_checks": ["Check identifiers inside each training fold."],
        }),
    )
    leaderboard = await audit_leaderboard_risk(
        competition_desc="IID binary classification.",
        plan_data=plan,
        validation_result=validation,
        retrieved_documents=docs,
        client=JsonClient({
            "confidence": "medium",
            "evidence_ids": ["source-001"],
            "shake_up_risk": "medium",
            "submission_selection_rule": "Select by repeated fixed-fold CV.",
            "public_lb_trust": "low",
            "warnings": ["Do not tune against the public leaderboard."],
        }),
    )
    experiments = await plan_experiments(
        validation_result=validation,
        leakage_result=leakage,
        metric_result=metric,
        retrieved_documents=docs,
        eda_hypotheses=evidence_pack.testable_hypotheses,
        client=JsonClient({
            "experiments": [{
                "priority": "P0",
                "experiment": "Train a baseline on fixed validation folds",
                "why": "The validation policy needs a reproducible performance anchor.",
                "cost": "low",
                "expected_gain": "diagnostic",
                "risk": "Changing folds would make comparisons unreliable.",
                "evidence_ids": ["validation_policy", "source-001"],
                "source_hypothesis_ids": [eda_hypothesis_id],
            }]
        }),
    )
    assert experiments[0].experiment_id
    reviewer = await review(
        draft_sections={"experiments": [item.model_dump(mode="json") for item in experiments]},
        retrieved_documents=docs,
        client=JsonClient({
            "confidence": "medium",
            "evidence_ids": ["source-001"],
            "unsupported_claims": [],
            "too_generic": [],
            "unnecessary_experiments": [],
            "approved_experiment_ids": [experiments[0].experiment_id],
            "rejected_experiment_ids": [],
            "reviewed_experiment_ids": [experiments[0].experiment_id],
            "revised_sections": {},
        }),
    )

    reasoning_outputs = {
        "metric": metric.model_dump(mode="json"),
        "validation": validation.model_dump(mode="json"),
        "leakage": leakage.model_dump(mode="json"),
        "leaderboard": leaderboard.model_dump(mode="json"),
        "experiments": [item.model_dump(mode="json") for item in experiments],
        "review": reviewer.model_dump(mode="json"),
    }
    strategy = await synthesize_final_strategy(
        competition_desc="IID binary classification.",
        plan_data=plan,
        retrieved_documents=docs,
        domain_patterns=[],
        research_hypotheses=hypotheses,
        eda_evidence_pack=evidence_pack,
        reasoning_outputs=reasoning_outputs,
        eda_summary_text=eda_result.summary_path.read_text(encoding="utf-8"),
        client=FinalProtocolClient(
            hypothesis_id=eda_hypothesis_id,
            approved_experiment_id=experiments[0].experiment_id,
        ),
        model="mock-model",
    )
    rendered = render_final_strategy(strategy)
    assert "secondary_validation" not in rendered
    assert "Metric" in rendered and "Validation" in rendered
    linked_actions = [
        action for action in strategy.actions
        if experiments[0].experiment_id in action.experiment_ids
    ]
    assert linked_actions
    assert eda_hypothesis_id in linked_actions[0].hypothesis_ids
    assert strategy.synthesis_status == "llm_success"
    assert strategy.rendering_status == "llm_success"
    assert strategy.reference_repairs == []

    run_dir = tmp_path / "full-run"
    for directory in (run_dir / "research", run_dir / "eda", run_dir / "reasoning", run_dir / "final"):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixture / "research_hypotheses.json", run_dir / "research" / "research_hypotheses.json")
    shutil.copy2(fixture / "eda_task_plan.json", run_dir / "research" / "eda_task_plan.json")
    (run_dir / "research" / "retrieved_documents.json").write_text(
        json.dumps([document.model_dump(mode="json")], indent=2), encoding="utf-8"
    )
    shutil.copy2(eda_result.evidence_pack_path, run_dir / "eda" / "eda_evidence_pack.json")
    (run_dir / "reasoning" / "validation_result.json").write_text(validation.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "reasoning" / "experiment_plan.json").write_text(json.dumps([item.model_dump(mode="json") for item in experiments], indent=2), encoding="utf-8")
    (run_dir / "reasoning" / "skeptical_review.json").write_text(reviewer.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "final" / "final_strategy.json").write_text(strategy.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "final" / "final_report.md").write_text(rendered, encoding="utf-8")

    validate_full_run_artifacts(run_dir)
