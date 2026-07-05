from __future__ import annotations

from kaggle_researcher.main import validate_full_roadmap
from kaggle_researcher.reasoning.provenance import attach_default_provenance, normalize_provenance
from kaggle_researcher.reasoning.report_composer import SECTION_HEADINGS, compose_report
from kaggle_researcher.schemas import (
    ExperimentItem,
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    PlanData,
    ReviewResult,
    ValidationResult,
)
from kaggle_researcher.schemas import RetrievedDocument


def retrieved(source: str) -> RetrievedDocument:
    return RetrievedDocument(
        id=f"{source}-1",
        competition_id="comp",
        source=source,
        title="source",
        url="https://example.com",
        content="content",
        score=1.0,
        rrf_score=0.1,
    )


def test_normalize_provenance_filters_invalid_values() -> None:
    assert normalize_provenance(["kaggle", "bad", "ARXIV", "kaggle"]) == ["kaggle", "arxiv"]
    assert normalize_provenance(None) == []


def test_attach_default_provenance_adds_not_verified_to_leakage_claims() -> None:
    result = attach_default_provenance(
        "leakage",
        {"claim": "Possible target leakage in time features"},
        [retrieved("kaggle")],
    )

    assert "not_verified_on_data" in result["provenance"]
    assert "kaggle" in result["provenance"]


def test_metric_claims_get_arxiv_when_arxiv_docs_exist() -> None:
    result = attach_default_provenance(
        "metric",
        {"claim": "Gini is related to AUC and ranking."},
        [retrieved("arxiv")],
    )

    assert "arxiv" in result["provenance"]


def test_competition_specific_claims_get_kaggle_when_kaggle_docs_exist() -> None:
    result = attach_default_provenance(
        "validation",
        {"claim": "Kaggle notebooks mention public LB instability."},
        [retrieved("kaggle")],
    )

    assert "kaggle" in result["provenance"]


def test_full_roadmap_validation_accepts_provenance_markers() -> None:
    roadmap = "\n\n".join(
        f"## {heading}\nKey claim. _Provenance: Kaggle + heuristic; not verified on data._ "
        + ("More detail. " * 25)
        for heading in SECTION_HEADINGS
    )

    validate_full_roadmap(roadmap)


def test_report_composer_prompt_includes_provenance_and_temporal_policy() -> None:
    import asyncio

    class FakeClient:
        def __init__(self) -> None:
            self.system_prompt = ""

        async def chat_text(self, **kwargs):
            self.system_prompt = kwargs["system_prompt"]
            return "ok"

    client = FakeClient()
    asyncio.run(
        compose_report(
            competition_desc="desc",
            plan_data=PlanData(task_type="classification", metric="gini stability", domain="credit"),
            domain_patterns=[],
            validation_result=ValidationResult(
                confidence="medium",
                evidence_ids=[],
                recommended_cv="temporal",
                validation_risk="high",
                likely_split="time",
                reasoning="reason",
            ),
            leakage_result=LeakageRiskResult(confidence="low", evidence_ids=[], risk_level="medium"),
            metric_result=MetricResult(
                confidence="medium",
                evidence_ids=[],
                metric_explanation="gini",
                needs_calibration=False,
                rank_averaging_useful=True,
                threshold_search_needed=False,
                surrogate_loss_suggestion="auc",
            ),
            experiments=[
                ExperimentItem(
                    priority="P0",
                    experiment="baseline",
                    why="why",
                    cost="low",
                    expected_gain="medium",
                    risk="low",
                )
            ],
            lb_audit=LeaderboardAuditResult(
                confidence="medium",
                evidence_ids=[],
                shake_up_risk="high",
                submission_selection_rule="cv",
                public_lb_trust="low",
            ),
            review=ReviewResult(confidence="medium", evidence_ids=[]),
            client=client,
            model="model",
        )
    )

    assert "_Provenance:" in client.system_prompt
    assert "out-of-time holdout" in client.system_prompt
