from __future__ import annotations

import asyncio
import json

import pytest

from kaggle_researcher.reasoning.report_composer import (
    SECTION_HEADINGS,
    compose_report,
    validate_composed_report,
)
from kaggle_researcher.schemas import (
    ExperimentItem,
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    PlanData,
    ReviewResult,
    ValidationResult,
)


def _full_report(extra_body: str = "") -> str:
    return "\n\n".join(
        f"## {heading}\nConfidence: medium. _Provenance: Kaggle + heuristic; not verified on data._ "
        f"{extra_body} Actionable roadmap text."
        for heading in SECTION_HEADINGS
    )


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.kwargs: dict[str, object] = {}

    async def chat_text(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _validation() -> ValidationResult:
    return ValidationResult(
        confidence="medium",
        evidence_ids=[],
        recommended_cv="Out-of-time holdout",
        validation_risk="high",
        likely_split="time",
        reasoning="Use temporal validation.",
    )


def _leakage() -> LeakageRiskResult:
    return LeakageRiskResult(
        confidence="low",
        evidence_ids=[],
        risk_level="medium",
        possible_issues=["Possible timestamp risk."],
        recommended_checks=["Check timestamp availability."],
    )


def _metric() -> MetricResult:
    return MetricResult(
        confidence="medium",
        evidence_ids=[],
        metric_explanation="Gini rewards ranking.",
        needs_calibration=False,
        rank_averaging_useful=True,
        threshold_search_needed=False,
        surrogate_loss_suggestion="AUC-like validation.",
    )


def _experiment() -> ExperimentItem:
    return ExperimentItem(
        priority="P0",
        experiment="Train baseline",
        why="Anchor the roadmap.",
        cost="low",
        expected_gain="medium",
        risk="Low.",
    )


def _lb_audit() -> LeaderboardAuditResult:
    return LeaderboardAuditResult(
        confidence="medium",
        evidence_ids=[],
        shake_up_risk="high",
        submission_selection_rule="Select by CV.",
        public_lb_trust="low",
        warnings=["Avoid public LB overfitting."],
    )


def _run_compose(client: FakeClient) -> str:
    return asyncio.run(
        compose_report(
            competition_desc="Credit competition.",
            plan_data=PlanData(task_type="classification", metric="gini", domain="credit"),
            domain_patterns=[{"competition_family": "credit_risk_tabular"}],
            validation_result=_validation(),
            leakage_result=_leakage(),
            metric_result=_metric(),
            experiments=[_experiment()],
            lb_audit=_lb_audit(),
            review=ReviewResult(
                confidence="medium",
                evidence_ids=[],
                unnecessary_experiments=["Large ensemble before baseline."],
                revised_sections={"validation": "Keep temporal validation."},
            ),
            client=client,
            model="reasoning-model",
        )
    )


def test_compose_report_returns_report_with_all_15_required_headings() -> None:
    client = FakeClient(_full_report())

    report = _run_compose(client)

    assert [heading for heading in SECTION_HEADINGS if f"## {heading}" in report] == SECTION_HEADINGS
    assert "## Чего не делать" in report

    system_prompt = str(client.kwargs["system_prompt"])
    assert "Use exactly the 15 required section headings in order" in system_prompt
    assert "Do not claim real EDA, train/test analysis" in system_prompt
    assert "Do not include chain-of-thought" in system_prompt

    payload = json.loads(str(client.kwargs["user_prompt"]))
    assert payload["required_sections"] == SECTION_HEADINGS
    assert payload["plan_data"]["metric"] == "gini"
    assert payload["review_revised_sections_for_prompt"]["validation"] == "Keep temporal validation."


def test_validate_composed_report_rejects_missing_sections() -> None:
    with pytest.raises(RuntimeError, match="15 required section headings"):
        validate_composed_report("## Executive summary\nConfidence: medium.")


def test_validate_composed_report_rejects_forbidden_data_execution_claims() -> None:
    report = _full_report("EDA showed strong leakage.")

    with pytest.raises(RuntimeError, match="forbidden data-execution"):
        validate_composed_report(report)


def test_compose_report_rejects_forbidden_llm_output() -> None:
    client = FakeClient(_full_report("We analyzed train/test data."))

    with pytest.raises(RuntimeError, match="forbidden data-execution"):
        _run_compose(client)
