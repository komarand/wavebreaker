from __future__ import annotations

import asyncio
import json

import pytest

from kaggle_researcher.reasoning.report_composer import (
    SECTION_HEADINGS,
    compose_report,
    extract_report_sections,
    repair_composed_report,
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


class SequentialFakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def chat_text(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _validation() -> ValidationResult:
    return ValidationResult(
        confidence="medium",
        evidence_ids=[],
        recommended_cv="Out-of-time holdout",
        validation_risk="high",
        likely_split="time",
        reasoning="Use temporal validation.",
        primary_validation={"method": "temporal_cv"},
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


def _repair_inputs() -> dict[str, object]:
    return {
        "competition_desc": "Credit competition with tabular applications.",
        "plan_data": PlanData(task_type="classification", metric="gini", domain="credit").model_dump(),
        "domain_patterns": [{"competition_family": "credit_risk_tabular", "typical_validation": "time split"}],
        "validation_result": _validation().model_dump(),
        "leakage_result": _leakage().model_dump(),
        "metric_result": _metric().model_dump(),
        "experiments": [_experiment().model_dump()],
        "lb_audit": _lb_audit().model_dump(),
        "review": ReviewResult(
            confidence="medium",
            evidence_ids=[],
            unnecessary_experiments=["Large ensemble before baseline."],
            unsupported_claims=["Claiming train/test data was inspected."],
        ).model_dump(),
    }


def test_validate_composed_report_accepts_exact_15_sections() -> None:
    validate_composed_report(_full_report())


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


def test_compose_report_canonicalizes_common_heading_variants() -> None:
    variant_headings = SECTION_HEADINGS.copy()
    variant_headings[2] = "Анатомия датасета"
    variant_headings[11] = "Очередь экспериментов"
    variant_headings[13] = "Что не делать"
    report = "\n\n".join(
        f"## {heading}\nConfidence: medium. _Provenance: Kaggle + heuristic; not verified on data._"
        for heading in variant_headings
    )
    client = FakeClient(report)

    result = _run_compose(client)

    assert f"## {SECTION_HEADINGS[2]}" in result
    assert f"## {SECTION_HEADINGS[11]}" in result
    assert f"## {SECTION_HEADINGS[13]}" in result
    validate_composed_report(result)


def test_compose_report_repairs_missing_headings_once() -> None:
    client = SequentialFakeClient(
        [
            "## Executive summary\nToo short.",
            _full_report(),
        ]
    )

    result = _run_compose(client)

    assert len(client.calls) == 2
    assert "Repair the markdown roadmap structure only" in str(client.calls[1]["system_prompt"])
    validate_composed_report(result)


def test_validate_composed_report_rejects_missing_sections() -> None:
    with pytest.raises(RuntimeError, match="15 required section headings"):
        validate_composed_report("## Executive summary\nConfidence: medium.")


def test_repair_composed_report_adds_missing_sections() -> None:
    malformed = "## Executive summary\nLLM summary.\n\n## РџР»Р°РЅ baseline\nStart simple."

    repaired = repair_composed_report(malformed, **_repair_inputs())

    validate_composed_report(repaired)
    assert [heading for heading in SECTION_HEADINGS if f"## {heading}" in repaired] == SECTION_HEADINGS
    assert repaired.count("This section was reconstructed because") >= 13
    assert "Dataset anatomy is based on source descriptions only" in repaired
    assert "Public notebooks were analyzed only as text/source material; notebooks were not executed." in repaired


def test_repair_composed_report_preserves_recovered_content() -> None:
    malformed = (
        "## Executive summary\nKeep this executive content.\n\n"
        "## РћС‡РµСЂРµРґСЊ СЌРєСЃРїРµСЂРёРјРµРЅС‚РѕРІ\nRecovered experiment queue."
    )

    repaired = repair_composed_report(malformed, **_repair_inputs())
    sections = extract_report_sections(repaired)

    validate_composed_report(repaired)
    assert "Keep this executive content." in sections[SECTION_HEADINGS[0]]
    assert "Recovered experiment queue." in sections[SECTION_HEADINGS[11]]
    assert f"## {SECTION_HEADINGS[11]}" in repaired


def test_validate_composed_report_rejects_forbidden_data_execution_claims() -> None:
    report = _full_report("EDA showed strong leakage.")

    with pytest.raises(RuntimeError, match="forbidden data-execution"):
        validate_composed_report(report)


def test_compose_report_rejects_forbidden_llm_output() -> None:
    client = FakeClient(_full_report("We analyzed train/test data."))

    with pytest.raises(RuntimeError, match="forbidden data-execution"):
        _run_compose(client)


def test_compose_report_repairs_after_invalid_llm_retry() -> None:
    first = "\n\n".join(
        [
            "## Executive summary\nInitial short answer.",
            "## РџР»Р°РЅ baseline\nBaseline text.",
        ]
    )
    retry = "\n\n".join(
        f"## {heading}\nRecovered body for {index}."
        for index, heading in enumerate(SECTION_HEADINGS[:12], start=1)
    )
    client = SequentialFakeClient([first, retry])

    result = _run_compose(client)

    assert len(client.calls) == 2
    validate_composed_report(result)
    assert "Recovered body for 12." in result
    assert "This section was reconstructed because the LLM response did not provide a valid canonical section." in result
