from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel

from kaggle_researcher.reasoning.common import call_reasoning_json
from kaggle_researcher.reasoning.report_composer import compose_report, format_section_for_prompt
from kaggle_researcher.schemas import (
    ExperimentItem,
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    PlanData,
    ReviewResult,
    ValidationResult,
)


def test_review_result_accepts_string_revised_sections() -> None:
    result = ReviewResult.model_validate({"revised_sections": {"validation": "text"}})

    assert result.revised_sections["validation"] == "text"


def test_review_result_accepts_dict_revised_sections_and_preserves_provenance() -> None:
    result = ReviewResult.model_validate(
        {
            "revised_sections": {
                "validation": {
                    "recommended_cv": "out-of-time holdout",
                    "confidence": "high",
                    "provenance": ["kaggle", "heuristic"],
                }
            }
        }
    )

    assert result.revised_sections["validation"]["recommended_cv"] == "out-of-time holdout"
    assert result.revised_sections["validation"]["provenance"] == ["kaggle", "heuristic"]


def test_review_result_accepts_list_revised_sections() -> None:
    result = ReviewResult.model_validate(
        {
            "revised_sections": {
                "experiments": [
                    {
                        "priority": "P0",
                        "experiment": "Build temporal validation",
                        "provenance": ["heuristic"],
                    }
                ]
            }
        }
    )

    assert result.revised_sections["experiments"][0]["priority"] == "P0"
    assert result.revised_sections["experiments"][0]["provenance"] == ["heuristic"]


def test_format_section_for_prompt_serializes_structured_sections() -> None:
    section = {
        "recommended_cv": "out-of-time holdout",
        "provenance": ["kaggle", "heuristic"],
    }

    formatted = format_section_for_prompt(section)

    assert json.loads(formatted) == section
    assert format_section_for_prompt("already text") == "already text"


def test_report_composer_serializes_structured_revised_sections_without_crashing() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.user_prompt = ""

        async def chat_text(self, **kwargs):
            self.user_prompt = kwargs["user_prompt"]
            return "ok"

    client = FakeClient()
    review = ReviewResult(
        revised_sections={
            "validation": {
                "recommended_cv": "out-of-time holdout",
                "confidence": "high",
                "provenance": ["kaggle", "heuristic"],
            },
            "experiments": [
                {
                    "priority": "P0",
                    "experiment": "Build temporal validation",
                    "provenance": ["heuristic"],
                }
            ],
        }
    )

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
            review=review,
            client=client,
            model="model",
        )
    )

    payload = json.loads(client.user_prompt)
    assert payload["review_result"]["revised_sections"]["validation"]["provenance"] == [
        "kaggle",
        "heuristic",
    ]
    assert (
        json.loads(payload["review_revised_sections_for_prompt"]["experiments"])[0]["experiment"]
        == "Build temporal validation"
    )


def test_call_reasoning_json_reports_schema_and_saves_raw_response(tmp_path) -> None:
    class ResultModel(BaseModel):
        value: str

    class FakeClient:
        async def chat_json(self, **kwargs):
            return {"value": {"not": "a string"}, "extra": True}

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            call_reasoning_json(
                client=FakeClient(),
                model="reasoning-model",
                system_prompt="system",
                user_payload={},
                result_model=ResultModel,
                artifact_dir=tmp_path,
                raw_artifact_name="raw.json",
            )
        )

    message = str(exc_info.value)
    assert "reasoning-model" in message
    assert "ResultModel" in message
    assert "Returned keys: ['value', 'extra']" in message
    assert json.loads((tmp_path / "raw.json").read_text(encoding="utf-8")) == {
        "value": {"not": "a string"},
        "extra": True,
    }
