from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_researcher.contracts.artifacts import (
    load_eda_task_plan,
    load_research_hypotheses,
    write_eda_task_plan_atomic,
    write_research_hypotheses_atomic,
)
from kaggle_researcher.contracts.errors import ArtifactContractError, ContractError
from kaggle_researcher.contracts.research_to_eda import validate_research_to_eda_contract
from tests.contracts.factories import (
    make_valid_eda_task_plan,
    make_valid_research_hypotheses,
)


pytestmark = [pytest.mark.contract, pytest.mark.offline, pytest.mark.unit]


def test_utf8_json_roundtrip_preserves_order_optional_fields_and_versions(tmp_path: Path) -> None:
    hypotheses = make_valid_research_hypotheses()
    hypotheses.hypotheses[0].claim = "Проверить целевой столбец — без потери Unicode."
    hypotheses.hypotheses[0].rationale = None
    plan = make_valid_eda_task_plan()
    research_path = tmp_path / "research_hypotheses.json"
    plan_path = tmp_path / "eda_task_plan.json"

    write_research_hypotheses_atomic(research_path, hypotheses)
    write_eda_task_plan_atomic(plan_path, plan)
    loaded_hypotheses, _ = load_research_hypotheses(research_path)
    loaded_plan, _ = load_eda_task_plan(plan_path, hypotheses=loaded_hypotheses)

    assert loaded_hypotheses == hypotheses
    assert loaded_plan == plan
    assert loaded_hypotheses.hypotheses[0].claim.startswith("Проверить")
    assert [item.hypothesis_id for item in loaded_hypotheses.hypotheses] == [
        item.hypothesis_id for item in hypotheses.hypotheses
    ]
    assert [item.task_id for item in loaded_plan.eda_tasks] == [item.task_id for item in plan.eda_tasks]
    assert loaded_hypotheses.schema_version == loaded_plan.schema_version == "1.0"
    assert loaded_hypotheses.hypotheses[0].rationale is None
    assert validate_research_to_eda_contract(loaded_hypotheses, loaded_plan).valid

    raw = research_path.read_text(encoding="utf-8")
    assert "ResearchHypotheses(" not in raw
    assert "null" in raw
    assert json.loads(raw)["structured_findings"] == []


def test_atomic_serialization_is_deterministic_for_same_model(tmp_path: Path) -> None:
    model = make_valid_research_hypotheses()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_research_hypotheses_atomic(first, model)
    write_research_hypotheses_atomic(second, model)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("content", ["{not-json", "[]", '"string"'])
def test_malformed_or_non_object_json_is_a_typed_bounded_error(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "research_hypotheses.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ContractError) as caught:
        load_research_hypotheses(path)
    assert path.name in str(caught.value)
    assert content not in str(caught.value)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "research_hypotheses.json"
    path.write_text('{"schema_version":"1.0","schema_version":"1.0"}', encoding="utf-8")
    with pytest.raises(ArtifactContractError, match="duplicate JSON keys"):
        load_research_hypotheses(path)


def test_large_invalid_value_is_not_echoed_in_error(tmp_path: Path) -> None:
    payload = make_valid_research_hypotheses().model_dump(mode="json")
    secret = "x" * 10_000
    payload["hypotheses"][0]["priority"] = secret
    path = tmp_path / "research_hypotheses.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError) as caught:
        load_research_hypotheses(path)
    assert secret not in str(caught.value)
    assert len(str(caught.value)) < 1_000


def test_schema_snapshots_capture_public_shape_without_titles() -> None:
    from kaggle_researcher.contracts.eda import EdaTask, EdaTaskPlan
    from kaggle_researcher.contracts.research import ResearchHypothesis, ResearchHypotheses
    from kaggle_researcher.contracts.research_to_eda import ResearchToEdaContractValidationResult

    expected_fields = {
        ResearchHypothesis: {
            "hypothesis_id", "category", "claim", "rationale", "expected_eda_checks",
            "priority", "confidence_before_eda", "source_refs", "status", "limitations",
        },
        ResearchHypotheses: {
            "contract_family", "schema_version", "competition_id", "created_at", "hypotheses",
            "eda_tasks", "structured_findings", "scout_limitations", "models_used",
        },
        EdaTask: {
            "task_id", "module", "priority", "blocking", "related_hypothesis_ids",
            "dependencies", "expected_outputs", "params",
        },
        EdaTaskPlan: {
            "contract_family", "schema_version", "competition_id", "task_type", "metric",
            "dataset", "eda_tasks", "hypothesis_index", "recommended_module_sequence",
            "recommended_human_checklist", "blocking_tasks",
        },
        ResearchToEdaContractValidationResult: {"valid", "errors", "warnings"},
    }
    for model, fields in expected_fields.items():
        schema = model.model_json_schema()
        assert set(schema["properties"]) == fields
        assert schema["additionalProperties"] is False
