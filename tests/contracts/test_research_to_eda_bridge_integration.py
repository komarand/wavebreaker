from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_researcher.contracts.artifacts import (
    load_eda_task_plan,
    load_research_hypotheses,
    write_eda_task_plan_atomic,
    write_research_hypotheses_atomic,
)
from kaggle_researcher.contracts.research_to_eda import validate_research_to_eda_contract
from tests.contracts.factories import make_valid_eda_task_plan, make_valid_research_hypotheses


pytestmark = [pytest.mark.contract, pytest.mark.integration, pytest.mark.offline]


def test_reasoner_files_load_and_validate_without_starting_eda_modules(tmp_path: Path) -> None:
    hypotheses = make_valid_research_hypotheses()
    plan = make_valid_eda_task_plan()
    hypotheses_path = tmp_path / "research_hypotheses.json"
    plan_path = tmp_path / "eda_task_plan.json"
    write_research_hypotheses_atomic(hypotheses_path, hypotheses)
    write_eda_task_plan_atomic(plan_path, plan)

    loaded_hypotheses, _ = load_research_hypotheses(hypotheses_path)
    loaded_plan, _ = load_eda_task_plan(plan_path, hypotheses=loaded_hypotheses)
    validation = validate_research_to_eda_contract(loaded_hypotheses, loaded_plan)

    assert hypotheses_path.is_file() and plan_path.is_file()
    assert loaded_hypotheses == hypotheses
    assert loaded_plan == plan
    assert validation.valid
    assert loaded_hypotheses.competition_id == loaded_plan.competition_id
    assert not list(tmp_path.glob("**/eda_evidence_pack.json"))

