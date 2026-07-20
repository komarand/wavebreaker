from __future__ import annotations

import json
from pathlib import Path

import pytest

import kaggle_researcher.eda.orchestrator as orchestrator
from kaggle_researcher.contracts.research_to_eda import ResearchToEdaContractError
from kaggle_researcher.eda.schemas import EdaRunConfig
from tests.contracts.factories import valid_research_payload, valid_task_plan_payload


pytestmark = [pytest.mark.contract, pytest.mark.offline]


def _write(tmp_path: Path, research: dict, plan: dict) -> EdaRunConfig:
    hypotheses_path = tmp_path / "research_hypotheses.json"
    task_plan_path = tmp_path / "eda_task_plan.json"
    hypotheses_path.write_text(json.dumps(research), encoding="utf-8")
    task_plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return EdaRunConfig(
        competition_id=research.get("competition_id", "fixture-competition"),
        hypotheses_path=hypotheses_path,
        task_plan_path=task_plan_path,
        local_dataset_path=tmp_path / "must-not-be-read",
        output_dir=tmp_path / "runs",
        download_dataset=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("competition", "competition_id_mismatch"),
        ("dangling", "unknown_hypothesis_reference"),
        ("unknown_module", "unknown_eda_module"),
        ("invalid_check", "unknown_eda_check"),
        ("metric", "metric_task_type_mismatch"),
    ],
)
async def test_invalid_contract_stops_before_dataset_resolution_and_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    research, plan = valid_research_payload(), valid_task_plan_payload()
    if mutation == "competition":
        plan["competition_id"] = "other"
    elif mutation == "dangling":
        plan["eda_tasks"][0]["related_hypothesis_ids"] = ["schema_missing"]
    elif mutation == "unknown_module":
        plan["eda_tasks"][0]["module"] = "future_module"
    elif mutation == "invalid_check":
        research["hypotheses"][0]["expected_eda_checks"] = ["schema_inferer.future"]
    elif mutation == "metric":
        plan["task_type"], plan["metric"] = "regression", {"name": "f1"}
    config = _write(tmp_path, research, plan)
    dataset_touched = False

    def fail_dataset(*_args: object, **_kwargs: object) -> None:
        nonlocal dataset_touched
        dataset_touched = True
        raise AssertionError("dataset resolution must not run")

    monkeypatch.setattr(orchestrator, "resolve_dataset", fail_dataset)
    with pytest.raises(ValueError) as caught:
        await orchestrator.run_eda(config)
    result = getattr(caught.value, "result", None)
    if isinstance(caught.value, ResearchToEdaContractError):
        assert expected_code in {issue.code for issue in caught.value.result.errors}
    else:
        assert expected_code == "competition_id_mismatch"
    assert result is None or not result.valid
    assert not dataset_touched
    assert not (tmp_path / "runs").exists()


@pytest.mark.asyncio
async def test_malformed_json_stops_before_dataset_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write(tmp_path, valid_research_payload(), valid_task_plan_payload())
    config.hypotheses_path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(
        orchestrator,
        "resolve_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dataset touched")),
    )
    with pytest.raises(ValueError, match="Could not read contract artifact"):
        await orchestrator.run_eda(config)
    assert not (tmp_path / "runs").exists()


@pytest.mark.asyncio
async def test_valid_contract_reaches_dataset_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write(tmp_path, valid_research_payload(), valid_task_plan_payload())

    class ReachedDataset(RuntimeError):
        pass

    def reached(*_args: object, **_kwargs: object) -> None:
        raise ReachedDataset

    monkeypatch.setattr(orchestrator, "resolve_dataset", reached)
    with pytest.raises(ReachedDataset):
        await orchestrator.run_eda(config)
