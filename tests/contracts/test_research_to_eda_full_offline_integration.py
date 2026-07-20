from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_researcher.contracts.artifacts import (
    write_eda_task_plan_atomic,
    write_research_hypotheses_atomic,
)
from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import EdaEvidencePack, EdaRunConfig
from tests.contracts.factories import make_valid_eda_task_plan, make_valid_research_hypotheses


pytestmark = [pytest.mark.contract, pytest.mark.integration, pytest.mark.offline]


@pytest.mark.asyncio
async def test_full_reasoner_to_eda_mvp_bridge_runs_offline(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.csv").write_text(
        "row_id,target,feature_num,feature_cat\n"
        "1,0,0.1,a\n2,1,0.9,b\n3,0,0.2,a\n4,1,0.8,b\n"
        "5,0,0.3,a\n6,1,0.7,b\n7,0,0.4,a\n8,1,0.6,b\n"
        "9,0,0.2,a\n10,1,0.8,b\n11,0,0.1,a\n12,1,0.9,b\n",
        encoding="utf-8",
    )
    (dataset / "test.csv").write_text(
        "row_id,feature_num,feature_cat\n13,0.15,a\n14,0.85,b\n15,0.45,a\n",
        encoding="utf-8",
    )
    (dataset / "sample_submission.csv").write_text(
        "row_id,prediction\n13,0.2\n14,0.8\n15,0.5\n",
        encoding="utf-8",
    )

    hypotheses = make_valid_research_hypotheses(competition_id="offline-bridge")
    plan = make_valid_eda_task_plan(competition_id="offline-bridge")
    hypotheses_path = tmp_path / "research_hypotheses.json"
    plan_path = tmp_path / "eda_task_plan.json"
    write_research_hypotheses_atomic(hypotheses_path, hypotheses)
    write_eda_task_plan_atomic(plan_path, plan)

    result = await run_eda(EdaRunConfig(
        competition_id="offline-bridge",
        hypotheses_path=hypotheses_path,
        task_plan_path=plan_path,
        local_dataset_path=dataset,
        output_dir=tmp_path / "runs",
        download_dataset=False,
        enable_p1_modules=False,
        enable_baseline=False,
        enable_notebook_static_analysis=False,
    ))

    evidence = EdaEvidencePack.model_validate(
        json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))
    )
    assert result.evidence_pack_path.is_file()
    assert evidence.competition_id == "offline-bridge"
    assert len(evidence.hypothesis_results) == len(hypotheses.hypotheses)
    assert {item.hypothesis_id for item in evidence.hypothesis_results} == {
        item.hypothesis_id for item in hypotheses.hypotheses
    }
    assert evidence.recommended_next_actions or evidence.testable_hypotheses
    for item in evidence.hypothesis_results:
        if item.status in {"confirmed", "partially_confirmed", "rejected"}:
            assert item.evidence_refs
    primary = evidence.validation_evidence.get("primary_validation", {})
    assert primary.get("method") not in {
        "temporal", "time_series", "rolling", "expanding", "out_of_time"
    }
    assert evidence.notebook_static_analysis == {}

