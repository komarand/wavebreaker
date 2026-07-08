from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import EdaRunConfig


FIXTURE_ROOT = Path("tests/fixtures/eda")


def test_offline_home_credit_fixture_creates_evidence_pack(tmp_path: Path) -> None:
    result = asyncio.run(_run_fixture("home_credit_tiny", tmp_path / "runs"))

    assert result.evidence_pack_path.is_file()
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))
    assert payload["competition_id"] == "home_credit_tiny"
    assert payload["validation_evidence"]["primary_validation"]["method"] == "temporal_holdout"
    assert payload["recommended_next_actions"]


def test_iid_binary_tiny_fixture_selects_stratified_kfold(tmp_path: Path) -> None:
    result = asyncio.run(_run_fixture("iid_binary_tiny", tmp_path / "runs"))
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert payload["validation_evidence"]["primary_validation"]["method"] == "stratified_kfold"


def test_regression_tiny_fixture_selects_kfold(tmp_path: Path) -> None:
    result = asyncio.run(_run_fixture("regression_tiny", tmp_path / "runs"))
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert payload["validation_evidence"]["primary_validation"]["method"] == "kfold"


async def _run_fixture(competition_id: str, output_dir: Path):
    fixture_dir = FIXTURE_ROOT / competition_id
    return await run_eda(
        EdaRunConfig(
            competition_id=competition_id,
            hypotheses_path=fixture_dir / "research_hypotheses.json",
            task_plan_path=fixture_dir / "eda_task_plan.json",
            local_dataset_path=fixture_dir,
            output_dir=output_dir,
            download_dataset=False,
            profile_sample_rows=1000,
        )
    )
