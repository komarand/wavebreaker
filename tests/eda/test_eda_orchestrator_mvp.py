from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import EdaRunConfig
from kaggle_researcher.contracts.evidence_manifest import PublishedEdaEvidenceBundle
from kaggle_researcher.contracts.hashing import sha256_contract


FIXTURE_ROOT = Path("tests/fixtures/eda")


def test_offline_home_credit_fixture_creates_evidence_pack(tmp_path: Path) -> None:
    result = asyncio.run(_run_fixture("home_credit_tiny", tmp_path / "runs"))

    assert result.evidence_pack_path.is_file()
    assert result.evidence_manifest_path is not None
    assert result.evidence_manifest_path.is_file()
    assert result.published_bundle_path is not None
    assert result.published_bundle_path.is_file()
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))
    bundle = PublishedEdaEvidenceBundle.model_validate_json(
        result.published_bundle_path.read_text(encoding="utf-8")
    )
    assert bundle.evidence_pack.model_dump(mode="json") == payload
    assert bundle.pack_hash == sha256_contract(bundle.evidence_pack)
    assert bundle.evidence_manifest.generated_at_stage == "eda_publication_boundary"
    assert payload["competition_id"] == "home_credit_tiny"
    assert payload["validation_evidence"]["primary_validation"]["method"] == "temporal_holdout"
    assert payload["recommended_next_actions"]
    assert payload["experiment_candidates"] == payload["testable_hypotheses"]
    assert len(payload["testable_hypotheses"]) <= 10
    assert payload["safety_constraints"]
    assert payload["validation_requirements"]
    assert payload["deprecated_outputs"]["experiment_candidates"]["replacement"] == "testable_hypotheses"


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
