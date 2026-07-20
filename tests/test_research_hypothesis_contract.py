from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_researcher.contracts.research_hypotheses import (
    ResearchHypotheses,
    UnsupportedSchemaVersionError,
    load_research_hypotheses,
    migrate_research_hypotheses_payload,
    write_research_hypotheses_atomic,
)
from kaggle_researcher.eda.schemas import ResearchHypotheses as EdaResearchHypotheses
from kaggle_researcher.orchestration.full_run import _canonicalize_research_artifact_bundle


pytestmark = pytest.mark.contract


def _legacy_payload(category: str = "relationships") -> dict[str, object]:
    return {
        "competition_id": "demo",
        "hypotheses": [{
            "id": "rel_001",
            "category": category,
            "claim": "Entity links may need group-safe validation.",
            "priority": "P1",
            "confidence": "high",
            "status": "untested",
            "how_to_verify": ["relationship_inferer.keys"],
        }],
    }


def test_canonical_contract_is_shared_with_eda() -> None:
    assert EdaResearchHypotheses is ResearchHypotheses
    canonical = ResearchHypotheses.model_validate({
        "schema_version": "1.0", "competition_id": "demo", "hypotheses": [{
            "hypothesis_id": "rel_001", "category": "relationship", "claim": "Check links.",
            "priority": "P1", "confidence_before_eda": "medium",
        }],
    })
    assert canonical.hypotheses[0].hypothesis_id == "rel_001"


@pytest.mark.parametrize(("legacy", "canonical"), [
    ("relationships", "relationship"),
    ("feature_engineering", "feature"),
    ("dataset_schema", "schema"),
    ("notebook_reverse_engineering", "notebook"),
])
def test_migration_normalizes_known_legacy_aliases(legacy: str, canonical: str) -> None:
    result = migrate_research_hypotheses_payload(_legacy_payload(legacy))

    parsed = ResearchHypotheses.model_validate(result.canonical_payload)
    hypothesis = parsed.hypotheses[0]
    assert hypothesis.hypothesis_id == "rel_001"
    assert hypothesis.category == canonical
    assert hypothesis.confidence_before_eda == "high"
    assert hypothesis.status == "needs_eda"
    assert result.migrated is True


def test_migration_defaults_missing_confidence_and_rejects_conflicting_ids() -> None:
    payload = _legacy_payload()
    hypothesis = payload["hypotheses"][0]
    assert isinstance(hypothesis, dict)
    hypothesis.pop("confidence")
    result = migrate_research_hypotheses_payload(payload)
    assert result.canonical_payload["hypotheses"][0]["confidence_before_eda"] == "medium"
    assert result.warnings == ["hypotheses[0].confidence_before_eda defaulted to medium"]

    hypothesis["hypothesis_id"] = "different"
    with pytest.raises(ValueError, match="conflicting"):
        migrate_research_hypotheses_payload(payload)


def test_unknown_categories_are_not_fuzzy_mapped_and_future_versions_fail() -> None:
    with pytest.raises(Exception):
        ResearchHypotheses.model_validate(migrate_research_hypotheses_payload(_legacy_payload("magical_features")).canonical_payload)
    with pytest.raises(UnsupportedSchemaVersionError):
        migrate_research_hypotheses_payload({"schema_version": "99.0", "competition_id": "demo"})


def test_full_run_migrates_legacy_artifact_with_backup(tmp_path: Path) -> None:
    research = tmp_path / "research"; research.mkdir()
    path = research / "research_hypotheses.json"
    path.write_text(json.dumps(_legacy_payload()), encoding="utf-8")
    (research / "eda_task_plan.json").write_text(json.dumps({
        "competition_id": "demo",
        "eda_tasks": [{"id": "eda_relationship_001", "module": "relationship_inferer", "priority": "P1", "related_hypothesis_ids": ["rel_001"]}],
        "hypothesis_index": {"rel_001": {"category": "relationships"}},
    }), encoding="utf-8")

    _canonicalize_research_artifact_bundle(research)

    canonical, migration = load_research_hypotheses(path)
    assert canonical.schema_version == "1.0"
    assert canonical.hypotheses[0].category == "relationship"
    assert migration.migrated is False
    assert (research / "research_hypotheses.legacy.json").is_file()
    assert (research / "eda_task_plan.legacy.json").is_file()
    assert (research / "research_artifact_migrations.json").is_file()


def test_atomic_canonical_write_exposes_only_canonical_fields(tmp_path: Path) -> None:
    result = migrate_research_hypotheses_payload(_legacy_payload())
    canonical = ResearchHypotheses.model_validate(result.canonical_payload)
    path = tmp_path / "research_hypotheses.json"
    write_research_hypotheses_atomic(path, canonical)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert "id" not in payload["hypotheses"][0]
    assert not path.with_suffix(".json.tmp").exists()
