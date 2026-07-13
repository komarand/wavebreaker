from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field

from kaggle_researcher.contracts.normalization import HYPOTHESIS_CATEGORY_ALIASES


SCHEMA_VERSION = "1.0"
HypothesisCategory = Literal[
    "schema", "metric", "validation", "leakage", "relationship", "drift",
    "baseline", "feature", "notebook", "leaderboard", "data_quality",
]
Confidence = Literal["low", "medium", "high"]
Priority = Literal["P0", "P1", "P2", "P3"]
HypothesisStatus = Literal["needs_eda", "supported_by_source", "analogous_only", "not_testable"]
ALLOWED_HYPOTHESIS_CATEGORIES = tuple(HypothesisCategory.__args__)

LEGACY_CATEGORY_ALIASES = HYPOTHESIS_CATEGORY_ALIASES
LEGACY_STATUS_ALIASES = {
    "needs_eda": "needs_eda",
    "pending": "needs_eda",
    "untested": "needs_eda",
    "source_supported": "supported_by_source",
    "supported_by_source": "supported_by_source",
    "heuristic": "analogous_only",
    "analogous_only": "analogous_only",
    "not_testable": "not_testable",
}


class UnsupportedSchemaVersionError(ValueError):
    pass


class ResearchHypothesis(BaseModel):
    hypothesis_id: str
    category: HypothesisCategory
    claim: str
    rationale: str | None = None
    expected_eda_checks: list[str] = Field(default_factory=list)
    priority: Priority = "P1"
    confidence_before_eda: Confidence
    source_refs: list[str] = Field(default_factory=list)
    status: HypothesisStatus = "needs_eda"


class ResearchHypotheses(BaseModel):
    schema_version: str = SCHEMA_VERSION
    competition_id: str
    created_at: str | None = None
    hypotheses: list[ResearchHypothesis] = Field(default_factory=list)
    eda_tasks: list[dict[str, Any]] = Field(default_factory=list)
    structured_findings: list[dict[str, Any]] = Field(default_factory=list)
    scout_limitations: list[str] = Field(default_factory=list)
    models_used: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class HypothesisMigrationResult:
    canonical_payload: dict[str, Any]
    source_schema_version: str | None
    target_schema_version: str
    migrated: bool
    applied_migrations: list[str]
    warnings: list[str]


def migrate_research_hypotheses_payload(payload: Mapping[str, Any]) -> HypothesisMigrationResult:
    source_version = payload.get("schema_version")
    if source_version not in {None, SCHEMA_VERSION}:
        raise UnsupportedSchemaVersionError(
            f"Unsupported ResearchHypotheses schema version: {source_version!r}."
        )
    migrations: list[str] = []
    warnings: list[str] = []
    canonical: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "competition_id": payload.get("competition_id"),
        "created_at": payload.get("created_at"),
        "hypotheses": [],
        "eda_tasks": list(payload.get("eda_tasks") or []),
        "structured_findings": list(payload.get("structured_findings") or []),
        "scout_limitations": list(payload.get("scout_limitations") or []),
        "models_used": dict(payload.get("models_used") or {}),
    }
    if source_version is None:
        migrations.append("schema_version:missing->1.0")
    for index, item in enumerate(payload.get("hypotheses") or []):
        if not isinstance(item, Mapping):
            canonical["hypotheses"].append(item)
            continue
        value = dict(item)
        hypothesis_id = value.get("hypothesis_id")
        legacy_id = value.get("id")
        if hypothesis_id is None and legacy_id is not None:
            hypothesis_id = legacy_id
            migrations.append(f"hypotheses[{index}].id->hypothesis_id")
        elif hypothesis_id is not None and legacy_id is not None and hypothesis_id != legacy_id:
            raise ValueError(
                f"hypotheses[{index}] has conflicting id={legacy_id!r} and hypothesis_id={hypothesis_id!r}."
            )
        category = value.get("category")
        if category in LEGACY_CATEGORY_ALIASES:
            category = LEGACY_CATEGORY_ALIASES[category]
            migrations.append(f"hypotheses[{index}].category:{value.get('category')}->{category}")
        confidence = value.get("confidence_before_eda")
        if confidence is None:
            legacy_confidence = value.get("confidence", value.get("source_confidence"))
            if legacy_confidence in {"low", "medium", "high"}:
                confidence = legacy_confidence
                migrations.append(f"hypotheses[{index}].confidence->confidence_before_eda")
            else:
                confidence = "medium"
                warnings.append(f"hypotheses[{index}].confidence_before_eda defaulted to medium")
        status = value.get("status", "needs_eda")
        if status in LEGACY_STATUS_ALIASES:
            normalized_status = LEGACY_STATUS_ALIASES[status]
            if normalized_status != status:
                migrations.append(f"hypotheses[{index}].status:{status}->{normalized_status}")
            status = normalized_status
        claim = value.get("claim") or value.get("statement") or value.get("hypothesis") or value.get("text")
        if claim != value.get("claim"):
            migrations.append(f"hypotheses[{index}].text_alias->claim")
        canonical["hypotheses"].append({
            "hypothesis_id": hypothesis_id,
            "category": category,
            "claim": claim,
            "rationale": value.get("rationale") or value.get("why_it_matters"),
            "expected_eda_checks": value.get("expected_eda_checks") or value.get("how_to_verify") or [],
            "priority": value.get("priority") or "P1",
            "confidence_before_eda": confidence,
            "source_refs": value.get("source_refs") or value.get("supporting_source_ids") or [],
            "status": status,
        })
    migrated = bool(migrations or warnings)
    return HypothesisMigrationResult(canonical, source_version, SCHEMA_VERSION, migrated, migrations, warnings)


def load_research_hypotheses(path: Path) -> tuple[ResearchHypotheses, HypothesisMigrationResult]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    migration = migrate_research_hypotheses_payload(payload)
    return ResearchHypotheses.model_validate(migration.canonical_payload), migration


def write_research_hypotheses_atomic(path: Path, hypotheses: ResearchHypotheses) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(hypotheses.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
