from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaggle_researcher.brief_schemas import (
    Claim,
    CompetitionBrief,
    EdaTask as BriefEdaTask,
    ResearchHypothesis as BriefResearchHypothesis,
)
from kaggle_researcher.research_scout_schemas import EdaTask, ResearchHypothesis


def test_brief_schemas_reuse_research_scout_classes() -> None:
    assert BriefResearchHypothesis is ResearchHypothesis
    assert BriefEdaTask is EdaTask


@pytest.mark.parametrize("kind", ["fact", "claim", "inference"])
def test_claim_accepts_all_declared_kinds(kind: str) -> None:
    claim = Claim(text="Supported statement.", source_ids=[], kind=kind)

    assert claim.kind == kind
    assert claim.source_ids == []


def test_claim_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        Claim(text="Unsupported kind.", source_ids=["facts"], kind="opinion")


def test_competition_brief_round_trips_through_json() -> None:
    brief = CompetitionBrief(
        competition_id="example-comp",
        thesis="Validation design should drive the initial strategy.",
        thesis_support=["facts", "topic-101"],
        validation=[
            Claim(
                text="Grouped validation appears in the strongest notebook lineage.",
                source_ids=["author/notebook"],
                kind="fact",
            )
        ],
        metric_notes=[
            Claim(
                text="Metric behavior should be verified on real predictions.",
                source_ids=["facts"],
                kind="inference",
            )
        ],
        leakage_risks=[],
        what_works=[
            Claim(
                text="Entity aggregates are repeatedly discussed.",
                source_ids=["topic-101"],
                kind="claim",
            )
        ],
        time_wasters=[],
        hypotheses=[_hypothesis()],
        eda_tasks=[_eda_task()],
        first_moves=["Audit entity overlap across candidate folds."],
        unknowns=["The true temporal boundary is not yet known."],
        limitations=["No train or test data was downloaded."],
    )

    restored = CompetitionBrief.model_validate_json(brief.model_dump_json())

    assert restored == brief
    assert restored.schema_version == "1.0"
    assert isinstance(restored.hypotheses[0], ResearchHypothesis)
    assert isinstance(restored.eda_tasks[0], EdaTask)


def test_competition_brief_requires_every_content_section() -> None:
    with pytest.raises(ValidationError):
        CompetitionBrief.model_validate(
            {
                "competition_id": "example-comp",
                "thesis": "Incomplete brief.",
                "thesis_support": [],
            }
        )


def _hypothesis() -> ResearchHypothesis:
    return ResearchHypothesis(
        id="val_001",
        category="validation",
        priority="P0",
        claim="Grouped validation is required.",
        why_it_matters="Entity overlap may inflate random cross-validation.",
        how_to_verify=["Measure entity overlap between folds."],
        provenance=["kaggle"],
        supporting_source_ids=["author/notebook"],
        confidence="high",
    )


def _eda_task() -> EdaTask:
    return EdaTask(
        id="eda_val_001",
        priority="P0",
        module="validation_analyzer",
        question="Which entity column defines independent groups?",
        rationale="Independent groups are required for trustworthy validation.",
        required_inputs=["train"],
        expected_outputs=["validation_evidence.group_column"],
        related_hypothesis_ids=["val_001"],
        blocking=True,
    )
