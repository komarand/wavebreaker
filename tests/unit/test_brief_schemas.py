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
    claim = Claim(
        claim_id=f"claim_{kind}",
        text="Supported statement.",
        source_ids=[],
        kind=kind,
    )

    assert claim.kind == kind
    assert claim.source_ids == []


def test_claim_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        Claim(
            claim_id="claim_unknown",
            text="Unsupported kind.",
            source_ids=["facts"],
            kind="opinion",
        )


@pytest.mark.parametrize("claim_id", ["", "claim with spaces", "fact_001"])
def test_claim_rejects_unstable_or_blank_identifier(claim_id: str) -> None:
    with pytest.raises(ValidationError):
        Claim(
            claim_id=claim_id,
            text="Statement.",
            source_ids=["facts"],
            kind="fact",
        )


def test_competition_brief_round_trips_through_json() -> None:
    brief = CompetitionBrief(
        competition_id="example-comp",
        thesis="Validation design should drive the initial strategy.",
        thesis_support=["claim_validation", "claim_entities"],
        validation=[
            Claim(
                claim_id="claim_validation",
                text="Grouped validation appears in the strongest notebook lineage.",
                source_ids=["author/notebook"],
                kind="fact",
            )
        ],
        metric_notes=[
            Claim(
                claim_id="claim_metric",
                text="Metric behavior should be verified on real predictions.",
                source_ids=["facts"],
                kind="inference",
            )
        ],
        leakage_risks=[],
        what_works=[
            Claim(
                claim_id="claim_entities",
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


def test_duplicate_claim_ids_across_sections_are_rejected() -> None:
    with pytest.raises(ValidationError, match="globally unique"):
        _brief(
            thesis_support=["claim_shared"],
            validation=[_claim("claim_shared", ["facts"])],
            metric_notes=[_claim("claim_shared", ["topic-1"])],
        )


def test_unknown_thesis_support_claim_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown claim_id"):
        _brief(
            thesis_support=["claim_missing"],
            validation=[_claim("claim_known", ["facts"])],
        )


def test_direct_source_identifier_cannot_replace_claim_identifier() -> None:
    with pytest.raises(ValidationError, match="unknown claim_id"):
        _brief(
            thesis_support=["topic-101"],
            validation=[_claim("claim_validation", ["topic-101"])],
        )


def test_thesis_supporting_claim_requires_a_source() -> None:
    with pytest.raises(ValidationError, match="at least one source_id"):
        _brief(
            thesis_support=["claim_unsupported"],
            validation=[_claim("claim_unsupported", [])],
        )


def test_nonempty_thesis_requires_supporting_claims() -> None:
    with pytest.raises(ValidationError, match="non-empty thesis"):
        _brief(thesis_support=[])


def test_empty_thesis_may_have_no_supporting_claims() -> None:
    brief = _brief(thesis_support=[], thesis="")

    assert brief.thesis == ""
    assert brief.thesis_support == []


def test_thesis_can_reference_claims_from_multiple_sections() -> None:
    brief = _brief(
        thesis_support=["claim_validation", "claim_metric", "claim_leakage"],
        validation=[_claim("claim_validation", ["author/notebook"])],
        metric_notes=[_claim("claim_metric", ["facts"])],
        leakage_risks=[_claim("claim_leakage", ["topic-101"])],
    )

    assert brief.thesis_support == [
        "claim_validation",
        "claim_metric",
        "claim_leakage",
    ]


def test_thesis_claim_source_chain_is_reconstructable() -> None:
    brief = _brief(
        thesis_support=["claim_validation", "claim_entities"],
        validation=[_claim("claim_validation", ["facts", "author/notebook"])],
        what_works=[_claim("claim_entities", ["topic-101"])],
    )
    claims = {
        claim.claim_id: claim
        for section in (
            brief.validation,
            brief.metric_notes,
            brief.leakage_risks,
            brief.what_works,
            brief.time_wasters,
        )
        for claim in section
    }

    assert {
        claim_id: claims[claim_id].source_ids for claim_id in brief.thesis_support
    } == {
        "claim_validation": ["facts", "author/notebook"],
        "claim_entities": ["topic-101"],
    }


def test_competition_brief_requires_every_content_section() -> None:
    with pytest.raises(ValidationError):
        CompetitionBrief.model_validate(
            {
                "competition_id": "example-comp",
                "thesis": "Incomplete brief.",
                "thesis_support": [],
            }
        )


def _brief(
    *,
    thesis_support: list[str],
    thesis: str = "Validation should drive the initial strategy.",
    validation: list[Claim] | None = None,
    metric_notes: list[Claim] | None = None,
    leakage_risks: list[Claim] | None = None,
    what_works: list[Claim] | None = None,
    time_wasters: list[Claim] | None = None,
) -> CompetitionBrief:
    return CompetitionBrief(
        competition_id="example-comp",
        thesis=thesis,
        thesis_support=thesis_support,
        validation=validation or [],
        metric_notes=metric_notes or [],
        leakage_risks=leakage_risks or [],
        what_works=what_works or [],
        time_wasters=time_wasters or [],
        hypotheses=[],
        eda_tasks=[],
        first_moves=[],
        unknowns=[],
        limitations=[],
    )


def _claim(claim_id: str, source_ids: list[str]) -> Claim:
    return Claim(
        claim_id=claim_id,
        text="Grounded statement.",
        source_ids=source_ids,
        kind="fact",
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
