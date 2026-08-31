from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from kaggle_researcher import brief_validate
from kaggle_researcher.brief_schemas import Claim, CompetitionBrief
from kaggle_researcher.facts.models import (
    CompetitionFacts,
    CompetitionMetadata,
    CvLbPair,
    DiscussionFacts,
    FileManifest,
    NotebookFacts,
    ScoreObservation,
    UserConstraints,
)
from kaggle_researcher.research_scout_schemas import EdaTask, ResearchHypothesis


def test_fabricated_source_is_removed_and_recorded() -> None:
    claim = _claim(
        "claim_validation",
        ["facts", "fabricated-source"],
        text="Metric evidence comes from official facts.",
    )
    original = _brief(validation=[claim])

    validated = brief_validate.validate_brief(original, _facts())

    assert validated.validation[0].source_ids == ["facts"]
    assert validated.validation[0].text == claim.text
    assert any(
        "claim_validation" in item and "fabricated-source" in item for item in validated.limitations
    )


def test_claim_with_only_invalid_sources_moves_to_unknowns() -> None:
    unsupported = _claim(
        "claim_unsupported",
        ["made-up"],
        text="An unsupported performance claim.",
    )
    original = _brief(
        validation=[_claim("claim_validation", ["facts"])],
        what_works=[unsupported],
    )

    validated = brief_validate.validate_brief(original, _facts())

    assert validated.what_works == []
    assert "unsupported: An unsupported performance claim." in validated.unknowns
    assert all(
        claim.text != unsupported.text for claim in validated.validation + validated.what_works
    )


@pytest.mark.parametrize(
    "section_name",
    ["validation", "metric_notes", "leakage_risks", "what_works", "time_wasters"],
)
def test_all_claim_sections_apply_the_same_source_validation(section_name: str) -> None:
    kwargs = {section_name: [_claim("claim_section", ["invalid-source"])]}
    if section_name == "validation":
        kwargs["validation"] = [
            _claim("claim_validation", ["facts"]),
            _claim("claim_section", ["invalid-source"]),
        ]

    validated = brief_validate.validate_brief(_brief(**kwargs), _facts())

    section = getattr(validated, section_name)
    assert all(claim.claim_id != "claim_section" for claim in section)
    assert "unsupported: Grounded claim text." in validated.unknowns


def test_facts_notebook_and_discussion_ids_are_valid_sources() -> None:
    facts = _facts(with_sources=True)
    claim = _claim(
        "claim_validation",
        ["facts", "author/notebook", "topic-101"],
    )

    validated = brief_validate.validate_brief(
        _brief(validation=[claim]),
        facts,
    )

    assert validated.validation[0].source_ids == [
        "facts",
        "author/notebook",
        "topic-101",
    ]
    assert validated.limitations == ["Existing limitation."]


@pytest.mark.parametrize("source_id", ["cv_lb", "notebook_ast"])
def test_canonical_trusted_source_id_is_preserved(source_id: str) -> None:
    validated = brief_validate.validate_brief(
        _brief(validation=[_claim("claim_validation", [source_id])]),
        _facts(),
    )

    assert validated.validation[0].source_ids == [source_id]
    assert validated.validation[0].kind == "fact"
    assert validated.limitations == ["Existing limitation."]


@pytest.mark.parametrize("source_type", ["discussion", "winner_writeup"])
def test_discussion_only_fact_is_downgraded_without_rewriting(
    source_type: str,
) -> None:
    source_id = f"topic-{source_type}"
    facts = _facts(discussions=[_discussion(topic_id=source_id, source_type=source_type)])
    claim = _claim(
        "claim_discussion_only",
        [source_id],
        text="The source makes this assertion.",
    )

    validated = brief_validate.validate_brief(
        _brief(
            thesis_support=["claim_discussion_only"],
            validation=[claim],
        ),
        facts,
    )

    result = validated.validation[0]
    assert result.claim_id == claim.claim_id
    assert result.text == claim.text
    assert result.source_ids == claim.source_ids
    assert result.kind == "claim"
    assert (
        "Claim claim_discussion_only was downgraded from fact to claim because it is "
        "supported only by discussion or writeup sources."
    ) in validated.limitations


def test_mixed_trusted_and_discussion_fact_is_not_downgraded() -> None:
    facts = _facts(discussions=[_discussion()])

    validated = brief_validate.validate_brief(
        _brief(validation=[_claim("claim_validation", ["facts", "topic-101"])]),
        facts,
    )

    assert validated.validation[0].kind == "fact"
    assert validated.validation[0].source_ids == ["facts", "topic-101"]
    assert not any("downgraded" in item for item in validated.limitations)


def test_implausible_observation_only_fact_is_downgraded_without_rewriting() -> None:
    facts = _facts(with_sources=True)
    facts.notebooks[0].score_observations = [
        ScoreObservation(
            value=100.0,
            value_raw="100.0",
            metric_raw="ROC AUC",
            metric_canonical=None,
            locator="cell_0",
            raw_text="ROC AUC: 100.0",
            source="markdown",
            plausible=False,
            implausible_reason="value_out_of_range",
        )
    ]
    claim = _claim(
        "claim_bad_observation",
        [facts.notebooks[0].ref],
        text="The reported score is 100.0.",
    )

    validated = brief_validate.validate_brief(
        _brief(validation=[claim], thesis_support=[claim.claim_id]),
        facts,
    )

    result = validated.validation[0]
    assert result.kind == "claim"
    assert result.text == claim.text
    assert result.source_ids == claim.source_ids
    assert (
        "Claim claim_bad_observation was downgraded from fact to claim: supporting "
        "observations failed plausibility checks (value_out_of_range)."
    ) in validated.limitations


def test_validation_returns_a_new_brief_without_mutating_input() -> None:
    original = _brief(validation=[_claim("claim_validation", ["facts", "invalid-source"])])

    validated = brief_validate.validate_brief(original, _facts())

    assert validated is not original
    assert original.validation[0].source_ids == ["facts", "invalid-source"]
    assert original.limitations == ["Existing limitation."]


def test_claim_stats_describe_only_final_validated_claims() -> None:
    facts = _facts(with_sources=True)
    invalid = _claim("claim_removed", ["invalid-source"])
    original = _brief(
        validation=[_claim("claim_validation", ["facts"])],
        metric_notes=[
            _claim("claim_metric", ["topic-101"]).model_copy(
                update={"kind": "claim"}
            )
        ],
        leakage_risks=[
            _claim("claim_leakage", ["author/notebook"]).model_copy(
                update={"kind": "inference", "evidence_strength": "inference"}
            )
        ],
        what_works=[invalid],
        time_wasters=[_claim("claim_waste", ["facts", "topic-101"])],
    )

    validated = brief_validate.validate_brief(original, facts)

    assert validated.claim_stats is not None
    assert validated.claim_stats.model_dump() == {
        "fact": 2,
        "claim": 1,
        "inference": 1,
        "total": 4,
        "grounded": 4,
        "ungrounded": 0,
        "grounding_rate": 1.0,
        "distinct_sources": 3,
        "by_evidence_strength": {
            "measured_with_protocol": 3,
            "reported_score": 0,
            "prevalence": 0,
            "inference": 1,
        },
        "hypotheses_total": 0,
        "hypotheses_dropped_unverifiable": 0,
    }
    assert validated.what_works == []
    assert "unsupported: Grounded claim text." in validated.unknowns
    assert original.claim_stats is None


def test_empty_validated_brief_has_zero_grounding_rate() -> None:
    original = _brief(
        thesis="",
        thesis_support=[],
        validation=[],
    )

    validated = brief_validate.validate_brief(original, _facts())

    assert validated.claim_stats is not None
    assert validated.claim_stats.total == 0
    assert validated.claim_stats.grounded == 0
    assert validated.claim_stats.ungrounded == 0
    assert validated.claim_stats.grounding_rate == 0.0
    assert validated.claim_stats.distinct_sources == 0
    assert sum(validated.claim_stats.by_evidence_strength.values()) == 0
    assert validated.limitations == original.limitations


def test_insufficient_cv_lb_reliability_is_recorded_deterministically() -> None:
    facts = _facts()
    facts.cv_lb_pairs = [
        CvLbPair(
            notebook_ref="author/notebook",
            declared_cv=0.8382,
            public_score=0.797,
            lineage_cluster_id="lineage_one",
        )
    ]

    validated = brief_validate.validate_brief(_brief(), facts)

    assert (
        "CV/LB reliability is insufficient (n=1): single pair; gap is not evidence "
        "of a systematic pattern."
    ) in validated.limitations


def test_thesis_is_rejected_when_a_number_is_absent_from_support_and_facts() -> None:
    claim = _claim(
        "claim_cv_lb_gap",
        ["facts"],
        text="The observed pair was CV 0.8382 and LB 0.797.",
    )
    original = _brief(
        thesis="Scores reach up to ~0.94.",
        thesis_support=[claim.claim_id],
        validation=[claim],
    )

    validated = brief_validate.validate_brief(original, _facts())

    assert validated.thesis == ""
    assert validated.thesis_support == []
    assert "unsupported: Scores reach up to ~0.94." in validated.unknowns
    assert any(
        "numeric literals were absent" in item and "0.94" in item for item in validated.limitations
    )
    assert validated.validation == [claim]


def test_equivalent_number_spelling_in_support_keeps_thesis() -> None:
    claim = _claim(
        "claim_score",
        ["facts"],
        text="A supported source reports a score of 0.940.",
    )
    original = _brief(
        thesis="The supported score is ~0.94.",
        thesis_support=[claim.claim_id],
        validation=[claim],
    )

    validated = brief_validate.validate_brief(original, _facts())

    assert validated.thesis == original.thesis
    assert validated.thesis_support == [claim.claim_id]


def test_number_present_in_competition_facts_keeps_thesis() -> None:
    facts = _facts()
    facts.metadata.submissions_per_day = 5
    claim = _claim(
        "claim_submission_limit",
        ["facts"],
        text="The official submission limit applies.",
    )
    original = _brief(
        thesis="The competition allows 5 submissions per day.",
        thesis_support=[claim.claim_id],
        validation=[claim],
    )

    validated = brief_validate.validate_brief(original, facts)

    assert validated.thesis == original.thesis
    assert validated.thesis_support == [claim.claim_id]


def test_insufficient_numeric_cv_lb_claim_cannot_support_thesis() -> None:
    facts = _facts()
    facts.cv_lb_pairs = [
        CvLbPair(
            notebook_ref="author/notebook",
            declared_cv=0.8382,
            public_score=0.797,
            lineage_cluster_id="lineage_one",
        )
    ]
    gap_claim = _claim(
        "claim_cv_lb_gap",
        ["cv_lb"],
        text="CV 0.8382 exceeds LB 0.797 by 0.0412.",
    )
    original = _brief(
        thesis="CV reaches 0.8382.",
        thesis_support=[gap_claim.claim_id],
        metric_notes=[gap_claim],
    )

    validated = brief_validate.validate_brief(original, facts)

    assert validated.thesis == ""
    assert validated.thesis_support == []
    assert validated.metric_notes == [gap_claim]
    assert original.thesis == "CV reaches 0.8382."
    assert any(
        "claim_cv_lb_gap" in item and "reliability=insufficient" in item
        for item in validated.limitations
    )


def test_reliable_numeric_claim_can_remain_after_insufficient_support_is_dropped() -> None:
    facts = _facts()
    facts.cv_lb_pairs = [
        CvLbPair(
            notebook_ref="author/notebook",
            declared_cv=0.8382,
            public_score=0.797,
            lineage_cluster_id="lineage_one",
        )
    ]
    gap_claim = _claim(
        "claim_cv_lb_gap",
        ["cv_lb"],
        text="One CV/LB pair has a gap of 0.0412.",
    )
    reliable_claim = _claim(
        "claim_notebook_score",
        ["facts"],
        text="A separate supported observation reports 0.940.",
    )
    original = _brief(
        thesis="A supported score reaches 0.94.",
        thesis_support=[gap_claim.claim_id, reliable_claim.claim_id],
        validation=[reliable_claim],
        metric_notes=[gap_claim],
    )

    validated = brief_validate.validate_brief(original, facts)

    assert validated.thesis == original.thesis
    assert validated.thesis_support == [reliable_claim.claim_id]
    assert validated.metric_notes == [gap_claim]


def test_removed_thesis_support_is_dropped_and_thesis_moves_to_unknowns() -> None:
    original = _brief(
        thesis="Unsupported thesis text.",
        thesis_support=["claim_validation"],
        validation=[_claim("claim_validation", ["fabricated-source"])],
    )

    validated = brief_validate.validate_brief(original, _facts())

    assert validated.thesis == ""
    assert validated.thesis_support == []
    assert validated.validation == []
    assert "unsupported: Unsupported thesis text." in validated.unknowns
    assert any("Thesis was moved to unknowns" in item for item in validated.limitations)


def test_thesis_keeps_remaining_supported_claim_identifier() -> None:
    original = _brief(
        thesis_support=["claim_validation", "claim_secondary"],
        validation=[
            _claim("claim_validation", ["invalid-source"]),
            _claim("claim_secondary", ["facts"]),
        ],
    )

    validated = brief_validate.validate_brief(original, _facts())

    assert validated.thesis == original.thesis
    assert validated.thesis_support == ["claim_secondary"]
    assert [claim.claim_id for claim in validated.validation] == ["claim_secondary"]


def test_hypotheses_and_eda_tasks_remain_pydantic_valid_and_unchanged() -> None:
    hypothesis = _hypothesis()
    eda_task = _eda_task()
    original = _brief(hypotheses=[hypothesis], eda_tasks=[eda_task])

    validated = brief_validate.validate_brief(original, _facts())

    assert validated.hypotheses == [hypothesis]
    assert validated.eda_tasks == [eda_task]
    assert isinstance(validated.hypotheses[0], ResearchHypothesis)
    assert isinstance(validated.eda_tasks[0], EdaTask)


def test_reported_score_fact_is_downgraded_without_rewriting_text() -> None:
    claim = _claim(
        "claim_reported",
        ["facts"],
        text="A notebook reports LB 0.805.",
        evidence_strength="reported_score",
    )

    validated = brief_validate.validate_brief(
        _brief(validation=[claim], thesis_support=[claim.claim_id]),
        _facts(),
    )

    result = validated.validation[0]
    assert result.kind == "claim"
    assert result.text == claim.text
    assert result.evidence_strength == "reported_score"
    assert any(
        "claim_reported" in item and "evidence_strength='reported_score'" in item
        for item in validated.limitations
    )


def test_inference_with_prevalence_evidence_is_downgraded() -> None:
    claim = _claim(
        "claim_mismatch",
        ["facts"],
        evidence_strength="prevalence",
    ).model_copy(update={"kind": "inference"})

    validated = brief_validate.validate_brief(
        _brief(validation=[claim], thesis_support=[claim.claim_id]),
        _facts(),
    )

    assert validated.validation[0].kind == "claim"
    assert validated.validation[0].evidence_strength == "prevalence"


def test_claim_stats_evidence_strength_counts_sum_to_total() -> None:
    claims = [
        _claim("claim_measured", ["facts"]),
        _claim(
            "claim_reported",
            ["facts"],
            evidence_strength="reported_score",
        ).model_copy(update={"kind": "claim"}),
        _claim(
            "claim_prevalence",
            ["facts"],
            evidence_strength="prevalence",
        ),
        _claim(
            "claim_inference",
            ["facts"],
            evidence_strength="inference",
        ).model_copy(update={"kind": "inference"}),
    ]

    validated = brief_validate.validate_brief(
        _brief(validation=claims, thesis_support=["claim_measured"]),
        _facts(),
    )

    assert validated.claim_stats is not None
    assert sum(validated.claim_stats.by_evidence_strength.values()) == (
        validated.claim_stats.total
    )


def test_hypothesis_missing_success_condition_moves_to_unknowns() -> None:
    hypothesis = _hypothesis().model_copy(update={"success_condition": " "})

    validated = brief_validate.validate_brief(
        _brief(hypotheses=[hypothesis]),
        _facts(),
    )

    assert validated.hypotheses == []
    assert f"unverifiable hypothesis: {hypothesis.claim}" in validated.unknowns
    assert any(
        hypothesis.id in item and "success_condition" in item
        for item in validated.limitations
    )
    assert validated.claim_stats is not None
    assert validated.claim_stats.hypotheses_total == 1
    assert validated.claim_stats.hypotheses_dropped_unverifiable == 1


def test_nonquantitative_hypothesis_moves_to_unknowns() -> None:
    hypothesis = _hypothesis().model_copy(
        update={
            "success_condition": "Improves CV.",
            "failure_condition": "Does not improve CV.",
        }
    )

    validated = brief_validate.validate_brief(
        _brief(hypotheses=[hypothesis]),
        _facts(),
    )

    assert validated.hypotheses == []
    assert f"unverifiable hypothesis: {hypothesis.claim}" in validated.unknowns
    assert any("non-quantitative acceptance criteria" in item for item in validated.limitations)


def test_quantitative_hypothesis_is_retained() -> None:
    hypothesis = _hypothesis()

    validated = brief_validate.validate_brief(
        _brief(hypotheses=[hypothesis]),
        _facts(),
    )

    assert validated.hypotheses == [hypothesis]
    assert validated.claim_stats is not None
    assert validated.claim_stats.hypotheses_total == 1
    assert validated.claim_stats.hypotheses_dropped_unverifiable == 0


def test_unspecified_objective_is_recorded_without_rejecting_thesis() -> None:
    facts = _facts()
    original = _brief(thesis="Pursue a medal with robust validation.")

    validated = brief_validate.validate_brief(original, facts)

    assert validated.thesis == original.thesis
    assert (
        "Thesis names a competition objective that user_constraints.objective does not specify."
        in validated.limitations
    )


def test_explicit_objective_does_not_create_unspecified_objective_limitation() -> None:
    facts = _facts()
    facts.user_constraints = UserConstraints(objective="medal")
    validated = brief_validate.validate_brief(
        _brief(thesis="Pursue a medal with robust validation."),
        facts,
    )

    assert not any(
        "objective that user_constraints.objective does not specify" in item
        for item in validated.limitations
    )


def test_validator_contains_no_content_rewriting_helpers() -> None:
    source = inspect.getsource(brief_validate)

    for forbidden in ("enforce_", "correct_", "cleanup_", "_rewrite_"):
        assert forbidden not in source


def _brief(
    *,
    thesis: str = "Use the official metric to anchor validation.",
    thesis_support: list[str] | None = None,
    validation: list[Claim] | None = None,
    metric_notes: list[Claim] | None = None,
    leakage_risks: list[Claim] | None = None,
    what_works: list[Claim] | None = None,
    time_wasters: list[Claim] | None = None,
    hypotheses: list[ResearchHypothesis] | None = None,
    eda_tasks: list[EdaTask] | None = None,
) -> CompetitionBrief:
    validation_claims = validation
    if validation_claims is None:
        validation_claims = [_claim("claim_validation", ["facts"])]
    support = thesis_support
    if support is None:
        support = [validation_claims[0].claim_id]
    return CompetitionBrief(
        competition_id="current-comp",
        thesis=thesis,
        thesis_support=support,
        validation=validation_claims,
        metric_notes=metric_notes or [],
        leakage_risks=leakage_risks or [],
        what_works=what_works or [],
        time_wasters=time_wasters or [],
        hypotheses=hypotheses or [],
        eda_tasks=eda_tasks or [],
        first_moves=[],
        unknowns=[],
        limitations=["Existing limitation."],
    )


def _claim(
    claim_id: str,
    source_ids: list[str],
    *,
    text: str = "Grounded claim text.",
    evidence_strength: str = "measured_with_protocol",
) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        source_ids=source_ids,
        kind="fact",
        evidence_strength=evidence_strength,
    )


def _facts(
    *,
    with_sources: bool = False,
    discussions: list[DiscussionFacts] | None = None,
) -> CompetitionFacts:
    notebooks = [_notebook()] if with_sources else []
    fact_discussions = discussions
    if fact_discussions is None:
        fact_discussions = [_discussion()] if with_sources else []
    return CompetitionFacts(
        competition_id="current-comp",
        collected_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        metadata=CompetitionMetadata(
            competition_id="current-comp",
            metric_name="roc_auc",
            is_code_competition=True,
            unavailable_fields=[],
        ),
        files=FileManifest(
            files=[],
            sample_submission_columns=[],
            sample_submission_source="unavailable",
            limitations=[],
        ),
        notebooks=notebooks,
        discussions=fact_discussions,
        similar_competitions=[],
        cv_lb_pairs=[],
        user_constraints=UserConstraints(),
        collection_errors=[],
    )


def _notebook() -> NotebookFacts:
    return NotebookFacts(
        ref="author/notebook",
        title="Notebook",
        ast_fingerprint="a" * 64,
        lineage_cluster_id="lineage_one",
        splitters=[],
        models=[],
        metrics=[],
        feature_ops=[],
        declared_cv=[],
        parse_status="ok",
    )


def _discussion(
    *,
    topic_id: str = "topic-101",
    source_type: str = "discussion",
) -> DiscussionFacts:
    return DiscussionFacts(
        topic_id=topic_id,
        title="Discussion",
        author="author",
        author_is_host=False,
        votes=1,
        source_type=source_type,
        competition_id=("current-comp" if source_type == "discussion" else "past-comp"),
        text="Discussion text.",
    )


def _hypothesis() -> ResearchHypothesis:
    return ResearchHypothesis(
        id="val_001",
        category="validation",
        priority="P0",
        claim="Grouped validation may be required.",
        why_it_matters="Entity overlap may inflate validation.",
        how_to_verify=["Measure entity overlap between folds."],
        provenance=["heuristic", "not_verified_on_data"],
        supporting_source_ids=["facts"],
        confidence="medium",
        success_condition="OOF improves by at least 0.003 on 3 seeds.",
        failure_condition="OOF improves by less than 0.003 on 3 seeds.",
    )


def _eda_task() -> EdaTask:
    return EdaTask(
        id="eda_val_001",
        priority="P0",
        module="validation_analyzer",
        question="Which column defines independent validation groups?",
        rationale="Independent groups are needed for validation.",
        expected_outputs=["validation_evidence.group_column"],
        related_hypothesis_ids=["val_001"],
        blocking=True,
    )
