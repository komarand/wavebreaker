from __future__ import annotations

from typing import Any

import pytest

from kaggle_researcher.config import get_max_similar_verifications
from kaggle_researcher.facts.models import (
    CompetitionMetadata,
    DiscussionFacts,
    DiscussionLink,
    DiscussionMessageFacts,
    SimilarCompetition,
)
from kaggle_researcher.facts.similar import (
    build_similar_candidates,
    collect_mentions,
    verify_candidate,
)


def test_collect_mentions_counts_topics_and_total_and_excludes_self() -> None:
    discussions = [
        _discussion(
            "topic-2",
            "competitions/happywhale and competitions/current-comp",
        ),
        _discussion(
            "topic-1",
            "competitions/happywhale then competitions/happywhale",
        ),
    ]

    candidates = collect_mentions(discussions, "CURRENT-COMP")

    assert len(candidates) == 1
    assert candidates[0].slug == "happywhale"
    assert candidates[0].evidence_topic_ids == ["topic-1", "topic-2"]
    assert candidates[0].mention_topic_count == 2
    assert candidates[0].mention_total == 3


def test_collect_mentions_reads_structured_competition_links() -> None:
    discussion = _discussion("topic-1", "link source")
    discussion.messages = [
        DiscussionMessageFacts(
            evidence_id="message:1",
            message_id="1",
            topic_id="topic-1",
            content_text="link source",
            content_sha256="a" * 64,
            links=[
                DiscussionLink(
                    url="https://www.kaggle.com/c/happywhale",
                    kind="kaggle",
                    competition_slug="happywhale",
                )
            ],
        )
    ]

    candidates = collect_mentions([discussion], "self")

    assert candidates[0].slug == "happywhale"
    assert candidates[0].mention_total == 1


def test_verify_candidate_accepts_canonical_metric_match() -> None:
    candidate = SimilarCompetition(slug="candidate", discovered_by="discussion_mention")

    verified = verify_candidate(
        candidate,
        _metadata("self", "Roc Auc Score", category="playground"),
        ["id", "target"],
        metadata_fetcher=lambda slug: _metadata(
            slug,
            "AUC",
            category="playground",
        ),
    )

    assert verified.verification == "verified"
    assert verified.evidence.same_metric is True
    assert verified.evidence.same_submission_shape is None
    assert verified.evidence.metric_self == "roc_auc"
    assert verified.evidence.metric_candidate == "roc_auc"
    assert verified.evidence.matched_features == [
        "same_metric",
        "same_code_competition",
        "same_category",
    ]


def test_verify_candidate_rejects_metric_mismatch_with_both_metrics() -> None:
    rejected = verify_candidate(
        SimilarCompetition(slug="candidate", discovered_by="discussion_mention"),
        _metadata("self", "roc_auc"),
        [],
        metadata_fetcher=lambda slug: _metadata(slug, "rmse"),
    )

    assert rejected.verification == "rejected"
    assert rejected.evidence.same_metric is False
    assert rejected.rejection_reason == "metric mismatch: roc_auc vs rmse"


def test_verify_candidate_keeps_unknown_candidate_metric_as_none() -> None:
    rejected = verify_candidate(
        SimilarCompetition(slug="candidate", discovered_by="discussion_mention"),
        _metadata("self", "roc_auc"),
        [],
        metadata_fetcher=lambda slug: _metadata(slug, None),
    )

    assert rejected.verification == "rejected"
    assert rejected.evidence.same_metric is None
    assert rejected.rejection_reason == "metric unknown on candidate"


def test_verify_candidate_marks_missing_competition_not_found() -> None:
    result = verify_candidate(
        SimilarCompetition(slug="missing", discovered_by="discussion_mention"),
        _metadata("self", "roc_auc"),
        [],
        metadata_fetcher=lambda slug: _missing_metadata(slug),
    )

    assert result.verification == "not_found"
    assert result.rejection_reason == "competition not found"


def test_manual_candidates_are_verified_outside_limit_and_merge_mentions() -> None:
    discussions = [
        _discussion("topic-1", "competitions/manual-comp competitions/auto-one"),
        _discussion("topic-2", "competitions/auto-two"),
    ]
    lookups: list[str] = []

    def metadata(slug: str, api: Any | None = None) -> CompetitionMetadata:
        lookups.append(slug)
        return _metadata(slug, "roc_auc")

    candidates, diagnostics = build_similar_candidates(
        discussions,
        _metadata("self", "roc_auc"),
        [],
        manual=["manual-comp"],
        max_verifications=1,
        metadata_fetcher=metadata,
    )

    by_slug = {candidate.slug: candidate for candidate in candidates}
    assert lookups == ["manual-comp", "auto-one"]
    assert by_slug["manual-comp"].confirmed is True
    assert by_slug["manual-comp"].discovered_by == "manual"
    assert by_slug["manual-comp"].mention_topic_count == 1
    assert by_slug["auto-one"].verification == "verified"
    assert by_slug["auto-two"].verification == "unchecked"
    assert diagnostics.metadata_lookups == 2
    assert diagnostics.status == "found_by_mention"


def test_empty_discussions_return_no_candidates_diagnostic() -> None:
    candidates, diagnostics = build_similar_candidates(
        [],
        _metadata("self", "roc_auc"),
        [],
        manual=[],
        max_verifications=10,
    )

    assert candidates == []
    assert diagnostics.status == "no_candidates"
    assert diagnostics.metadata_lookups == 0


def test_all_metric_mismatches_return_all_rejected_diagnostic() -> None:
    candidates, diagnostics = build_similar_candidates(
        [_discussion("topic-1", "competitions/other-comp")],
        _metadata("self", "roc_auc"),
        [],
        manual=[],
        max_verifications=10,
        metadata_fetcher=lambda slug: _metadata(slug, "rmse"),
    )

    assert candidates[0].verification == "rejected"
    assert diagnostics.status == "all_rejected"
    assert diagnostics.rejected == 1


def test_max_similar_verifications_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_SIMILAR_VERIFICATIONS", "7")

    assert get_max_similar_verifications() == 7


def _discussion(topic_id: str, text: str) -> DiscussionFacts:
    return DiscussionFacts(
        topic_id=topic_id,
        title=topic_id,
        source_type="discussion",
        competition_id="self",
        text=text,
    )


def _metadata(
    slug: str,
    metric: str | None,
    *,
    category: str | None = "playground",
) -> CompetitionMetadata:
    unavailable = ["metric_name"] if metric is None else []
    return CompetitionMetadata(
        competition_id=slug,
        title=slug,
        metric_name=metric,
        metric_status="available" if metric is not None else "unavailable",
        is_code_competition=True,
        category=category,
        unavailable_fields=unavailable,
    )


def _missing_metadata(slug: str) -> CompetitionMetadata:
    return CompetitionMetadata(
        competition_id=slug,
        metric_status="unavailable",
        unavailable_fields=[
            "title",
            "metric_name",
            "is_code_competition",
            "submissions_per_day",
            "max_team_size",
            "deadline",
            "reward",
            "category",
            "num_teams",
        ],
    )
